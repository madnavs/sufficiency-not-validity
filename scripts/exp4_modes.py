"""Mitigation probe (paper section 8): label the question type before answering.

GSM8K items in three versions: intact, underspecified (a fact-bearing sentence deleted)
and contradictory (a fact restated with a conflicting number). Two prompts: a plain one,
and one that first asks the model to label the question ANSWER, ASK or REJECT.
Gemma-4-E2B. Writes results/exp4_modes.json.

Usage:  python scripts/exp4_modes.py
"""

from __future__ import annotations

import argparse, json, pathlib, re, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp1_generate import extract_pred, gold_answer, same

ROOT = pathlib.Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\d.$])(\d+(?:\.\d+)?)(?![\d.])")

PLAIN = ("\n\nAnswer the question. If you can determine a numeric answer, end with "
         "'The answer is <number>.'")

ROUTED = ("\n\nBefore answering, decide which ONE of these the question calls for:\n"
          "  ANSWER    - it is fully specified and has a determinate answer\n"
          "  ASK       - information needed to solve it is missing\n"
          "  REJECT    - the question contains contradictory or false information\n"
          "Begin your reply with exactly one of: MODE: ANSWER / MODE: ASK / "
          "MODE: REJECT.\n"
          "Then, only if MODE: ANSWER, end with 'The answer is <number>.'")


def mutate(question: str, kind: str, rng):
    """Return a mutated question, or None if this item cannot host the mutation."""
    sents = [s for s in re.split(r"(?<=[.?])\s+", question.strip()) if s]
    # sentences carrying a number, excluding the final question sentence
    carriers = [i for i, s in enumerate(sents[:-1]) if NUM.search(s)]
    if not carriers:
        return None

    if kind == "intact":
        return question

    if kind == "underspec":
        # delete a whole fact-bearing sentence: the problem becomes unsolvable
        i = int(rng.choice(carriers))
        return " ".join(s for j, s in enumerate(sents) if j != i)

    if kind == "contradictory":
        # restate one fact with a different number, creating a direct conflict
        i = int(rng.choice(carriers))
        s = sents[i]
        m = list(NUM.finditer(s))[-1]
        old = float(m.group(1))
        new = round(old * 2) + 3
        conflict = s[:m.start(1)] + str(int(new)) + s[m.end(1):]
        out = list(sents)
        out.insert(i + 1, conflict)
        return " ".join(out)
    raise ValueError(kind)


def classify(text: str, kind: str) -> dict:
    """Score the reply. Mode is read from the MODE: tag when present, else inferred."""
    m = re.search(r"MODE:\s*(ANSWER|ASK|REJECT)", text, re.I)
    if m:
        mode = m.group(1).upper()
    else:
        # no tag (plain arm): infer from surface behaviour
        asks = bool(re.search(r"missing|not (?:given|provided|specified|stated)|"
                              r"cannot determine|can't determine|need to know|"
                              r"insufficient|how many .*\?|what is the .*\?", text, re.I))
        rejects = bool(re.search(r"contradict|inconsist|conflict|both .* and|"
                                 r"discrepan|cannot be both", text, re.I))
        gave = extract_pred(text) is not None and "answer is" in text.lower()
        mode = "REJECT" if rejects else ("ASK" if asks and not gave else
                                         ("ANSWER" if gave else "ASK"))
    correct_mode = {"intact": "ANSWER", "underspec": "ASK",
                    "contradictory": "REJECT"}[kind]
    return {"mode": mode, "mode_correct": int(mode == correct_mode),
            "gave_number": int(extract_pred(text) is not None
                               and "answer is" in text.lower())}


@torch.no_grad()
def ask(model, tok, device, question, suffix, max_new=320):
    msg = [{"role": "user", "content": question + suffix}]
    enc = tok.apply_chat_template(msg, add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device).eval()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(args.seed)
    rows, t0, n = [], time.time(), 0

    for r in ds:
        if n >= args.limit:
            break
        muts = {k: mutate(r["question"], k, rng)
                for k in ("intact", "underspec", "contradictory")}
        if any(v is None for v in muts.values()):
            continue
        n += 1
        for kind, q in muts.items():
            for arm, suffix in (("plain", PLAIN), ("routed", ROUTED)):
                text = ask(model, tok, device, q, suffix)
                c = classify(text, kind)
                c.update({"idx": n, "kind": kind, "arm": arm})
                if kind == "intact":
                    c["answer_correct"] = int(same(extract_pred(text),
                                                   gold_answer(r["answer"])))
                rows.append(c)
        if n % 20 == 0:
            print(f"  {n}/{args.limit}  ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "results" / "exp4_modes.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out} ({len(rows)} rows)\n")

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"{'kind':>15} {'arm':>8} {'n':>5} {'mode_correct':>13} {'gave_number':>12}")
    print("-" * 58)
    for kind in ("intact", "underspec", "contradictory"):
        for arm in ("plain", "routed"):
            s = df[(df.kind == kind) & (df.arm == arm)]
            if len(s):
                print(f"{kind:>15} {arm:>8} {len(s):>5} "
                      f"{s.mode_correct.mean():>13.4f} {s.gave_number.mean():>12.4f}")

    print("\n--- the key number ---")
    for arm in ("plain", "routed"):
        u = df[(df.kind == "underspec") & (df.arm == arm)]
        c = df[(df.kind == "contradictory") & (df.arm == arm)]
        print(f"  {arm:>6}: confabulates a number for an UNSOLVABLE problem "
              f"{u.gave_number.mean():.1%}   for a CONTRADICTORY one {c.gave_number.mean():.1%}")
    ok = df[(df.kind == "intact")]
    for arm in ("plain", "routed"):
        s = ok[ok.arm == arm]
        if "answer_correct" in s:
            print(f"  {arm:>6}: accuracy on INTACT problems "
                  f"{s.answer_correct.mean():.4f}  (routing must not cost this)")


if __name__ == "__main__":
    main()
