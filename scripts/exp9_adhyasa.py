"""Assertion vs interrogation on GSM8K (paper section 5, Table 2).

The final question of each problem is replaced, giving five variants:
  answerable      the original question (sanity check)
  viparyaya       asks for an unsupported quantity about an entity in the problem
  adhyasa         asserts an unsupported numeric premise about that entity, then asks
                  for the total
  vikalpa         as viparyaya, about an entity absent from the problem
  adhyasa_absent  as adhyasa, about an entity absent from the problem
Table 2 compares viparyaya with adhyasa. The asked-about quantity and the asserted
premise are drawn independently, so they are different unsupported facts about the same
entity. Condition names come from Sanskrit epistemology and do not appear in the paper.
Writes results/exp9_adhyasa_<tag>.json.

Usage:  python scripts/exp9_adhyasa.py --model <hf model id> --tag <tag>
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

ABSENT = ["Marcus", "Priya", "Tomas", "Aisha", "Dmitri", "Lena", "Kwame", "Ingrid"]
PROPS  = ["spend on supplies", "pay in taxes", "save for the winter",
          "give away to neighbours", "lose to spoilage"]
SCHEMA = ["sells them at the market for $3 each",
          "works 8 hours a day at $12 per hour",
          "receives a 20% discount on everything",
          "splits the total equally among 4 friends",
          "pays $5 in delivery costs per trip"]

SUFFIX = ("\n\nIf the question can be answered from the information given, answer it "
          "and end with 'The answer is <number>.' If it cannot be answered, explain why.")


def names_in(text: str) -> list[str]:
    out = []
    for s in re.split(r"(?<=[.?])\s+", text.strip()):
        for w in re.findall(r"\b([A-Z][a-z]{2,})\b", s[1:]):
            if w not in out:
                out.append(w)
    return out


def build(question: str, rng):
    """Split off the final question sentence; every variant reuses the same context."""
    sents = [s for s in re.split(r"(?<=[.?])\s+", question.strip()) if s]
    if len(sents) < 2:
        return None
    ctx, orig_q = " ".join(sents[:-1]), sents[-1]
    present = names_in(ctx)
    if not present:
        return None
    p, ab, sc = present[0], rng.choice([a for a in ABSENT if a not in present]), rng.choice(SCHEMA)
    prop = rng.choice(PROPS)
    return {
        "answerable":     f"{ctx} {orig_q}",
        "vikalpa":        f"{ctx} How much does {ab} {prop}?",
        "viparyaya":      f"{ctx} How much does {p} {prop}?",
        "adhyasa":        f"{ctx} Given that {p} {sc}, what is the total in dollars?",
        "adhyasa_absent": f"{ctx} Given that {ab} {sc}, what is the total in dollars?",
    }


REFUSE = re.compile(
    r"cannot be answered|cannot determine|can't determine|cannot be determined|"
    r"not (?:given|provided|specified|stated|mentioned|possible|enough|answerable)|"
    r"no information|insufficient|does not (?:say|state|mention|provide|specify)|"
    r"doesn't (?:say|state|mention|provide|specify)|unable to|"
    r"is not (?:mentioned|referred|named|introduced|present)|"
    r"there is no (?:mention|information|reference)", re.I)


@torch.no_grad()
def ask(model, tok, device, q, max_new=400):
    msg = [{"role": "user", "content": q + SUFFIX}]
    enc = tok.apply_chat_template(msg, add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="gemma4e2b")
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
        v = build(r["question"], rng)
        if v is None:
            continue
        n += 1
        for kind, q in v.items():
            text = ask(model, tok, device, q)
            pred = extract_pred(text)
            rec = {"idx": n, "kind": kind,
                   "gave_number": int(pred is not None and "answer is" in text.lower()),
                   "refused": int(bool(REFUSE.search(text))),
                   "question": q, "response": text}      # raw text saved for audit
            if kind == "answerable":
                rec["answer_correct"] = int(same(pred, gold_answer(r["answer"])))
            rows.append(rec)
        if n % 25 == 0:
            print(f"  {n}/{args.limit}  ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "results" / f"exp9_adhyasa_{args.tag}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out} ({len(rows)} rows)\n")

    import pandas as pd
    from sigtest import mcnemar
    df = pd.DataFrame(rows)
    print(f"{'kind':>16} {'n':>5} {'refused':>9} {'confabulated':>13}")
    print("-" * 48)
    for k in ("answerable", "vikalpa", "viparyaya", "adhyasa", "adhyasa_absent"):
        s = df[df.kind == k]
        if len(s):
            print(f"{k:>16} {len(s):>5} {s.refused.mean():>9.4f} {s.gave_number.mean():>13.4f}")
    ac = df[df.kind == "answerable"]
    if "answer_correct" in ac:
        print(f"\n  sanity: accuracy on answerable = {ac.answer_correct.mean():.4f}")

    print("\n=== THE CRUX: same surface form, different substrate ===")
    a = df[df.kind == "adhyasa"].set_index("idx").refused
    b = df[df.kind == "adhyasa_absent"].set_index("idx").refused
    c = a.index.intersection(b.index)
    n10, n01, p = mcnemar(a.loc[c].values, b.loc[c].values)
    print(f"  adhyasa (present substrate) {a.loc[c].mean():.4f}  vs  "
          f"adhyasa_absent {b.loc[c].mean():.4f}   ({n10}/{n01})  p={p:.4g}")
    print("  Compares an asserted premise about a present vs an absent entity.")

    print("\n=== other pairs ===")
    for x, y in (("vikalpa", "viparyaya"), ("viparyaya", "adhyasa"), ("vikalpa", "adhyasa")):
        a = df[df.kind == x].set_index("idx").refused
        b = df[df.kind == y].set_index("idx").refused
        c = a.index.intersection(b.index)
        n10, n01, p = mcnemar(a.loc[c].values, b.loc[c].values)
        print(f"  {x:>10} {a.loc[c].mean():.4f} vs {y:<15} {b.loc[c].mean():.4f}  "
              f"({n10}/{n01})  p={p:.4g}")


if __name__ == "__main__":
    main()
