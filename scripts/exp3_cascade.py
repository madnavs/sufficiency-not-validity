"""Cascade (paper section 5): is a corrupted intermediate value worse than a missing one?

Take a correct chain of thought from results/exp1_traces.json, choose one computational
span in the early, middle or late third, and either corrupt its final number (keeping the
trace up to and including that span) or truncate the trace just before it. The model then
continues from that prefix. Records whether it recovers the gold answer and whether it
explicitly flags a problem. Gemma-4-E2B. Writes results/exp3_cascade.json.

Usage:  python scripts/exp3_cascade.py
"""

from __future__ import annotations

import argparse, json, pathlib, re, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp1_generate import PROMPT_SUFFIX, extract_pred, same
from cot_segment import SUSTAINING, segment

ROOT = pathlib.Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")


def corrupt_last_number(text: str, rng):
    """Perturb the FINAL number in a span -- the computed result, not an input.

    Returns (new_text, original_value, new_value) or None.
    """
    ms = list(NUM.finditer(text))
    if not ms:
        return None
    m = ms[-1]
    old = float(m.group(1))
    if old == 0:
        return None
    # a plausible wrong value: shift by 10-60%, never equal, keep integers integral
    delta = rng.uniform(0.10, 0.60) * (1 if rng.random() < 0.5 else -1)
    new = old * (1 + delta)
    new = round(new) if float(old).is_integer() else round(new, 2)
    if new == old or new <= 0:
        new = old + (1 if old > 0 else -1)
    s = f"{int(new)}" if float(new).is_integer() else f"{new}"
    return text[:m.start(1)] + s + text[m.end(1):], old, new


@torch.no_grad()
def continue_from(model, tok, device, question, prefix, max_new=220):
    """Let the model keep reasoning from a (possibly corrupted) prefix."""
    msg = [{"role": "user", "content": question + PROMPT_SUFFIX}]
    head = tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tok(head + prefix + "\n", return_tensors="pt",
              add_special_tokens=False).to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device).eval()

    traces = [t for t in json.loads((ROOT / "results/exp1_traces.json").read_text())
              if t["correct"]][:args.limit]
    rng = np.random.default_rng(args.seed)
    rows, t0 = [], time.time()

    for n, tr in enumerate(traces):
        spans = segment(tr["cot"])
        sus = [i for i, s in enumerate(spans) if s["label"] == SUSTAINING]
        if not sus:
            continue

        for band, lo, hi in [("early", 0.0, 0.34), ("mid", 0.34, 0.67),
                             ("late", 0.67, 1.01)]:
            pool = [i for i in sus if lo <= i / max(len(spans) - 1, 1) < hi]
            if not pool:
                continue
            i = int(rng.choice(pool))
            c = corrupt_last_number(spans[i]["text"], rng)
            if c is None:
                continue
            new_text, old, new = c

            for cond in ("corrupt", "truncate"):
                if cond == "corrupt":
                    prefix = "\n".join(
                        (new_text if j == i else s["text"])
                        for j, s in enumerate(spans[:i + 1]))
                else:
                    prefix = "\n".join(s["text"] for s in spans[:i])

                gen = continue_from(model, tok, device, tr["question"], prefix)
                pred = extract_pred(gen)
                rows.append({
                    "idx": tr["idx"], "band": band, "cond": cond,
                    "old": old, "new": new,
                    "recovered": int(same(pred, tr["gold"])),
                    # did the final answer move in the direction the corruption pushed?
                    "shifted": int(pred is not None and not same(pred, tr["gold"])),
                    "pred": pred,
                    # did the model ever explicitly flag a problem?
                    "flagged": int(bool(re.search(
                        r"mistake|error|incorrect|wait|recalculat|actually|"
                        r"correction|should be|let me re", gen, re.I))),
                })
        if n % 20 == 0:
            print(f"  {n}/{len(traces)}  ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "results" / "exp3_cascade.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out} ({len(rows)} rows)\n")

    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"{'band':>7} {'cond':>10} {'n':>5} {'recovered':>11} {'flagged':>9}")
    print("-" * 48)
    for band in ("early", "mid", "late"):
        for cond in ("corrupt", "truncate"):
            s = df[(df.band == band) & (df.cond == cond)]
            if len(s):
                print(f"{band:>7} {cond:>10} {len(s):>5} "
                      f"{s.recovered.mean():>11.4f} {s.flagged.mean():>9.4f}")

    c = df[df.cond == "corrupt"]
    print(f"\nOverall with a corrupted value in view:")
    print(f"  recovered the true answer : {c.recovered.mean():.4f}")
    print(f"  explicitly flagged a problem: {c.flagged.mean():.4f}")
    print("\nSecond arrow = low recovery AND low flagging: the model reasons "
          "fluently onward from a value it never questions.")


if __name__ == "__main__":
    main()
