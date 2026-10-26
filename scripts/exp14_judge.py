"""Judging (paper section 7, Table 3).

Correct chains of thought from results/exp1_traces.json are shown to a judge with the
gold answer, in three versions: intact (supported), one computational step deleted
(omitted), or one plausible but unsupported supporting fact inserted (false). The judge is
asked: "Is the final answer adequately supported by the reasoning shown? Reply VALID if
the reasoning fully justifies the answer, or INVALID if it does not."
Writes results/exp14_judge_<tag>.json.

Usage:  python scripts/exp14_judge.py --model <hf model id> --tag <tag>
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cot_segment import SUSTAINING, segment, render

ROOT = pathlib.Path(__file__).resolve().parents[1]

# plausible, concrete, and NOT derivable from the problem statement
FABRICATED = [
    "Since the standard rate for this is $2 per unit, we apply it here.",
    "Each of these normally comes in packs of 6, so we account for that.",
    "The usual weekly allowance of 4 is included in this figure.",
    "Applying the customary 10% adjustment gives the same result.",
    "Because there are 3 of them per batch, the count carries over.",
]

JUDGE = ("\n\nIs the final answer adequately supported by the reasoning shown? "
         "Reply VALID if the reasoning fully justifies the answer, or INVALID if "
         "it does not. Begin your reply with VALID or INVALID.")


@torch.no_grad()
def judge(model, tok, device, q, reasoning, ans, max_new=200):
    p = (f"Question: {q}\n\nProposed reasoning:\n{reasoning}\n\n"
         f"Proposed answer: {ans}{JUDGE}")
    enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True, return_dict=True,
                                  return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def verdict(t):
    m = re.search(r"\b(INVALID|VALID)\b", t, re.I)
    return (m.group(1).upper() if m else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--tag", default="gemma4e2b")
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device).eval()

    traces = [t for t in json.loads((ROOT/"results/exp1_traces.json").read_text())
              if t["correct"]][:args.limit]
    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()

    for n, tr in enumerate(traces):
        spans = segment(tr["cot"])
        sus = [i for i, s in enumerate(spans) if s["label"] == SUSTAINING]
        if not sus:
            continue
        i = int(rng.choice(sus))
        full = render(spans, set())
        variants = {
            "supported": full,
            "omitted":   render(spans, {i}),                       # a gap
            "false":     "\n".join(
                (rng.choice(FABRICATED) + "\n" + s["text"]) if j == i else s["text"]
                for j, s in enumerate(spans)),
        }
        for kind, reasoning in variants.items():
            text = judge(model, tok, device, tr["question"], reasoning, tr["gold"])
            v = verdict(text)
            rows.append({"idx": n, "kind": kind, "verdict": v,
                         "flagged_invalid": int(v == "INVALID"),
                         "response": text})
        if n % 25 == 0:
            print(f"  {n}/{len(traces)}  ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "results" / f"exp14_judge_{args.tag}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out} ({len(rows)} rows)\n")

    import pandas as pd
    from sigtest import mcnemar
    df = pd.DataFrame(rows)
    print(f"{'condition':>12} {'n':>5} {'flagged INVALID':>17} {'unparsed':>10}")
    for k in ("supported", "omitted", "false"):
        s = df[df.kind == k]
        print(f"{k:>12} {len(s):>5} {s.flagged_invalid.mean():>17.4f} "
              f"{s.verdict.isna().mean():>10.3f}")
    a = df[df.kind == "omitted"].set_index("idx").flagged_invalid
    b = df[df.kind == "false"].set_index("idx").flagged_invalid
    c = a.index.intersection(b.index)
    n10, n01, p = mcnemar(a.loc[c].values, b.loc[c].values)
    print(f"\nJUDGE ASYMMETRY: omitted {a.loc[c].mean():.3f} vs false "
          f"{b.loc[c].mean():.3f}  ({n10}/{n01})  p={p:.4g}")
    print("Prediction holds if a GAP is flagged more than an INVENTED supporting fact.")


if __name__ == "__main__":
    main()
