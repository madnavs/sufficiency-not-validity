"""Generate chain-of-thought traces on GSM8K (input to the cascade and judging experiments).

Greedy decoding with Gemma-4-E2B over the first 400 GSM8K test problems with a 768-token
cap. Writes results/exp1_traces.json.

Usage:  python scripts/exp1_generate.py
"""

from __future__ import annotations

import argparse, json, pathlib, re, sys, time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]

PROMPT_SUFFIX = ("\n\nThink step by step, then give the final answer as "
                 "'The answer is <number>.'")


def gold_answer(a: str) -> str:
    """GSM8K gold answers end with '#### <number>'."""
    return a.split("####")[-1].strip().replace(",", "")


def extract_pred(text: str) -> str | None:
    """Last number appearing after 'answer is', else the last number in the text."""
    m = re.findall(r"answer is\D*(-?[\d,]+(?:\.\d+)?)", text, flags=re.I)
    if not m:
        m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def same(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    # Defaults reproduce the released traces. A 320-token cap truncates about half
    # of the chains of thought and cuts measured accuracy from ~91% to ~55%.
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--out", default="results/exp1_traces.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device).eval()

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.limit))
    out, t0 = [], time.time()

    for i, r in enumerate(ds):
        msg = [{"role": "user", "content": r["question"] + PROMPT_SUFFIX}]
        enc = tok.apply_chat_template(msg, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False)
        cot = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        gold = gold_answer(r["answer"])
        out.append({
            "idx": i,
            "question": r["question"],
            "gold": gold,
            "cot": cot,
            "pred": extract_pred(cot),
            "correct": int(same(extract_pred(cot), gold)),
        })
        if i % 25 == 0:
            acc = sum(o["correct"] for o in out) / len(out)
            print(f"  {i}/{len(ds)}  acc={acc:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    path = ROOT / args.out
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    acc = sum(o["correct"] for o in out) / len(out)
    print(f"\nwrote {path}  n={len(out)}  baseline accuracy={acc:.4f}")
    print(f"correct traces: {sum(o['correct'] for o in out)}")


if __name__ == "__main__":
    main()
