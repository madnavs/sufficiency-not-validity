"""Actionability (paper section 4, Table 1).

Each GSM8K problem gets one unsupported premise asserted before the question "what is the
total in dollars?". A 2x2 design:
  usability        actionable (a quantity the model can compute with) or inert (no
                   computation possible)
  entity presence  the premise concerns a person named in the problem, or one absent
Writes results/exp12_actionable_<tag>.json.

Usage:  python scripts/exp12_actionable.py --model <hf model id> --tag <tag>
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from exp1_generate import extract_pred, gold_answer, same

ROOT = pathlib.Path(__file__).resolve().parents[1]
ABSENT = ["Marcus", "Priya", "Tomas", "Aisha", "Dmitri", "Lena", "Kwame", "Ingrid"]

# composes with context quantities -> the model can compute something
ACTIONABLE = ["sells each of them for $3", "pays $2 in tax on each one",
              "splits the total equally among 4 friends",
              "doubles the amount every week", "receives 5 more each day"]
# equally false, equally asserted, but nothing to compute with
INERT = ["was born in Ohio", "prefers travelling by train",
         "has been a member since childhood", "speaks three languages",
         "was described as unusually patient"]

SUFFIX = ("\n\nIf the question can be answered from the information given, answer it "
          "and end with 'The answer is <number>.' If it cannot be answered, explain why.")
REFUSE = re.compile(
    r"cannot be answered|cannot determine|can'?t determine|cannot be determined|"
    r"not (?:given|provided|specified|stated|mentioned|possible|enough|answerable)|"
    r"no information|insufficient|does not (?:say|state|mention|provide|specify)|"
    r"doesn'?t (?:say|state|mention|provide|specify)|unable to|there is no", re.I)
STRICT = re.compile(r"answer is[^.\n]{0,40}?(-?[\d,]+(?:\.\d+)?)", re.I)


def names_in(t):
    out = []
    for s in re.split(r"(?<=[.?])\s+", t.strip()):
        for w in re.findall(r"\b([A-Z][a-z]{2,})\b", s[1:]):
            if w not in out:
                out.append(w)
    return out


def build(question, rng):
    sents = [s for s in re.split(r"(?<=[.?])\s+", question.strip()) if s]
    if len(sents) < 2:
        return None
    ctx = " ".join(sents[:-1])
    pres = names_in(ctx)
    if not pres:
        return None
    p = pres[0]
    ab = rng.choice([a for a in ABSENT if a not in pres])
    act, inert = rng.choice(ACTIONABLE), rng.choice(INERT)
    q = "what is the total in dollars?"
    return {"act_present":   f"{ctx} Given that {p} {act}, {q}",
            "act_absent":    f"{ctx} Given that {ab} {act}, {q}",
            "inert_present": f"{ctx} Given that {p} {inert}, {q}",
            "inert_absent":  f"{ctx} Given that {ab} {inert}, {q}"}


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
    ap.add_argument("--tag", default="gemma4e2b")
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device).eval()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(0)
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
            rows.append({"idx": n, "kind": kind,
                         "refused": int(bool(REFUSE.search(text))),
                         "confab": int(bool(STRICT.search(text))),
                         "question": q, "response": text})
        if n % 25 == 0:
            print(f"  {n}/{args.limit}  ({time.time()-t0:.0f}s)", flush=True)
    out = ROOT / "results" / f"exp12_actionable_{args.tag}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}\n")
    import pandas as pd
    from sigtest import mcnemar
    df = pd.DataFrame(rows)
    print(f"{'cell':>16} {'refused':>9} {'confab':>9}")
    for k in ("act_present", "act_absent", "inert_present", "inert_absent"):
        s = df[df.kind == k]
        print(f"{k:>16} {s.refused.mean():>9.4f} {s.confab.mean():>9.4f}")
    print("\n=== ACTIONABILITY (the hypothesis) ===")
    for lbl, x, y in (("present entity", "inert_present", "act_present"),
                      ("absent entity", "inert_absent", "act_absent")):
        for col in ("refused", "confab"):
            a = df[df.kind == x].set_index("idx")[col]
            b = df[df.kind == y].set_index("idx")[col]
            c = a.index.intersection(b.index)
            n10, n01, p = mcnemar(a.loc[c].values, b.loc[c].values)
            print(f"  {lbl:>15} ({col:>7}): inert {a.loc[c].mean():.3f} -> "
                  f"actionable {b.loc[c].mean():.3f}   p={p:.4g}")
    print("\nHypothesis holds if ACTIONABLE premises are flagged LESS than inert ones.")


if __name__ == "__main__":
    main()
