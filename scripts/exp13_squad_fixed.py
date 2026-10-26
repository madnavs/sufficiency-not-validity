"""Assertion vs interrogation on SQuAD 2.0 (paper section 5, Table 2).

SQuAD passages with constructed questions over a named entity, in the same five variants
as exp9_adhyasa.py. Every variant asks for a numeric answer, and the prompt suffix is the
same as in exp9_adhyasa.py. Writes results/exp13_squad_<tag>.json.

Usage:  python scripts/exp13_squad_fixed.py --model <hf model id> --tag <tag>
"""

from __future__ import annotations

import argparse, json, pathlib, re, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]

ABSENT = ["Marcus Feld", "Priya Raman", "Tomas Berg", "Aisha Nour",
          "Dmitri Vasev", "Lena Hartl", "Kwame Osei", "Ingrid Sallow"]
# Passive participles so "When was <X> <prop>?" is grammatical for both persons
# and groups. Ungrammatical phrasing would itself invite extra scrutiny and
# confound the comparison.
# Entity-type-neutral: must read sensibly for a person OR a group/thing, since
# the same property is applied to both present and absent entities in the 2x2.
# Numeric quantities, absent from every passage, so the QUESTION cells ask for
# the same answer type as the ASSERTION cells.
PROPS = ["settlements were controlled", "coins were levied per household",
         "garrisons were maintained", "charters were recorded each year",
         "provinces were governed"]
# Asserted premises are numeric and combinable, as on GSM8K, so that a model can
# compose them with the passage to produce an answer.
SCHEMA = ["controlled 12 settlements, each with 400 inhabitants",
          "levied a tax of 30 coins per household across 250 households",
          "maintained 8 garrisons of 150 soldiers each",
          "recorded 24 charters per year over 15 years",
          "governed 6 provinces averaging 3,000 residents each"]

# Same suffix as exp9_adhyasa.py, so the prompt does not differ between domains.
SUFFIX = ("\n\nIf the question can be answered from the information given, answer it "
          "and end with 'The answer is <answer>.' If it cannot be answered, explain why.")

REFUSE = re.compile(
    r"does not (?:say|state|mention|provide|specify|contain|indicate)|"
    r"doesn'?t (?:say|state|mention|provide|specify|contain)|"
    r"not (?:mentioned|stated|specified|provided|given|contained|discussed|"
    r"present|in the passage|answerable)|no (?:mention|information|reference|"
    r"indication)|cannot be (?:answered|determined|found)|"
    r"can(?:no|')t be (?:answered|determined)|insufficient|unable to (?:answer|determine)|"
    r"there is no", re.I)


def entities(ctx: str) -> list[str]:
    """Capitalised multi-word or single-word proper nouns, excluding sentence starts."""
    out = []
    for s in re.split(r"(?<=[.?])\s+", ctx.strip()):
        for w in re.findall(r"\b([A-Z][a-z]{3,})\b", s[1:]):
            if w not in out and w not in ("The", "This", "That", "These", "There"):
                out.append(w)
    return out


def build(ctx: str, question: str, rng):
    ents = entities(ctx)
    if len(ents) < 1:
        return None
    p = ents[0]
    ab = rng.choice(ABSENT)
    prop, sc = rng.choice(PROPS), rng.choice(SCHEMA)
    return {
        "answerable":     question,
        "vikalpa":        f"How many {prop} by {ab}?",
        "viparyaya":      f"How many {prop} by {p}?",
        "adhyasa":        f"Given that {p} {sc}, what is the total?",
        "adhyasa_absent": f"Given that {ab} {sc}, what is the total?",
    }


@torch.no_grad()
def ask(model, tok, device, ctx, q, max_new=220):
    msg = [{"role": "user", "content": f"Passage:\n{ctx}\n\nQuestion: {q}{SUFFIX}"}]
    enc = tok.apply_chat_template(msg, add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--tag", default="gemma4e2b")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(device).eval()

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = np.random.default_rng(args.seed)
    rows, t0, n, seen = [], time.time(), 0, set()

    for r in ds:
        if n >= args.limit:
            break
        if not r["answers"]["text"] or r["context"] in seen:
            continue          # keep only answerable originals, one item per passage
        v = build(r["context"], r["question"], rng)
        if v is None:
            continue
        seen.add(r["context"]); n += 1
        for kind, q in v.items():
            text = ask(model, tok, device, r["context"], q)
            refused = int(bool(REFUSE.search(text)))
            rows.append({
                "idx": n, "kind": kind, "refused": refused,
                # confabulation: produced a substantive answer without flagging
                "confab": int(bool(re.search(r"answer is[^.\n]{0,60}?[A-Za-z0-9]", text, re.I)) and not refused),
                "question": q, "response": text,
                "gold": r["answers"]["text"][0] if kind == "answerable" else None,
            })
        if n % 25 == 0:
            print(f"  {n}/{args.limit}  ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "results" / f"exp13_squad_{args.tag}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out} ({len(rows)} rows)\n")

    import pandas as pd
    from sigtest import mcnemar
    df = pd.DataFrame(rows)
    print(f"{'kind':>16} {'n':>5} {'refused':>9} {'confab':>9}")
    print("-" * 44)
    for k in ("answerable", "vikalpa", "viparyaya", "adhyasa", "adhyasa_absent"):
        s = df[df.kind == k]
        print(f"{k:>16} {len(s):>5} {s.refused.mean():>9.4f} {s.confab.mean():>9.4f}")

    print("\n=== does the assertion effect transfer to prose? ===")
    for lbl, x, y in (("ASSERTION", "viparyaya", "adhyasa_absent"),
                      ("substrate", "adhyasa_absent", "adhyasa"),
                      ("vikalpa vs viparyaya", "vikalpa", "viparyaya")):
        for col in ("refused", "confab"):
            a = df[df.kind == x].set_index("idx")[col]
            b = df[df.kind == y].set_index("idx")[col]
            c = a.index.intersection(b.index)
            n10, n01, p = mcnemar(a.loc[c].values, b.loc[c].values)
            print(f"  {lbl:>22} ({col:>7}): {a.loc[c].mean():.3f} -> {b.loc[c].mean():.3f}"
                  f"   ({n10}/{n01})  p={p:.4g}")


if __name__ == "__main__":
    main()
