"""Run the actionability (section 4) and judging (section 7) experiments through Ollama.

Used for Gemma-4-26B, 4-bit quantised (Ollama tag gemma4:26b). Items, prompts and
scoring are imported from exp12_actionable.py and exp14_judge.py, so they are identical
to the Hugging Face runs. Requires a running Ollama server with the model pulled.

Usage:  python scripts/ollama_runner.py --exp 12 --model gemma4:26b --tag gemma4_26b_q4
        python scripts/ollama_runner.py --exp 14 --model gemma4:26b --tag gemma4_26b_q4
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys, time, urllib.request
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from datasets import load_dataset

import exp12_actionable as E12
import exp14_judge as E14
from cot_segment import SUSTAINING, segment, render

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = "http://localhost:11434/api/chat"


def chat(model, prompt, num_predict=400, retries=3):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "think": False,   # required: else response is empty
                       "options": {"temperature": 0, "num_predict": num_predict}}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            if a == retries - 1:
                # Fail loudly: an empty string would be scored as "not flagged" and,
                # written at the end of the run, would overwrite the released results.
                raise RuntimeError(f"Ollama request failed after {retries} attempts: {e}")
            time.sleep(5)


def check_server(model):
    """Abort before any generation if Ollama is down or the model is not pulled."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as r:
            names = [m["name"] for m in json.loads(r.read())["models"]]
    except Exception as e:
        sys.exit(f"Ollama server not reachable at localhost:11434 ({e}). Start it with "
                 f"`ollama serve`.")
    if model not in names:
        sys.exit(f"Model {model!r} not found in Ollama. Pull it with `ollama pull {model}`.")


def run12(model, tag, limit):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.default_rng(0)
    rows, t0, n = [], time.time(), 0
    for r in ds:
        if n >= limit:
            break
        v = E12.build(r["question"], rng)
        if v is None:
            continue
        n += 1
        for kind, q in v.items():
            text = chat(model, q + E12.SUFFIX)
            rows.append({"idx": n, "kind": kind,
                         "refused": int(bool(E12.REFUSE.search(text))),
                         "confab": int(bool(E12.STRICT.search(text))),
                         "question": q, "response": text})
        if n % 25 == 0:
            print(f"  {n}/{limit}  ({time.time()-t0:.0f}s)", flush=True)
    return rows, ROOT / "results" / f"exp12_actionable_{tag}.json"


def run14(model, tag, limit):
    traces = [t for t in json.loads((ROOT / "results/exp1_traces.json").read_text())
              if t["correct"]][:limit]
    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()
    for n, tr in enumerate(traces):
        spans = segment(tr["cot"])
        sus = [i for i, s in enumerate(spans) if s["label"] == SUSTAINING]
        if not sus:
            continue
        i = int(rng.choice(sus))
        variants = {
            "supported": render(spans, set()),
            "omitted": render(spans, {i}),
            "false": "\n".join((rng.choice(E14.FABRICATED) + "\n" + s["text"])
                               if j == i else s["text"] for j, s in enumerate(spans)),
        }
        for kind, reasoning in variants.items():
            p = (f"Question: {tr['question']}\n\nProposed reasoning:\n{reasoning}\n\n"
                 f"Proposed answer: {tr['gold']}{E14.JUDGE}")
            text = chat(model, p, num_predict=200)
            v = E14.verdict(text)
            rows.append({"idx": n, "kind": kind, "verdict": v,
                         "flagged_invalid": int(v == "INVALID"), "response": text})
        if n % 25 == 0:
            print(f"  {n}/{len(traces)}  ({time.time()-t0:.0f}s)", flush=True)
    return rows, ROOT / "results" / f"exp14_judge_{tag}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, required=True, choices=[12, 14])
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--tag", default="gemma4_26b_q4")
    ap.add_argument("--limit", type=int, default=150)
    a = ap.parse_args()
    check_server(a.model)
    rows, out = (run12 if a.exp == 12 else run14)(a.model, a.tag, a.limit)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out} ({len(rows)} rows)")
    import pandas as pd
    from sigtest import mcnemar
    df = pd.DataFrame(rows)
    if a.exp == 12:
        for lbl, x, y in (("present", "inert_present", "act_present"),
                          ("absent", "inert_absent", "act_absent")):
            for col in ("refused", "confab"):
                p_ = df[df.kind == x].set_index("idx")[col]
                q_ = df[df.kind == y].set_index("idx")[col]
                c = p_.index.intersection(q_.index)
                n10, n01, pv = mcnemar(p_.loc[c].values, q_.loc[c].values)
                print(f"  {lbl:>8} ({col:>7}): inert {p_.loc[c].mean():.3f} -> "
                      f"actionable {q_.loc[c].mean():.3f}  p={pv:.4g}")
    else:
        for k in ("supported", "omitted", "false"):
            print(f"  {k:>10}: flagged INVALID {df[df.kind==k].flagged_invalid.mean():.4f}")
        a_ = df[df.kind == "omitted"].set_index("idx").flagged_invalid
        b_ = df[df.kind == "false"].set_index("idx").flagged_invalid
        c = a_.index.intersection(b_.index)
        n10, n01, pv = mcnemar(a_.loc[c].values, b_.loc[c].values)
        print(f"  asymmetry: {a_.loc[c].mean():.3f} vs {b_.loc[c].mean():.3f}  p={pv:.4g}")


if __name__ == "__main__":
    main()
