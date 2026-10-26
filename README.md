# Models Verify Sufficiency, Not Validity

Code and released model outputs for **"Models Verify Sufficiency, Not Validity: Usable
Falsehoods Evade the Checks that Gaps Trigger"**, EvoRobust Workshop at NeurIPS 2026.

Paper: `<OpenReview URL>`

Unanswerable-question benchmarks are built by omission: delete the fact a question needs,
then score refusal. We hold source support fixed and vary whether an unsupported premise
is *usable*. On GSM8K, across four models from two families (2B–26B), usable unsupported
premises are flagged 8–30 points less often and answered 1.5–2.3× more often than inert
ones, and the drop is carried by premises that supply the quantity the question asks for.
Two further manipulations point the same way: a fact asserted rather than asked about (on
GSM8K for three models, on SQuAD 2.0 for one of three), and an intermediate value
corrupted rather than removed.

## Verify every number in the paper (seconds, no GPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python verify_paper_numbers.py
```

This recomputes every table cell, p-value and in-text figure from the released files in
`results/`, using the paper's own metric definitions, and exits non-zero on any mismatch.
It should report `70/70 checks passed` and end with
`Every number checked matches the paper.` Installing the pinned requirements includes
PyTorch, so the install step takes a few minutes; the check itself takes seconds.

## Repository layout

```
verify_paper_numbers.py   recompute the paper's numbers from results/
scripts/                  experiment code (regenerates results/)
  make_tables.py          metric definitions and the paper's tables
  sigtest.py              exact paired McNemar test
  cot_segment.py          splits a chain of thought into labelled spans
results/                  released prompts, model outputs and scores (see below)
```

## What produced each result

All generation uses greedy decoding. Hugging Face models run in bf16; the 26B model runs
4-bit quantised through Ollama.

| Paper | Experiment | Data file(s) |
|---|---|---|
| §4, Table 1 (actionability) | `exp12_actionable.py`, `ollama_runner.py --exp 12` | `exp12_actionable_*.json` |
| §5, Table 2 (assertion, GSM8K) | `exp9_adhyasa.py` | `exp9_adhyasa_*.json` |
| §5, Table 2 (assertion, SQuAD) | `exp13_squad_fixed.py` | `exp13_squad_*.json` |
| §5 (cascade) | `exp3_cascade.py` | `exp3_cascade.json` |
| §7, Table 3 (judging) | `exp14_judge.py`, `ollama_runner.py --exp 14` | `exp14_judge_*.json` |
| §8 (mitigation) | `exp4_modes.py` | `exp4_modes.json` |
| input to §5 cascade and §7 | `exp1_generate.py` (chains of thought) | `exp1_traces.json` |

### What the released files contain

| Files | Contents |
|---|---|
| `exp12_*`, `exp9_*` | every problem as shown to the model (`question`; the fixed instruction suffix is in the script) and raw model output (`response`) |
| `exp13_*` | every question (`question`) and raw model output (`response`); the SQuAD passage is not stored and is rebuilt by `exp13_squad_fixed.py` from `rajpurkar/squad_v2` |
| `exp14_*` | every raw judge output; the prompts are rebuilt deterministically from `exp1_traces.json` by `exp14_judge.py` |
| `exp1_traces.json` | the first 400 GSM8K test problems and each raw chain of thought (768-token cap; the prompt suffix is in `exp1_generate.py`) |
| `exp3_cascade.json`, `exp4_modes.json` | **scored outcomes only**; the raw text of these two runs was not saved, so their labels cannot be re-scored |

### Regenerating from scratch

**Every script overwrites its file in `results/`.** Regenerate in a separate clone, or
copy `results/` first, so the released files are kept for comparison. Run from the
repository root; step 1 produces the chains of thought that steps 2 and 3 consume.

```bash
# 1. chains of thought on GSM8K (Gemma-4-E2B, 400 problems, 768-token cap)
python scripts/exp1_generate.py

# 2. cascade (§5) and mitigation (§8), Gemma-4-E2B
python scripts/exp3_cascade.py
python scripts/exp4_modes.py

# 3. actionability, assertion (GSM8K and SQuAD) and judging, per Hugging Face model
python scripts/exp12_actionable.py  --model google/gemma-4-E2B-it    --tag gemma4e2b
python scripts/exp9_adhyasa.py      --model google/gemma-4-E2B-it    --tag gemma4e2b
python scripts/exp13_squad_fixed.py --model google/gemma-4-E2B-it    --tag gemma4e2b
python scripts/exp14_judge.py       --model google/gemma-4-E2B-it    --tag gemma4e2b

python scripts/exp12_actionable.py  --model Qwen/Qwen2.5-3B-Instruct --tag qwen25_3b
python scripts/exp9_adhyasa.py      --model Qwen/Qwen2.5-3B-Instruct --tag qwen25_3b
python scripts/exp13_squad_fixed.py --model Qwen/Qwen2.5-3B-Instruct --tag qwen25_3b
python scripts/exp14_judge.py       --model Qwen/Qwen2.5-3B-Instruct --tag qwen25_3b

python scripts/exp12_actionable.py  --model Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b
python scripts/exp9_adhyasa.py      --model Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b
python scripts/exp13_squad_fixed.py --model Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b
python scripts/exp14_judge.py       --model Qwen/Qwen2.5-7B-Instruct --tag qwen25_7b

# 4. Gemma-4-26B, 4-bit, via a local Ollama server (`ollama pull gemma4:26b`)
python scripts/ollama_runner.py --exp 12 --model gemma4:26b --tag gemma4_26b_q4
python scripts/ollama_runner.py --exp 14 --model gemma4:26b --tag gemma4_26b_q4

# 5. tables
python scripts/make_tables.py
```

The released results took roughly a day of compute on an Apple M4 Max with 36 GB of
unified memory (PyTorch MPS backend). `ollama_runner.py` stops before generating anything
if the Ollama server is not running or the model is not pulled.

**Versions.** The scripts load each model and dataset at its current default revision.
To reproduce exactly, pin the revisions listed under *Models and data* (for example by
passing `revision=` to `from_pretrained` and `load_dataset`) and use the listed Ollama
digest. The Ollama tag `gemma4:26b` may move.

**Determinism.** Decoding is greedy, but outputs can still differ slightly across
hardware, PyTorch versions and Ollama builds. `verify_paper_numbers.py` checks the
released files exactly, so on a fresh regeneration expect the pattern of results, not
every digit.

## Condition names used in the code

Some condition names come from the Sanskrit epistemology that motivated the design. They
do not appear in the paper.

| Experiment | Field and values | Meaning |
|---|---|---|
| actionability | `kind`: `act_present`, `act_absent` | actionable premise, entity present / absent |
| | `kind`: `inert_present`, `inert_absent` | inert premise, entity present / absent |
| assertion | `kind`: `viparyaya` | a **question** asking for an unsupported quantity about an entity in the problem |
| | `kind`: `adhyasa` | an unsupported numeric premise **asserted** about the same entity, followed by a request for the total |
| | `kind`: `vikalpa`, `adhyasa_absent` | the same two framings for an entity absent from the problem (not in Table 2) |
| | `kind`: `answerable` | the original question (sanity check) |
| judging | `kind`: `supported` / `omitted` / `false` | intact solution / one computational step deleted / a plausible unsupported fact inserted |
| cascade | `band`: `early` / `mid` / `late`; `cond`: `truncate` / `corrupt` | where in the chain of thought; intermediate value removed / corrupted |
| mitigation | `kind`: `intact` / `underspec` / `contradictory`; `arm`: `plain` / `routed` | item version; no labelling step / answer-ask-reject labelling first |

In the assertion experiment the asked-about quantity (`viparyaya`) and the asserted premise
(`adhyasa`) are drawn independently: they are different unsupported facts about the same
entity, not one fact in two framings.

## Metrics

Responses are scored by pattern matching over the saved text (definitions in
`scripts/make_tables.py`):

- **Flagged** (`refused_v2`): the response says the question cannot be answered from the
  information given.
- **Committed** (`answered`): the response states an answer ("the answer is *n*") or an
  equation with a numeric result; the latter also fires on intermediate arithmetic in
  responses that go on to decline, so *committed* and *flagged* overlap.
- **Confabulated** (Table 2): an explicit answer statement. On GSM8K this is "the answer
  is *n*"; on SQuAD it is "the answer is ..." in a response that the script's own refusal
  pattern does not match (stored as `confab` by `exp13_squad_fixed.py`).

As the paper states, these rules were audited on only 40 responses. The audit found a
condition-dependent error in the first version of the *committed* rule, which is
corrected here; an audit of that size cannot bound residual per-condition error. Where raw
outputs are released (see above), the labels can be re-scored.

## Models and data

| | Revision | Licence |
|---|---|---|
| `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | Apache-2.0 |
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Qwen Research License |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | Apache-2.0 |
| `gemma4:26b` (Ollama, Q4_K_M) | digest `5571076f3d70` | Apache-2.0 |
| `openai/gsm8k` (main, test) | `740312add88f781978c0658806c59bc2815b9866` | MIT |
| `rajpurkar/squad_v2` (validation) | `3ffb306f725f7d2ce8394bc1873b24868140c412` | CC BY-SA 4.0 |

## Licence

Code is released under the MIT License (see `LICENSE`).

The files in `results/` contain text from two datasets, which keep their own terms (see
`NOTICE`): GSM8K problems (MIT, Copyright (c) 2021 OpenAI) and text derived from
SQuAD 2.0 (Rajpurkar et al., 2018; CC BY-SA 4.0, built from Wikipedia). The model outputs
are released under CC BY-SA 4.0. Outputs of Qwen2.5-3B-Instruct
(`*_qwen25_3b.json`) are also subject to the Qwen Research License, which restricts
commercial use.

## Citation

```bibtex
@inproceedings{madiraju2026sufficiency,
  title     = {Models Verify Sufficiency, Not Validity: Usable Falsehoods Evade the
               Checks that Gaps Trigger},
  author    = {Madiraju, Naveen},
  booktitle = {EvoRobust Workshop at the 40th Conference on Neural Information
               Processing Systems (NeurIPS)},
  year      = {2026}
}
```
