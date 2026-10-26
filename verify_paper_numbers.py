"""Recompute every number and quoted example in the camera-ready paper from the released data.

    python verify_paper_numbers.py

Runs in seconds, needs no GPU and no model weights. Uses the paper's own metric
definitions and significance test (imported from scripts/make_tables.py and
scripts/sigtest.py) so it cannot drift from what produced the paper. Exits 0 only
if every check passes.

p-values are printed in the paper either as decimals, as "<0.001", or as an order of
magnitude "10^k" (the exponent of the value rounded to one significant figure).
"""
from __future__ import annotations

import json, math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_tables as MT          # noqa: E402  metric regexes + load() + pair()
from sigtest import mcnemar  # noqa: E402  exact binomial McNemar

FAILS: list[str] = []
N_CHECKS = 0


def check(name: str, ok: bool, got: str = "") -> None:
    global N_CHECKS
    N_CHECKS += 1
    if not ok:
        FAILS.append(f"{name}  (got {got})")
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{got}]" if got else ""))


def rate_eq(x: float, expected: float) -> bool:
    return abs(round(x, 3) - expected) < 1e-9


def p_eq(p: float, shown: str) -> bool:
    if shown.startswith("<"):
        return p < float(shown[1:])
    if shown.startswith("1e"):
        return int(f"{p:.0e}".split("e")[1]) == int(shown[2:])
    return abs(round(p, 3) - float(shown)) < 1e-9


def pair_rates(df, x, y, col):
    a = df[df.kind == x].set_index("idx")[col]
    b = df[df.kind == y].set_index("idx")[col]
    i = a.index.intersection(b.index)
    return a.loc[i], b.loc[i]


# --------------------------------------------------------------------------- Table 1
print("\nTable 1  Actionability (GSM8K, 2x2: usability x entity presence)")
T1 = [  # tag, entity, flag inert, flag act, p, commit inert, commit act, p
    ("gemma4e2b", "present", .853, .553, "1e-12", .367, .547, "<0.001"),
    ("gemma4e2b", "absent", .867, .640, "1e-7", .293, .513, "1e-6"),
    ("qwen25_3b", "present", .673, .473, "<0.001", .467, .753, "1e-10"),
    ("qwen25_3b", "absent", .693, .553, "0.009", .280, .653, "1e-13"),
    ("qwen25_7b", "present", .887, .740, "1e-5", .187, .400, "1e-7"),
    ("qwen25_7b", "absent", .907, .793, "0.003", .187, .280, "0.016"),
    ("gemma4_26b_q4", "present", .907, .787, "0.001", .233, .487, "1e-8"),
    ("gemma4_26b_q4", "absent", .920, .840, "0.036", .233, .387, "<0.001"),
]
all_p, flag_gap, commit_gap, commit_ratio = [], {}, [], []
for tag, ent, fi, fa, pf, ci, ca, pc in T1:
    df = MT.load("exp12_actionable", tag)
    x, y = f"inert_{ent}", f"act_{ent}"
    for col, ei, ea, ep in (("refused_v2", fi, fa, pf), ("answered", ci, ca, pc)):
        a, b = pair_rates(df, x, y, col)
        _, _, p = mcnemar(a.values, b.values)
        all_p.append(p)
        check(f"{tag:>13} {ent:>7} {col:>10}: {ei:.3f} -> {ea:.3f}, p {ep}",
              len(a) == 150 and rate_eq(a.mean(), ei) and rate_eq(b.mean(), ea) and p_eq(p, ep),
              f"n={len(a)} {a.mean():.3f}->{b.mean():.3f} p={p:.3g}")
        if col == "refused_v2":
            flag_gap[(tag, ent)] = (a.mean() - b.mean()) * 100
        else:
            commit_gap.append((b.mean() - a.mean()) * 100)
            commit_ratio.append(b.mean() / a.mean())

print("\nSection 4  significance, effect sizes, scale")
holm = sorted(all_p)
holm_ok = all(p <= 0.05 / (len(holm) - k) for k, p in enumerate(holm))
check("all 16 contrasts p < 0.05", sum(p < .05 for p in all_p) == 16, f"{sum(p < .05 for p in all_p)}/16")
check("8 of them p < 1e-4", sum(p < 1e-4 for p in all_p) == 8, f"{sum(p < 1e-4 for p in all_p)}")
check("all 16 survive Holm correction", holm_ok)
fg = list(flag_gap.values())
check("flagging falls 8-30 points", round(min(fg)) == 8 and round(max(fg)) == 30, f"{min(fg):.1f}-{max(fg):.1f}")
check("commitment rises 9-37 points", round(min(commit_gap)) == 9 and round(max(commit_gap)) == 37,
      f"{min(commit_gap):.1f}-{max(commit_gap):.1f}")
check("commitment ratio 1.5-2.3x", round(min(commit_ratio), 1) == 1.5 and round(max(commit_ratio), 1) == 2.3,
      f"{min(commit_ratio):.2f}-{max(commit_ratio):.2f}")
fam = {k: round(v) for k, v in flag_gap.items()}
check("within family, entity present: Gemma 30->12, Qwen 20->15",
      (fam[("gemma4e2b", "present")], fam[("gemma4_26b_q4", "present")],
       fam[("qwen25_3b", "present")], fam[("qwen25_7b", "present")]) == (30, 12, 20, 15), str(fam))
check("within family, entity absent: 23->8 and 14->11",
      (fam[("gemma4e2b", "absent")], fam[("gemma4_26b_q4", "absent")],
       fam[("qwen25_3b", "absent")], fam[("qwen25_7b", "absent")]) == (23, 8, 14, 11))
check("95% CI half-width at most +/-8 points at n=150", round(1.96 * math.sqrt(.25 / 150) * 100) == 8)

# --------------------------------------------------------------------------- Table 2
print("\nTable 2  Assertion vs interrogation (entity present)")
T2 = [  # exp, domain, tag, flag quest, flag assert, p, confab quest, confab assert, p
    ("exp9_adhyasa", "GSM8K", "gemma4e2b", .867, .487, "1e-13", .033, .267, "1e-8"),
    ("exp9_adhyasa", "GSM8K", "qwen25_3b", .673, .393, "1e-8", .220, .433, "1e-6"),
    ("exp9_adhyasa", "GSM8K", "qwen25_7b", .827, .720, "0.020", .127, .273, "0.001"),
    ("exp13_squad", "SQuAD", "gemma4e2b", .947, .527, "1e-16", .020, .280, "1e-10"),
    ("exp13_squad", "SQuAD", "qwen25_3b", .980, 1.000, "0.250", .020, .000, "0.250"),
    ("exp13_squad", "SQuAD", "qwen25_7b", .940, .967, "0.344", .033, .000, "0.062"),
]
drops = {}
for exp, dom, tag, fq, fa, pf, cq, ca, pc in T2:
    df = MT.load(exp, tag)
    for col, eq, ea, ep in (("refused_v2", fq, fa, pf), ("confab", cq, ca, pc)):
        a, b = pair_rates(df, "viparyaya", "adhyasa", col)
        _, _, p = mcnemar(a.values, b.values)
        check(f"{dom} {tag:>10} {col:>10}: {eq:.3f} -> {ea:.3f}, p {ep}",
              rate_eq(a.mean(), eq) and rate_eq(b.mean(), ea) and p_eq(p, ep),
              f"{a.mean():.3f}->{b.mean():.3f} p={p:.3g}")
        if col == "refused_v2":
            drops[(dom, tag)] = (a.mean() - b.mean()) * 100
gd = [v for (dom, _), v in drops.items() if dom == "GSM8K"]
check("assertion on GSM8K: flagged 11-38 points less", round(min(gd)) == 11 and round(max(gd)) == 38,
      f"{min(gd):.1f}-{max(gd):.1f}")
check("SQuAD Gemma 42 points vs GSM8K 38",
      round(drops[("SQuAD", "gemma4e2b")]) == 42 and round(drops[("GSM8K", "gemma4e2b")]) == 38)

# --------------------------------------------------------------------------- Cascade
print("\nSection 5  Cascade (Gemma-4-E2B)")
import pandas as pd  # noqa: E402
c = pd.DataFrame(json.loads((ROOT / "results" / "exp3_cascade.json").read_text()))
ratios = []
for band, removed, corrupted in (("early", .416, .186), ("mid", .775, .225), ("late", .979, .345)):
    t = c[(c.band == band) & (c.cond == "truncate")].recovered.mean()
    k = c[(c.band == band) & (c.cond == "corrupt")].recovered.mean()
    ratios.append(t / k)
    check(f"{band:>5}: recovery {removed:.3f} removed -> {corrupted:.3f} corrupted",
          rate_eq(t, removed) and rate_eq(k, corrupted), f"{t:.3f}->{k:.3f}")
check("penalty 2.2-3.4x", round(min(ratios), 1) == 2.2 and round(max(ratios), 1) == 3.4,
      f"{min(ratios):.2f}-{max(ratios):.2f}")
check("corrupted value flagged 34% of the time", round(c[c.cond == "corrupt"].flagged.mean() * 100) == 34)

# --------------------------------------------------------------------------- Table 3
print("\nTable 3  Judging seat")
T3 = [("gemma4e2b", .033, .220, .213, "1.000", 78.0), ("qwen25_3b", .000, .013, .007, "1.000", 98.7),
      ("qwen25_7b", .000, .080, .073, "1.000", 92.0), ("gemma4_26b_q4", .020, .047, .293, "1e-10", 95.3)]
for tag, ei, es, ef, ep, epass in T3:
    d = pd.DataFrame(json.loads((ROOT / "results" / f"exp14_judge_{tag}.json").read_text()))
    g = lambda k: d[d.kind == k].flagged_invalid.mean()  # noqa: E731
    a, b = pair_rates(d, "omitted", "false", "flagged_invalid")
    _, _, p = mcnemar(a.values, b.values)
    check(f"{tag:>13}: intact {ei:.3f}, step deleted {es:.3f}, fact invented {ef:.3f}, p {ep}, passes {epass}%",
          rate_eq(g("supported"), ei) and rate_eq(g("omitted"), es) and rate_eq(g("false"), ef)
          and p_eq(p, ep) and abs(round((1 - g("omitted")) * 100, 1) - epass) < 1e-9,
          f"{g('supported'):.3f} {g('omitted'):.3f} {g('false'):.3f} p={p:.3g}")
q = json.loads((ROOT / "results" / "exp14_judge_qwen25_3b.json").read_text())
bare = sum(1 for r in q if r["kind"] == "omitted" and r["response"].strip().upper() == "VALID")
check("Qwen2.5-3B returns bare VALID for 148 of 150 items", bare == 148, f"{bare}/150")
d26 = pd.DataFrame(json.loads((ROOT / "results" / "exp14_judge_gemma4_26b_q4.json").read_text()))
a, b = pair_rates(d26, "omitted", "false", "flagged_invalid")
check("26B: fabricated 0.293 vs deleted 0.047, p < 1e-9", mcnemar(a.values, b.values)[2] < 1e-9)

# --------------------------------------------------------------------------- Mitigation
print("\nSection 8  Mitigation (Gemma-4-E2B)")
m = pd.DataFrame(json.loads((ROOT / "results" / "exp4_modes.json").read_text()))
i, k = m[m.kind == "intact"], m[m.kind == "contradictory"]
check("answerable accuracy 0.608 -> 0.392",
      rate_eq(i[i.arm == "plain"].answer_correct.mean(), .608) and rate_eq(i[i.arm == "routed"].answer_correct.mean(), .392))
check("correct REJECT on contradictory 0.300 -> 0.050",
      rate_eq(k[k.arm == "plain"].mode_correct.mean(), .300) and rate_eq(k[k.arm == "routed"].mode_correct.mean(), .050))

# --------------------------------------------------------------------------- Audit
print("\nSection 3  Metric audit (first-version rules, Gemma-4-E2B)")
raw = json.loads((ROOT / "results" / "exp12_actionable_gemma4e2b.json").read_text())
miss = lambda kind: sum(1 for r in raw if r["kind"] == kind and not r["refused"]  # noqa: E731
                        and not MT.STRICT.search(r["response"])) / 150 * 100
check("first rule missed 21.3% of actionable vs 4.0% of inert",
      round(miss("act_present"), 1) == 21.3 and round(miss("inert_present"), 1) == 4.0,
      f"{miss('act_present'):.1f}% vs {miss('inert_present'):.1f}%")

# --------------------------------------------------------------------------- Section 4 breakdown
print("\nSection 4  Flagging effect by premise type (price/tax vs other actionable premises)")
PRICE = ("sells each of them for $3", "pays $2 in tax on each one")
pt_drop, other_move, other_sig = [], [], []
for tag in ("gemma4e2b", "qwen25_3b", "qwen25_7b", "gemma4_26b_q4"):
    df = MT.load("exp12_actionable", tag)
    act = df[df.kind.str.startswith("act")].drop_duplicates("idx").set_index("idx").question
    price_ids = {i for i, q in act.items() if any(s in q for s in PRICE)}
    for ent in ("present", "absent"):
        a, b = pair_rates(df, f"inert_{ent}", f"act_{ent}", "refused_v2")
        for ids, store in ((price_ids, pt_drop), (set(a.index) - price_ids, other_move)):
            x, y = a.loc[sorted(ids)], b.loc[sorted(ids)]
            store.append((x.mean() - y.mean()) * 100)
            if store is other_move and mcnemar(x.values, y.values)[2] < .05:
                other_sig.append((tag, ent, mcnemar(x.values, y.values)[2]))
    check(f"{tag:>13}: 50 items with a price or tax premise", len(price_ids) == 50, str(len(price_ids)))
check("price/tax premises: flagging falls 32-74 points in every cell",
      round(min(pt_drop)) == 32 and round(max(pt_drop)) == 74, f"{min(pt_drop):.1f}-{max(pt_drop):.1f}")
check("other premises: flagging moves by at most 12 points",
      round(max(abs(v) for v in other_move)) == 12, f"{max(abs(v) for v in other_move):.1f}")
check("other premises: significant only for Gemma-4-E2B, entity present, p=0.004",
      len(other_sig) == 1 and other_sig[0][:2] == ("gemma4e2b", "present") and round(other_sig[0][2], 3) == .004,
      str(other_sig))

# --------------------------------------------------------------------------- Section 3 rules
print("\nSection 3  Corrected vs first-version rules (Table 1 gaps)")
larger = 0
for tag in ("gemma4e2b", "qwen25_3b", "qwen25_7b", "gemma4_26b_q4"):
    df = MT.load("exp12_actionable", tag)
    df["first_commit"] = [int(bool(MT.STRICT.search(r))) for r in df.response]
    for ent in ("present", "absent"):
        i_, a_ = df[df.kind == f"inert_{ent}"], df[df.kind == f"act_{ent}"]
        larger += (i_.refused_v2.mean() - a_.refused_v2.mean()) > (i_.refused.mean() - a_.refused.mean())
        larger += (a_.answered.mean() - i_.answered.mean()) > (a_.first_commit.mean() - i_.first_commit.mean())
check("Table 1 effect larger under the corrected rules in 13 of 16 contrasts", larger == 13, f"{larger}/16")

# --------------------------------------------------------------------------- Section 5 additions
print("\nSection 5  Holm across Table 2, cascade baseline")
t2p = []
for exp, dom, tag, *_ in T2:
    df = MT.load(exp, tag)
    for col in ("refused_v2", "confab"):
        a, b = pair_rates(df, "viparyaya", "adhyasa", col)
        t2p.append(((dom, tag, col), mcnemar(a.values, b.values)[2]))
t2p.sort(key=lambda t: t[1])
survive, running = set(), 0.0
for k, (key, pv) in enumerate(t2p):
    running = max(running, min(1.0, (len(t2p) - k) * pv))
    if running < .05:
        survive.add(key)
check("Table 2 under Holm: 7 of 12 contrasts survive", len(survive) == 7, f"{len(survive)}/12")
check("GSM8K Qwen2.5-7B flagging (p=0.020) does not survive", ("GSM8K", "qwen25_7b", "refused_v2") not in survive)
check("truncation flagged 2.5% of the time", round(c[c.cond == "truncate"].flagged.mean() * 100, 1) == 2.5)
check("cascade continuation capped at 220 new tokens",
      "max_new=220" in (ROOT / "scripts" / "exp3_cascade.py").read_text())

# --------------------------------------------------------------------------- examples
print("\nAbstract, Sections 4, 6 and 7  Quoted examples and judge setup")
sq = [r for r in json.loads((ROOT / "results" / "exp13_squad_gemma4e2b.json").read_text()) if r["idx"] == 10]
sq = {r["kind"]: r for r in sq}
check("opening example asked: 'How many charters were recorded each year by Normans?' is flagged",
      sq["viparyaya"]["question"] == "How many charters were recorded each year by Normans?"
      and bool(MT.REFUSED2.search(sq["viparyaya"]["response"])))
check("opening example asserted: 24 charters per year over 15 years -> 360, premise quoted as the passage",
      sq["adhyasa"]["question"].startswith("Given that Normans recorded 24 charters per year over 15 years")
      and sq["adhyasa"]["response"].startswith(
          'The provided passage mentions that "Normans recorded 24 charters per year over 15 years."')
      and "The total is 360." in sq["adhyasa"]["response"])
jo = {r["kind"]: r for r in raw if r["idx"] == 11}
check("Jones item: premises 'sells each of them for $3' / 'was born in Ohio'",
      "Jones sells each of them for $3" in jo["act_present"]["question"]
      and "Jones was born in Ohio" in jo["inert_present"]["question"])
check("Jones item: quoted Gemma-4-E2B responses",
      "Since there is no financial information provided in the problem, the total in dollars cannot be determined."
      in jo["inert_present"]["response"] and "The answer is 42." in jo["act_present"]["response"])
import ast  # noqa: E402
fab = next(n for n in ast.walk(ast.parse((ROOT / "scripts" / "exp14_judge.py").read_text()))
           if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FABRICATED")
check("judge: five generic invented sentences", len(fab.value.elts) == 5)

# --------------------------------------------------------------------------- verdict
print(f"\n{N_CHECKS - len(FAILS)}/{N_CHECKS} checks passed")
if FAILS:
    print("FAILED:\n  " + "\n  ".join(FAILS))
    sys.exit(1)
print("Every number checked matches the paper.")
