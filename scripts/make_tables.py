"""Generate the paper's tables (paper/tables_auto.tex) from results/.

Every number in Tables 1-3 is produced here; captions were edited in the paper source."""
from __future__ import annotations
import json, pathlib, re, sys
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sigtest import mcnemar

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Metrics revised after a 40-response manual audit. The original pair
# under-counted "answered" ~5x more often for actionable items than inert ones,
# biasing the headline contrast toward zero. These are stricter/broader and the
# effect is larger under them.
STRICT = re.compile(r"answer is[^.\n]{0,40}?(-?[\d,]+(?:\.\d+)?)", re.I)
ANSWERED = re.compile(r"answer is[^.\n]{0,40}?[\d]|=\s*\$?\s*[\d,]+(?:\.\d+)?\s*(?:\.|$|\s)", re.I|re.M)
REFUSED2 = re.compile(
    r"(?:cannot|can'?t|unable to) (?:be )?(?:answer|determin|calculat|comput)|"
    r"no information (?:is )?(?:given|provided|about)|"
    r"does not (?:give|provide|state|mention|specify|contain) (?:any )?information|"
    r"not (?:enough|sufficient) information|insufficient information", re.I)
MODELS = [("gemma4e2b", "Gemma-4-E2B"), ("qwen25_3b", "Qwen2.5-3B"),
          ("qwen25_7b", "Qwen2.5-7B"), ("gemma4_26b_q4", "Gemma-4-26B$^{q}$")]


def load(exp, tag):
    p = ROOT / "results" / f"{exp}_{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    for r in d:
        if "confab" not in r:
            r["confab"] = int(bool(STRICT.search(r["response"])))
        if "response" in r:
            r["answered"] = int(bool(ANSWERED.search(r["response"])))
            r["refused_v2"] = int(bool(REFUSED2.search(r["response"])))
    return pd.DataFrame(d)


def pair(df, x, y, col):
    a = df[df.kind == x].set_index("idx")[col]
    b = df[df.kind == y].set_index("idx")[col]
    c = a.index.intersection(b.index)
    n10, n01, p = mcnemar(a.loc[c].values, b.loc[c].values)
    return a.loc[c].mean(), b.loc[c].mean(), p


def fp(p):
    # Matches the camera-ready: order of magnitude below 1e-4, otherwise three decimals,
    # with "<0.001" where three decimals would print 0.000.
    if p < 1e-4:
        m, e = f"{p:.0e}".split("e")
        return f"$10^{{{int(e)}}}$"
    s = f"{p:.3f}"
    return "$<$0.001" if s == "0.000" else s


def t_action():
    rows = []
    for tag, name in MODELS:
        df = load("exp12_actionable", tag)
        if df is None:
            continue
        for ent, x, y in (("present", "inert_present", "act_present"),
                          ("absent", "inert_absent", "act_absent")):
            ar, br, pr = pair(df, x, y, "refused_v2")
            ac, bc, pc = pair(df, x, y, "answered")
            rows.append(f"{name} & {ent} & {ar:.3f} & {br:.3f} & {fp(pr)} & "
                        f"{ac:.3f} & {bc:.3f} & {fp(pc)} \\\\")
    if not rows: return ""
    return (r"""\begin{table}[t]\centering
\caption{\textbf{Actionability.} Identical framing, entity and prompt; only whether
the false premise can be \emph{used} varies. Paired McNemar over 150 items.
Metrics are the audited ones (\S\ref{sec:metrics}). $^{q}$4-bit quantised.}
\label{tab:action}\small
\begin{tabular}{llcccccc}
\toprule
& & \multicolumn{3}{c}{flagged unanswerable} & \multicolumn{3}{c}{committed to answer} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Model & Entity & inert & action. & $p$ & inert & action. & $p$ \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule\end{tabular}\end{table}
""")


def t_assert():
    rows = []
    for exp, dom in (("exp9_adhyasa", "GSM8K"), ("exp13_squad", "SQuAD")):
        for tag, name in MODELS:
            df = load(exp, tag)
            if df is None:
                continue
            # Unified with Table 1 on the audited rule. The per-experiment `refused`
            # field used each script's original regex, so the two tables previously
            # reported different measurements under an identical column header.
            ar, br, pr = pair(df, "viparyaya", "adhyasa", "refused_v2")
            ac, bc, pc = pair(df, "viparyaya", "adhyasa", "confab")
            rows.append(f"{dom} & {name} & {ar:.3f} & {br:.3f} & {fp(pr)} & "
                        f"{ac:.3f} & {bc:.3f} & {fp(pc)} \\\\")
    if not rows: return ""
    return (r"""\begin{table}[t]\centering
\caption{\textbf{Assertion vs.\ interrogation.} The same missing fact posed as a
question versus asserted as a premise, entity held present.}
\label{tab:assert}\small
\begin{tabular}{llcccccc}
\toprule
& & \multicolumn{3}{c}{flagged unanswerable} & \multicolumn{3}{c}{confabulated} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Domain & Model & quest. & assert. & $p$ & quest. & assert. & $p$ \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule\end{tabular}\end{table}
""")


def t_cascade():
    p = ROOT / "results" / "exp3_cascade.json"
    if not p.exists(): return ""
    df = pd.DataFrame(json.loads(p.read_text()))
    rows = []
    for band in ("early", "mid", "late"):
        c = df[(df.band == band) & (df.cond == "corrupt")]
        t = df[(df.band == band) & (df.cond == "truncate")]
        rows.append(f"{band} & {len(c)} & {c.recovered.mean():.3f} & "
                    f"{t.recovered.mean():.3f} & {t.recovered.mean()/max(c.recovered.mean(),1e-9):.1f}$\\times$ & "
                    f"{c.flagged.mean():.3f} \\\\")
    return (r"""\begin{table}[t]\centering
\caption{\textbf{A false value is worse than no value.} Recovery of the correct
final answer after one intermediate quantity is corrupted, versus simply removed.
Gemma-4-E2B on GSM8K.}
\label{tab:cascade}\small
\begin{tabular}{lccccc}
\toprule
Position & $n$ & corrupted & removed & ratio & flagged \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule\end{tabular}\end{table}
""")


def t_judge():
    rows=[]
    for tag,name in MODELS:
        f=ROOT/"results"/f"exp14_judge_{tag}.json"
        if not f.exists(): continue
        df=pd.DataFrame(json.loads(f.read_text()))
        r={k:df[df.kind==k].flagged_invalid.mean() for k in ("supported","omitted","false")}
        _,_,pv=pair(df,"omitted","false","flagged_invalid")
        rows.append(f"{name} & {r['supported']:.3f} & {r['omitted']:.3f} & {r['false']:.3f} & "
                    f"{fp(pv)} & {1-r['omitted']:.1%} \\\\".replace("%","\\%"))
    if not rows: return ""
    return (r"""\begin{table}[t]\centering
\caption{\textbf{The judging seat.} Rate at which a judge marks a proof INVALID.
The predicted asymmetry is absent at $\le$7B ($p=1$) and inverts at 26B.
Acceptance of a deleted-step proof stays high at every scale.}
\label{tab:judge}\small
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{flagged INVALID} & & accepts a \\
\cmidrule(lr){2-4}
Judge & intact & step deleted & fact invented & $p$ & broken proof \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule\end{tabular}\end{table}
""")


def main():
    out = ["% AUTO-GENERATED by scripts/make_tables.py -- do not edit by hand.\n",
           t_action(), t_assert(), t_judge()]
    (ROOT / "paper").mkdir(exist_ok=True)
    (ROOT / "paper" / "tables_auto.tex").write_text("\n".join(out))
    print("wrote paper/tables_auto.tex")
    for e in ("exp12_actionable", "exp9_adhyasa", "exp13_squad"):
        print(f"  {e}: {[n for t,n in MODELS if load(e,t) is not None]}")


if __name__ == "__main__":
    main()
