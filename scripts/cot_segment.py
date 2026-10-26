"""Segment a chain of thought into labelled spans.

  DIRECTING   names a step without performing it ("**Calculate the total:**")
  SUSTAINING  performs a computation ("Total = $3 + 4 = 7$")
  OTHER       restatements of the problem, preamble, and the final answer line

The cascade (paper section 5) and judging (section 7) experiments treat SUSTAINING spans
as the computational steps they corrupt or delete.
"""

from __future__ import annotations

import re

DIRECTING, SUSTAINING, OTHER = "directing", "sustaining", "other"

# LaTeX must be normalised before arithmetic detection, or spans like
#   $$\text{Total Cost} = \$80,000 + \$50,000 = \$130,000$$
# are missed entirely because of the \$ escapes and \text{} wrappers.
_TEXCMD = re.compile(r"\\text\{([^}]*)\}")
_TEXOPS = [(r"\\times", "*"), (r"\\cdot", "*"), (r"\\div", "/"),
           (r"\\frac", " "), (r"\\%", "%")]


def normalise(s: str) -> str:
    s = _TEXCMD.sub(r"\1", s)
    for pat, rep in _TEXOPS:
        s = re.sub(pat, rep, s)
    s = s.replace("\\$", "$").replace("$", " ")
    s = re.sub(r"[\\{}]", " ", s)
    return re.sub(r"\s+", " ", s)


# a digit-operator-digit computation, or an equation yielding a number
_ARITH = re.compile(r"\d[\d,\.]*\s*[\+\-\*/×÷]\s*\d|"
                    r"=\s*-?\s*[\d,]+(?:\.\d+)?\s*%?")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ANSWER = re.compile(r"the answer is", re.I)
_LIST_PREFIX = re.compile(r"^\s*(?:\d+[\.\)]|[\*\-•])\s*")
_IMPERATIVE = re.compile(
    r"^\s*(calculate|determine|identify|find|compute|figure out|work out|"
    r"first|next|then|now|step)\b", re.I)


def has_arith(s: str) -> bool:
    return bool(_ARITH.search(normalise(s)))


def reasoning_only(cot: str) -> str:
    """Drop the final answer statement -- the model must re-derive it."""
    m = _ANSWER.search(cot)
    return cot[:m.start()].rstrip() if m else cot.rstrip()


def segment(cot: str) -> list[dict]:
    """Split into labelled spans, in order. Each span keeps its exact text."""
    spans: list[dict] = []
    for line in reasoning_only(cot).split("\n"):
        if not line.strip():
            continue

        # split a bold directive away from any trailing content on the same line
        m = _BOLD.search(line)
        if m and not has_arith(line[:m.end()]):
            head = line[:m.end()]
            tail = line[m.end():]
            spans.append({"text": head, "label": DIRECTING})
            if tail.strip():
                spans.append({
                    "text": tail,
                    "label": SUSTAINING if has_arith(tail) else OTHER,
                })
            continue

        if has_arith(line):
            spans.append({"text": line, "label": SUSTAINING})
        else:
            body = _LIST_PREFIX.sub("", line)
            spans.append({
                "text": line,
                "label": DIRECTING if _IMPERATIVE.match(body) else OTHER,
            })
    return spans


def render(spans: list[dict], drop: set[int]) -> str:
    """Reassemble the trace with the given span indices removed."""
    return "\n".join(s["text"] for i, s in enumerate(spans) if i not in drop)
