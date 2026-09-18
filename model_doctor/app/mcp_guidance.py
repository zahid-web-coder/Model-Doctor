"""What a reader should be told before they read the evidence.

The tools answer questions; this module says how to ask them and what the
answers do not mean. It exists for the same reason :mod:`mcp_payloads` does:
two surfaces serve these tools — the engine's own MCP server and Ramanujan's —
and guidance that drifts between them is worse than none, because a reader
cannot tell which version they were given.

The same two rules apply here as in the payload layer, and for the same
reasons:

* **Nothing here imports ``mcp``.** Prompts are returned as plain dictionaries
  and each caller shapes them into its own protocol types.
* **Nothing here opens a database.** These are static texts; a prompt that
  needed to read the database to describe itself would be a tool.

One thing this module cannot do is guarantee delivery. The MCP ``instructions``
field is the cheapest way to put this in front of a model — clients may add it
to the system prompt — but it is optional for a client to use and is removed
from the protocol in ``2026-07-28``. So it is treated as a convenience, not a
foundation: the prompts below and the repository's skill file carry the same
content through mechanisms that do not expire.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

#: Every tool a reader may be told to call, by the name Ramanujan's surface
#: serves it under. The texts below are written with these names; a surface
#: that serves a tool under another name renders them with
#: :func:`for_surface` before sending them.
TOOL_NAMES: tuple[str, ...] = (
    "list_analyses",
    "get_analysis",
    "list_findings",
    "get_finding",
    "get_image",
    "get_heatmap_image",
)

#: The engine's own server exposes the same capabilities under older names.
#: Used to render the texts the engine sends, so its readers are told the names
#: its registry actually serves. The drift guard checks the rendered text, not
#: this mapping: a mapping can be right while the text sent is wrong.
ENGINE_TOOL_ALIASES: dict[str, str] = {
    "list_analyses": "list_runs",
    "get_analysis": "get_analysis",
    "list_findings": "list_findings",
    "get_finding": "get_finding",
    "get_image": "get_image",
    "get_heatmap_image": "get_heatmap_image",
}

# ---------------------------------------------------------------------------
# What every caller is told up front
# ---------------------------------------------------------------------------

#: Sent as the MCP server's ``instructions``. Deliberately about *reading*
#: rather than about the API: the schemas already describe the arguments, and
#: the thing a model gets wrong is not which field to pass but what a number
#: means once it has it.
INSTRUCTIONS = """\
Model Doctor diagnoses why an object-detection model fails. These tools expose
its stored analysis, read-only. Nothing here starts work, changes a run or
retrains anything.

Work outward from the cheapest evidence. `list_analyses` says which analyses
exist and what evidence each carries. `get_analysis` returns one analysis — or
several, compared — with outcome counts, factors, failure groups and
recommendations. That answers most questions on its own. Only when a specific
case is in doubt go on to `list_findings` for the individual cases behind a
rate, then `get_finding` for one case's evidence. Do not call a later tool when
an earlier answer already settles the question.

Treat every figure as evidence and keep your own reasoning visibly separate
from it.

Outcome counts are not COCO mAP and will not agree with it; they are the
matcher's verdicts at this run's thresholds. A factor is a correlation against
a control rate, never a proven cause — quote its failure rate and its correct
rate together or quote neither, and read `qualifies: false` as "did not meet
the rule", not "unimportant". `null` means not measured; it never means zero. A
recommendation's `status` bounds how much weight it bears: only `replicated`,
`reproduced` and `provisional` are actionable, and `reproduced` means the only
runs that agreed were re-executions of the same configuration. The `relations`
block is measured geometry — which prediction covers which object — not
causality. One finding is an anecdote; rates come from `get_analysis`.

`get_analysis`, `list_findings` and `get_finding` carry `caveats`, and
`get_analysis` also carries `evidence_gaps`; `list_analyses` and the two image
tools carry neither. Read them and pass on what they say. If the stored evidence
does not answer the question, say so and name the pass that would answer it.
Never fill a gap with a plausible guess.

Images are for genuinely visual questions — whether an object is visible at
all, occluded, or too dark to detect — not for counting or ranking, where the
stored numbers are better than looking. Call `get_image` only when such a
question is open, and only for as many images as the question needs.
`get_heatmap_image` is worth calling only when `get_finding` has already
reported that a heatmap exists for that finding: explanation is implemented for
some detector families and not others, so a run without heatmaps is a normal
state, not a transient failure. Do not retry, and do not probe other findings
hoping to find one. Read an overlay as where the model attended, not as why it
was right or wrong.

Everything is asked for by id. There is no path, URL or bucket argument, and a
refusal is an answer: an analysis that is not visible to this credential does
not become visible by trying variations. Image retrieval is bounded per
credential and every retrieval is recorded. The photographs are of real sites —
look at them for the analysis in hand, do not infer identity, location or
ownership, and do not copy their contents into anything that outlives the
question."""


# ---------------------------------------------------------------------------
# Rendering for a surface, and the drift check
# ---------------------------------------------------------------------------

#: What a tool name looks like in prose, backticked or not. Every analysis
#: tool either surface serves starts ``list_`` or ``get_``, and no field or
#: word in these texts does, so the pattern finds the references and only them.
_TOOL_LIKE = re.compile(r"\b(?:list|get)_[a-z0-9_]+\b")


def for_surface(text: str, tool_names: Mapping[str, str] | None = None) -> str:
    """Render ``text`` with each shared tool name replaced by a surface's own.

    ``tool_names`` maps a name in :data:`TOOL_NAMES` to the name that surface
    serves; ``None`` leaves the text as written, which is Ramanujan's naming.
    Whole words only, so a name is never rewritten inside a longer one.
    """
    renames = {old: new for old, new in (tool_names or {}).items() if old != new}
    if not renames:
        return text
    names = sorted(renames, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
    return pattern.sub(lambda match: renames[match.group(1)], text)


def tool_references(text: str) -> set[str]:
    """Every tool name ``text`` points a reader at, backticked or not."""
    return set(_TOOL_LIKE.findall(text))


def unserved_tool_references(texts: Iterable[str], served: Iterable[str]) -> set[str]:
    """The tool names these texts mention that ``served`` does not contain.

    Checked against the registry a client actually sees, never against
    :data:`TOOL_NAMES` or an alias table: a name can be "known" to this module
    and still absent from the server that sent it.
    """
    registry = set(served)
    mentioned: set[str] = set()
    for text in texts:
        mentioned |= tool_references(text)
    return mentioned - registry


#: The same instructions, naming the tools the engine's own server serves.
ENGINE_INSTRUCTIONS = for_surface(INSTRUCTIONS, ENGINE_TOOL_ALIASES)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
#
# Three, because three workflows are genuinely distinct and a longer list would
# be a menu nobody reads. Each is a starting move that a person picks
# deliberately, not an instruction the model receives unasked.


def _diagnose(arguments: dict[str, Any]) -> str:
    run_id = arguments["run_id"]
    return f"""\
Diagnose analysis {run_id} using the Model Doctor tools, working outward from
the cheapest evidence.

1. `get_analysis` with run_ids [{run_id}]. Report the outcome counts, the
   headline evaluation if one exists, and the factors that qualify — each with
   its failure rate *and* its control rate, so the comparison is visible.
2. Read `evidence_gaps` and `caveats` and say plainly what this analysis cannot
   answer.
3. Only if a particular failure mode is unclear, use `list_findings` filtered
   to that outcome to look at the individual cases, and `get_finding` for one
   or two of them.
4. Use `get_image` only if the open question is whether something was visible
   at all. Use `get_heatmap_image` only if `get_finding` reported one exists.

Finish with what the evidence supports, what it does not, and which unrun pass
would settle anything still open. Do not offer a cause the factors only
correlate with."""


def _compare(arguments: dict[str, Any]) -> str:
    run_ids = arguments["run_ids"]
    baseline = arguments.get("baseline")
    against = f" against baseline {baseline}" if baseline else ""
    return f"""\
Compare Model Doctor analyses {run_ids}{against}.

Call `get_analysis` once with all of them; it returns the comparison in one
structure rather than requiring a call each. Report:

- what differs in the outcome counts and evaluation, as deltas;
- which factors and failure groups replicate across the runs and which appear
  in only one;
- what `comparability` says about whether these runs can fairly be compared at
  all, and what `evidence_gaps` says each run lacks.

Two runs of the same model at different settings are the comparison that can
support a conclusion. Runs of different models, or runs measured under
different protocols, mostly cannot — say so rather than comparing anyway.
`reproduced` means only re-executions of one configuration agreed, which is
weaker than it sounds."""


def _investigate(arguments: dict[str, Any]) -> str:
    run_id = arguments["run_id"]
    outcome = arguments["outcome"]
    return f"""\
Investigate the `{outcome}` findings in analysis {run_id}.

1. `get_analysis` first, for the rate this sits inside. A count means nothing
   without the population it came from.
2. `list_findings` with run_id {run_id} and outcome `{outcome}`. Note the total
   against the analysis's whole outcome counts, which travel with every page.
3. `get_finding` on a few of them — not all — and report what the stored passes
   attributed: factors, the outline result if masks were measured, the groups
   the finding was placed in.
4. If the open question is visual, `get_image` for one or two of them.
   `get_heatmap_image` only where `get_finding` said a heatmap exists.

Report what these cases have in common and how strongly the analysis-level
factors support that reading. Individual cases illustrate a pattern; they do
not establish one."""


#: name -> definition and builder. The builder returns the user-message text;
#: the caller wraps it in whatever its protocol layer expects.
PROMPTS: dict[str, dict[str, Any]] = {
    "diagnose_analysis": {
        "definition": {
            "name": "diagnose_analysis",
            "title": "Diagnose one analysis",
            "description": (
                "Work one analysis from its rates to its causes, in the order "
                "the evidence supports, and say what it cannot answer."
            ),
            "arguments": [
                {
                    "name": "run_id",
                    "description": "Which analysis, by run_id from list_analyses.",
                    "required": True,
                }
            ],
        },
        "build": _diagnose,
    },
    "compare_analyses": {
        "definition": {
            "name": "compare_analyses",
            "title": "Compare analyses",
            "description": (
                "Compare two or more analyses in one call, reporting deltas, "
                "what replicates, and whether they are comparable at all."
            ),
            "arguments": [
                {
                    "name": "run_ids",
                    "description": "The analyses to compare, e.g. `6, 7`.",
                    "required": True,
                },
                {
                    "name": "baseline",
                    "description": (
                        "Which analysis the deltas are measured against. "
                        "Defaults to the first."
                    ),
                    "required": False,
                },
            ],
        },
        "build": _compare,
    },
    "investigate_failure_mode": {
        "definition": {
            "name": "investigate_failure_mode",
            "title": "Investigate one failure mode",
            "description": (
                "Drill from a rate into the individual findings behind it "
                "without mistaking the cases for the pattern."
            ),
            "arguments": [
                {
                    "name": "run_id",
                    "description": "Which analysis, by run_id from list_analyses.",
                    "required": True,
                },
                {
                    "name": "outcome",
                    "description": (
                        "Which outcome to investigate: wrong_class, "
                        "poor_localization, false_positive or false_negative."
                    ),
                    "required": True,
                },
            ],
        },
        "build": _investigate,
    },
}


class PromptError(Exception):
    """A prompt could not be built, with a message written for the caller.

    Deliberately not ``McpError``: this module cannot import ``mcp``. Each
    surface maps it into its own vocabulary, exactly as ``PayloadError`` is
    mapped in the payload layer.
    """


def _render(value: Any, tool_names: Mapping[str, str] | None) -> Any:
    """Apply :func:`for_surface` to every string inside a definition."""
    if isinstance(value, str):
        return for_surface(value, tool_names)
    if isinstance(value, list):
        return [_render(item, tool_names) for item in value]
    if isinstance(value, dict):
        return {key: _render(item, tool_names) for key, item in value.items()}
    return value


def prompt_definitions(
    tool_names: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Every prompt as ``prompts/list`` advertises it, named for a surface."""
    return [_render(entry["definition"], tool_names) for entry in PROMPTS.values()]


def build_prompt(
    name: str,
    arguments: dict[str, Any] | None,
    tool_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one prompt, validating its arguments first.

    Arguments are checked the way tool arguments are — a missing required one
    is refused with its name rather than rendered as ``None`` into the text,
    which would produce a confident instruction to look at analysis ``None``.
    ``tool_names`` renders the text and any refusal for a surface that serves
    tools under other names; see :func:`for_surface`.
    """
    try:
        return _build_prompt(name, arguments, tool_names)
    except PromptError as error:
        raise PromptError(for_surface(str(error), tool_names)) from None


def _build_prompt(
    name: str,
    arguments: dict[str, Any] | None,
    tool_names: Mapping[str, str] | None,
) -> dict[str, Any]:
    entry = PROMPTS.get(name)
    if entry is None:
        raise PromptError(
            f"No such prompt: {name!r}. Available: "
            f"{', '.join(sorted(PROMPTS))}."
        )
    supplied = dict(arguments or {})
    missing = [
        a["name"]
        for a in entry["definition"]["arguments"]
        if a.get("required") and not str(supplied.get(a["name"], "")).strip()
    ]
    if missing:
        raise PromptError(
            f"{name} needs {', '.join(missing)}. "
            "Call list_analyses first if you do not have the id."
        )
    return {
        "description": for_surface(entry["definition"]["description"], tool_names),
        "text": for_surface(entry["build"](supplied), tool_names),
    }
