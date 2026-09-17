"""The shared guidance: what both surfaces tell a reader before they read.

Two properties matter more than the prose. The first is that this module stays
free of the protocol and the database, like the payload layer it sits beside —
Ramanujan vendors it and cannot import `mcp`. The second is drift: guidance
that names a tool which has been renamed is worse than no guidance, because it
sends a reader after something that is not there. The test for that compares
the names in the text against this server's own registry rather than against a
list written here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import anyio
import pytest

from model_doctor.app import mcp_guidance
from model_doctor.app.mcp_guidance import PromptError
from model_doctor.app.mcp_server import create_server


def flat(text: str) -> str:
    """Collapse wrapping, so assertions are about words not line breaks.

    The texts are hard-wrapped for reading; a phrase that happens to straddle
    a newline is still present, and a test that missed it would be testing the
    formatter rather than the content.
    """
    return " ".join(text.split())


class TestItCarriesNoProtocolAndOpensNoDatabase:
    """The two rules that let a second surface vendor this."""

    def test_it_imports_neither_mcp_nor_pydantic(self):
        """Ramanujan forbids `mcp`, and a test there enforces the ban."""
        source = Path(mcp_guidance.__file__).read_text()
        imported: list[str] = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not [m for m in imported if m.split(".")[0] in {"mcp", "pydantic"}]

    def test_it_opens_no_database(self):
        """A prompt that had to read the database would be a tool."""
        source = Path(mcp_guidance.__file__).read_text()
        for forbidden in ("sqlite3", "storage.connect", "connect_read_only"):
            assert forbidden not in source


class TestTheInstructions:
    """What every client is told before it calls anything."""

    def test_the_server_sends_them(self):
        """Empty instructions are the same as none."""
        assert create_server().instructions == mcp_guidance.INSTRUCTIONS
        assert len(mcp_guidance.INSTRUCTIONS) > 500

    @pytest.mark.parametrize("rule", [
        "not COCO mAP",
        "correlation against a control rate",
        "never means zero",
        "anecdote",
        "measured geometry",
        "Never fill a gap",
    ])
    def test_every_epistemic_rule_is_stated(self, rule):
        """These are the readings that go wrong without being told."""
        assert rule in flat(mcp_guidance.INSTRUCTIONS)

    def test_it_states_the_cheap_before_expensive_order(self):
        """The workflow is the point, not a list of tool names."""
        text = mcp_guidance.INSTRUCTIONS
        assert text.index("list_analyses") < text.index("get_analysis")
        assert text.index("get_analysis") < text.index("list_findings")
        assert text.index("list_findings") < text.index("get_finding")
        assert text.index("get_finding") < text.index("get_image")
        assert "Do not call a later tool" in text

    def test_it_says_when_not_to_ask_for_a_heatmap(self):
        """The retry loop this prevents is the reason it is written down."""
        text = flat(mcp_guidance.INSTRUCTIONS)
        assert "Do not retry" in text
        assert "not a transient failure" in text

    def test_it_carries_the_security_rules(self):
        """Ids only, refusals respected, nothing inferred about people."""
        text = flat(mcp_guidance.INSTRUCTIONS)
        assert "no path, URL or bucket argument" in text
        assert "a refusal is an answer" in text
        assert "identity, location or ownership" in text

    def test_it_contains_no_path_credential_or_host(self):
        """Guidance travels to every client; it must carry nothing internal."""
        text = flat(mcp_guidance.INSTRUCTIONS)
        for forbidden in ("/pod", "/workspace", "rmd_", "Bearer", "mongodb", "http://"):
            assert forbidden not in text


class TestTheDriftGuard:
    """Guidance naming a tool that does not exist is worse than none."""

    def test_every_tool_named_in_the_guidance_exists_on_this_server(self):
        """Compared against the live registry, not a hand-kept list."""
        live = {t.name for t in anyio.run(create_server().list_tools)}
        for shared in mcp_guidance.TOOL_NAMES:
            engine = mcp_guidance.ENGINE_TOOL_ALIASES[shared]
            assert engine in live, f"{shared} -> {engine} is not served here"

    def test_every_tool_name_in_the_prose_is_one_of_those(self):
        """Catches a tool invented in the text but never built."""
        args = {"run_id": 1, "run_ids": "1, 2", "outcome": "false_negative"}
        text = mcp_guidance.INSTRUCTIONS + "".join(
            entry["build"](args) for entry in mcp_guidance.PROMPTS.values()
        )
        mentioned = set(re.findall(r"`([a-z_]+)`", text))
        known = set(mcp_guidance.TOOL_NAMES) | set(
            mcp_guidance.ENGINE_TOOL_ALIASES.values()
        )
        looks_like_a_tool = {m for m in mentioned if m.startswith(("list_", "get_"))}
        assert looks_like_a_tool <= known, looks_like_a_tool - known


class TestThePrompts:
    """Three workflows, each a starting move somebody picks deliberately."""

    def test_the_server_offers_exactly_these(self):
        """A longer list is a menu nobody reads."""
        offered = {p.name for p in anyio.run(create_server().list_prompts)}
        assert offered == set(mcp_guidance.PROMPTS)
        assert len(offered) == 3

    def test_each_definition_is_complete(self):
        """A prompt without a description is one nobody will choose."""
        for definition in mcp_guidance.prompt_definitions():
            assert definition["name"]
            assert definition["title"]
            assert len(definition["description"]) > 30
            for argument in definition["arguments"]:
                assert argument["name"]
                assert argument["description"]
                assert isinstance(argument["required"], bool)

    def test_a_built_prompt_names_the_analysis_asked_for(self):
        """The argument has to reach the text, or it is decoration."""
        built = mcp_guidance.build_prompt("diagnose_analysis", {"run_id": 7})
        assert "analysis 7" in built["text"]
        assert built["description"]

    def test_the_comparison_prompt_carries_its_baseline(self):
        """Optional arguments change the text when given."""
        with_base = mcp_guidance.build_prompt(
            "compare_analyses", {"run_ids": "6, 7", "baseline": "6"}
        )["text"]
        without = mcp_guidance.build_prompt(
            "compare_analyses", {"run_ids": "6, 7"}
        )["text"]
        assert "baseline 6" in with_base
        assert "baseline" not in without.split("Call `get_analysis`")[0]

    def test_a_missing_required_argument_is_refused_by_name(self):
        """Rendering `None` would instruct somebody to read analysis None."""
        with pytest.raises(PromptError, match="needs run_id"):
            mcp_guidance.build_prompt("diagnose_analysis", {})
        with pytest.raises(PromptError, match="needs outcome"):
            mcp_guidance.build_prompt("investigate_failure_mode", {"run_id": 7})

    def test_a_blank_argument_counts_as_missing(self):
        """Whitespace is not an analysis id."""
        with pytest.raises(PromptError, match="needs run_id"):
            mcp_guidance.build_prompt("diagnose_analysis", {"run_id": "   "})

    def test_an_unknown_prompt_names_what_does_exist(self):
        """A refusal that lists the alternatives saves a second round trip."""
        with pytest.raises(PromptError, match="No such prompt"):
            mcp_guidance.build_prompt("retrain_everything", {})

    def test_every_prompt_keeps_the_evidence_first_order(self):
        """A prompt that jumped to images would undo the instructions."""
        for name in mcp_guidance.PROMPTS:
            args = {"run_id": 7, "run_ids": "6, 7", "outcome": "false_negative"}
            text = mcp_guidance.build_prompt(name, args)["text"]
            if "get_image" in text:
                assert text.index("get_analysis") < text.index("get_image")
            if "get_heatmap_image" in text:
                assert "get_finding" in text

    def test_no_prompt_invites_a_conclusion_the_data_cannot_support(self):
        """Each one says where to stop as well as where to start."""
        for name in mcp_guidance.PROMPTS:
            args = {"run_id": 7, "run_ids": "6, 7", "outcome": "false_negative"}
            text = flat(mcp_guidance.build_prompt(name, args)["text"]).lower()
            assert any(
                phrase in text
                for phrase in ("do not", "cannot", "does not", "weaker than")
            ), name
