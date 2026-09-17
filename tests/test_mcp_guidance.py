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
import json
import sys
from pathlib import Path

import anyio
import pytest

from model_doctor.app import mcp_guidance, mcp_payloads, mcp_server
from model_doctor.app.mcp_guidance import PromptError
from model_doctor.app.mcp_server import create_server

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mcp_findings import db  # noqa: E402

__all__ = ["db"]


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
        assert create_server().instructions == mcp_guidance.ENGINE_INSTRUCTIONS
        assert len(mcp_guidance.ENGINE_INSTRUCTIONS) > 500

    def test_they_are_the_shared_text_named_for_this_server(self):
        """One text, with only the tool names this server serves differently."""
        rendered = mcp_guidance.for_surface(
            mcp_guidance.INSTRUCTIONS, mcp_guidance.ENGINE_TOOL_ALIASES
        )
        assert rendered == mcp_guidance.ENGINE_INSTRUCTIONS
        assert "list_analyses" not in mcp_guidance.ENGINE_INSTRUCTIONS
        assert "`list_runs`" in mcp_guidance.ENGINE_INSTRUCTIONS

    def test_they_do_not_claim_every_response_carries_caveats(self):
        """The listing and the images carry none; saying otherwise was false."""
        text = flat(mcp_guidance.INSTRUCTIONS)
        assert "Every response carries" not in text
        assert (
            "`get_analysis`, `list_findings` and `get_finding` carry `caveats`"
            in text
        )
        assert "`list_analyses` and the two image tools carry neither" in text

    def test_the_caveats_claim_matches_the_payloads(self, db):
        """Checked against what the payloads return, not against the prose."""
        carries = {
            "list_analyses": mcp_payloads.list_runs_payload(db.conn),
            "get_analysis": mcp_payloads.get_analysis_payload(db.conn, [db.run_a]),
            "list_findings": mcp_payloads.list_findings_payload(db.conn, db.run_a),
            "get_finding": mcp_payloads.get_finding_payload(
                db.conn, db.run_a, db.ids_a[0]
            ),
        }
        assert {name for name, p in carries.items() if p.get("caveats")} == {
            "get_analysis", "list_findings", "get_finding",
        }
        assert "caveats" not in carries["list_analyses"]
        assert "evidence_gaps" not in carries["list_analyses"]

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


def _served_texts(server) -> list[str]:
    """Everything this server puts in front of a client that can name a tool.

    The instructions, every tool's title, description and schema, every
    prompt's advertised text, and each prompt rendered the way this server's
    prompt handler renders it — refusals included, since a refusal tells a
    reader what to call next. Prompts are rendered through
    ``mcp_server.TOOL_NAMES`` because that is what the handler passes;
    ``MCPServer.get_prompt`` cannot yet drive the handler (see the report).
    """
    texts = [server.instructions or ""]
    for tool in anyio.run(server.list_tools):
        texts += [tool.title or "", tool.description or ""]
        texts.append(json.dumps(tool.input_schema))
    for prompt in anyio.run(server.list_prompts):
        texts += [prompt.title or "", prompt.description or ""]
        texts += [a.description or "" for a in prompt.arguments or []]
    args = {"run_id": 7, "run_ids": "6, 7", "outcome": "false_negative"}
    for name in mcp_guidance.PROMPTS:
        texts.append(
            mcp_guidance.build_prompt(name, args, mcp_server.TOOL_NAMES)["text"]
        )
        with pytest.raises(PromptError) as refused:
            mcp_guidance.build_prompt(name, {}, mcp_server.TOOL_NAMES)
        texts.append(str(refused.value))
    return texts


def _served_tools(server) -> set[str]:
    return {t.name for t in anyio.run(server.list_tools)}


class TestTheDriftGuard:
    """Guidance naming a tool that does not exist is worse than none.

    The guard reads the text a client receives and compares every tool name in
    it with the registry a client sees. The earlier guard compared an alias
    table with the registry, and passed while this server told readers to call
    `list_analyses`, which it does not serve.
    """

    def test_nothing_this_server_sends_names_a_tool_it_does_not_serve(self):
        server = create_server()
        texts = _served_texts(server)
        assert mcp_guidance.tool_references(server.instructions), "names no tools"
        assert mcp_guidance.unserved_tool_references(
            texts, _served_tools(server)
        ) == set()

    def test_every_renamed_tool_is_one_this_server_serves(self):
        """The mapping itself must point at real tools, or rendering hides a gap."""
        live = _served_tools(create_server())
        assert set(mcp_server.TOOL_NAMES.values()) <= live

    def test_the_guard_fails_on_the_text_this_server_used_to_send(self):
        """The regression: the unrendered shared text names `list_analyses`."""
        live = _served_tools(create_server())
        assert mcp_guidance.unserved_tool_references(
            [mcp_guidance.INSTRUCTIONS], live
        ) == {"list_analyses"}

    def test_the_guard_fails_when_the_instructions_name_an_unserved_tool(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            mcp_server,
            "INSTRUCTIONS",
            mcp_guidance.ENGINE_INSTRUCTIONS + " Then call `get_everything`.",
        )
        server = create_server()
        assert mcp_guidance.unserved_tool_references(
            _served_texts(server), _served_tools(server)
        ) == {"get_everything"}

    def test_the_guard_fails_when_the_prompts_lose_the_rename(self, monkeypatch):
        """A prompt refusal naming `list_analyses` here is caught too."""
        monkeypatch.setattr(mcp_server, "TOOL_NAMES", {})
        server = create_server()
        assert mcp_guidance.unserved_tool_references(
            _served_texts(server), _served_tools(server)
        ) == {"list_analyses"}

    def test_references_are_found_without_backticks(self):
        """`Call list_analyses first` is as much an instruction as a backticked one."""
        assert mcp_guidance.tool_references(
            "Call list_analyses first, then `get_finding`."
        ) == {"list_analyses", "get_finding"}

    def test_rendering_replaces_whole_names_only(self):
        rendered = mcp_guidance.for_surface(
            "`list_analyses`, xlist_analyses, list_analyses_old",
            mcp_guidance.ENGINE_TOOL_ALIASES,
        )
        assert rendered == "`list_runs`, xlist_analyses, list_analyses_old"

    def test_no_mapping_leaves_the_text_as_written(self):
        assert mcp_guidance.for_surface(mcp_guidance.INSTRUCTIONS) == (
            mcp_guidance.INSTRUCTIONS
        )


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
