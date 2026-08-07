# Coding Standards

Enforced mechanically where possible. `pyproject.toml` holds the tool
configuration, so these rules are checkable rather than aspirational:

```bash
./.venv/bin/python -m ruff check .
./.venv/bin/python -m pytest
```

Both must pass with zero findings before a task is considered complete.

---

## Style

PEP 8, with one deliberate deviation:

- **Line length 88**, not 79. PEP 8 explicitly permits teams to raise this.
  Recorded as [DECISIONS.md](DECISIONS.md) D-012.

Enabled lint rule families, each chosen for a reason:

| Family | Catches |
| --- | --- |
| `E`, `W` | PEP 8 layout |
| `F` | Unused imports, undefined names |
| `I` | Deterministic import ordering |
| `N` | Naming conventions |
| `UP` | Outdated syntax for the target version |
| `B` | Likely bugs, not merely style |
| `SIM` | Needless complexity |
| `PTH` | `pathlib` over `os.path` |
| `ARG` | Unused arguments |
| `D` | Docstring presence and format |

`B` (bugbear) earns its place: it found a real correctness defect during
Milestone 2 — a `zip()` over parallel tensors without `strict=`, which would
have silently discarded data. See [DECISIONS.md](DECISIONS.md) D-011.

Disabled rules must carry a documented reason in `pyproject.toml`. A silently
ignored rule is debt; a documented one is a decision.

---

## Type hints

Required on every function signature — parameters and return type.

- `from __future__ import annotations` at the top of every module, so modern
  syntax (`X | None`, `list[str]`) works uniformly.
- Import `Iterable`, `Sequence`, `Iterator` from `collections.abc`, not
  `typing`.
- `Any` is permitted only at a third-party boundary where the real type is not
  usefully expressible. Annotate it and move on rather than inventing stubs.
- Prefer `Path` over `str` for filesystem paths, everywhere.

---

## Docstrings

Google convention, enforced by lint. Every module, class, and public function.

A docstring states **why**, not what the code already says:

```python
# Poor — restates the signature
def find_images(directory: Path) -> list[Path]:
    """Find images in directory."""

# Good — explains the contract and the reasoning
def find_images(directory: Path) -> list[Path]:
    """Collect image files under ``directory``.

    Extensions come from configuration rather than a literal list, and matching
    is case-insensitive so uppercase extensions are not silently skipped.

    Args:
        directory: Root to search. A missing directory yields ``[]`` rather
            than raising — callers report absence via the status report.

    Returns:
        Image paths sorted by name, for reproducible ordering.
    """
```

Document `Raises:` whenever a function raises deliberately. Non-obvious return
conventions — an empty list meaning a legitimate negative case, `None` meaning
not-found — must be stated explicitly.

Test functions take a single-line docstring naming the behaviour being pinned
down.

---

## Comments

Comment the **non-obvious decision**, never the mechanics.

```python
# Poor
i += 1  # increment i

# Good
# DEBUG, not INFO: discovery runs on every resource check, and the chosen
# weights file is already reported once in the status report.
logger.debug(...)
```

If a line looks wrong but is correct, explain why. That comment prevents a
future "fix" that reintroduces a bug.

---

## Errors

- Raise project exceptions from `utils/exceptions.py`, never bare `Exception`.
- Error messages state what is wrong **and what to do about it**. A message
  that names a missing file without naming the expected location is only half
  written.
- Catch narrowly. Broad `except Exception` is acceptable only when isolating a
  failure that must not abort a batch, or when probing an optional backend —
  and must carry a comment saying which.
- Never silence an exception without logging it.

Distinguish the two kinds of failure:

| Kind | Handling |
| --- | --- |
| Anticipated (missing resource, bad label line) | Report clearly, continue or exit cleanly |
| Genuine bug (`AttributeError`, `KeyError`) | Let it propagate loudly |

---

## Logging

- Obtain loggers via `utils.logging_utils.get_logger(__name__)`. Never call
  `basicConfig` outside that module.
- Use lazy `%s` formatting, not f-strings — the formatting cost is skipped when
  the level is disabled:

```python
logger.info("Loaded %d classes", count)      # yes
logger.info(f"Loaded {count} classes")       # no
```

| Level | Use for |
| --- | --- |
| `DEBUG` | Internal detail useful when diagnosing |
| `INFO` | Normal progress a user wants to see |
| `WARNING` | Recoverable problem; work continued |
| `ERROR` | An operation failed |

Never use `print` for diagnostics. `print` is only for deliberate, formatted
user-facing report output.

---

## Structure

- Configurable value → `config.py`. Reusable logic → `utils/`. Application
  behaviour → `app/`.
- No hardcoded class names, class counts, dataset paths, or model paths.
- Prefer immutable data (`@dataclass(frozen=True)`) for values passed between
  modules.
- Derive values rather than storing them twice. Two sources of truth will
  disagree eventually.
- Import heavy third-party libraries inside functions when a module must stay
  importable without them — this is what keeps the health check working in a
  broken environment.

---

## Tests

- Test behavioural contracts, not implementation details.
- Name tests as the behaviour asserted:
  `test_load_ground_truth_missing_label_is_a_negative_sample`.
- Cover the failure path as seriously as the success path. While resources are
  unavailable, absence handling is the primary contract.
- Tests must not require a model, a dataset, or network access.
- Use `tmp_path` for filesystem work. Never write into the project tree.
