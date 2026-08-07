# Development Rules

Process rules. For how code should be *written*, see
[CODING_STANDARDS.md](CODING_STANDARDS.md).

---

## Existing work

- Improve or extend existing files. Do not recreate them.
- Never overwrite work without asking first.
- Preserve the established project structure.

## Version control

Version control is **not** managed by tooling on this project.

- Do not initialise a repository.
- Do not create remote repositories.
- Do not commit. Do not push.

## Scope discipline

- Work only on the current milestone.
- Do not implement future roadmap features, however small they seem.
- If something outside scope seems necessary, raise it and record the decision
  rather than quietly building it.

Unused code written "for later" is a liability: it is untested, undocumented by
use, and constrains future design.

## Isolation

- All dependencies install into the project-local environment. No global
  installs.
- Never read from or write to unrelated projects on the machine.
- Temporary experiments belong outside the project tree, not in `models/` or
  `datasets/`.

## Resources

**Never fabricate a model or a dataset.** Not to demonstrate a feature, not to
make a test pass, not to make output look complete.

- Always verify a resource before using it.
- Missing resources produce guidance, never a crash.
- If a capability cannot be exercised because a resource is absent, report it
  as unproven. Do not describe it as verified.

## Never hardcode

Under any circumstances:

- Class names
- Number of classes
- Dataset paths
- Model paths

Class names come from the model's own mapping or from `data.yaml`. Paths come
from `config.py`. Class counts are derived, never written down.

Rationale: a model retrained with reordered classes must not silently produce
mislabelled output.

## Code placement

| Kind of code | Location |
| --- | --- |
| Configurable value | `config.py` |
| Reusable logic | `utils/` |
| Application behaviour | `app/` |
| Tests | `tests/` |

Duplicated logic is a defect. When the same thing is implemented twice, extract
it — the second occurrence is the signal.

## Architecture review

Before implementing any new module:

1. Review the proposed design.
2. Identify future maintenance risks.
3. Propose a better structure if one exists.
4. State the trade-offs explicitly.
5. Implement only after the design is reviewed.

Record the outcome in [DECISIONS.md](DECISIONS.md) if the architecture changed.

Prefer maintainability, readability, simplicity, and modularity. Avoid
premature optimisation and overengineering in equal measure.

## Documentation

Documentation is part of the work, not a follow-up.

- Keep documentation synchronised with the implementation.
- Update it whenever architecture or behaviour changes.
- **Never document features that are not implemented.**
- Always distinguish Completed / In Progress / Planned.
- Record architectural decisions in [DECISIONS.md](DECISIONS.md).
- Update [CHANGELOG.md](CHANGELOG.md) after each completed milestone.
- Keep [ROADMAP.md](ROADMAP.md) aligned with the current plan.
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) is the source of truth.
- Keep documentation generic. No dataset, model, or class names except where a
  concrete example is genuinely required — and then mark it as an example.

---

## Quality gates

Verify **before** claiming any task complete:

- [ ] Code executes successfully
- [ ] No crashes on expected inputs
- [ ] Invalid inputs handled gracefully
- [ ] Type hints present
- [ ] Meaningful docstrings present
- [ ] Logging where appropriate
- [ ] No duplicated code
- [ ] Configuration externalised
- [ ] Reusable logic in the right module
- [ ] Conforms to the project's coding standards
- [ ] Design decisions documented
- [ ] Trade-offs explained
- [ ] Future extensibility considered
- [ ] No unnecessary dependencies introduced

## Definition of Done

A milestone is complete only when **all** of the following hold:

- [ ] Code executes successfully
- [ ] Edge cases handled
- [ ] User-friendly error handling implemented
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Design decisions recorded if architecture changed
- [ ] No unfinished work remains in the milestone
- [ ] Roadmap remains consistent
- [ ] Future modules remain compatible
- [ ] Code is modular and reusable
- [ ] The implementation has been explained and understood

If an item cannot be satisfied, the milestone is not complete. Say so plainly
and state what is outstanding — do not mark it done with a caveat buried in the
text.
