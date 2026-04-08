# rubrify — Philosophy and Guard Rails

> **rubrify is a Python toolchain for prompt programs whose source is XML and whose runtime is an LLM.**

A rubric is not a scoring form. A rubric is a prompt program. The XML `<LLM_JUDGE_SPEC>` is the source code. The LLM is the runtime. The Python kernel types are the AST. The library's job is to author, mutate, validate, render, execute, test, refactor, version, and verify those programs faithfully.

This document is the product framing. Every design decision, every phase, every module, and every PR must serve it. Violations of either list below are blocking issues.

---

## Part 1 — Philosophy Anchors

These ten anchors are derived directly from the main reference repos and are binding.

1. **Rubrics are prompt programs.** XML is the source. The LLM is the runtime. Python objects are the AST.

2. **(Roleplaying == jailbreak == context following) == rubrics.** All four are the same mechanism viewed from different angles. A rubric that judges is a rubric that role-plays a judge. A rubric that forces output structure is a rubric that jailbreaks the model into producing that structure. A rubric that extracts entities is a rubric that asks the model to follow an extraction context.

3. **XML is first-class.** Python exists to manipulate XML, not to replace it. The model-facing canonical form is always XML. Round-trip fidelity is mandatory.

4. **Constraints are jailbreaks.** Forcing rubrics, transformation rubrics, extraction rubrics, and adversarial rubrics are members of the same family. They are all behavioral constraints expressed as context. There is no meaningful distinction between "structured output forcing" and "jailbreak" beyond intent.

5. **Failure modes are valid artifacts.** `slurs.xml` failing to steer modern models is information, not a bug. The library asserts that failure rather than trying to "fix" it. Adversarial rubrics are first-class artifacts worth documenting.

6. **Calibration is mandatory for behavioral claims.** A rubric without calibration is unverified. Compliance-style rubrics ship with expected cases. Detection rubrics ship with discriminant invariants. A rubric that makes a behavioral claim without test cases is a draft, not an artifact.

7. **Models are not equal.** GPT-class and Claude-class models are the supported targets. Other OpenAI-compatible models are experimental until evaluated against reference calibration suites. The library is honest about this instead of pretending all models follow XML contracts equally.

8. **Steering anchors are not rituals.** Constructs like `BECAUSE:` prefixes, exact word counts, `FIX:` markers, and fixed key ordering are context-following anchors that lock the model into the rubric's role. They are runtime-critical, not decorative. They are the mechanism by which rubrics force compliance.

9. **Meta-rubrics are rubrics.** `META_EVALUATOR` follows every rule in this list. It is itself authored, mutated, validated, calibrated, refined, and tested against known-good rubrics. The dogfood is not exempt.

10. **Categorical and control-theory inspirations are design lenses, not Python machinery.** They explain *why* the runtime has feedback loops, composition, and algebraic operations. They never appear as typeclasses, protocols, runtime objects, or abstract base classes. The category-theoretic vocabulary stays in documentation and design discussion.

---

## Part 2 — Over-Engineering Guard Rails

These rules prevent drift over time. Every phase, every PR, and every new module must respect them.

1. **No abstraction without a reference citation.** If a class, protocol, function, or module does not trace back to a behavior in the three main reference repos, it does not belong in the library. "Might be useful later" is not a reference.

2. **No formal categorical machinery.** Category theory and control theory remain explanatory tools. They never become Python types, protocols, or runtime machinery.

3. **No metadata-driven dispatch.** Fields like `behaviors` are documentation, not branch points. Runtime never branches on them.

4. **No plugin / registry / hook systems.** Add a function. Export it. Done. The library is small and concrete.

5. **No new XML tags without reference precedent.** Extending the canonical XML format requires a reference rubric using the same shape. Invented tags are vetoed.

6. **No silent fallbacks.** Repair must report. Parsing must report. Refinement must report stopping reasons. Anything the library does that the user did not directly ask for must be visible in the result.

7. **No backwards-incompatible API changes.** All evolution is additive. Defaults preserve current behavior. The existing test suite must keep passing throughout.

8. **No hypothetical future-proofing.** No provider abstractions, no async, no fine-tuning hooks, no multi-modal scaffolding "for later". Features arrive when a reference behavior demands them.

9. **Adversarial / failure-mode rubrics are first-class artifacts.** They are not bugs and they are not fixed. They are asserted as expected failures.

10. **META_EVALUATOR is not exempt.** It is a rubric. It is calibrated, validated, evolved, and tested like any other rubric.

---

## Part 3 — Toolchain Mapping

Every part of rubrify maps to a familiar concept from a normal language toolchain. This mapping is the structural backbone of the library.

| Compiler / runtime concept | rubrify equivalent |
|---|---|
| Source format | XML `<LLM_JUDGE_SPEC>` |
| AST | Python kernel dataclasses in `_types.py` |
| Type checking / linting | property lattice, `validate()`, `META_EVALUATOR` |
| Compiler / code generator | `to_xml()` plus input renderers |
| Runtime / interpreter | `Rubric.evaluate()` and `ConstraintRubric.apply()` |
| Runtime error recovery | repair layer |
| Observability / debugger | `EvaluationTrace` |
| Unit tests | calibration cases, calibration runner |
| Conformance suite | reference-faithfulness tests |
| Refactoring tools | mutations, `evolve`, algebra (`\|`, `&`, `project`, `reweight`) |
| Version control / lineage | provenance |
| Compiler target selection | model policy |
| Standard library scaffolds | meta-rubric instruction primitives and profiles |
| Code review bot | `META_EVALUATOR` |
| Dogfooding | `META_EVALUATOR` is itself a rubric |

---

## Closing

**Violations of Part 1 or Part 2 are blocking issues.** Any PR that introduces an abstraction without a reference citation, branches runtime on metadata, hides a failure silently, or tries to "fix" a documented failure-mode rubric must be rejected or reworked before merge.

The three main reference repos are the ground truth:

- `references/main/rubrics/` — books2rubrics (Zinsser v1/v2/v3), special_ones (anti_slop, completeness, slurs)
- `references/main/gist-ae3976ad/` — "How to Build a Good LLM Judge" playbook
- `references/main/gist-ff87ac23/` — red_team_rubric.py (ComplianceJudge with calibration cases)

When in doubt, re-read them.
