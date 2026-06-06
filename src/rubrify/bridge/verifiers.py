"""Bridge between rubrify and Prime Intellect's ``verifiers`` library.

Provides two functions:

* ``make_rubrify_rubric`` -- wraps a compiled RubricBundle + Judge into a
  ``verifiers.Rubric`` with one reward function (aggregated score) and one
  metric per criterion (individual unit scores).

* ``make_rubrify_env`` -- convenience wrapper that builds a ready-to-use
  ``verifiers.SingleTurnEnv``.

Both functions import ``verifiers`` lazily so that rubrify itself never
hard-depends on it.  If ``verifiers`` is not installed the import will
fail with a clear ``ImportError`` at call time.

Usage::

    from rubrify import compile_rubric, Judge, JudgeConfig
    from rubrify.bridge.verifiers import make_rubrify_rubric

    bundle = compile_rubric(spec).bundle
    judge  = Judge(JudgeConfig(model=model))
    rubric = make_rubrify_rubric(bundle, judge)

    # Use with verifiers training loop
    env = vf.SingleTurnEnv(dataset=ds, rubric=rubric)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import verifiers as vf

    from rubrify.engine.judge import Judge
    from rubrify.ir.bundle import RubricBundle


def _import_verifiers() -> Any:
    """Lazily import verifiers, raising a clear error if missing."""
    try:
        import verifiers as vf
        return vf
    except ImportError as exc:
        raise ImportError(
            "The 'verifiers' package is required for rubrify.bridge.verifiers. "
            "Install it with: pip install verifiers"
        ) from exc


def make_rubrify_rubric(bundle: RubricBundle, judge: Judge) -> vf.Rubric:
    """Create a ``verifiers.Rubric`` from a compiled rubrify bundle and judge.

    The returned rubric contains:

    * **One reward function** (weight 1.0) that calls ``judge.evaluate()``
      on the completion text and returns the aggregated normalized score
      mapped to [0, 1].  The full ``Judgment`` object is stashed on
      ``state["rubrify_judgment"]`` so downstream metrics (and user code)
      can inspect it without re-running the LLM.

    * **One metric per criterion** (weight 0) whose name is
      ``rubrify_<criterion_id>``.  Each reads its ``unit_score`` from the
      stashed judgment.  Because verifiers evaluates reward functions
      before metrics, these never trigger an extra LLM call.

    Parameters
    ----------
    bundle:
        A locked ``RubricBundle`` (output of ``compile_rubric``).
    judge:
        A configured ``Judge`` instance.

    Returns
    -------
    vf.Rubric
        A verifiers-compatible rubric ready for use in a training env.
    """
    vf = _import_verifiers()

    rubric = vf.Rubric()

    # ── Main reward: aggregated rubric score ──────────────────────
    async def rubrify_reward(completion: list[dict[str, str]],
                             state: dict[str, Any],
                             **kwargs: Any) -> float:
        response_text = completion[-1]["content"] if completion else ""
        judgment = await judge.evaluate(bundle, response_text)
        # Stash for downstream metric functions (and user inspection).
        state["rubrify_judgment"] = judgment
        return judgment.aggregation.normalized_score / 100.0

    rubric.add_reward_func(rubrify_reward, weight=1.0)

    # ── Per-criterion metrics (weight=0, visible in state["metrics"]) ─
    for criterion in bundle.rubric.criteria:
        cid = criterion.id

        # The default-arg trick (_cid=cid) captures the loop variable by
        # value so each closure references its own criterion ID.
        async def criterion_metric(completion: list[dict[str, str]],
                                   state: dict[str, Any],
                                   _cid: str = cid,
                                   **kwargs: Any) -> float:
            j = state.get("rubrify_judgment")
            if j is None:
                return 0.0
            for cj in j.criterion_judgments:
                if cj.criterion_id == _cid:
                    return cj.unit_score
            return 0.0

        # verifiers uses __name__ as the metric key in state["metrics"].
        criterion_metric.__name__ = f"rubrify_{cid}"
        rubric.add_metric(criterion_metric)

    return rubric


def make_rubrify_env(
    bundle: RubricBundle,
    judge: Judge,
    dataset: Any,
    eval_dataset: Any = None,
    system_prompt: str = "",
) -> vf.SingleTurnEnv:
    """Create a ``verifiers.SingleTurnEnv`` wired to a rubrify judge.

    This is a thin convenience wrapper around ``make_rubrify_rubric``.

    Parameters
    ----------
    bundle:
        A locked ``RubricBundle``.
    judge:
        A configured ``Judge``.
    dataset:
        Training dataset (passed through to ``SingleTurnEnv``).
    eval_dataset:
        Optional evaluation dataset.
    system_prompt:
        System prompt prepended to every rollout.

    Returns
    -------
    vf.SingleTurnEnv
        Ready-to-use environment for verifiers training loops.
    """
    vf = _import_verifiers()

    rubric = make_rubrify_rubric(bundle, judge)
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        system_prompt=system_prompt,
    )


__all__ = [
    "make_rubrify_rubric",
    "make_rubrify_env",
]
