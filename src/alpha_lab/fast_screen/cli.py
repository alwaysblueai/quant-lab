"""Fast-screen CLI: ``alpha-lab fast-screen run|deep-dive|list-modules``.

The ``run`` and ``deep-dive`` subcommands both consume the existing
single-factor spec format so the preprocessing pipeline (universe mask,
neutralization, coverage gating) stays in one place.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from alpha_lab.real_cases.common_io import (
    apply_universe_to_factor,
    apply_universe_to_prices,
    load_prices,
    load_tabular_frame,
    load_universe_mask,
)
from alpha_lab.real_cases.single_factor.pipeline import (
    _maybe_neutralize_factor,  # reuse private prep to avoid drift
    _prepare_factor,
)
from alpha_lab.real_cases.single_factor.spec import (
    SingleFactorCaseSpec,
    load_single_factor_case_spec,
)

from .artifacts import (
    load_tier1_result,
    save_tier1_result,
)
from .tier1 import Tier1Inputs, run_tier1
from .tier2 import TIER2_MODULES, run_tier2_modules

DEFAULT_ARTIFACT_ROOT = Path("outputs") / "fast_screen"


def _prepare_inputs(spec: SingleFactorCaseSpec) -> Tier1Inputs:
    """Reproduce preprocessing done by ``run_single_factor_case`` without heavy integrity checks.

    We deliberately skip the full integrity suite here and set
    ``integrity_passed=True`` unless a prep step itself raises. The Tier-2
    ``integrity_full`` module runs the detailed checks on demand.
    """
    universe_mask = load_universe_mask(spec.universe)
    prices = load_prices(spec.prices_path)
    if universe_mask is not None:
        prices = apply_universe_to_prices(prices, universe_mask)

    raw_factor = load_tabular_frame(spec.factor_path, object_name="factor")
    factor_df = _prepare_factor(raw_factor, spec=spec)
    if universe_mask is not None:
        factor_df = apply_universe_to_factor(factor_df, universe_mask)
    factor_df, _ = _maybe_neutralize_factor(
        factor_df,
        spec=spec,
        universe_mask=universe_mask,
    )

    return Tier1Inputs(
        factor_name=spec.factor_name,
        factor_df=factor_df,
        prices=prices,
        horizon=spec.target.horizon,
        n_quantiles=spec.n_quantiles,
        cost_rate=spec.transaction_cost.one_way_rate,
        universe=spec.universe.name,
        frequency="daily",
    )


def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_single_factor_case_spec(Path(args.spec).resolve())
    inputs = _prepare_inputs(spec)
    result = run_tier1(inputs, run_id=args.run_id or None)
    artifact_root = Path(args.artifact_root).resolve()
    paths = save_tier1_result(artifact_root, result)
    print(
        json.dumps(
            {
                "ok": True,
                "factor_name": result.factor_name,
                "run_id": result.run_id,
                "verdict": result.verdict.status,
                "tier1_dir": str(paths.tier1_dir),
                "inputs_hash": result.inputs_hash,
            },
            indent=2,
        )
    )
    return 0


def _cmd_deep_dive(args: argparse.Namespace) -> int:
    spec = load_single_factor_case_spec(Path(args.spec).resolve())
    artifact_root = Path(args.artifact_root).resolve()
    # Load the Tier-1 result to pick up run_id + inputs_hash for staleness.
    tier1 = load_tier1_result(artifact_root, spec.factor_name, args.run_id)
    if tier1.verdict.status == "fail" and not args.force:
        print(
            f"Tier-1 verdict is FAIL ({tier1.verdict.triggered_rules}). "
            "Re-run with --force to compute Tier-2 anyway.",
            file=sys.stderr,
        )
        return 2

    inputs = _prepare_inputs(spec)
    modules = (
        [m.strip() for m in args.modules.split(",") if m.strip()]
        if args.modules
        else [m.key for m in TIER2_MODULES]
    )
    statuses = run_tier2_modules(
        inputs,
        artifact_root=artifact_root,
        factor_name=spec.factor_name,
        run_id=args.run_id,
        modules=modules,
        inputs_hash=tier1.inputs_hash,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "factor_name": spec.factor_name,
                "run_id": args.run_id,
                "results": {k: v.to_dict() for k, v in statuses.items()},
            },
            indent=2,
            default=str,
        )
    )
    any_fail = any(s.status.value == "failed" for s in statuses.values())
    return 1 if any_fail else 0


def _cmd_list_modules(_args: argparse.Namespace) -> int:
    rows = [
        {
            "key": m.key,
            "label": m.label,
            "est_seconds": m.estimated_seconds,
        }
        for m in TIER2_MODULES
    ]
    print(json.dumps(rows, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alpha-lab fast-screen",
        description="Tier-1 fast-screen and Tier-2 deep-dive for single factors.",
    )
    sub = p.add_subparsers(dest="action", required=True)

    run = sub.add_parser("run", help="Run Tier-1 fast screen against a single-factor spec.")
    run.add_argument("--spec", required=True, help="Path to single-factor spec (YAML/JSON).")
    run.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    run.add_argument("--run-id", default=None, help="Optional run-id; defaults to inputs_hash.")
    run.set_defaults(func=_cmd_run)

    dd = sub.add_parser("deep-dive", help="Run Tier-2 modules for an existing Tier-1 run.")
    dd.add_argument("--spec", required=True)
    dd.add_argument("--run-id", required=True)
    dd.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    dd.add_argument(
        "--modules",
        default=None,
        help="Comma-separated module keys; omit to run all registered modules.",
    )
    dd.add_argument("--force", action="store_true", help="Run even if Tier-1 verdict is FAIL.")
    dd.set_defaults(func=_cmd_deep_dive)

    ls = sub.add_parser("list-modules", help="List Tier-2 modules available.")
    ls.set_defaults(func=_cmd_list_modules)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
