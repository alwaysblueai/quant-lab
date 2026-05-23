"""Codebase snapshot for Stage 0 idea distribution.

Both Stage 1 engines receive the same generator + reviewer task. They need to
see what is already in the repo before judging whether a candidate mechanism is
executable in the v1 contract. This module produces a small, frozen snapshot used by
``engine_prompts.build_prompt`` to inject:

- existing factor / model_candidate names (so reviewer flags duplication)
- existing case yaml names
- ``factor.json`` required keys and ``ModelFactorCaseSpec`` top-level fields
- a validator hard-rule cheatsheet (paraphrased; full source in
  ``draft_factor_validation`` / ``draft_model_validation``)

The snapshot is intentionally cheap to build (directory listing only); it does
not parse YAML / JSON contents to keep distribution fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Static cheatsheets — sourced from the validators. Update them when the
# validator hard rules change. These are paraphrased for prompt readability;
# reviewer always defers to the validator's exact behaviour at Stage 3.

_FACTOR_JSON_REQUIRED_KEYS: tuple[str, ...] = (
    "name",
    "description",
    "required_columns",
    "optional_columns",
    "frequency",
    "unavailable_data_policy",
    "pit_assumption",
    "code",
)

_MODEL_CASE_SPEC_TOP_KEYS: tuple[str, ...] = (
    "name",
    "factor_name",
    "features_path",
    "prices_path",
    "feature_columns",
    "rebalance_frequency",
    "n_quantiles",
    "direction",
    "universe",
    "target",
    "feature_availability",
    "feature_preprocess",
    "feature_importance",
    "model",
    "model_selection",
    "training",
    "neutralization",
    "transaction_cost",
    "output",
)

_FACTOR_VALIDATOR_RULES: tuple[str, ...] = (
    "name must be snake_case (3-64 chars) and not collide with BUILTIN_FACTOR_NAMES",
    "frequency must be 'daily'",
    (
        "code must define build_factor(frame) returning index-aligned pd.Series, "
        "or legacy builder(prices, ...)"
    ),
    "forbidden imports: os/pathlib/glob/pickle/shutil/socket/subprocess/urllib/requests/httpx",
    "forbidden calls: __import__/eval/exec/open/compile/breakpoint/input",
    (
        "forbidden attr calls: mkdir/read_csv/read_parquet/to_csv/to_parquet/"
        "popen/system/run/urlopen/write_text/..."
    ),
    (
        "future-leakage tokens forbidden in identifiers: "
        "future_return/forward_return/fwd_return/next_return/label_return/target_return"
    ),
    "shift(-n) and negative pct_change rejected as future leakage",
    "rolling/expanding must group by 'asset'",
    "required_columns must be real factor_recipe.py registered names (prices + intraday-derived)",
)

_MODEL_VALIDATOR_RULES: tuple[str, ...] = (
    "contract_version must be 'stage2_model_candidate_v1'",
    "implementation_status must be 'draft_for_stage3'",
    (
        "implementation_type must be 'spec_variant' "
        "(no custom feature/estimator/sample_weight code in v1)"
    ),
    "candidate_name is snake_case (3-64)",
    "case_spec_payload must parse via model_factor_case_spec_from_mapping (no patch-only)",
    "feature_columns must all appear in features file header",
    (
        "fundamental-like features cannot use feature_availability.mode='required_timestamp' "
        "with column=null"
    ),
    (
        "payload must not contain Level 3 / execution_replay / fill_simulation / "
        "portfolio_construction / live_trading tokens"
    ),
)


@dataclass(frozen=True)
class CodebaseSnapshot:
    """Read-only snapshot of repo state used by Stage 0 reviewer prompts."""

    factors_promoted: tuple[str, ...] = ()
    factors_research: tuple[str, ...] = ()
    model_candidates_promoted: tuple[str, ...] = ()
    model_candidates_research: tuple[str, ...] = ()
    single_factor_cases: tuple[str, ...] = ()
    model_factor_cases: tuple[str, ...] = ()
    factor_json_required_keys: tuple[str, ...] = field(
        default_factory=lambda: _FACTOR_JSON_REQUIRED_KEYS
    )
    model_case_spec_top_keys: tuple[str, ...] = field(
        default_factory=lambda: _MODEL_CASE_SPEC_TOP_KEYS
    )
    factor_validator_rules: tuple[str, ...] = field(
        default_factory=lambda: _FACTOR_VALIDATOR_RULES
    )
    model_validator_rules: tuple[str, ...] = field(
        default_factory=lambda: _MODEL_VALIDATOR_RULES
    )

    def to_payload(self) -> dict[str, object]:
        return {
            "factors": {
                "promoted": list(self.factors_promoted),
                "research": list(self.factors_research),
            },
            "model_candidates": {
                "promoted": list(self.model_candidates_promoted),
                "research": list(self.model_candidates_research),
            },
            "cases": {
                "single_factor": list(self.single_factor_cases),
                "model_factor": list(self.model_factor_cases),
            },
            "schema": {
                "factor_json_required_keys": list(self.factor_json_required_keys),
                "model_case_spec_top_keys": list(self.model_case_spec_top_keys),
            },
            "validator_rules": {
                "factor": list(self.factor_validator_rules),
                "model": list(self.model_validator_rules),
            },
        }


def _list_subdir_names(parent: Path) -> tuple[str, ...]:
    if not parent.exists() or not parent.is_dir():
        return ()
    return tuple(
        sorted(
            child.name
            for child in parent.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        )
    )


def _list_yaml_stems(parent: Path) -> tuple[str, ...]:
    if not parent.exists() or not parent.is_dir():
        return ()
    return tuple(
        sorted(
            path.stem
            for path in parent.iterdir()
            if path.is_file() and path.suffix in {".yaml", ".yml"}
        )
    )


def build_codebase_snapshot(workspace_root: str | Path) -> CodebaseSnapshot:
    """Walk the workspace root and produce a frozen snapshot.

    Cheap directory listing only — does not parse contents.
    """

    root = Path(workspace_root).expanduser().resolve()
    return CodebaseSnapshot(
        factors_promoted=_list_subdir_names(root / "custom_factors" / "promoted"),
        factors_research=_list_subdir_names(root / "custom_factors" / "research"),
        model_candidates_promoted=_list_subdir_names(
            root / "model_candidates" / "promoted"
        ),
        model_candidates_research=_list_subdir_names(
            root / "model_candidates" / "research"
        ),
        single_factor_cases=_list_yaml_stems(
            root / "configs" / "real_cases" / "single_factor"
        ),
        model_factor_cases=_list_yaml_stems(
            root / "configs" / "real_cases" / "model_factor"
        ),
    )
