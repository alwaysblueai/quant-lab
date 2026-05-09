from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineFactorSpec:
    """Research baseline factor with fixed recipe semantics.

    ``factor_registry`` stores reusable base methods.  Baseline specs are one
    layer above that: each spec fixes the method, lookback, skip, required input
    columns, and baseline family so candidate factors can be compared against a
    stable research benchmark instead of a loose method name.
    """

    name: str
    family: str
    label: str
    description: str
    recipe: Mapping[str, object]
    required_columns: tuple[str, ...]
    tags: tuple[str, ...] = ()
    enabled_by_default: bool = True

    def to_dict(self) -> dict[str, object]:
        base = self.recipe.get("base")
        base_method = base.get("method") if isinstance(base, Mapping) else None
        return {
            "name": self.name,
            "family": self.family,
            "label": self.label,
            "description": self.description,
            "recipe": dict(self.recipe),
            "base_method": base_method,
            "required_columns": list(self.required_columns),
            "tags": list(self.tags),
            "enabled_by_default": self.enabled_by_default,
        }


_CLOSE_ONLY = ("date", "asset", "close")
_OHLC = ("date", "asset", "close", "high", "low")
_AMOUNT_OR_VOLUME = ("date", "asset", "close", "amount")


BASELINE_FACTOR_SUITE: tuple[BaselineFactorSpec, ...] = (
    BaselineFactorSpec(
        name="mom_20d",
        family="momentum",
        label="20日动量",
        description="最近20个交易日简单收益，作为短中期趋势基准。",
        recipe={"base": {"method": "momentum", "window": 20}},
        required_columns=_CLOSE_ONLY,
        tags=("price", "trend", "short_mid_horizon"),
    ),
    BaselineFactorSpec(
        name="mom_60d_skip5d",
        family="momentum",
        label="60日动量，跳过最近5日",
        description="60日趋势并跳过最近5日，降低短期反转污染。",
        recipe={"base": {"method": "momentum", "window": 60, "skip_recent": 5}},
        required_columns=_CLOSE_ONLY,
        tags=("price", "trend", "skip_recent"),
    ),
    BaselineFactorSpec(
        name="mom_120d_skip20d",
        family="momentum",
        label="120日动量，跳过最近20日",
        description="更长趋势基准，跳过最近20日以贴近经典中期动量口径。",
        recipe={"base": {"method": "momentum", "window": 120, "skip_recent": 20}},
        required_columns=_CLOSE_ONLY,
        tags=("price", "trend", "medium_horizon", "skip_recent"),
    ),
    BaselineFactorSpec(
        name="rev_3d",
        family="reversal",
        label="3日反转",
        description="最近3日收益取负，捕捉极短期反转。",
        recipe={"base": {"method": "reversal", "window": 3}},
        required_columns=_CLOSE_ONLY,
        tags=("price", "reversal", "short_horizon"),
    ),
    BaselineFactorSpec(
        name="rev_5d",
        family="reversal",
        label="5日反转",
        description="最近5日收益取负，作为默认短期反转基准。",
        recipe={"base": {"method": "reversal", "window": 5}},
        required_columns=_CLOSE_ONLY,
        tags=("price", "reversal", "short_horizon"),
    ),
    BaselineFactorSpec(
        name="rev_20d",
        family="reversal",
        label="20日反转",
        description="最近20日收益取负，用于识别候选因子是否只是月度反转变体。",
        recipe={"base": {"method": "reversal", "window": 20}},
        required_columns=_CLOSE_ONLY,
        tags=("price", "reversal", "monthly"),
    ),
    BaselineFactorSpec(
        name="retvol_20d",
        family="volatility",
        label="20日低波动",
        description="20日 close-to-close 波动率取负，作为低波动基准。",
        recipe={"base": {"method": "low_volatility", "window": 20}},
        required_columns=_CLOSE_ONLY,
        tags=("price", "volatility", "defensive"),
    ),
    BaselineFactorSpec(
        name="downside_vol_20d",
        family="volatility",
        label="20日下行低波动",
        description="20日下行波动率取负，区分普通波动和坏波动暴露。",
        recipe={"base": {"method": "downside_volatility", "window": 20}},
        required_columns=_CLOSE_ONLY,
        tags=("price", "downside_risk", "defensive"),
    ),
    BaselineFactorSpec(
        name="amplitude_20d",
        family="amplitude",
        label="20日低振幅",
        description="20日日内振幅均值取负，作为A股常用波动/拥挤代理。",
        recipe={"base": {"method": "amplitude", "window": 20}},
        required_columns=_OHLC,
        tags=("price", "intraday_range", "a_share"),
    ),
    BaselineFactorSpec(
        name="vcimom_20_5",
        family="volume_confirmed_momentum",
        label="量能确认残差动量",
        description="残差动量 + 成交额确认 + 正向冲击惩罚；成本更高，默认不跑。",
        recipe={
            "base": {
                "method": "vcimom",
                "residual_window": 60,
                "momentum_window": 20,
                "skip_recent": 4,
                "confirm_window": 10,
                "penalty_window": 5,
                "amount_window": 20,
                "confirm_weight": 0.6,
                "penalty_weight": 0.4,
            }
        },
        required_columns=_AMOUNT_OR_VOLUME,
        tags=("price", "volume", "residual_momentum", "heavier"),
        enabled_by_default=False,
    ),
)


def iter_baseline_factor_specs(
    *,
    include_non_default: bool = False,
    families: Iterable[str] | None = None,
) -> tuple[BaselineFactorSpec, ...]:
    family_filter = {item.strip().lower() for item in families or () if item.strip()}
    specs = []
    for spec in BASELINE_FACTOR_SUITE:
        if not include_non_default and not spec.enabled_by_default:
            continue
        if family_filter and spec.family.lower() not in family_filter:
            continue
        specs.append(spec)
    return tuple(specs)


def baseline_factor_suite_payload(*, include_non_default: bool = True) -> list[dict[str, object]]:
    return [
        spec.to_dict()
        for spec in iter_baseline_factor_specs(include_non_default=include_non_default)
    ]


def baseline_required_columns_available(
    spec: BaselineFactorSpec,
    available_columns: Iterable[str],
) -> bool:
    available = {str(column) for column in available_columns}
    if spec.name == "vcimom_20_5":
        return {"date", "asset", "close"}.issubset(available) and (
            "amount" in available or "volume" in available
        )
    return set(spec.required_columns).issubset(available)
