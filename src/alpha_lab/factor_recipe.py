from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabExperimentError

from alpha_lab.factors import amplitude, downside_volatility, low_volatility, momentum, reversal
from alpha_lab.interfaces import validate_factor_output
from alpha_lab.signal_transforms import (
    apply_min_coverage_gate,
    rank_cross_section,
    winsorize_cross_section,
    zscore_cross_section,
)


# ---------------------------------------------------------------------------
# Factor registry
# ---------------------------------------------------------------------------

_FactorBuilderFn = Callable[[pd.DataFrame], pd.DataFrame]


class FactorRegistry:
    """Extensible registry for base factor builder functions.

    Built-in factors (momentum, reversal, low_volatility) are registered at
    module load time.  External code can register additional builders via
    :meth:`register` or the :meth:`register_factor` decorator::

        from alpha_lab.factor_recipe import factor_registry

        @factor_registry.register_factor("my_custom_factor")
        def my_custom_factor(prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
            ...
    """

    def __init__(self) -> None:
        self._builders: dict[str, _FactorBuilderFn] = {}

    def register(self, name: str, fn: _FactorBuilderFn) -> None:
        """Register a factor builder function under *name* (lowercase)."""
        key = name.strip().lower()
        if not key:
            raise AlphaLabConfigError("factor name must be non-empty")
        self._builders[key] = fn

    def register_factor(self, name: str) -> Callable[[_FactorBuilderFn], _FactorBuilderFn]:
        """Decorator form of :meth:`register`."""
        def decorator(fn: _FactorBuilderFn) -> _FactorBuilderFn:
            self.register(name, fn)
            return fn
        return decorator

    def get(self, name: str) -> _FactorBuilderFn | None:
        """Look up a builder by name (case-insensitive)."""
        return self._builders.get(name.strip().lower())

    def supported_methods(self) -> list[str]:
        """Return sorted list of registered method names."""
        return sorted(self._builders)

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._builders


# Module-level singleton — external code imports this to register factors.
factor_registry = FactorRegistry()
factor_registry.register("momentum", momentum)
factor_registry.register("reversal", reversal)
factor_registry.register("low_volatility", low_volatility)
factor_registry.register("amplitude", amplitude)
factor_registry.register("downside_volatility", downside_volatility)


class FactorRecipeError(ValueError):
    """Raised when factor-recipe payload is invalid."""


def load_recipe_mapping(path: str | Path) -> dict[str, object]:
    """Load factor recipe mapping from JSON/YAML file."""
    recipe_path = Path(path).resolve()
    if not recipe_path.exists() or not recipe_path.is_file():
        raise FileNotFoundError(f"recipe file does not exist: {recipe_path}")

    text = recipe_path.read_text(encoding="utf-8")
    suffix = recipe_path.suffix.lower()
    payload: object
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        payload = _yaml_load(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = _yaml_load(text)

    if not isinstance(payload, dict):
        raise FactorRecipeError("factor recipe payload must be a mapping")
    return dict(payload)


def build_factor_from_recipe_mapping(
    *,
    prices: pd.DataFrame,
    recipe: Mapping[str, object],
    factor_name: str,
) -> pd.DataFrame:
    """Build canonical factor table from prices and recipe mapping.

    Required recipe keys:
    - ``base.method``: one of ``momentum``, ``reversal``, ``low_volatility``

    Optional sections:
    - ``preprocess``: winsorize / standardization / min_coverage
    - ``orthogonalize``: residualize base signal against exposure recipes
    - ``signal``: reserved for future non-directional metadata
    """
    factor_token = _required_non_empty_string(factor_name, "factor_name")
    price_frame = _normalize_prices(prices)
    recipe_map = _as_mapping(recipe, field="recipe")

    base_cfg = _as_mapping(recipe_map.get("base"), field="base")
    base_signal = _compute_step_signal(prices=price_frame, step_cfg=base_cfg)

    preprocess_cfg = _optional_mapping(recipe_map.get("preprocess"), field="preprocess")
    if preprocess_cfg is not None:
        base_signal = _apply_preprocess(base_signal, preprocess_cfg)

    ortho_cfg = _optional_mapping(recipe_map.get("orthogonalize"), field="orthogonalize")
    if ortho_cfg is not None and bool(ortho_cfg.get("enabled", True)):
        base_signal = _apply_orthogonalization(
            signal=base_signal,
            prices=price_frame,
            orthogonalize_cfg=ortho_cfg,
        )

    signal_cfg = _optional_mapping(recipe_map.get("signal"), field="signal")
    if signal_cfg is not None:
        if "direction" in signal_cfg:
            raise FactorRecipeError(
                "recipe.signal.direction is not supported; use case-level direction instead"
            )

    out = base_signal.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out["factor"] = factor_token
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "asset"]).copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out[["date", "asset", "factor", "value"]]
    out = out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)

    validate_factor_output(out)
    return out


def build_factor_from_recipe_file(
    *,
    prices_path: str | Path,
    recipe_path: str | Path,
    factor_name: str,
) -> pd.DataFrame:
    """Load prices/recipe from paths and build canonical factor output."""
    px_path = Path(prices_path).resolve()
    if not px_path.exists() or not px_path.is_file():
        raise FileNotFoundError(f"prices file does not exist: {px_path}")

    prices = pd.read_csv(px_path)
    recipe = load_recipe_mapping(recipe_path)
    return build_factor_from_recipe_mapping(
        prices=prices,
        recipe=recipe,
        factor_name=factor_name,
    )


def _compute_step_signal(*, prices: pd.DataFrame, step_cfg: Mapping[str, object]) -> pd.DataFrame:
    raw_method = step_cfg.get("method")
    if not isinstance(raw_method, str):
        raise FactorRecipeError("base.method must be a string")

    method = raw_method.strip().lower()
    if method == "vcimom":
        return _build_vcimom_signal(prices=prices, step_cfg=step_cfg)
    builder = factor_registry.get(method)
    if builder is None:
        supported = sorted([*factor_registry.supported_methods(), "vcimom"])
        raise FactorRecipeError(
            f"unsupported base.method={raw_method!r}; supported methods: {supported}"
        )

    window = _optional_positive_int(step_cfg.get("window"), field="base.window")
    skip_recent = _optional_non_negative_int(
        step_cfg.get("skip_recent"),
        field="base.skip_recent",
    )
    min_periods = _optional_positive_int(step_cfg.get("min_periods"), field="base.min_periods")

    kwargs: dict[str, int] = {}
    if window is not None:
        kwargs["window"] = window
    if skip_recent is not None:
        kwargs["skip_recent"] = skip_recent
    if min_periods is not None:
        kwargs["min_periods"] = min_periods

    factor_df = builder(prices, **kwargs)
    return factor_df[["date", "asset", "value"]].copy()


def _build_vcimom_signal(
    *,
    prices: pd.DataFrame,
    step_cfg: Mapping[str, object],
) -> pd.DataFrame:
    residual_window = _optional_positive_int(
        step_cfg.get("residual_window"),
        field="base.residual_window",
    )
    momentum_window = _optional_positive_int(
        step_cfg.get("momentum_window"),
        field="base.momentum_window",
    )
    skip_recent = _optional_non_negative_int(
        step_cfg.get("skip_recent"),
        field="base.skip_recent",
    )
    confirm_window = _optional_positive_int(
        step_cfg.get("confirm_window"),
        field="base.confirm_window",
    )
    penalty_window = _optional_positive_int(
        step_cfg.get("penalty_window"),
        field="base.penalty_window",
    )
    amount_window = _optional_positive_int(
        step_cfg.get("amount_window"),
        field="base.amount_window",
    )
    confirm_weight = _optional_float(
        step_cfg.get("confirm_weight"),
        field="base.confirm_weight",
    )
    penalty_weight = _optional_float(
        step_cfg.get("penalty_weight"),
        field="base.penalty_weight",
    )

    residual_window = 60 if residual_window is None else residual_window
    momentum_window = 20 if momentum_window is None else momentum_window
    skip_recent = 4 if skip_recent is None else skip_recent
    confirm_window = 10 if confirm_window is None else confirm_window
    penalty_window = 5 if penalty_window is None else penalty_window
    amount_window = 20 if amount_window is None else amount_window
    confirm_weight = 0.6 if confirm_weight is None else confirm_weight
    penalty_weight = 0.4 if penalty_weight is None else penalty_weight

    if momentum_window <= skip_recent:
        raise FactorRecipeError("base.momentum_window must be > base.skip_recent")

    frame = prices.copy()
    frame["return"] = (
        frame.groupby("asset", sort=False)["close"].pct_change(fill_method=None)
    )
    market_return = (
        frame.groupby("date", sort=False)["return"].mean().rename("market_return").reset_index()
    )
    frame = frame.merge(market_return, on="date", how="left", validate="many_to_one")
    frame["beta"] = _rolling_beta(
        returns=frame["return"],
        market_returns=frame["market_return"],
        assets=frame["asset"],
        window=residual_window,
    )
    frame["residual_return"] = frame["return"] - frame["beta"] * frame["market_return"]

    amount_series = _resolve_amount_series(frame)
    frame["amount_proxy"] = amount_series
    log_amount = np.log(amount_series.where(amount_series > 0))
    amount_mean = (
        log_amount.groupby(frame["asset"], sort=False)
        .rolling(
            amount_window,
            min_periods=min(amount_window, max(3, amount_window // 2)),
        )
        .mean()
        .reset_index(level=0, drop=True)
    )
    amount_std = (
        log_amount.groupby(frame["asset"], sort=False)
        .rolling(
            amount_window,
            min_periods=min(amount_window, max(3, amount_window // 2)),
        )
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )
    frame["amount_z"] = (log_amount - amount_mean).div(amount_std.replace(0.0, np.nan))

    lagged_residual = frame.groupby("asset", sort=False)["residual_return"].shift(skip_recent + 1)
    momentum_signal_window = momentum_window - skip_recent
    frame["momentum_leg"] = (
        lagged_residual.groupby(frame["asset"], sort=False)
        .rolling(
            momentum_signal_window,
            min_periods=min(momentum_signal_window, max(3, momentum_signal_window // 2)),
        )
        .sum()
        .reset_index(level=0, drop=True)
    )
    frame["confirm_leg"] = _rolling_group_corr(
        x=frame["residual_return"],
        y=frame["amount_z"],
        assets=frame["asset"],
        window=confirm_window,
    )
    positive_shock = frame["residual_return"].clip(lower=0.0) * frame["amount_z"].clip(lower=0.0)
    frame["penalty_leg"] = (
        positive_shock.groupby(frame["asset"], sort=False)
        .rolling(penalty_window, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    out = frame[["date", "asset"]].copy()
    out["value"] = (
        _cross_sectional_zscore(frame, "momentum_leg")
        + confirm_weight * _cross_sectional_zscore(frame, "confirm_leg")
        - penalty_weight * _cross_sectional_zscore(frame, "penalty_leg")
    )
    return out


def _apply_preprocess(signal: pd.DataFrame, cfg: Mapping[str, object]) -> pd.DataFrame:
    out = signal.copy()

    winsorize_cfg = _optional_mapping(cfg.get("winsorize"), field="preprocess.winsorize")
    if winsorize_cfg is not None and bool(winsorize_cfg.get("enabled", True)):
        lower = _optional_float(winsorize_cfg.get("lower"), field="preprocess.winsorize.lower")
        upper = _optional_float(winsorize_cfg.get("upper"), field="preprocess.winsorize.upper")
        min_group_size = _optional_positive_int(
            winsorize_cfg.get("min_group_size"),
            field="preprocess.winsorize.min_group_size",
        )
        out = winsorize_cross_section(
            out,
            lower=0.01 if lower is None else lower,
            upper=0.99 if upper is None else upper,
            min_group_size=3 if min_group_size is None else min_group_size,
        )

    standardization = cfg.get("standardization")
    std_method = "none"
    std_group_size = 3
    if isinstance(standardization, str):
        std_method = standardization.strip().lower()
    elif isinstance(standardization, Mapping):
        standardization_cfg = _as_mapping(standardization, field="preprocess.standardization")
        raw_method = standardization_cfg.get("method", "none")
        if not isinstance(raw_method, str):
            raise FactorRecipeError("preprocess.standardization.method must be a string")
        std_method = raw_method.strip().lower()
        parsed_group_size = _optional_positive_int(
            standardization_cfg.get("min_group_size"),
            field="preprocess.standardization.min_group_size",
        )
        if parsed_group_size is not None:
            std_group_size = parsed_group_size
    elif standardization is not None:
        raise FactorRecipeError(
            "preprocess.standardization must be a string or mapping when provided"
        )

    if std_method not in {"none", "zscore", "rank"}:
        raise FactorRecipeError(
            "preprocess.standardization method must be one of ['none', 'zscore', 'rank']"
        )
    if std_method == "zscore":
        out = zscore_cross_section(out, min_group_size=std_group_size)
    elif std_method == "rank":
        out = rank_cross_section(out, min_group_size=std_group_size, pct=True)

    min_coverage = _optional_float(cfg.get("min_coverage"), field="preprocess.min_coverage")
    if min_coverage is not None:
        out = apply_min_coverage_gate(out, min_coverage=min_coverage)

    return out


def _apply_orthogonalization(
    *,
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    orthogonalize_cfg: Mapping[str, object],
) -> pd.DataFrame:
    exposures_raw = orthogonalize_cfg.get("exposures")
    if not isinstance(exposures_raw, Sequence) or isinstance(exposures_raw, (str, bytes)):
        raise FactorRecipeError("orthogonalize.exposures must be a non-empty list")
    exposure_cfgs = [
        _as_mapping(item, field=f"orthogonalize.exposures[{idx}]")
        for idx, item in enumerate(exposures_raw)
    ]
    if not exposure_cfgs:
        raise FactorRecipeError("orthogonalize.exposures must be non-empty")

    min_obs = _optional_positive_int(
        orthogonalize_cfg.get("min_obs"),
        field="orthogonalize.min_obs",
    )
    ridge = _optional_float(orthogonalize_cfg.get("ridge"), field="orthogonalize.ridge")

    panel = signal.copy()
    exposure_cols: list[str] = []
    for idx, cfg in enumerate(exposure_cfgs):
        step_signal = _compute_step_signal(prices=prices, step_cfg=cfg)
        col = f"exposure_{idx + 1}"
        exposure_cols.append(col)
        panel = panel.merge(
            step_signal.rename(columns={"value": col}),
            on=["date", "asset"],
            how="left",
            validate="one_to_one",
        )

    out = _cross_sectional_residualize(
        panel=panel,
        value_col="value",
        exposure_cols=exposure_cols,
        min_obs=20 if min_obs is None else min_obs,
        ridge=1e-8 if ridge is None else ridge,
    )
    return out[["date", "asset", "value"]].copy()


def _cross_sectional_residualize(
    *,
    panel: pd.DataFrame,
    value_col: str,
    exposure_cols: Sequence[str],
    min_obs: int,
    ridge: float,
) -> pd.DataFrame:
    if min_obs <= 0:
        raise FactorRecipeError("orthogonalize.min_obs must be > 0")
    if ridge < 0:
        raise FactorRecipeError("orthogonalize.ridge must be >= 0")

    out = panel.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    for col in exposure_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for _, idx in out.groupby("date", sort=False).groups.items():
        row_idx = pd.Index(idx)
        group = out.loc[row_idx, [value_col, *exposure_cols]].copy()
        valid = group[value_col].notna()
        for col in exposure_cols:
            valid &= group[col].notna()

        n_obs = int(valid.sum())
        if n_obs < min_obs:
            continue

        valid_index = group.index[valid]
        y = group.loc[valid_index, value_col].to_numpy(dtype=float)
        x_raw = group.loc[valid_index, list(exposure_cols)].to_numpy(dtype=float)
        x = np.column_stack([np.ones(n_obs, dtype=float), x_raw])

        xtx = x.T @ x
        if ridge > 0:
            reg = np.eye(xtx.shape[0], dtype=float) * ridge
            reg[0, 0] = 0.0
            xtx = xtx + reg

        xty = x.T @ y
        beta = np.linalg.pinv(xtx) @ xty
        resid = y - (x @ beta)
        out.loc[valid_index, value_col] = resid

    return out


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise FactorRecipeError(f"prices is missing required columns: {sorted(missing)}")

    keep_cols = ["date", "asset", "close"]
    for optional_col in ("high", "low", "volume", "amount"):
        if optional_col in prices.columns:
            keep_cols.append(optional_col)

    out = prices[keep_cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["asset"] = out["asset"].astype(str).str.strip()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "high" in out.columns:
        out["high"] = pd.to_numeric(out["high"], errors="coerce")
    if "low" in out.columns:
        out["low"] = pd.to_numeric(out["low"], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["date", "asset", "close"]).copy()
    out = out[out["asset"] != ""].copy()
    out = out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)

    if out.empty:
        raise FactorRecipeError("prices became empty after normalization")
    return out


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FactorRecipeError(f"{field} must be a mapping")
    return value


def _optional_mapping(value: object, *, field: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _as_mapping(value, field=field)


def _required_non_empty_string(value: str, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise FactorRecipeError(f"{field} must be non-empty")
    return text


def _optional_positive_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FactorRecipeError(f"{field} must be a positive integer")
    if isinstance(value, int):
        out = value
    elif isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            out = int(token)
        except ValueError as exc:
            raise FactorRecipeError(f"{field} must be a positive integer") from exc
    else:
        raise FactorRecipeError(f"{field} must be a positive integer")

    if out <= 0:
        raise FactorRecipeError(f"{field} must be > 0")
    return out


def _optional_non_negative_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FactorRecipeError(f"{field} must be a non-negative integer")
    if isinstance(value, int):
        out = value
    elif isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            out = int(token)
        except ValueError as exc:
            raise FactorRecipeError(f"{field} must be a non-negative integer") from exc
    else:
        raise FactorRecipeError(f"{field} must be a non-negative integer")

    if out < 0:
        raise FactorRecipeError(f"{field} must be >= 0")
    return out


def _optional_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FactorRecipeError(f"{field} must be numeric")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            return float(token)
        except ValueError as exc:
            raise FactorRecipeError(f"{field} must be numeric") from exc
    raise FactorRecipeError(f"{field} must be numeric")


def _yaml_load(text: str) -> object:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - import guard
        raise AlphaLabExperimentError("PyYAML is required to parse YAML factor recipes") from exc

    return yaml.safe_load(text)


def _resolve_amount_series(frame: pd.DataFrame) -> pd.Series:
    if "amount" in frame.columns:
        amount = pd.to_numeric(frame["amount"], errors="coerce")
        if amount.notna().any():
            return amount
    if "volume" in frame.columns:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        if volume.notna().any():
            return volume * pd.to_numeric(frame["close"], errors="coerce")
    raise FactorRecipeError("base.method='vcimom' requires prices with amount or volume columns")


def _rolling_beta(
    *,
    returns: pd.Series,
    market_returns: pd.Series,
    assets: pd.Series,
    window: int,
) -> pd.Series:
    out = pd.Series(np.nan, index=returns.index, dtype=float)
    min_periods = min(window, max(3, window // 3))
    market_var = (
        market_returns.groupby(assets, sort=False)
        .rolling(window, min_periods=min_periods)
        .var(ddof=1)
        .reset_index(level=0, drop=True)
    )
    cov = (
        (returns * market_returns).groupby(assets, sort=False)
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
        - (
            returns.groupby(assets, sort=False)
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
            * market_returns.groupby(assets, sort=False)
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
    )
    out.loc[:] = cov.div(market_var.replace(0.0, np.nan))
    return out


def _rolling_group_corr(
    *,
    x: pd.Series,
    y: pd.Series,
    assets: pd.Series,
    window: int,
) -> pd.Series:
    out = pd.Series(np.nan, index=x.index, dtype=float)
    for asset, idx in assets.groupby(assets, sort=False).groups.items():
        del asset
        asset_idx = pd.Index(idx)
        x_asset = pd.to_numeric(x.loc[asset_idx], errors="coerce")
        y_asset = pd.to_numeric(y.loc[asset_idx], errors="coerce")
        min_periods = min(window, max(3, window // 2))
        out.loc[asset_idx] = x_asset.rolling(
            window,
            min_periods=min_periods,
        ).corr(y_asset)
    return out


def _cross_sectional_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    mean = values.groupby(frame["date"], sort=False).transform("mean")
    std = values.groupby(frame["date"], sort=False).transform("std").replace(0.0, np.nan)
    return values.sub(mean).div(std)
