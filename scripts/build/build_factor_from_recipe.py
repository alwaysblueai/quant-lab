from __future__ import annotations

import argparse
from pathlib import Path

from alpha_lab.factor_recipe import (
    FactorRecipeError,
    build_factor_from_recipe_file,
    load_recipe_mapping,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build canonical factor CSV from a declarative recipe and prices CSV.")
    )
    parser.add_argument("--recipe", required=True, help="Path to recipe YAML/JSON")
    parser.add_argument("--prices", required=True, help="Path to prices.csv")
    parser.add_argument("--output", required=True, help="Output factor CSV path")
    parser.add_argument(
        "--factor-name",
        default="",
        help=(
            "Output factor name. If omitted, use recipe.output_factor; "
            "error if neither is provided."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    recipe_path = Path(args.recipe).resolve()
    prices_path = Path(args.prices).resolve()
    output_path = Path(args.output).resolve()

    recipe = load_recipe_mapping(recipe_path)
    factor_name = str(args.factor_name or "").strip()
    if not factor_name:
        raw = recipe.get("output_factor")
        if not isinstance(raw, str) or not raw.strip():
            raise FactorRecipeError(
                "factor name is required: pass --factor-name or set recipe.output_factor"
            )
        factor_name = raw.strip()

    factor_df = build_factor_from_recipe_file(
        prices_path=prices_path,
        recipe_path=recipe_path,
        factor_name=factor_name,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    factor_df.to_csv(output_path, index=False)

    print(f"factor_name: {factor_name}")
    print(f"rows: {len(factor_df)}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
