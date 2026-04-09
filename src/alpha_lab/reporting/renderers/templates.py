from __future__ import annotations

import math
from collections.abc import Sequence

CASE_SECTION_TITLES: tuple[str, ...] = (
    "1. Title",
    "2. Objective",
    "3. Signal / Factor Definition",
    "4. Data & Setup",
    "5. Methodology",
    "6. Key Diagnostics",
    "7. Interpretation",
    "8. Risks & Limitations",
    "9. Next Steps",
)

CAMPAIGN_SECTION_TITLES: tuple[str, ...] = (
    "1. Title",
    "2. Campaign Objective",
    "3. Included Cases",
    "4. Comparison Table",
    "5. Case-by-Case Highlights",
    "6. Cross-Case Insights",
    "7. Failure Cases / Data Issues",
    "8. Conclusions",
    "9. Next Steps",
)

COMPARISON_TABLE_COLUMNS: tuple[str, ...] = (
    "Rank",
    "Case Name",
    "Type (single/composite)",
    "IC / ICIR",
    "IC 95% CI",
    "Long-Short Performance",
    "Turnover",
    "Coverage",
    "Uncertainty",
    "Neutralization Comparison",
    "Verdict",
    "Campaign Triage",
    "Triage Reasons",
    "Level 2 Promotion",
    "Promotion Reasons",
    "Promotion Blockers",
    "L1->L2 Transition",
    "Level 2 Portfolio Validation",
    "Portfolio Robustness",
    "Portfolio Benchmark Relative",
    "Portfolio Validation Risks",
    "Status",
)

PLACEHOLDER_INTERPRETATION = (
    "[Placeholder] Add an interpretation of statistical and economic significance."
)
PLACEHOLDER_NEXT_STEPS = (
    "[Placeholder] Add concrete follow-up experiments or research-validation checks."
)
PLACEHOLDER_OBJECTIVE = (
    "[Placeholder] Objective not found in artifacts. Add research intent manually."
)


def format_metric(value: object, *, precision: int = 6, na: str = "N/A") -> str:
    if value is None:
        return na
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return na
        return f"{value:.{precision}f}"
    return str(value)


def format_text(value: object, *, na: str = "N/A") -> str:
    if value is None:
        return na
    text = str(value).strip()
    return text if text else na


def section_lines(title: str, body_lines: Sequence[str]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.extend(body_lines)
    lines.append("")
    return lines


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_row, separator]
    for row in rows:
        rendered = [format_text(value) for value in row]
        lines.append("| " + " | ".join(rendered) + " |")
    return lines
