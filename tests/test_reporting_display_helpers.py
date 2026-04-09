from __future__ import annotations

from alpha_lab.reporting.display_helpers import (
    as_object_dict,
    as_object_list,
    format_ci,
    format_text_list,
    parse_text_list,
    portfolio_validation_benchmark_note,
    portfolio_validation_note,
    to_finite_float,
)


def test_parse_text_list_normalizes_semicolon_strings_and_sequences() -> None:
    assert parse_text_list("a; b ; ; c") == ["a", "b", "c"]
    assert parse_text_list(["a", " ", "b"]) == ["a", "b"]
    assert parse_text_list(("x", None, "y")) == ["x", "y"]


def test_format_text_list_respects_empty_and_separator() -> None:
    assert format_text_list(None, empty="none") == "none"
    assert format_text_list("a;b", empty="none") == "a, b"
    assert (
        format_text_list(["a", "b"], empty="none", separator="; ", split_semicolon=False)
        == "a; b"
    )


def test_numeric_and_mapping_helpers() -> None:
    assert to_finite_float(1) == 1.0
    assert to_finite_float(True) is None
    assert to_finite_float(float("inf")) is None
    assert format_ci(0.1, 0.2) == "[0.100000, 0.200000]"
    assert format_ci("x", 0.2) == "N/A"
    assert as_object_dict({"k": 1}) == {"k": 1}
    assert as_object_dict(None) == {}
    assert as_object_list([1, 2]) == [1, 2]
    assert as_object_list((1, 2)) == []


def test_portfolio_validation_notes_preserve_expected_text_shape() -> None:
    assert portfolio_validation_note(None, None) == "N/A"
    assert portfolio_validation_note("completed", "Credible at portfolio level") == (
        "completed (Credible at portfolio level)"
    )

    benchmark = portfolio_validation_benchmark_note(
        "available",
        "supports_standalone_strength",
        0.0005,
        0.019,
        format_metric=lambda value: "N/A" if value is None else f"{float(value):.6f}",
    )
    assert (
        benchmark
        == "available (supports_standalone_strength), excess=0.000500, tracking_error=0.019000"
    )
