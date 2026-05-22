"""``alpha_lab.reporting.renderers`` — HTML renderer entry points.

Public names listed in ``__all__`` are resolved lazily via ``__getattr__`` so
that simply touching this package (e.g. from CLI parser registration or other
report-adjacent code paths) does not eagerly import
``campaign_profile_dashboard`` (~6.4kLoC of template code) or pull in the
case / campaign report modules' jinja / chart-rendering dependencies. Each
renderer module is loaded only when one of its names is actually accessed.
"""

from typing import Any

__all__ = [
    "render_case_report",
    "write_case_report",
    "render_campaign_report",
    "write_campaign_report",
    "render_campaign_profile_dashboard_html",
    "write_campaign_profile_dashboard_html",
]


def __getattr__(name: str) -> Any:
    if name in {"render_case_report", "write_case_report"}:
        from alpha_lab.reporting.renderers.case_report import (
            render_case_report,
            write_case_report,
        )

        return {
            "render_case_report": render_case_report,
            "write_case_report": write_case_report,
        }[name]

    if name in {"render_campaign_report", "write_campaign_report"}:
        from alpha_lab.reporting.renderers.campaign_report import (
            render_campaign_report,
            write_campaign_report,
        )

        return {
            "render_campaign_report": render_campaign_report,
            "write_campaign_report": write_campaign_report,
        }[name]

    if name in {
        "render_campaign_profile_dashboard_html",
        "write_campaign_profile_dashboard_html",
    }:
        from alpha_lab.reporting.renderers.campaign_profile_dashboard import (
            render_campaign_profile_dashboard_html,
            write_campaign_profile_dashboard_html,
        )

        return {
            "render_campaign_profile_dashboard_html": (render_campaign_profile_dashboard_html),
            "write_campaign_profile_dashboard_html": (write_campaign_profile_dashboard_html),
        }[name]

    raise AttributeError(name)
