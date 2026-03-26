__all__ = [
    "render_case_report",
    "write_case_report",
    "render_campaign_report",
    "write_campaign_report",
]


def __getattr__(name: str):
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

    raise AttributeError(name)
