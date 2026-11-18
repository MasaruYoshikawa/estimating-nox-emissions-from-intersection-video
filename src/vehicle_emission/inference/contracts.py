"""Shared vocabulary for the reviewed inference preparation contracts."""

from __future__ import annotations

LOCKED_RETROSPECTIVE_SCOPE = "locked_retrospective_two_site"
# This exact-calendar-hour scaffold accepts the end-labelled candidate only;
# the Tokyo-specific source convention remains to be established for empirical use.
MONITOR_LABEL_SEMANTICS = frozenset({"end-labelled-hour"})
VIDEO_COVERAGE_GAP_RULES = frozenset({"require_contiguous_coverage", "allow_internal_gaps"})

EXPLORATORY_DECISION_CATEGORIES = frozenset(
    {
        "exploratory_practically_strong",
        "exploratory_positive",
        "exploratory_non_positive",
        "exploratory_meaningful_moderation",
        "exploratory_rules_out_proposed_magnitude",
        "exploratory_meaningful_added_value",
        "exploratory_meaningful_count_advantage",
        "exploratory_practical_equivalence",
        "exploratory_inconclusive",
        "exploratory_no_confirmatory_decision",
    }
)
RESULT_DECISION_CATEGORIES = frozenset(
    {*EXPLORATORY_DECISION_CATEGORIES, "descriptive_only", "not_estimable"}
)

__all__ = [
    "EXPLORATORY_DECISION_CATEGORIES",
    "LOCKED_RETROSPECTIVE_SCOPE",
    "MONITOR_LABEL_SEMANTICS",
    "RESULT_DECISION_CATEGORIES",
    "VIDEO_COVERAGE_GAP_RULES",
]
