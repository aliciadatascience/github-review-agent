"""
agent/event_handler.py
Reads GitHub Actions event context and builds a direct task for the agent.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ReviewMode(Enum):
    FULL_PR_REVIEW = "full_pr_review"
    INCREMENTAL_PR_REVIEW = "incremental_pr_review"
    THOROUGH_PR_REVIEW = "thorough_pr_review"
    COMMIT_CHECK = "commit_check"
    UNKNOWN = "unknown"


@dataclass
class GitHubEventContext:
    event_name: str
    event_action: str
    pr_number: Optional[int]
    commit_sha: str
    repo_owner: str
    repo_name: str
    review_mode: ReviewMode


def parse_event_context() -> GitHubEventContext:
    event_name   = os.getenv("GH_EVENT_NAME", "").lower()
    event_action = os.getenv("GH_EVENT_ACTION", "").lower()
    pr_number_str = os.getenv("GH_PR_NUMBER", "")
    commit_sha   = os.getenv("GH_COMMIT_SHA", "")
    repo_owner   = os.getenv("GITHUB_OWNER", "")
    repo_name    = os.getenv("GITHUB_REPO", "")

    pr_number = None
    if pr_number_str and pr_number_str.strip().isdigit():
        pr_number = int(pr_number_str.strip())

    review_mode = _determine_review_mode(event_name, event_action, pr_number)

    ctx = GitHubEventContext(
        event_name=event_name,
        event_action=event_action,
        pr_number=pr_number,
        commit_sha=commit_sha,
        repo_owner=repo_owner,
        repo_name=repo_name,
        review_mode=review_mode,
    )

    logger.info(
        "Event: event=%s action=%s pr=#%s mode=%s",
        event_name, event_action, pr_number or "N/A", review_mode.value,
    )
    return ctx


def _determine_review_mode(
    event_name: str,
    event_action: str,
    pr_number: Optional[int]
) -> ReviewMode:
    if event_name == "pull_request":
        if event_action == "opened":
            return ReviewMode.FULL_PR_REVIEW
        elif event_action == "synchronize":
            return ReviewMode.INCREMENTAL_PR_REVIEW
        elif event_action == "review_requested":
            return ReviewMode.THOROUGH_PR_REVIEW
    elif event_name == "push":
        return ReviewMode.COMMIT_CHECK
    return ReviewMode.UNKNOWN


def build_agent_task(ctx: GitHubEventContext) -> str:
    """
    Build an extremely direct task so the agent wastes zero steps.
    Tell it exactly what to do and in what order.
    """

    if ctx.review_mode in (
        ReviewMode.FULL_PR_REVIEW,
        ReviewMode.INCREMENTAL_PR_REVIEW,
        ReviewMode.THOROUGH_PR_REVIEW,
    ):
        pr = ctx.pr_number
        return f"""Review pull request #{pr}. Follow these steps IN ORDER. Do NOT repeat any step.

Step 1: call get_pull_request_details with input: {pr}
Step 2: call get_pull_request_files with input: {pr}
Step 3: call get_pull_request_commits with input: {pr}
Step 4: Based ONLY on what you found in steps 1-3, call ONE of:
  - approve_pull_request with input: {pr}|your detailed reason
  - request_changes_on_pr with input: {pr}|your detailed reason
  - leave_review_comment with input: {pr}|your detailed reason

STRICT RULES:
- Input is always just the number {pr}, never quoted like '{pr}'
- Do NOT call list_open_pull_requests — you already know PR number is {pr}
- Do NOT repeat a step you already completed
- After step 4 you are DONE — stop immediately
- If a step returns an error, move on to the next step anyway
"""

    elif ctx.review_mode == ReviewMode.COMMIT_CHECK:
        return f"""Check recent commit quality for commit {ctx.commit_sha[:7]}.

Step 1: call list_recent_commits with no input
Step 2: call leave_review_comment with input: 1|your analysis of commit message quality

STRICT RULES:
- Do NOT repeat steps
- Stop after step 2
"""

    else:
        pr = ctx.pr_number or 1
        return f"""Review pull request #{pr}.

Step 1: call get_pull_request_details with input: {pr}
Step 2: call get_pull_request_files with input: {pr}
Step 3: call get_pull_request_commits with input: {pr}
Step 4: call approve_pull_request OR request_changes_on_pr with input: {pr}|your reason

Stop after step 4.
"""
