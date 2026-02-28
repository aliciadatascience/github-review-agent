"""
agent/event_handler.py

Reads the GitHub Actions event context (injected via environment variables)
and routes to the correct agent behavior.

WHY THIS FILE EXISTS:
  When GitHub Actions runs our agent, it tells us WHAT happened via env vars:
    - GH_EVENT_NAME   = "pull_request" or "push"
    - GH_EVENT_ACTION = "opened", "synchronize", "review_requested", etc.
    - GH_PR_NUMBER    = "42" (if it's a PR event)
    - GH_COMMIT_SHA   = the commit SHA (for push events)

  This file reads those vars and decides:
    - PR opened       → full review (files + commits → approve/request changes)
    - PR synchronized → re-review changed files only
    - review_requested → full review with extra thoroughness
    - push to main    → commit message quality check + comment

EVENT ROUTING TABLE:
  pull_request / opened          → full_pr_review()
  pull_request / synchronize     → incremental_pr_review()
  pull_request / review_requested → full_pr_review() (thorough mode)
  push                           → commit_check()
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
    """Parsed GitHub Actions event context."""
    event_name: str          # "pull_request" or "push"
    event_action: str        # "opened", "synchronize", "review_requested", etc.
    pr_number: Optional[int] # PR number, if this is a PR event
    commit_sha: str          # The triggering commit SHA
    repo_owner: str
    repo_name: str
    review_mode: ReviewMode  # What the agent should do


def parse_event_context() -> GitHubEventContext:
    """
    Read GitHub Actions environment variables and determine what the agent
    should do in response to the event.
    """
    event_name = os.getenv("GH_EVENT_NAME", "").lower()
    event_action = os.getenv("GH_EVENT_ACTION", "").lower()
    pr_number_str = os.getenv("GH_PR_NUMBER", "")
    commit_sha = os.getenv("GH_COMMIT_SHA", "")
    repo_owner = os.getenv("GITHUB_OWNER", "")
    repo_name = os.getenv("GITHUB_REPO", "")

    # Parse PR number safely
    pr_number = None
    if pr_number_str and pr_number_str.strip().isdigit():
        pr_number = int(pr_number_str.strip())

    # Route to the right review mode
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
        "Event context: event=%s action=%s pr=#%s mode=%s",
        event_name, event_action, pr_number or "N/A", review_mode.value,
    )

    return ctx


def _determine_review_mode(event_name: str, event_action: str, pr_number: Optional[int]) -> ReviewMode:
    """Map GitHub event + action to a ReviewMode."""

    if event_name == "pull_request":
        if event_action == "opened":
            # Brand new PR — do a full review
            return ReviewMode.FULL_PR_REVIEW

        elif event_action == "synchronize":
            # New commits pushed to an existing PR — re-review
            return ReviewMode.INCREMENTAL_PR_REVIEW

        elif event_action == "review_requested":
            # Someone explicitly asked for a review — be thorough
            return ReviewMode.THOROUGH_PR_REVIEW

    elif event_name == "push":
        # Direct push to a branch — check commit quality
        return ReviewMode.COMMIT_CHECK

    return ReviewMode.UNKNOWN


def build_agent_task(ctx: GitHubEventContext) -> str:
    """
    Build the natural-language task description for the LangChain agent
    based on the event context.

    The LLM reads this as its instruction. Be specific — the clearer the
    task, the better the agent performs.
    """

    if ctx.review_mode == ReviewMode.FULL_PR_REVIEW:
        return (
            f"A new pull request #{ctx.pr_number} has been opened. "
            f"Please do a complete code review:\n"
            f"1. Get the PR details (title, description, author)\n"
            f"2. Check which files were changed\n"
            f"3. Check the commit messages for clarity\n"
            f"4. Decide: APPROVE if the changes look correct and safe, "
            f"REQUEST CHANGES if there are bugs/missing tests/unclear code, "
            f"or COMMENT if you have suggestions but it's generally acceptable.\n"
            f"5. Write specific, helpful feedback explaining your decision.\n"
            f"Be constructive and explain your reasoning clearly."
        )

    elif ctx.review_mode == ReviewMode.INCREMENTAL_PR_REVIEW:
        return (
            f"New commits were pushed to pull request #{ctx.pr_number}. "
            f"Please re-review the PR:\n"
            f"1. Get the updated PR details\n"
            f"2. Check the changed files — focus on what's new\n"
            f"3. Check the latest commit messages\n"
            f"4. Update your review: APPROVE, REQUEST CHANGES, or COMMENT.\n"
            f"If previous issues have been addressed, say so in your feedback."
        )

    elif ctx.review_mode == ReviewMode.THOROUGH_PR_REVIEW:
        return (
            f"A code review was explicitly requested for PR #{ctx.pr_number}. "
            f"Please do a thorough review:\n"
            f"1. Get full PR details\n"
            f"2. Review ALL changed files carefully\n"
            f"3. Check every commit message\n"
            f"4. Look for: bugs, missing error handling, missing tests, "
            f"unclear variable names, security issues, performance concerns.\n"
            f"5. APPROVE only if confident everything is correct. "
            f"REQUEST CHANGES with specific line-level feedback if anything is wrong.\n"
            f"This review was explicitly requested — be thorough and detailed."
        )

    elif ctx.review_mode == ReviewMode.COMMIT_CHECK:
        return (
            f"Code was pushed directly to a branch (commit: {ctx.commit_sha[:7]}). "
            f"Please check the recent commits:\n"
            f"1. List the 5 most recent commits\n"
            f"2. Check if commit messages are descriptive (not just 'fix' or 'update')\n"
            f"3. If there are open PRs for this branch, leave a comment with your findings.\n"
            f"4. Note any concerns about the commit quality."
        )

    else:
        return (
            "Check the repository for any open pull requests that need review. "
            "Review each one and decide: APPROVE, REQUEST CHANGES, or COMMENT."
        )
