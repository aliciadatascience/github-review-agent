
import json
import logging
from langchain.tools import Tool
from mcp_client import GitHubMCPClient

logger = logging.getLogger(__name__)


def create_github_tools(client: GitHubMCPClient) -> list[Tool]:
    """
    Build the list of LangChain Tools the agent can use.
    Each tool wraps one of our MCP client convenience methods.
    """

    def _safe_json(data) -> str:
        """Convert any result to a clean string for the LLM."""
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, indent=2)
        except Exception:
            return str(data)

    # ------------------------------------------------------------------
    # Tool: List open pull requests
    # ------------------------------------------------------------------
    def list_open_prs(input_str: str = "") -> str:
        """List all open pull requests in the repository."""
        try:
            prs = client.list_pull_requests(state="open")
            if not prs:
                return "No open pull requests found."
            # Return concise summary for the LLM
            summaries = []
            for pr in (prs if isinstance(prs, list) else [prs]):
                summaries.append(
                    f"PR #{pr.get('number')}: {pr.get('title')} "
                    f"| Author: {pr.get('user', {}).get('login', '?')} "
                    f"| Branch: {pr.get('head', {}).get('ref', '?')} → {pr.get('base', {}).get('ref', '?')}"
                )
            return "\n".join(summaries)
        except Exception as e:
            logger.error("list_open_prs error: %s", e)
            return f"Error listing PRs: {e}"

    # ------------------------------------------------------------------
    # Tool: Get PR details
    # ------------------------------------------------------------------
    def get_pr_details(pr_number: str) -> str:
        """Get detailed information about a specific pull request."""
        try:
            pr = client.get_pull_request(int(pr_number.strip()))
            if not pr:
                return f"PR #{pr_number} not found."
            return (
                f"PR #{pr.get('number')}: {pr.get('title')}\n"
                f"Author: {pr.get('user', {}).get('login')}\n"
                f"State: {pr.get('state')}\n"
                f"Mergeable: {pr.get('mergeable')}\n"
                f"Commits: {pr.get('commits')}\n"
                f"Additions: +{pr.get('additions')} / Deletions: -{pr.get('deletions')}\n"
                f"Changed files: {pr.get('changed_files')}\n"
                f"Body:\n{pr.get('body', '(no description)')}"
            )
        except Exception as e:
            logger.error("get_pr_details error: %s", e)
            return f"Error getting PR details: {e}"

    # ------------------------------------------------------------------
    # Tool: Get PR file changes/diff
    # ------------------------------------------------------------------
    def get_pr_files(pr_number: str) -> str:
        """Get the list of files changed in a pull request."""
        try:
            files = client.get_pull_request_diff(int(pr_number.strip()))
            if not files:
                return "No file changes found."
            if isinstance(files, list):
                summaries = []
                for f in files:
                    summaries.append(
                        f"  [{f.get('status', '?').upper()}] {f.get('filename')} "
                        f"(+{f.get('additions', 0)} / -{f.get('deletions', 0)})"
                    )
                return "Files changed:\n" + "\n".join(summaries)
            return _safe_json(files)
        except Exception as e:
            logger.error("get_pr_files error: %s", e)
            return f"Error getting PR files: {e}"

    # ------------------------------------------------------------------
    # Tool: Get PR commits
    # ------------------------------------------------------------------
    def get_pr_commits(pr_number: str) -> str:
        """Get all commits included in a pull request."""
        try:
            commits = client.get_pull_request_commits(int(pr_number.strip()))
            if not commits:
                return "No commits found."
            if isinstance(commits, list):
                lines = []
                for c in commits:
                    sha = c.get("sha", "")[:7]
                    msg = c.get("commit", {}).get("message", "").split("\n")[0]
                    author = c.get("commit", {}).get("author", {}).get("name", "?")
                    lines.append(f"  {sha} — {msg} ({author})")
                return "Commits:\n" + "\n".join(lines)
            return _safe_json(commits)
        except Exception as e:
            logger.error("get_pr_commits error: %s", e)
            return f"Error getting PR commits: {e}"

    # ------------------------------------------------------------------
    # Tool: Approve PR
    # ------------------------------------------------------------------
    def approve_pr(input_str: str) -> str:
        """
        Approve a pull request. Input format: '<pr_number>|<review_comment>'
        Example: '42|Looks great, all checks pass!'
        """
        try:
            parts = input_str.split("|", 1)
            pr_number = int(parts[0].strip())
            body = parts[1].strip() if len(parts) > 1 else "✅ Approved by AI Review Agent."
            client.create_review(pr_number=pr_number, body=body, event="APPROVE")
            return f"✅ PR #{pr_number} approved with comment: {body}"
        except Exception as e:
            logger.error("approve_pr error: %s", e)
            return f"Error approving PR: {e}"

    # ------------------------------------------------------------------
    # Tool: Request changes on PR
    # ------------------------------------------------------------------
    def request_changes(input_str: str) -> str:
        """
        Request changes on a pull request. Input: '<pr_number>|<reason>'
        Example: '42|Missing unit tests for the new auth module.'
        """
        try:
            parts = input_str.split("|", 1)
            pr_number = int(parts[0].strip())
            body = parts[1].strip() if len(parts) > 1 else "Changes requested by AI Review Agent."
            client.create_review(pr_number=pr_number, body=body, event="REQUEST_CHANGES")
            return f"❌ Changes requested on PR #{pr_number}: {body}"
        except Exception as e:
            logger.error("request_changes error: %s", e)
            return f"Error requesting changes: {e}"

    # ------------------------------------------------------------------
    # Tool: Leave a neutral comment
    # ------------------------------------------------------------------
    def leave_comment(input_str: str) -> str:
        """
        Leave a review comment on a PR (neutral, does not approve or block).
        Input: '<pr_number>|<comment_text>'
        Example: '42|Nice work overall. Consider adding docstrings to new functions.'
        """
        try:
            parts = input_str.split("|", 1)
            pr_number = int(parts[0].strip())
            body = parts[1].strip() if len(parts) > 1 else "Reviewed by AI Agent."
            client.create_review(pr_number=pr_number, body=body, event="COMMENT")
            return f"💬 Comment left on PR #{pr_number}: {body}"
        except Exception as e:
            logger.error("leave_comment error: %s", e)
            return f"Error leaving comment: {e}"

    # ------------------------------------------------------------------
    # Tool: List recent commits
    # ------------------------------------------------------------------
    def list_recent_commits(input_str: str = "") -> str:
        """List the most recent commits on the default branch."""
        try:
            commits = client.list_commits(per_page=5)
            if not commits:
                return "No commits found."
            if isinstance(commits, list):
                lines = []
                for c in commits:
                    sha = c.get("sha", "")[:7]
                    msg = c.get("commit", {}).get("message", "").split("\n")[0]
                    author = c.get("commit", {}).get("author", {}).get("name", "?")
                    date = c.get("commit", {}).get("author", {}).get("date", "")[:10]
                    lines.append(f"  {sha} [{date}] {msg} ({author})")
                return "Recent commits:\n" + "\n".join(lines)
            return _safe_json(commits)
        except Exception as e:
            logger.error("list_recent_commits error: %s", e)
            return f"Error listing commits: {e}"

    # ------------------------------------------------------------------
    # Assemble and return all tools
    # ------------------------------------------------------------------
    return [
        Tool(
            name="list_open_pull_requests",
            func=list_open_prs,
            description=(
                "List all open pull requests in the GitHub repository. "
                "Use this first to see what PRs need review. No input required."
            ),
        ),
        Tool(
            name="get_pull_request_details",
            func=get_pr_details,
            description=(
                "Get detailed information about a specific pull request. "
                "Input: the PR number as a string (e.g. '42'). "
                "Returns title, author, state, size, and description."
            ),
        ),
        Tool(
            name="get_pull_request_files",
            func=get_pr_files,
            description=(
                "Get the list of files changed in a pull request. "
                "Input: the PR number as a string (e.g. '42'). "
                "Use this to understand what code was modified."
            ),
        ),
        Tool(
            name="get_pull_request_commits",
            func=get_pr_commits,
            description=(
                "Get all commits included in a pull request. "
                "Input: the PR number as a string (e.g. '42'). "
                "Use this to understand the commit history and messages."
            ),
        ),
        Tool(
            name="approve_pull_request",
            func=approve_pr,
            description=(
                "Approve a pull request after reviewing it. "
                "Input format: '<pr_number>|<approval_comment>'. "
                "Example: '42|All changes look good. Tests pass. Ready to merge.' "
                "Only approve if you are confident the PR is correct and safe."
            ),
        ),
        Tool(
            name="request_changes_on_pr",
            func=request_changes,
            description=(
                "Request changes on a pull request that has issues. "
                "Input format: '<pr_number>|<reason_for_changes>'. "
                "Example: '42|Missing error handling in the login function.' "
                "Use when the PR needs fixes before it can be merged."
            ),
        ),
        Tool(
            name="leave_review_comment",
            func=leave_comment,
            description=(
                "Leave a neutral review comment on a pull request. "
                "Input format: '<pr_number>|<comment_text>'. "
                "Example: '42|Good approach. Consider adding unit tests.' "
                "Use for feedback that does not block or approve the PR."
            ),
        ),
        Tool(
            name="list_recent_commits",
            func=list_recent_commits,
            description=(
                "List the 5 most recent commits on the default branch. "
                "Use to get a quick overview of recent repository activity. "
                "No input required."
            ),
        ),
    ]
