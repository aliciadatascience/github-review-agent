import json
import logging
from langchain.tools import Tool
from mcp_client import GitHubMCPClient

logger = logging.getLogger(__name__)


def create_github_tools(client: GitHubMCPClient) -> list[Tool]:

    def _safe_json(data) -> str:
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, indent=2)
        except Exception:
            return str(data)

    # ── shared helper — strips quotes LLM adds around numbers ────────
    def clean_int(val) -> int:
        """Convert '9', "9", 9 all safely to integer 9."""
        return int(str(val).strip().strip("'\""))

    # ── Tool: List open PRs ───────────────────────────────────────────
    def list_open_prs(input_str: str = "") -> str:
        try:
            prs = client.list_pull_requests(state="open")
            # handle string response
            if isinstance(prs, str):
                return prs
            if not prs:
                return "No open pull requests found."
            summaries = []
            for pr in (prs if isinstance(prs, list) else [prs]):
                if isinstance(pr, str):
                    summaries.append(pr)
                    continue
                summaries.append(
                    f"PR #{pr.get('number')}: {pr.get('title')} "
                    f"| Author: {pr.get('user', {}).get('login', '?')} "
                    f"| Branch: {pr.get('head', {}).get('ref', '?')} "
                    f"→ {pr.get('base', {}).get('ref', '?')}"
                )
            return "\n".join(summaries)
        except Exception as e:
            logger.error("list_open_prs error: %s", e)
            return f"Error listing PRs: {e}"

    # ── Tool: Get PR details ──────────────────────────────────────────
    def get_pr_details(pr_number: str) -> str:
        try:
            result = client.get_pull_request(clean_int(pr_number))
            # handle string response
            if isinstance(result, str):
                return f"PR details: {result[:500]}"
            if isinstance(result, list):
                result = result[0] if result else {}
            return (
                f"PR #{result.get('number', 'unknown')}: "
                f"{result.get('title', 'no title')} | "
                f"Author: {result.get('user', {}).get('login', 'unknown')} | "
                f"State: {result.get('state', 'unknown')} | "
                f"Files changed: {result.get('changed_files', 'unknown')} | "
                f"Additions: +{result.get('additions', 0)} | "
                f"Deletions: -{result.get('deletions', 0)} | "
                f"Description: {result.get('body', '(no description)')[:200]}"
            )
        except Exception as e:
            logger.error("get_pr_details error: %s", e)
            return f"Error getting PR details: {e}"

    # ── Tool: Get PR files ────────────────────────────────────────────
    def get_pr_files(pr_number: str) -> str:
        try:
            files = client.get_pull_request_diff(clean_int(pr_number))  # ← fixed
            if not files:
                return "No file changes found."
            if isinstance(files, list):
                summaries = []
                for f in files:
                    summaries.append(
                        f"  [{f.get('status', '?').upper()}] "
                        f"{f.get('filename')} "
                        f"(+{f.get('additions', 0)} / -{f.get('deletions', 0)})"
                    )
                return "Files changed:\n" + "\n".join(summaries)
            return _safe_json(files)
        except Exception as e:
            logger.error("get_pr_files error: %s", e)
            return f"Error getting PR files: {e}"

    # ── Tool: Get PR commits ──────────────────────────────────────────
    def get_pr_commits(pr_number: str) -> str:
        try:
            commits = client.get_pull_request_commits(clean_int(pr_number))  # ← fixed
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

    # ── Tool: Approve PR ──────────────────────────────────────────────
    def approve_pr(input_str: str) -> str:
        try:
            parts = input_str.split("|", 1)
            pr_number = clean_int(parts[0])  # ← fixed
            body = parts[1].strip() if len(parts) > 1 else "✅ Approved by AI Review Agent."
            client.create_review(pr_number=pr_number, body=body, event="APPROVE")
            return f"✅ PR #{pr_number} approved with comment: {body}"
        except Exception as e:
            logger.error("approve_pr error: %s", e)
            return f"Error approving PR: {e}"

    # ── Tool: Request changes ─────────────────────────────────────────
    def request_changes(input_str: str) -> str:
        try:
            parts = input_str.split("|", 1)
            pr_number = clean_int(parts[0])  # ← fixed
            body = parts[1].strip() if len(parts) > 1 else "Changes requested by AI Review Agent."
            client.create_review(pr_number=pr_number, body=body, event="REQUEST_CHANGES")
            return f"❌ Changes requested on PR #{pr_number}: {body}"
        except Exception as e:
            logger.error("request_changes error: %s", e)
            return f"Error requesting changes: {e}"

    # ── Tool: Leave comment ───────────────────────────────────────────
    def leave_comment(input_str: str) -> str:
        try:
            parts = input_str.split("|", 1)
            pr_number = clean_int(parts[0])  # ← fixed
            body = parts[1].strip() if len(parts) > 1 else "Reviewed by AI Agent."
            client.create_review(pr_number=pr_number, body=body, event="COMMENT")
            return f"💬 Comment left on PR #{pr_number}: {body}"
        except Exception as e:
            logger.error("leave_comment error: %s", e)
            return f"Error leaving comment: {e}"

    # ── Tool: List recent commits ─────────────────────────────────────
    def list_recent_commits(input_str: str = "") -> str:
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

    # ── Assemble tools ────────────────────────────────────────────────
    return [
        Tool(
            name="list_open_pull_requests",
            func=list_open_prs,
            description=(
                "List all open pull requests in the repository. "
                "No input required."
            ),
        ),
        Tool(
            name="get_pull_request_details",
            func=get_pr_details,
            description=(
                "Get details about a specific pull request. "
                "Input: PR number only, no quotes. Example: 9"
            ),
        ),
        Tool(
            name="get_pull_request_files",
            func=get_pr_files,
            description=(
                "Get files changed in a pull request. "
                "Input: PR number only, no quotes. Example: 9"
            ),
        ),
        Tool(
            name="get_pull_request_commits",
            func=get_pr_commits,
            description=(
                "Get commits in a pull request. "
                "Input: PR number only, no quotes. Example: 9"
            ),
        ),
        Tool(
            name="approve_pull_request",
            func=approve_pr,
            description=(
                "Approve a pull request. "
                "Input format: pr_number|comment "
                "Example: 9|LGTM, tests present and commits are clean."
            ),
        ),
        Tool(
            name="request_changes_on_pr",
            func=request_changes,
            description=(
                "Request changes on a pull request. "
                "Input format: pr_number|reason "
                "Example: 9|Missing unit tests for the new function."
            ),
        ),
        Tool(
            name="leave_review_comment",
            func=leave_comment,
            description=(
                "Leave a neutral comment on a pull request. "
                "Input format: pr_number|comment "
                "Example: 9|Good approach, consider adding docstrings."
            ),
        ),
        Tool(
            name="list_recent_commits",
            func=list_recent_commits,
            description=(
                "List 5 most recent commits on the default branch. "
                "No input required."
            ),
        ),
    ]
