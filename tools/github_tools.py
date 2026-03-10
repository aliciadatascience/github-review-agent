import logging
from langchain.tools import Tool
from mcp_client import GitHubMCPClient

logger = logging.getLogger(__name__)


def create_github_tools(client: GitHubMCPClient) -> list[Tool]:

    def clean_int(val) -> int:
        return int(str(val).strip().strip("'\""))

    # ── Tool: List open PRs ───────────────────────────────────────────
    def list_open_prs(input_str: str = "") -> str:
        try:
            prs = client.list_pull_requests(state="open")
            if not prs:
                return "No open pull requests found."
            summaries = []
            for pr in prs:
                if not isinstance(pr, dict):
                    continue
                user = pr.get("user") or {}
                head = pr.get("head") or {}
                base = pr.get("base") or {}
                summaries.append(
                    f"PR #{pr.get('number')}: {pr.get('title')} "
                    f"| Author: {user.get('login', '?')} "
                    f"| Branch: {head.get('ref', '?')} "
                    f"→ {base.get('ref', '?')}"
                )
            return "\n".join(summaries) if summaries else "No open pull requests found."
        except Exception as e:
            logger.error("list_open_prs error: %s", e)
            return f"Error listing PRs: {e}"

    # ── Tool: Get PR details ──────────────────────────────────────────
    def get_pr_details(pr_number: str) -> str:
        try:
            num = clean_int(pr_number)
            pr = client.get_pull_request(num)
            if not isinstance(pr, dict):
                return f"PR details (raw): {str(pr)[:300]}"
            user = pr.get("user") or {}
            head = pr.get("head") or {}
            base = pr.get("base") or {}
            return (
                f"PR #{pr.get('number')}: {pr.get('title')} | "
                f"Author: {user.get('login', 'unknown')} | "
                f"State: {pr.get('state', 'unknown')} | "
                f"Branch: {head.get('ref', '?')} → {base.get('ref', '?')} | "
                f"Files changed: {pr.get('changed_files', 0)} | "
                f"Additions: +{pr.get('additions', 0)} | "
                f"Deletions: -{pr.get('deletions', 0)} | "
                f"Description: {pr.get('body') or '(no description)'}"
            )
        except Exception as e:
            logger.error("get_pr_details error: %s", e)
            return f"Error getting PR details: {e}"

    # ── Tool: Get PR files ────────────────────────────────────────────
    def get_pr_files(pr_number: str) -> str:
        try:
            num = clean_int(pr_number)
            files = client.get_pull_request_diff(num)
            if not files:
                return "No files changed."
            summaries = []
            for f in files:
                if not isinstance(f, dict):
                    summaries.append(str(f))
                    continue
                summaries.append(
                    f"  [{(f.get('status') or '?').upper()}] "
                    f"{f.get('filename') or 'unknown'} "
                    f"(+{f.get('additions') or 0} / -{f.get('deletions') or 0})"
                )
            return "Files changed:\n" + "\n".join(summaries)
        except Exception as e:
            logger.error("get_pr_files error: %s", e)
            return f"Error getting PR files: {e}"

    # ── Tool: Get PR commits ──────────────────────────────────────────
    def get_pr_commits(pr_number: str) -> str:
        try:
            num = clean_int(pr_number)
            commits = client.get_pull_request_commits(num)
            if not commits:
                return "No commits found."
            lines = []
            for c in commits:
                if not isinstance(c, dict):
                    lines.append(str(c))
                    continue
                sha = (c.get("sha") or "")[:7]
                commit = c.get("commit") or {}
                msg = (commit.get("message") or "").split("\n")[0]
                author = (commit.get("author") or {}).get("name") or "?"
                lines.append(f"  {sha} — {msg} ({author})")
            return "Commits:\n" + "\n".join(lines)
        except Exception as e:
            logger.error("get_pr_commits error: %s", e)
            return f"Error getting PR commits: {e}"

    # ── Tool: Approve PR ──────────────────────────────────────────────
    def approve_pr(input_str: str) -> str:
        try:
            parts = input_str.split("|", 1)
            pr_number = clean_int(parts[0])
            body = parts[1].strip() if len(parts) > 1 else "✅ Approved by AI Review Agent."
            client.create_review(pr_number=pr_number, body=body, event="APPROVE")
            return f"✅ PR #{pr_number} approved: {body}"
        except Exception as e:
            logger.error("approve_pr error: %s", e)
            return f"Error approving PR: {e}"

    # ── Tool: Request changes ─────────────────────────────────────────
    def request_changes(input_str: str) -> str:
        try:
            parts = input_str.split("|", 1)
            pr_number = clean_int(parts[0])
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
            pr_number = clean_int(parts[0])
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
            lines = []
            for c in commits:
                if not isinstance(c, dict):
                    continue
                sha = (c.get("sha") or "")[:7]
                commit = c.get("commit") or {}
                msg = (commit.get("message") or "").split("\n")[0]
                author = (commit.get("author") or {}).get("name") or "?"
                date = (commit.get("author") or {}).get("date", "")[:10]
                lines.append(f"  {sha} [{date}] {msg} ({author})")
            return "Recent commits:\n" + "\n".join(lines)
        except Exception as e:
            logger.error("list_recent_commits error: %s", e)
            return f"Error listing commits: {e}"

    # ── Assemble tools ────────────────────────────────────────────────
    return [
        Tool(
            name="list_open_pull_requests",
            func=list_open_prs,
            description="List all open pull requests. No input required.",
        ),
        Tool(
            name="get_pull_request_details",
            func=get_pr_details,
            description="Get details about a PR. Input: PR number only. Example: 15",
        ),
        Tool(
            name="get_pull_request_files",
            func=get_pr_files,
            description="Get files changed in a PR. Input: PR number only. Example: 15",
        ),
        Tool(
            name="get_pull_request_commits",
            func=get_pr_commits,
            description="Get commits in a PR. Input: PR number only. Example: 15",
        ),
        Tool(
            name="approve_pull_request",
            func=approve_pr,
            description=(
                "Approve a pull request. "
                "Input format: pr_number|comment "
                "Example: 15|LGTM, safe to merge."
            ),
        ),
        Tool(
            name="request_changes_on_pr",
            func=request_changes,
            description=(
                "Request changes on a pull request. "
                "Input format: pr_number|reason "
                "Example: 15|Missing unit tests."
            ),
        ),
        Tool(
            name="leave_review_comment",
            func=leave_comment,
            description=(
                "Leave a neutral comment on a pull request. "
                "Input format: pr_number|comment "
                "Example: 15|Good approach, consider adding docstrings."
            ),
        ),
        Tool(
            name="list_recent_commits",
            func=list_recent_commits,
            description="List 5 most recent commits. No input required.",
        ),
    ]
