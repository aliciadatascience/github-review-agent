import json
import logging
import requests
import os
from langchain.tools import Tool
from mcp_client import GitHubMCPClient

logger = logging.getLogger(__name__)

# GitHub API base URL
GITHUB_API = "https://api.github.com"


def create_github_tools(client: GitHubMCPClient) -> list[Tool]:

    # ── shared helpers ────────────────────────────────────────────────
    def clean_int(val) -> int:
        return int(str(val).strip().strip("'\""))

    def gh_headers() -> dict:
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def owner_repo() -> tuple:
        from config.settings import settings
        return settings.GITHUB_OWNER, settings.GITHUB_REPO

    # ── Tool: List open PRs — direct GitHub API ───────────────────────
    def list_open_prs(input_str: str = "") -> str:
        try:
            owner, repo = owner_repo()
            url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls?state=open&per_page=10"
            response = requests.get(url, headers=gh_headers(), timeout=15)
            prs = response.json()
            if not prs:
                return "No open pull requests found."
            if isinstance(prs, dict) and "message" in prs:
                return f"GitHub API error: {prs['message']}"
            summaries = []
            for pr in prs:
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

    # ── Tool: Get PR details — direct GitHub API ──────────────────────
    def get_pr_details(pr_number: str) -> str:
        try:
            owner, repo = owner_repo()
            num = clean_int(pr_number)
            url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{num}"
            response = requests.get(url, headers=gh_headers(), timeout=15)
            pr = response.json()
            if "message" in pr:
                return f"GitHub API error: {pr['message']}"
            return (
                f"PR #{pr.get('number')}: {pr.get('title')} | "
                f"Author: {pr.get('user', {}).get('login', 'unknown')} | "
                f"State: {pr.get('state', 'unknown')} | "
                f"Files changed: {pr.get('changed_files', 0)} | "
                f"Additions: +{pr.get('additions', 0)} | "
                f"Deletions: -{pr.get('deletions', 0)} | "
                f"Description: {pr.get('body', '(no description)')[:300]}"
            )
        except Exception as e:
            logger.error("get_pr_details error: %s", e)
            return f"Error getting PR details: {e}"

    # ── Tool: Get PR files — direct GitHub API ────────────────────────
    def get_pr_files(pr_number: str) -> str:
        try:
            owner, repo = owner_repo()
            num = clean_int(pr_number)
            url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{num}/files"
            response = requests.get(url, headers=gh_headers(), timeout=15)
            files = response.json()
            if isinstance(files, dict) and "message" in files:
                return f"GitHub API error: {files['message']}"
            if not files:
                return "No files changed."
            summaries = []
            for f in files:
                summaries.append(
                    f"  [{f.get('status', '?').upper()}] "
                    f"{f.get('filename')} "
                    f"(+{f.get('additions', 0)} / -{f.get('deletions', 0)})"
                )
            return "Files changed:\n" + "\n".join(summaries)
        except Exception as e:
            logger.error("get_pr_files error: %s", e)
            return f"Error getting PR files: {e}"

    # ── Tool: Get PR commits — direct GitHub API ──────────────────────
    def get_pr_commits(pr_number: str) -> str:
        try:
            owner, repo = owner_repo()
            num = clean_int(pr_number)
            url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{num}/commits"
            response = requests.get(url, headers=gh_headers(), timeout=15)
            commits = response.json()
            if isinstance(commits, dict) and "message" in commits:
                return f"GitHub API error: {commits['message']}"
            if not commits:
                return "No commits found."
            lines = []
            for c in commits:
                sha = c.get("sha", "")[:7]
                msg = c.get("commit", {}).get("message", "").split("\n")[0]
                author = c.get("commit", {}).get("author", {}).get("name", "?")
                lines.append(f"  {sha} — {msg} ({author})")
            return "Commits:\n" + "\n".join(lines)
        except Exception as e:
            logger.error("get_pr_commits error: %s", e)
            return f"Error getting PR commits: {e}"

    # ── Tool: Approve PR — uses MCP for writing ───────────────────────
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

    # ── Tool: Request changes — uses MCP for writing ──────────────────
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

    # ── Tool: Leave comment — uses MCP for writing ────────────────────
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

    # ── Tool: List recent commits — direct GitHub API ─────────────────
    def list_recent_commits(input_str: str = "") -> str:
        try:
            owner, repo = owner_repo()
            url = f"{GITHUB_API}/repos/{owner}/{repo}/commits?per_page=5"
            response = requests.get(url, headers=gh_headers(), timeout=15)
            commits = response.json()
            if isinstance(commits, dict) and "message" in commits:
                return f"GitHub API error: {commits['message']}"
            if not commits:
                return "No commits found."
            lines = []
            for c in commits:
                sha = c.get("sha", "")[:7]
                msg = c.get("commit", {}).get("message", "").split("\n")[0]
                author = c.get("commit", {}).get("author", {}).get("name", "?")
                date = c.get("commit", {}).get("author", {}).get("date", "")[:10]
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
                "Example: 15|LGTM, tests present and commits are clean."
            ),
        ),
        Tool(
            name="request_changes_on_pr",
            func=request_changes,
            description=(
                "Request changes on a pull request. "
                "Input format: pr_number|reason "
                "Example: 15|Missing unit tests for the new function."
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
            description="List 5 most recent commits on default branch. No input required.",
        ),
    ]
