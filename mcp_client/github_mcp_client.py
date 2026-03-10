from __future__ import annotations
import json
import logging
import subprocess
import threading
import time
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class MCPStdioTransport:
    def __init__(self, command: list, env: dict = None):
        self.command = command
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()

    def start(self):
        import os
        proc_env = os.environ.copy()
        proc_env.update(self.env)
        logger.info(f"Starting MCP server: {' '.join(self.command)}")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=proc_env,
            text=True,
            bufsize=1,
        )
        logger.info("MCP server process started (PID: %s)", self.process.pid)

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def send_request(self, method: str, params: dict = None) -> dict:
        if not self.process or self.process.poll() is not None:
            raise ConnectionError("MCP server process is not running")
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or {},
        }
        request_str = json.dumps(request) + "\n"
        logger.debug("→ MCP Request: %s", request_str.strip())
        try:
            self.process.stdin.write(request_str)
            self.process.stdin.flush()
            response_str = self.process.stdout.readline()
            logger.debug("← MCP Response: %s", response_str.strip())
            if not response_str:
                stderr = self.process.stderr.read()
                raise ConnectionError(f"MCP server closed. stderr: {stderr}")
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from MCP server: {response_str}") from e

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("MCP server process stopped")


class GitHubMCPClient:
    """
    MCP client using verified tool names from actual server log (Mar 2026):
      READ:  pull_request_read  (method: get / get_files / get_commits)
      WRITE: pull_request_review_write  (method: create_review)
      LIST:  list_pull_requests, list_commits
    """

    def __init__(self):
        self._transport: Optional[MCPStdioTransport] = None
        self._available_tools: list = []
        self._initialized = False
        self.owner = settings.GITHUB_OWNER
        self.repo = settings.GITHUB_REPO

    def connect(self):
        server_command, server_env = self._build_server_command()
        self._transport = MCPStdioTransport(
            command=server_command, env=server_env
        )
        self._transport.start()
        self._initialize_session()
        self._initialized = True
        logger.info("✅ Connected to GitHub MCP Server")

    def _build_server_command(self) -> tuple:
        env = {"GITHUB_TOKEN": settings.GITHUB_TOKEN}
        if self._command_exists("docker"):
            command = [
                "docker", "run", "--rm", "-i",
                "-e", "GITHUB_TOKEN",
                "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server",
            ]
            logger.info("Using Docker to run GitHub MCP Server")
            return command, env
        elif self._command_exists("npx"):
            command = ["npx", "-y", "@github/github-mcp-server"]
            logger.info("Using npx to run GitHub MCP Server")
            return command, env
        else:
            raise RuntimeError("Neither Docker nor npx found.")

    def _command_exists(self, cmd: str) -> bool:
        import shutil
        return shutil.which(cmd) is not None

    def _initialize_session(self):
        response = self._transport.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "github-review-agent", "version": "1.0.0"},
        })
        if "error" in response:
            raise ConnectionError(f"MCP initialization failed: {response['error']}")
        self._transport.send_request("notifications/initialized", {})

    def list_tools(self) -> list:
        response = self._transport.send_request("tools/list", {})
        if "error" in response:
            raise RuntimeError(f"Failed to list tools: {response['error']}")
        self._available_tools = response.get("result", {}).get("tools", [])
        logger.info("Discovered %d tools from GitHub MCP Server", len(self._available_tools))
        for tool in self._available_tools:
            logger.info("  TOOL: %s", tool["name"])
        return self._available_tools

    def get_tool_names(self) -> list:
        return [t["name"] for t in self._available_tools]

    def call_tool(self, tool_name: str, arguments: dict = None, retries: int = 2) -> Any:
        if not self._initialized:
            raise RuntimeError("Client not connected. Call connect() first.")
        arguments = arguments or {}
        last_error = None
        for attempt in range(retries + 1):
            try:
                response = self._transport.send_request("tools/call", {
                    "name": tool_name,
                    "arguments": arguments,
                })
                if "error" in response:
                    error = response["error"]
                    raise RuntimeError(f"MCP tool error: {error.get('message', error)}")
                result = response.get("result", {})
                content = result.get("content", [])
                if content and isinstance(content, list):
                    text_blocks = [
                        block.get("text", "")
                        for block in content
                        if block.get("type") == "text"
                    ]
                    combined = "\n".join(text_blocks)
                    try:
                        return json.loads(combined)
                    except json.JSONDecodeError:
                        return combined
                return result
            except (ConnectionError, RuntimeError) as e:
                last_error = e
                if attempt < retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Tool '%s' failed (attempt %d/%d): %s. Retrying in %ds...",
                        tool_name, attempt + 1, retries + 1, e, wait_time
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Tool '%s' failed after %d attempts: %s",
                                 tool_name, retries + 1, e)
        raise last_error

    # =========================================================================
    # Convenience wrappers — all using verified tool names from server log
    # =========================================================================

    def _parse(self, result) -> Any:
        """Parse string result to dict/list if possible."""
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
        return result

    def get_pull_request(self, pr_number: int) -> dict:
        """tool: pull_request_read, method: get"""
        result = self.call_tool("pull_request_read", {
            "owner": self.owner,
            "repo": self.repo,
            "pullNumber": pr_number,
            "method": "get",
        })
        return self._parse(result) or {}

    def list_pull_requests(self, state: str = "open") -> list:
        """tool: list_pull_requests ✅ verified in server log"""
        result = self.call_tool("list_pull_requests", {
            "owner": self.owner,
            "repo": self.repo,
            "state": state,
        })
        parsed = self._parse(result)
        return parsed if isinstance(parsed, list) else []

    def get_pull_request_diff(self, pr_number: int) -> list:
        """tool: pull_request_read, method: get_files"""
        result = self.call_tool("pull_request_read", {
            "owner": self.owner,
            "repo": self.repo,
            "pullNumber": pr_number,
            "method": "get_files",
        })
        parsed = self._parse(result)
        return parsed if isinstance(parsed, list) else []

    def get_pull_request_commits(self, pr_number: int) -> list:
        """tool: pull_request_read, method: get_commits"""
        result = self.call_tool("pull_request_read", {
            "owner": self.owner,
            "repo": self.repo,
            "pullNumber": pr_number,
            "method": "get_commits",
        })
        parsed = self._parse(result)
        return parsed if isinstance(parsed, list) else []

    def list_commits(self, branch: str = None, per_page: int = 10) -> list:
        """tool: list_commits ✅ verified in server log"""
        params = {
            "owner": self.owner,
            "repo": self.repo,
            "perPage": per_page,
        }
        if branch:
            params["sha"] = branch
        result = self.call_tool("list_commits", params)
        parsed = self._parse(result)
        return parsed if isinstance(parsed, list) else []

    def create_review(
        self,
        pr_number: int,
        body: str,
        event: str = "COMMENT",
        comments: list = None,
    ) -> dict:
        """
        tool: pull_request_review_write ✅ verified in server log
        method: create_review
        event: APPROVE | REQUEST_CHANGES | COMMENT
        """
        params = {
            "owner": self.owner,
            "repo": self.repo,
            "pullNumber": pr_number,
            "body": body,
            "event": event,
            "method": "create_review",
        }
        logger.info("Creating review on PR #%d via MCP: event=%s", pr_number, event)
        return self.call_tool("pull_request_review_write", params)

    def add_issue_comment(self, pr_number: int, body: str) -> dict:
        """tool: add_issue_comment ✅ verified in server log"""
        return self.call_tool("add_issue_comment", {
            "owner": self.owner,
            "repo": self.repo,
            "issue_number": pr_number,
            "body": body,
        })

    def merge_pull_request(
        self,
        pr_number: int,
        commit_title: str = None,
        merge_method: str = "merge",
    ) -> dict:
        """tool: merge_pull_request ✅ verified in server log"""
        params = {
            "owner": self.owner,
            "repo": self.repo,
            "pullNumber": pr_number,
            "merge_method": merge_method,
        }
        if commit_title:
            params["commit_title"] = commit_title
        logger.info("Merging PR #%d via %s", pr_number, merge_method)
        return self.call_tool("merge_pull_request", params)

    def close(self):
        if self._transport:
            self._transport.stop()
            self._initialized = False
            logger.info("MCP client connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
