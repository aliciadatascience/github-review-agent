import json
import logging
import subprocess
import threading
import time
from typing import Any, Optional
import requests

from config.settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: MCP Transport Layer
# This handles the low-level stdio communication with the GitHub MCP server
# process. Most of the time you won't touch this — it just works.
# =============================================================================

class MCPStdioTransport:
    def __init__(self, command: list[str], env: dict = None):
        self.command = command
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()

    def start(self):
        """Launch the MCP server subprocess."""
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
        """
        Send a JSON-RPC 2.0 request to the MCP server and wait for response.
        This is the actual 'wire' that carries all tool calls.
        """
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

            # Read response line (MCP uses newline-delimited JSON)
            response_str = self.process.stdout.readline()
            logger.debug("← MCP Response: %s", response_str.strip())

            if not response_str:
                stderr = self.process.stderr.read()
                raise ConnectionError(f"MCP server closed connection. stderr: {stderr}")

            return json.loads(response_str)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from MCP server: {response_str}") from e

    def stop(self):
        """Terminate the MCP server subprocess."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("MCP server process stopped")


# =============================================================================
# SECTION 2: The Main MCP Client
# =============================================================================

class GitHubMCPClient:
    """
    High-level MCP client for the GitHub MCP Server.
    
    USAGE:
        client = GitHubMCPClient()
        client.connect()
        
        # Discover what tools are available
        tools = client.list_tools()
        
        # Call any tool
        result = client.call_tool("get_pull_request", {"owner": "...", "repo": "...", "pullNumber": 1})
        
        # Or use convenience wrappers
        pr = client.get_pull_request(pr_number=1)
        client.create_review(pr_number=1, body="LGTM!", event="APPROVE")
        
        client.close()
    """

    def __init__(self):
        self._transport: Optional[MCPStdioTransport] = None
        self._available_tools: list[dict] = []
        self._initialized = False

        # GitHub context (set from settings)
        self.owner = settings.GITHUB_OWNER
        self.repo = settings.GITHUB_REPO

    # =========================================================================
    # CORE FUNCTION 1: connect()
    # Opens the connection to the MCP server.
    # =========================================================================

    def connect(self):
        """
        Start the GitHub MCP server and initialize the MCP session.
        
        The GitHub MCP server can run via:
          - Docker (recommended for production)
          - npx (good for development, requires Node.js)
          - Direct binary
        
        We try Docker first, then fall back to npx.
        """
        server_command, server_env = self._build_server_command()

        self._transport = MCPStdioTransport(command=server_command, env=server_env)
        self._transport.start()

        # MCP requires an initialization handshake before anything else
        self._initialize_session()
        self._initialized = True
        logger.info("✅ Connected to GitHub MCP Server")

    def _build_server_command(self) -> tuple[list[str], dict]:
        """
        Build the command to launch the GitHub MCP server.
        Returns (command_list, env_vars).
        """
        env = {"GITHUB_TOKEN": settings.GITHUB_TOKEN}

        # Try Docker first (most reliable)
        if self._command_exists("docker"):
            command = [
                "docker", "run",
                "--rm",           # Remove container when done
                "-i",             # Interactive (required for stdio)
                "-e", "GITHUB_TOKEN",
                "ghcr.io/github/github-mcp-server",
            ]
            logger.info("Using Docker to run GitHub MCP Server")
            return command, env

        # Fall back to npx (requires Node.js installed)
        elif self._command_exists("npx"):
            command = [
                "npx", "-y",
                "@modelcontextprotocol/server-github",
            ]
            logger.info("Using npx to run GitHub MCP Server")
            return command, env

        else:
            raise RuntimeError(
                "Neither Docker nor npx found. Please install one:\n"
                "  - Docker: https://docs.docker.com/get-docker/\n"
                "  - Node.js (for npx): https://nodejs.org/"
            )

    def _command_exists(self, cmd: str) -> bool:
        import shutil
        return shutil.which(cmd) is not None

    def _initialize_session(self):
        """Send MCP initialization handshake."""
        # Step 1: initialize
        response = self._transport.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "github-review-agent", "version": "1.0.0"},
        })

        if "error" in response:
            raise ConnectionError(f"MCP initialization failed: {response['error']}")

        # Step 2: initialized notification (required by MCP spec)
        self._transport.send_request("notifications/initialized", {})
        logger.debug("MCP session initialized: %s", response.get("result", {}).get("serverInfo", {}))

    # =========================================================================
    # CORE FUNCTION 2: list_tools()
    # Dynamically discover what tools the MCP server exposes.
    # The agent never hardcodes tool names — it discovers them here.
    # =========================================================================

    def list_tools(self) -> list[dict]:
        """
        Ask the MCP server what tools it has available.
        
        Returns a list like:
        [
          {
            "name": "get_pull_request",
            "description": "Get details of a pull request",
            "inputSchema": { "type": "object", "properties": {...} }
          },
          ...
        ]
        
        This is how your agent stays flexible — if GitHub MCP adds a new tool
        tomorrow, your agent can discover and use it without code changes.
        """
        response = self._transport.send_request("tools/list", {})

        if "error" in response:
            raise RuntimeError(f"Failed to list tools: {response['error']}")

        self._available_tools = response.get("result", {}).get("tools", [])
        logger.info("Discovered %d tools from GitHub MCP Server", len(self._available_tools))

        for tool in self._available_tools:
            logger.debug("  Tool: %s — %s", tool["name"], tool.get("description", "")[:60])

        return self._available_tools

    def get_tool_names(self) -> list[str]:
        """Return just the names of available tools."""
        return [t["name"] for t in self._available_tools]

    # =========================================================================
    # CORE FUNCTION 3: call_tool()
    # The real worker. Everything else eventually calls this.
    # Handles connection resilience and response parsing.
    # =========================================================================

    def call_tool(self, tool_name: str, arguments: dict = None, retries: int = 2) -> Any:
        """
        Call any GitHub MCP tool by name with arguments.
        
        This is the central function — all the convenience wrappers below call
        this. The tool-specific part is tiny (just the name + args). Almost
        everything else here is:
          - Retry logic for transient failures
          - Response parsing and error extraction
          - Logging for debugging
        
        The actual MCP call is just one line:
            self._transport.send_request("tools/call", {...})
        
        Args:
            tool_name: e.g. "create_pull_request_review"
            arguments: dict of tool-specific params
            retries: how many times to retry on transient error
            
        Returns:
            Parsed result from the tool (string, dict, or list)
        """
        if not self._initialized:
            raise RuntimeError("Client not connected. Call connect() first.")

        arguments = arguments or {}
        last_error = None

        for attempt in range(retries + 1):
            try:
                logger.debug("Calling tool '%s' (attempt %d)", tool_name, attempt + 1)

                # ---- THE REAL PART ----
                # Everything above is setup; this one line is the actual MCP call.
                response = self._transport.send_request("tools/call", {
                    "name": tool_name,
                    "arguments": arguments,
                })
                # ---- END REAL PART ----

                # Parse the response
                if "error" in response:
                    error = response["error"]
                    raise RuntimeError(f"MCP tool error: {error.get('message', error)}")

                result = response.get("result", {})
                content = result.get("content", [])

                # MCP returns content as a list of blocks; extract text
                if content and isinstance(content, list):
                    text_blocks = [
                        block.get("text", "")
                        for block in content
                        if block.get("type") == "text"
                    ]
                    combined = "\n".join(text_blocks)

                    # Try to parse as JSON for structured data
                    try:
                        return json.loads(combined)
                    except json.JSONDecodeError:
                        return combined

                return result

            except (ConnectionError, RuntimeError) as e:
                last_error = e
                if attempt < retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    logger.warning(
                        "Tool '%s' failed (attempt %d/%d): %s. Retrying in %ds...",
                        tool_name, attempt + 1, retries + 1, e, wait_time
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Tool '%s' failed after %d attempts: %s", tool_name, retries + 1, e)

        raise last_error

    # =========================================================================
    # CORE FUNCTION 4: Convenience Wrappers
    # Clean, readable methods built on top of call_tool().
    # These make the agent code readable and less error-prone.
    # =========================================================================

    def get_pull_request(self, pr_number: int) -> dict:
        """Get full details of a pull request."""
        return self.call_tool("get_pull_request", {
            "owner": self.owner,
            "repo": self.repo,
            "pullNumber": pr_number,
        })

    def list_pull_requests(self, state: str = "open") -> list:
        """List pull requests. state = 'open', 'closed', or 'all'."""
        return self.call_tool("list_pull_requests", {
            "owner": self.owner,
            "repo": self.repo,
            "state": state,
        })

    def get_pull_request_diff(self, pr_number: int) -> str:
        """Get the diff/files changed in a PR."""
        result = self.call_tool("get_pull_request_files", {
            "owner": self.owner,
            "repo": self.repo,
            "pullNumber": pr_number,
        })
        return result

    def get_pull_request_commits(self, pr_number: int) -> list:
        """Get all commits in a PR."""
        return self.call_tool("get_pull_request_commits", {
            "owner": self.owner,
            "repo": self.repo,
            "pullNumber": pr_number,
        })

    def list_commits(self, branch: str = None, per_page: int = 10) -> list:
        """List recent commits on a branch (or default branch)."""
        params = {
            "owner": self.owner,
            "repo": self.repo,
            "perPage": per_page,
        }
        if branch:
            params["sha"] = branch
        return self.call_tool("list_commits", params)

    def create_review(
        self,
        pr_number: int,
        body: str,
        event: str = "COMMENT",  # "APPROVE", "REQUEST_CHANGES", or "COMMENT"
        comments: list[dict] = None,
    ) -> dict:
        """
        Leave a review on a PR.
        
        event options:
          "APPROVE"          - Approve the PR ✅
          "REQUEST_CHANGES"  - Block merge, request changes ❌  
          "COMMENT"          - Neutral comment 💬
          
        comments: list of inline comments, each like:
          {"path": "file.py", "line": 42, "body": "Fix this"}
        """
        params = {
            "owner": self.owner,
            "repo": self.repo,
            "pullNumber": pr_number,
            "body": body,
            "event": event,
        }
        if comments:
            params["comments"] = comments

        logger.info("Creating review on PR #%d: event=%s", pr_number, event)
        return self.call_tool("create_pull_request_review", params)

    def add_issue_comment(self, pr_number: int, body: str) -> dict:
        """Add a general comment (not a review) to a PR/issue."""
        return self.call_tool("add_issue_comment", {
            "owner": self.owner,
            "repo": self.repo,
            "issueNumber": pr_number,
            "body": body,
        })

    def merge_pull_request(
        self,
        pr_number: int,
        commit_title: str = None,
        merge_method: str = "merge",  # "merge", "squash", or "rebase"
    ) -> dict:
        """Merge a pull request (only called if AUTO_MERGE_ENABLED=true)."""
        params = {
            "owner": self.owner,
            "repo": self.repo,
            "pullNumber": pr_number,
            "mergeMethod": merge_method,
        }
        if commit_title:
            params["commitTitle"] = commit_title

        logger.info("Merging PR #%d via %s", pr_number, merge_method)
        return self.call_tool("merge_pull_request", params)

    def get_repository_info(self) -> dict:
        """Get basic repo info."""
        return self.call_tool("get_repository", {
            "owner": self.owner,
            "repo": self.repo,
        })

    # =========================================================================
    # CORE FUNCTION 5: close()
    # Clean up the connection and stop the MCP server process.
    # =========================================================================

    def close(self):
        """Close the MCP connection and clean up resources."""
        if self._transport:
            self._transport.stop()
            self._initialized = False
            logger.info("MCP client connection closed")

    # Context manager support — lets you use: with GitHubMCPClient() as client:
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
