import argparse
import logging
import sys
import time
from datetime import datetime

from config.settings import settings
from mcp_client import GitHubMCPClient
from tools import create_github_tools
from agent import create_llm, create_review_agent, run_pr_review
from agent.event_handler import parse_event_context, build_agent_task, ReviewMode

import os
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# Main Agent Runner
# ============================================================

def run_once(pr_number: int = None):
    """
    Connect to GitHub MCP, run the review agent once, then disconnect.
    This is the core workflow.
    """
    logger.info("=" * 60)
    logger.info("GitHub AI Review Agent starting")
    logger.info("  Repo: %s/%s", settings.GITHUB_OWNER, settings.GITHUB_REPO)
    logger.info("  LLM:  %s", settings.HF_API_URL)
    logger.info("=" * 60)

    # Step 1: Connect to GitHub MCP Server
    client = GitHubMCPClient()
    try:
        client.connect()
    except Exception as e:
        logger.error("Failed to connect to GitHub MCP Server: %s", e)
        logger.error(
            "Make sure Docker is installed:\n"
            "  Docker: https://docs.docker.com/get-docker/"
        )
        return False

    try:
        # Step 2: Discover available tools (dynamic, not hardcoded)
        available_tools = client.list_tools()
        logger.info("Available GitHub MCP tools: %s", [t["name"] for t in available_tools])

        # Step 3: Create LangChain tool wrappers
        langchain_tools = create_github_tools(client)

        # Step 4: Set up the LLM
        llm = create_llm()

        # Step 5: Create the ReAct agent
        executor = create_review_agent(llm=llm, tools=langchain_tools)

        # Step 6: Run the review
        logger.info("Starting review...")
        result = run_pr_review(executor, pr_number=pr_number)

        if result["success"]:
            logger.info("✅ Review completed in %d steps", result["steps"])
            logger.info("Agent conclusion:\n%s", result["output"])
        else:
            logger.error("❌ Review failed: %s", result["output"])

        return result["success"]

    finally:
        # Always close the connection
        client.close()


def ci_mode():
    """
    CI mode: triggered by GitHub Actions.

    Reads the GitHub Actions event context from environment variables
    (injected by the workflow YAML), determines what happened, and runs
    the appropriate review task.

    This is what runs when the workflow calls: python main.py --ci
    """
    ctx = parse_event_context()

    if ctx.review_mode == ReviewMode.UNKNOWN:
        logger.warning(
            "Unknown event type: event=%s action=%s — nothing to do.",
            ctx.event_name, ctx.event_action,
        )
        return True

    # Build the natural-language task for the agent based on the event
    task = build_agent_task(ctx)

    logger.info("CI Mode | Event: %s/%s | PR: #%s | Mode: %s",
                ctx.event_name, ctx.event_action,
                ctx.pr_number or "N/A", ctx.review_mode.value)

    client = GitHubMCPClient()
    try:
        client.connect()
        client.list_tools()

        langchain_tools = create_github_tools(client)
        llm = create_llm()
        executor = create_review_agent(llm=llm, tools=langchain_tools)

        logger.info("Running task: %s", task[:120])
        result = executor.invoke({"input": task})

        output = result.get("output", "")
        steps = len(result.get("intermediate_steps", []))

        logger.info("✅ CI review complete in %d steps", steps)
        logger.info("Agent conclusion:\n%s", output)

        print("\n" + "="*60)
        print("AI REVIEW AGENT — RESULT")
        print("="*60)
        print(output)
        print("="*60 + "\n")

        return True

    except Exception as e:
        logger.error("CI mode failed: %s", e)
        return False

    finally:
        client.close()


def watch_mode():
    """
    Continuously poll for new PRs and review them automatically.
    Keeps track of which PRs have been reviewed to avoid duplicates.
    """
    logger.info("👀 Watch mode enabled — polling every %ds", settings.POLL_INTERVAL_SECONDS)
    logger.info("Press Ctrl+C to stop.")

    reviewed_prs: set[int] = set()

    while True:
        try:
            logger.info("[%s] Checking for new PRs...", datetime.now().strftime("%H:%M:%S"))

            # Quick check for new PRs without full agent run
            client = GitHubMCPClient()
            client.connect()

            try:
                open_prs = client.list_pull_requests(state="open")
                if not isinstance(open_prs, list):
                    open_prs = []

                new_prs = [
                    pr for pr in open_prs
                    if isinstance(pr, dict) and pr.get("number") not in reviewed_prs
                ]

                if new_prs:
                    logger.info("Found %d new PR(s) to review", len(new_prs))
                    for pr in new_prs:
                        pr_num = pr.get("number")
                        logger.info("Reviewing PR #%d: %s", pr_num, pr.get("title", ""))
                else:
                    logger.info("No new PRs. Next check in %ds.", settings.POLL_INTERVAL_SECONDS)

            finally:
                client.close()

            # Now do full reviews for new PRs
            for pr in new_prs:
                pr_num = pr.get("number")
                success = run_once(pr_number=pr_num)
                if success:
                    reviewed_prs.add(pr_num)

            time.sleep(settings.POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Watch mode stopped by user.")
            break
        except Exception as e:
            logger.error("Watch mode error: %s", e)
            logger.info("Retrying in %ds...", settings.POLL_INTERVAL_SECONDS)
            time.sleep(settings.POLL_INTERVAL_SECONDS)


def list_tools_mode():
    """Connect and print all available GitHub MCP tools."""
    client = GitHubMCPClient()
    client.connect()
    try:
        tools = client.list_tools()
        print(f"\n{'='*60}")
        print(f"GitHub MCP Server — Available Tools ({len(tools)} total)")
        print(f"{'='*60}")
        for tool in tools:
            print(f"\n🔧 {tool['name']}")
            print(f"   {tool.get('description', 'No description')}")
            schema = tool.get("inputSchema", {})
            props = schema.get("properties", {})
            if props:
                print(f"   Parameters: {', '.join(props.keys())}")
        print(f"\n{'='*60}\n")
    finally:
        client.close()


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GitHub AI Review Agent — automatically reviews PRs using LLM"
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: read GitHub Actions event context and review automatically"
    )
    parser.add_argument(
        "--pr", type=int, default=None,
        help="Review a specific PR by number (e.g. --pr 42)"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Run continuously, polling for new PRs"
    )
    parser.add_argument(
        "--list-tools", action="store_true",
        help="List all available GitHub MCP tools and exit"
    )

    args = parser.parse_args()

    # Validate config before doing anything
    try:
        settings.validate()
    except EnvironmentError as e:
        print(f"\n❌ Configuration error:\n{e}\n")
        sys.exit(1)

    if args.list_tools:
        list_tools_mode()
    elif args.ci:
        success = ci_mode()
        sys.exit(0 if success else 1)
    elif args.watch:
        watch_mode()
    else:
        success = run_once(pr_number=args.pr)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
