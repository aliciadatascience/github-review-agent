"""
agent/review_agent.py

The LangChain AI Agent that reviews GitHub PRs.

HOW IT WORKS:
  1. We give the LLM a list of Tools (GitHub actions it can take)
  2. We give it a system prompt explaining its job as a code reviewer
  3. LangChain's ReAct loop handles the reasoning:
       Thought: "I need to check the PR files"
       Action: get_pull_request_files
       Observation: [list of changed files]
       Thought: "The PR looks good, I'll approve it"
       Action: approve_pull_request
       Final Answer: "PR #42 approved"
  4. This repeats until the agent reaches a final answer

THE PROMPT IS THE BRAIN:
  distilgpt2 won't follow complex instructions well. With a better model
  (Mistral, Zephyr), the agent will reason much more reliably.
"""

import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


# The system prompt that defines how the agent thinks about PR reviews
REVIEW_AGENT_PROMPT = """You are an expert AI code reviewer for a GitHub repository.
Your job is to review pull requests (PRs), check commits, and decide whether to approve or request changes.

You have access to the following tools:
{tools}

Use this EXACT format for every response:

Question: the input question you must answer
Thought: think about what you need to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: your conclusion about the PR review

REVIEW GUIDELINES:
- Always start by listing open PRs to see what needs review
- For each PR, check: title/description, changed files, commit messages
- APPROVE if: changes are clear, commits are meaningful, no obvious bugs
- REQUEST CHANGES if: missing tests, unclear commits, risky changes, bad code
- COMMENT if: you have suggestions but the PR is generally acceptable
- Be specific in your feedback — explain WHY you made your decision
- Never approve a PR you haven't actually examined

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


def create_review_agent(llm: BaseLanguageModel, tools: list[Tool]) -> AgentExecutor:
    """
    Build and return the LangChain ReAct agent executor.
    
    ReAct = Reasoning + Acting
    The agent alternates between thinking (Thought) and doing (Action)
    until it reaches a conclusion.
    """
    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        template=REVIEW_AGENT_PROMPT,
    )

    # Create the ReAct agent (Reasoning + Acting loop)
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # AgentExecutor runs the ReAct loop and handles tool calls
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,           # Print Thought/Action/Observation to console
        max_iterations=20,      # Stop after 10 reasoning steps (prevents infinite loops)
        handle_parsing_errors=True,  # Don't crash on malformed LLM output
        return_intermediate_steps=True,
    )

    logger.info("Review agent created with %d tools", len(tools))
    return executor


def run_pr_review(executor: AgentExecutor, pr_number: int = None) -> dict:
    """
    Run the agent to review pull requests.
    
    If pr_number is given, review that specific PR.
    Otherwise, review all open PRs.
    """
    if pr_number:
        task = (
            f"Please review pull request #{pr_number}. "
            f"Check the PR details, files changed, and commits. "
            f"Then decide: should we APPROVE it, REQUEST CHANGES, or just leave a COMMENT? "
            f"Give specific reasons for your decision."
        )
    else:
        task = (
            "Please review all open pull requests in the repository. "
            "For each PR: check the details, files changed, and commits. "
            "Then decide for each: APPROVE, REQUEST CHANGES, or COMMENT. "
            "Give specific reasons for each decision."
        )

    logger.info("Starting PR review task: %s", task[:80])

    try:
        result = executor.invoke({"input": task})
        return {
            "success": True,
            "output": result.get("output", ""),
            "steps": len(result.get("intermediate_steps", [])),
        }
    except Exception as e:
        logger.error("Agent run failed: %s", e)
        return {
            "success": False,
            "output": f"Agent error: {e}",
            "steps": 0,
        }
