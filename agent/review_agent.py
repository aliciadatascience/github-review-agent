import logging
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


REVIEW_AGENT_PROMPT = """You are an AI code reviewer for GitHub pull requests.
Complete the task using the available tools. Follow the steps in exact order.

Available tools:
{tools}

Use this EXACT format — no deviations:

Question: the task you must complete
Thought: what step am I on and what do I do next
Action: exactly one tool from [{tool_names}]
Action Input: the input to the tool (PR number only, no quotes)
Observation: the result of the tool
Thought: what did I learn, what is next step
Action: next tool
Action Input: input
Observation: result
Thought: I have completed all steps, I know the final answer
Final Answer: summary of what you did and what decision was made on the PR

CRITICAL RULES:
- Never add quotes around PR numbers — use 9 not '9'
- Never repeat a step you already completed
- Never call list_open_pull_requests if you already know the PR number
- Move to Final Answer as soon as all steps are done
- If a tool returns an error, note it and move to the next step

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


def create_review_agent(
    llm: BaseLanguageModel,
    tools: list[Tool]
) -> AgentExecutor:

    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        template=REVIEW_AGENT_PROMPT,
    )

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=8,            # 4 steps + 4 buffer = enough
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    logger.info("Review agent created with %d tools", len(tools))
    return executor


def run_pr_review(
    executor: AgentExecutor,
    pr_number: int = None
) -> dict:

    if pr_number:
        task = f"""Review pull request #{pr_number}.
Step 1: get_pull_request_details — input: {pr_number}
Step 2: get_pull_request_files — input: {pr_number}
Step 3: get_pull_request_commits — input: {pr_number}
Step 4: approve_pull_request OR request_changes_on_pr — input: {pr_number}|your reason
Stop after step 4."""
    else:
        task = (
            "List open pull requests then review the most recent one. "
            "Check details, files, commits, then approve or request changes."
        )

    logger.info("Starting PR review: %s", task[:80])

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
