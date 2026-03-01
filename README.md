# GitHub AI Review Agent

An AI agent that automatically reviews GitHub Pull Requests using LangChain + HuggingFace LLM + GitHub MCP Server.

---

## How It Works

```
    └── main.py
         └── GitHubMCPClient          ← connects to GitHub MCP Server (Docker/npx)
              │    5 core functions:
              │      connect()         open the MCP connection
              │      list_tools()      discover GitHub tools dynamically
              │      call_tool()       the real worker (JSON-RPC to MCP)
              │      [wrappers]        get_pr(), create_review(), etc.
              │      close()           clean up
              │
              └── LangChain ReAct Agent
                   ├── HuggingFace LLM  (reads PRs, decides action)
                   └── Tools            (list PRs, get files, approve, etc.)
```

The agent follows a **Thought → Action → Observation** loop:
1. *Thought*: "I need to check the PR files"
2. *Action*: `get_pull_request_files` with input `"42"`
3. *Observation*: list of changed files returned from GitHub
4. *Thought*: "The changes look good, I'll approve"
5. *Action*: `approve_pull_request` with input `"42|LGTM!"`

---

## Project Structure

```
github-review-agent/          ← SEPARATE repo from the one you're reviewing
│
├── requirements.txt
├── main.py                   ← Entry point — run this
│
├── config/
│   └── settings.py           ← Loads all env vars in one place
│
├── mcp_client/
│   └── github_mcp_client.py  ← The 5-function MCP client
│
├── tools/
│   └── github_tools.py       ← LangChain Tool wrappers around MCP calls
│
├── agent/
│   ├── llm.py                ← HuggingFace LLM setup
│   └── review_agent.py       ← LangChain ReAct agent
│
└── logs/
    └── agent.log             ← Auto-created at runtime
```

---

## Setup

### Step 1: Prerequisites

Install **one** of these (for running the GitHub MCP Server):
- **Docker** : https://docs.docker.com/get-docker/


### Step 2: Clone and install

```bash
git clone <repo-url>
cd github-review-agent

python -m venv venv
source venv/bin/activate      

pip install -r requirements.txt
```

### Step 3: Get your tokens

GitHub Personal Access Token
HuggingFace Token


### Step 4: Configure .env

```bash
cp .env.example .env
```

Edit `.env`:
```
GITHUB_TOKEN=ghp_your_token_here
GITHUB_OWNER=your_github_username
GITHUB_REPO=the_repo_to_review

HF_API_URL=https://api-inference.huggingface.co/models/distilgpt2
HF_TOKEN=hf_your_token_here
```

### Step 5: Run!

```bash
# Review all open PRs once
python main.py

# Review a specific PR
python main.py --pr 42

# Watch mode: poll continuously for new PRs
python main.py --watch

# See all available GitHub MCP tools
python main.py --list-tools
```

---


## Troubleshooting

**"Docker not found"**  
→ Install Docker 

**"Missing required environment variables"**  
→ Check your `.env` file has all four required values

**Agent gives nonsensical output**  
→ Switch to Mistral-7B-Instruct (see LLM Model Note above)

**GitHub API rate limit errors**  
→ Increase `POLL_INTERVAL_SECONDS` in `.env` (try 300 for 5 minutes)
