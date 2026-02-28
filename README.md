# GitHub AI Review Agent

An AI agent that automatically reviews GitHub Pull Requests using LangChain + HuggingFace LLM + GitHub MCP Server.

---

## How It Works

```
Your Terminal
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
├── .env.example              ← Copy to .env and fill in your tokens
├── .env                      ← Your secrets (NEVER commit this)
├── .gitignore
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
- **Docker** (recommended): https://docs.docker.com/get-docker/
- **Node.js** (alternative): https://nodejs.org/

### Step 2: Clone and install

```bash
git clone <your-agent-repo-url>
cd github-review-agent

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 3: Get your tokens

**GitHub Personal Access Token:**
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token"
3. Select scopes: `repo`, `pull_requests`, `write:discussion`
4. Copy the token (starts with `ghp_`)

**HuggingFace Token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permission
3. Copy the token (starts with `hf_`)

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

## ⚠️ Important: LLM Model Note

`distilgpt2` is a tiny text-completion model — it will not reason well for agent tasks. 

For much better results, change `HF_API_URL` in your `.env` to:

| Model | URL | Notes |
|-------|-----|-------|
| Mistral 7B Instruct | `.../mistralai/Mistral-7B-Instruct-v0.2` | Best free option |
| Zephyr 7B | `.../HuggingFaceH4/zephyr-7b-beta` | Also excellent |
| Llama 2 Chat | `.../meta-llama/Llama-2-7b-chat-hf` | Needs HF approval |

Just update the URL — no code changes needed.

---

## How the MCP Client Works

The `GitHubMCPClient` implements exactly 5 core functions:

| Function | Purpose |
|----------|---------|
| `connect()` | Launches GitHub MCP server (Docker/npx) and handshakes |
| `list_tools()` | Asks MCP server what tools exist — discovers dynamically |
| `call_tool(name, args)` | The real worker: sends JSON-RPC, handles retries, parses response |
| convenience wrappers | `get_pull_request()`, `create_review()`, etc. — readable aliases for `call_tool()` |
| `close()` | Stops the MCP server process and cleans up |

The key insight: `call_tool()` does almost all the heavy lifting (connection resilience, response parsing, retries). The actual MCP call is just one line:
```python
self._transport.send_request("tools/call", {"name": tool_name, "arguments": arguments})
```

---

## Troubleshooting

**"Neither Docker nor npx found"**  
→ Install Docker or Node.js (see Step 1)

**"Missing required environment variables"**  
→ Check your `.env` file has all four required values

**Agent gives nonsensical output**  
→ Switch to Mistral-7B-Instruct (see LLM Model Note above)

**GitHub API rate limit errors**  
→ Increase `POLL_INTERVAL_SECONDS` in `.env` (try 300 for 5 minutes)
