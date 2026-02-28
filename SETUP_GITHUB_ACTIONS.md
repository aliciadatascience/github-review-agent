# GitHub Actions Setup Guide

This guide explains exactly how to connect your AI Review Agent to GitHub Actions
so it fires automatically on PR events.

---

## Architecture Overview

```
Developer opens PR
        │
        ▼
GitHub detects event
        │
        ▼
GitHub Actions runner starts (ubuntu-latest)
        │
        ├── Checks out TARGET repo (the code being reviewed)
        ├── Checks out AGENT repo (github-review-agent)
        ├── Pulls GitHub MCP Server Docker image
        │
        ▼
python main.py --ci
        │
        ├── event_handler.py reads env vars:
        │     GH_EVENT_NAME=pull_request
        │     GH_EVENT_ACTION=opened
        │     GH_PR_NUMBER=42
        │
        ├── GitHubMCPClient.connect()  → Docker container starts
        ├── GitHubMCPClient.list_tools() → discovers GitHub tools
        ├── LangChain ReAct agent runs
        │     Thought → Action → Observation loop
        │
        └── Agent posts review comment / approves / requests changes
                │
                ▼
          PR gets reviewed ✅
```

---

## Two Repos Involved

| Repo | Purpose | Contains |
|------|---------|----------|
| **Agent repo** (`github-review-agent`) | The AI agent code | This project |
| **Target repo** | The repo being reviewed | Your actual project |

The `.github/workflows/ai-review.yml` file goes into the **TARGET repo**.

---

## Step-by-Step Setup

### Step 1: Prepare your Agent Repo

Push `github-review-agent/` to GitHub as its own repo:

```bash
cd github-review-agent
git init
git add .
git commit -m "Initial AI review agent"
git remote add origin https://github.com/YOUR_USERNAME/github-review-agent.git
git push -u origin main
```

If it's a **private repo**, you'll need to create a PAT with repo read access
and add it as a secret `AGENT_REPO_TOKEN` in your target repo.

---

### Step 2: Add Secrets to Your TARGET Repo

Go to: **Target Repo → Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Value | Notes |
|-------------|-------|-------|
| `HF_TOKEN` | `hf_xxxxxxxxxxxx` | Your HuggingFace token |

> **Note:** `GITHUB_TOKEN` is provided automatically by GitHub Actions — you do NOT need to add it.

---

### Step 3: Add a Repository Variable

Go to: **Target Repo → Settings → Secrets and variables → Actions → Variables tab → New repository variable**

| Variable Name | Value | Example |
|---------------|-------|---------|
| `AGENT_REPO` | `your-username/github-review-agent` | `jsmith/github-review-agent` |

---

### Step 4: Copy the Workflow File

Copy `.github/workflows/ai-review.yml` from this agent repo into your **target repo**:

```bash
# In your target repo:
mkdir -p .github/workflows
cp /path/to/github-review-agent/.github/workflows/ai-review.yml .github/workflows/
git add .github/workflows/ai-review.yml
git commit -m "Add AI code review workflow"
git push
```

---

### Step 5: Grant Workflow Write Permissions

Go to: **Target Repo → Settings → Actions → General → Workflow permissions**

Select: ✅ **Read and write permissions**

This allows the agent to post review comments and approve PRs.

---

### Step 6: Test It

Open a Pull Request in your target repo. Within ~1 minute:

1. Go to **Actions** tab in your target repo
2. You'll see "AI Code Review Agent" workflow running
3. Click it to see the agent's Thought/Action/Observation loop in real time
4. The agent will post a review comment on your PR

---

## Event Routing

The agent reacts differently depending on what triggered it:

| GitHub Event | Action | What Agent Does |
|--------------|--------|-----------------|
| `pull_request` | `opened` | Full review: files + commits → approve/request changes |
| `pull_request` | `synchronize` | Re-review after new commits pushed |
| `pull_request` | `review_requested` | Thorough review (someone explicitly asked) |
| `push` | — | Commit message quality check |

---

## Environment Variables Flow

```
GitHub Actions workflow (ai-review.yml)
    sets env vars:
        GITHUB_TOKEN   ← auto-provided by GitHub, has PR write access
        GITHUB_OWNER   ← from github.repository_owner
        GITHUB_REPO    ← from github.event.repository.name
        HF_TOKEN       ← from secrets.HF_TOKEN
        HF_API_URL     ← hardcoded in workflow
        GH_EVENT_NAME  ← from github.event_name  (e.g. "pull_request")
        GH_EVENT_ACTION← from github.event.action (e.g. "opened")
        GH_PR_NUMBER   ← from github.event.pull_request.number
        GH_COMMIT_SHA  ← from github.sha

        │
        ▼

event_handler.py reads these and routes to:
    ReviewMode.FULL_PR_REVIEW        (PR opened)
    ReviewMode.INCREMENTAL_PR_REVIEW (PR synchronized)
    ReviewMode.THOROUGH_PR_REVIEW    (review requested)
    ReviewMode.COMMIT_CHECK          (push event)
```

---

## Customizing Which Events Trigger the Agent

Edit `.github/workflows/ai-review.yml` in your target repo:

```yaml
on:
  pull_request:
    types:
      - opened          # Remove any you don't want
      - synchronize
      - review_requested

  push:
    branches:
      - main            # Only watch these branches
      - develop
```

---

## Troubleshooting

**Workflow doesn't appear in Actions tab**
→ Make sure `ai-review.yml` is committed to the default branch of your target repo.

**"Resource not accessible by integration" error**
→ Go to Settings → Actions → General → set "Read and write permissions"

**Agent posts no comment**
→ Check the workflow logs (Actions tab → click the run → expand steps)
→ Look for the "AI REVIEW AGENT — RESULT" section at the bottom

**HuggingFace 503 error**
→ distilgpt2 may be cold. Switch to `mistralai/Mistral-7B-Instruct-v0.2` in the workflow YAML.

**Docker image pull fails**
→ `ghcr.io/github/github-mcp-server` requires Docker Hub login on some runners.
   Alternative: use npx by removing the Docker pull step and setting `USE_NPX=true`.
