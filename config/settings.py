"""
config/settings.py
Loads and validates all environment variables in one place.
All other modules import from here — never use os.environ directly elsewhere.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file automatically

class Settings:
    # GitHub
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_OWNER: str = os.getenv("GITHUB_OWNER", "")
    GITHUB_REPO: str = os.getenv("GITHUB_REPO", "")

    # HuggingFace LLM
    HF_API_URL: str = os.getenv("HF_API_URL", "HF_API_URL=https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Agent behavior
    POLL_INTERVAL_SECONDS: int = int(os.getenv("POLL_INTERVAL_SECONDS", "600"))
    AUTO_MERGE_ENABLED: bool = os.getenv("AUTO_MERGE_ENABLED", "false").lower() == "true"
    MIN_APPROVAL_SCORE: float = float(os.getenv("MIN_APPROVAL_SCORE", "0.85"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/agent.log")

    def validate(self):
        """Call this at startup to catch missing config early."""
        missing = []
        if not self.GITHUB_TOKEN:
            missing.append("GITHUB_TOKEN")
        if not self.GITHUB_OWNER:
            missing.append("GITHUB_OWNER")
        if not self.GITHUB_REPO:
            missing.append("GITHUB_REPO")
        if not self.HF_TOKEN:
            missing.append("HF_TOKEN")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Please copy .env.example to .env and fill in the values."
            )
        return True


settings = Settings()
