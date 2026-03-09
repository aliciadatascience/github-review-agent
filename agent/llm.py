import os
import logging
import requests
import time
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# ── 1. Test connection before starting agent ─────────────────────────
def _test_groq_connection(api_key: str) -> None:
    """
    Fire a lightweight test request to Groq before starting the agent.
    Catches auth and connectivity errors early with clear messages.
    """
    logger.info("Testing Groq connection → llama-3.1-8b-instant")

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            },
            timeout=15,
        )

        if response.status_code == 200:
            logger.info("✅ Groq connection successful")
            return

        elif response.status_code == 401:
            raise ValueError(
                "❌ 401 Unauthorized — GROQ_API_KEY is invalid.\n"
                "Fix: Check your Groq API key at console.groq.com"
            )

        elif response.status_code == 403:
            raise ValueError(
                "❌ 403 Forbidden — API key does not have access.\n"
                "Fix: Check your Groq account plan at console.groq.com"
            )

        elif response.status_code == 429:
            raise ValueError(
                "❌ 429 Rate Limited — too many requests.\n"
                "Fix: Wait a moment and retry, or upgrade Groq plan."
            )

        elif response.status_code == 404:
            raise ValueError(
                "❌ 404 Not Found — model does not exist.\n"
                "Fix: Check model name in llm.py"
            )

        else:
            raise ValueError(
                f"❌ Unexpected status {response.status_code}: "
                f"{response.text[:200]}"
            )

    except requests.exceptions.Timeout:
        raise ValueError(
            "❌ Connection timeout — Groq API not reachable.\n"
            "Fix: Check your internet connection."
        )

    except requests.exceptions.ConnectionError:
        raise ValueError(
            "❌ Connection error — cannot reach Groq API.\n"
            "Fix: Check your internet connection."
        )


# ── 2. Create the LLM object ─────────────────────────────────────────
def get_llm() -> ChatGroq:
    """
    Create and return a LangChain-compatible Groq LLM object.
    Tests the connection first so errors surface clearly.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "❌ GROQ_API_KEY not set.\n"
            "Fix: Add GROQ_API_KEY to your .env file or GitHub secrets."
        )

    # Test connection before creating agent
    _test_groq_connection(api_key)

    logger.info("Creating Groq LLM — llama-3.1-8b-instant")

    return ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1000,
    )


# ── 3. Retry wrapper ─────────────────────────────────────────────────
def create_llm_with_retry(max_retries: int = 3) -> ChatGroq:
    """
    Create LLM with retry logic for transient failures.
    Unlike HuggingFace, Groq rarely needs this — but good to have.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Creating LLM attempt {attempt}/{max_retries}")
            return get_llm()

        except ValueError as e:
            # Auth/config errors — no point retrying
            raise

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = attempt * 2   # 2s, 4s, 6s
                logger.warning(
                    f"Attempt {attempt} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"All {max_retries} attempts failed.")

    raise last_error


# import logging
# import requests
# from langchain_community.llms import HuggingFaceEndpoint
# from langchain_core.language_models import BaseLanguageModel
# from langchain_groq import ChatGroq
# from config.settings import settings

# logger = logging.getLogger(__name__)


# def _test_hf_connection(model_id: str) -> None:
#     """
#     Make a lightweight test call to HuggingFace BEFORE creating the LangChain
#     LLM object. This surfaces 401/403/503/404 errors immediately with a
#     clear, actionable message instead of a cryptic LangChain traceback.
#     """
#     url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
#     headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"}

#     logger.info("Testing HuggingFace connection → %s", url)

#     try:
#         # Send a minimal payload just to check auth + model availability
#         response = requests.post(
#             url,
#             headers=headers,
#             json={"inputs": "test", "options": {"wait_for_model": False}},
#             timeout=15,
#         )
#     except requests.exceptions.ConnectionError:
#         raise ConnectionError(
#             "❌ Cannot reach HuggingFace API.\n"
#             "   Check your internet connection."
#         )
#     except requests.exceptions.Timeout:
#         raise TimeoutError(
#             "❌ HuggingFace API timed out during connection test.\n"
#             "   Their servers may be slow — try again in a moment."
#         )

#     # ----------------------------------------------------------------
#     # Handle each HTTP error code with a clear, specific message
#     # ----------------------------------------------------------------
#     if response.status_code == 401:
#         raise PermissionError(
#             "❌ 401 Unauthorized — HuggingFace rejected your token.\n"
#             "   Fix: Check HF_TOKEN in your .env file.\n"
#             "   Get a valid token at: https://huggingface.co/settings/tokens\n"
#             f"   Current token starts with: '{settings.HF_TOKEN[:6]}...'"
#             if settings.HF_TOKEN else
#             "❌ 401 Unauthorized — HF_TOKEN is empty in your .env file."
#         )

#     elif response.status_code == 403:
#         raise PermissionError(
#             f"❌ 403 Forbidden — Your token doesn't have access to model: {model_id}\n"
#             "   Fix options:\n"
#             "     1. If this is a gated model (e.g. Llama), accept the license at:\n"
#             f"        https://huggingface.co/{model_id}\n"
#             "     2. Switch to a freely accessible model like:\n"
#             "        mistralai/Mistral-7B-Instruct-v0.3"
#         )

#     elif response.status_code == 404:
#         raise ValueError(
#             f"❌ 404 Not Found — Model does not exist: {model_id}\n"
#             "   Fix: Check HF_API_URL in your .env for typos.\n"
#             "   Example: https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3"
#         )

#     elif response.status_code == 503:
#         # 503 = model is cold/loading — this is normal, not a real error
#         logger.warning(
#             "⚠️  503 — Model '%s' is currently loading (cold start).\n"
#             "   This is normal. The agent will retry automatically.\n"
#             "   First request may take 20-30 seconds.",
#             model_id,
#         )
#         # Don't raise — 503 on a test ping is expected and recoverable

#     elif response.status_code == 200:
#         logger.info("✅ HuggingFace connection OK (model: %s)", model_id)

#     else:
#         # Catch-all for unexpected codes
#         raise RuntimeError(
#             f"❌ Unexpected response from HuggingFace: HTTP {response.status_code}\n"
#             f"   Response body: {response.text[:300]}"
#         )


# def create_llm() -> BaseLanguageModel:
#     """
#     Create and return the LangChain LLM connected to HuggingFace.

#     Runs a connection test first so any auth/model errors are shown
#     immediately with clear messages, not as cryptic mid-run failures.
#     """
#     model_id = settings.HF_API_URL.split("/models/")[-1]
#     logger.info("Initialising LLM: %s", model_id)

#     # Test the connection before handing back an LLM object
#     # This will raise with a clear message if anything is wrong
#     _test_hf_connection(model_id)

#     llm = HuggingFaceEndpoint(
#         endpoint_url=f"https://router.huggingface.co/hf-inference/models/{model_id}",
#         huggingfacehub_api_token=settings.HF_TOKEN,
#         max_new_tokens=512,
#         temperature=0.3,
#         top_p=0.9,
#         repetition_penalty=1.1,
#         timeout=60,  # Longer timeout to handle cold-start 503 waits
#     )

#     logger.info("✅ LLM ready (model: %s)", model_id)
#     return llm
