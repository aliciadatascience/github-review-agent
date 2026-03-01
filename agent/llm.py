
import logging
import requests
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.language_models import BaseLanguageModel
from config.settings import settings

logger = logging.getLogger(__name__)


def _test_hf_connection(model_id: str) -> None:
    """
    Make a lightweight test call to HuggingFace BEFORE creating the LangChain
    LLM object. This surfaces 401/403/503/404 errors immediately with a
    clear, actionable message instead of a cryptic LangChain traceback.
    """
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"}

    logger.info("Testing HuggingFace connection → %s", url)

    try:
        # Send a minimal payload just to check auth + model availability
        response = requests.post(
            url,
            headers=headers,
            json={"inputs": "test", "options": {"wait_for_model": False}},
            timeout=15,
        )
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "❌ Cannot reach HuggingFace API.\n"
            "   Check your internet connection."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            "❌ HuggingFace API timed out during connection test.\n"
            "   Their servers may be slow — try again in a moment."
        )

    # ----------------------------------------------------------------
    # Handle each HTTP error code with a clear, specific message
    # ----------------------------------------------------------------
    if response.status_code == 401:
        raise PermissionError(
            "❌ 401 Unauthorized — HuggingFace rejected your token.\n"
            "   Fix: Check HF_TOKEN in your .env file.\n"
            "   Get a valid token at: https://huggingface.co/settings/tokens\n"
            f"   Current token starts with: '{settings.HF_TOKEN[:6]}...'"
            if settings.HF_TOKEN else
            "❌ 401 Unauthorized — HF_TOKEN is empty in your .env file."
        )

    elif response.status_code == 403:
        raise PermissionError(
            f"❌ 403 Forbidden — Your token doesn't have access to model: {model_id}\n"
            "   Fix options:\n"
            "     1. If this is a gated model (e.g. Llama), accept the license at:\n"
            f"        https://huggingface.co/{model_id}\n"
            "     2. Switch to a freely accessible model like:\n"
            "        mistralai/Mistral-7B-Instruct-v0.3"
        )

    elif response.status_code == 404:
        raise ValueError(
            f"❌ 404 Not Found — Model does not exist: {model_id}\n"
            "   Fix: Check HF_API_URL in your .env for typos.\n"
            "   Example: https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3"
        )

    elif response.status_code == 503:
        # 503 = model is cold/loading — this is normal, not a real error
        logger.warning(
            "⚠️  503 — Model '%s' is currently loading (cold start).\n"
            "   This is normal. The agent will retry automatically.\n"
            "   First request may take 20-30 seconds.",
            model_id,
        )
        # Don't raise — 503 on a test ping is expected and recoverable

    elif response.status_code == 200:
        logger.info("✅ HuggingFace connection OK (model: %s)", model_id)

    else:
        # Catch-all for unexpected codes
        raise RuntimeError(
            f"❌ Unexpected response from HuggingFace: HTTP {response.status_code}\n"
            f"   Response body: {response.text[:300]}"
        )


def create_llm() -> BaseLanguageModel:
    """
    Create and return the LangChain LLM connected to HuggingFace.

    Runs a connection test first so any auth/model errors are shown
    immediately with clear messages, not as cryptic mid-run failures.
    """
    model_id = settings.HF_API_URL.split("/models/")[-1]
    logger.info("Initialising LLM: %s", model_id)

    # Test the connection before handing back an LLM object
    # This will raise with a clear message if anything is wrong
    _test_hf_connection(model_id)

    llm = HuggingFaceEndpoint(
        endpoint_url=f"https://router.huggingface.co/hf-inference/models/{model_id}",
        huggingfacehub_api_token=settings.HF_TOKEN,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        timeout=60,  # Longer timeout to handle cold-start 503 waits
    )

    logger.info("✅ LLM ready (model: %s)", model_id)
    return llm