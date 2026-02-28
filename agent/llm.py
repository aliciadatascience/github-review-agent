"""
agent/llm.py

Sets up the LLM connection to HuggingFace Inference API via LangChain.

IMPORTANT NOTE ON distilgpt2:
  distilgpt2 is a small text-completion model, NOT an instruction-following model.
  It will not reason well for agent tasks (it doesn't understand "choose a tool" prompts).
  
  For this agent to work well, swap to an instruction-tuned model:
    - mistralai/Mistral-7B-Instruct-v0.2     (best free option)
    - HuggingFaceH4/zephyr-7b-beta           (also great)
    - meta-llama/Llama-2-7b-chat-hf          (requires HF approval)
  
  Just update HF_API_URL in your .env file — no code changes needed.
"""

import logging
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.language_models import BaseLanguageModel
from config.settings import settings

logger = logging.getLogger(__name__)


def create_llm() -> BaseLanguageModel:
    """
    Create and return the LangChain LLM connected to HuggingFace.
    
    Uses HuggingFaceEndpoint which calls the Inference API — no local
    model download required, works with your HF token.
    """
    logger.info("Connecting to LLM: %s", settings.HF_API_URL)

    # Extract model ID from the URL for HuggingFaceEndpoint
    # URL format: https://api-inference.huggingface.co/models/{model_id}
    model_id = settings.HF_API_URL.split("/models/")[-1]

    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        huggingfacehub_api_token=settings.HF_TOKEN,
        
        # Generation parameters
        max_new_tokens=512,        # How long the response can be
        temperature=0.3,           # Lower = more focused/deterministic
        top_p=0.9,                 # Nucleus sampling
        repetition_penalty=1.1,   # Reduces repetitive output
        
        # Timeout for API calls (seconds)
        timeout=30,
    )

    logger.info("LLM ready (model: %s)", model_id)
    return llm
