import base64
import os
import json
import time
from typing import Optional
from easy_knowledge_retriever.llm.service import BaseLLMService
from easy_knowledge_retriever.utils.logger import logger
from easy_knowledge_retriever.kg.kv_storage.base import BaseKVStorage
from easy_knowledge_retriever.llm.utils import handle_cache, save_to_cache, CacheData
from easy_knowledge_retriever.utils.hashing import compute_args_hash
from easy_knowledge_retriever.utils.common_utils import statistic_data

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

from easy_knowledge_retriever.llm.prompts import PROMPTS

class ImageSummarizer:
    """
    Summarizes images using a Vision Language Model (VLM).
    """
    def __init__(self, llm_service: BaseLLMService, llm_response_cache: Optional[BaseKVStorage] = None):
        self.llm_service = llm_service
        self.llm_response_cache = llm_response_cache

    async def summarize(self, image_path: str, prompt: str = None) -> str:
        """
        Generate a summary of the image using the VLM.
        
        Args:
            image_path: Absolute path to the image file.
            prompt: Prompt to guide the summarization. If None, uses a default RAG-Anything inspired prompt.
            
        Returns:
            Textual summary of the image.
        """
        if prompt is None:
            # Use official RAG-Anything vision prompt
            # We construct the prompt by filling in available details.
            # Mineru may not provide captions/footnotes/entity_name separately found in 'images' list.
            # We use defaults for now.
            template = PROMPTS.get("vision_prompt", "")
            prompt = template.format(
                image_path=image_path,
                captions="None available",
                footnotes="None available",
                entity_name="Image" 
            )

        if not os.path.exists(image_path):
            logger.warning(f"Image not found at {image_path}, skipping summarization.")
            return ""

        try:
            base64_image = encode_image_to_base64(image_path)
            
            # Construct multimodal message for OpenAI-compatible API
            messages = [
                {
                    "role": "system",
                    "content": PROMPTS.get("IMAGE_ANALYSIS_SYSTEM", "You are an expert image analyst.")
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            # Cache check
            arg_hash = None
            dumped_messages = None
            
            if self.llm_response_cache:
                 # Create a unique representation for hashing
                 # We use the messages list which contains prompt and base64 image
                 dumped_messages = json.dumps(messages, sort_keys=True)
                 arg_hash = compute_args_hash(dumped_messages)
                 
                 # Try to get from cache
                 cached_result = await handle_cache(
                    self.llm_response_cache,
                    arg_hash,
                    dumped_messages,
                    "default",
                    cache_type="image_summary"
                 )
                 
                 if cached_result:
                     content, _ = cached_result
                     statistic_data["llm_cache"] += 1
                     logger.debug(f"Image summary found in cache for {image_path}")
                     return content

            statistic_data["llm_call"] += 1
            
            # Call LLM service with messages
            # Note: hashing_kv is typically bound to the llm_service function in retriever.py factories.
            # If llm_service is the one from retriever, it's already a partial with hashing_kv.
            # So caching should work automatically if the underlying function supports it.
            # We explicitly pass messages=messages.
            
            summary = await self.llm_service(prompt="", messages=messages)

            # Save to cache
            if self.llm_response_cache and summary and not summary.startswith("[Error") and arg_hash:
                 await save_to_cache(
                    self.llm_response_cache,
                    CacheData(
                        args_hash=arg_hash,
                        content=summary,
                        prompt=dumped_messages,
                        cache_type="image_summary",
                    )
                 )

            return summary

        except Exception as e:
            logger.error(f"Failed to summarize image {image_path}: {e}")
            return f"[Error processing image: {str(e)}]"
