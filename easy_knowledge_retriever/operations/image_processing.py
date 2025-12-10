import base64
import os
from typing import Optional
from easy_knowledge_retriever.llm.service import BaseLLMService
from easy_knowledge_retriever.utils.logger import logger

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

from easy_knowledge_retriever.llm.prompts import PROMPTS

class ImageSummarizer:
    """
    Summarizes images using a Vision Language Model (VLM).
    """
    def __init__(self, llm_service: BaseLLMService):
        self.llm_service = llm_service

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
            
            # Call LLM service with messages
            # Note: hashing_kv is typically bound to the llm_service function in retriever.py factories.
            # If llm_service is the one from retriever, it's already a partial with hashing_kv.
            # So caching should work automatically if the underlying function supports it.
            # We explicitly pass messages=messages.
            
            summary = await self.llm_service(prompt="", messages=messages)
            return summary

        except Exception as e:
            logger.error(f"Failed to summarize image {image_path}: {e}")
            return f"[Error processing image: {str(e)}]"
