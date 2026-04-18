"""
Vision Language Model Processing for Legal Documents
Generates text descriptions of images, charts, diagrams
"""
import logging
import base64
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI, OpenAI

from config import settings

logger = logging.getLogger(__name__)


class VisionProcessor:
    """
    Process images using Vision Language Models (VLM)
    Generates detailed text descriptions for non-text visual content
    """
    
    def __init__(self, model: str = None):
        self.model = model or settings.VISION_MODEL
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
    def describe_image(self, image_base64: str, context: str = "") -> str:
        """
        Generate detailed description of image using vision model
        
        Args:
            image_base64: Base64 encoded image
            context: Optional context about the document
            
        Returns:
            Text description of the image
        """
        try:
            prompt = self._build_prompt(context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024,
                temperature=0.3,  # Low temperature for factual descriptions
            )
            
            description = response.choices[0].message.content
            logger.info("Image description generated successfully")
            return description
            
        except Exception as e:
            logger.error(f"Error describing image: {e}")
            raise
    
    async def describe_image_async(self, image_base64: str, context: str = "") -> str:
        """Async version of describe_image"""
        try:
            prompt = self._build_prompt(context)
            
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in async image description: {e}")
            raise
    
    def describe_images_batch(self, images: List[Dict[str, Any]], context: str = "") -> List[Dict[str, Any]]:
        """
        Process multiple images in parallel
        """
        logger.info(f"Processing {len(images)} images in batch mode")
        
        updated_images = []
        
        for image_data in images:
            try:
                description = self.describe_image(image_data["image"], context)
                image_data["description"] = description
                updated_images.append(image_data)
                logger.info(f"Processed image from page {image_data['page']}")
                
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                image_data["description"] = f"[Failed to generate description: {str(e)}]"
                updated_images.append(image_data)
        
        return updated_images
    
    async def describe_images_batch_async(
        self, 
        images: List[Dict[str, Any]], 
        context: str = "",
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images asynchronously with concurrency limit
        """
        logger.info(f"Processing {len(images)} images asynchronously (max {max_concurrent} concurrent)")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(image_data):
            async with semaphore:
                try:
                    description = await self.describe_image_async(image_data["image"], context)
                    image_data["description"] = description
                    logger.info(f"Processed image from page {image_data['page']}")
                    return image_data
                except Exception as e:
                    logger.error(f"Failed to process image: {e}")
                    image_data["description"] = f"[Failed to generate description: {str(e)}]"
                    return image_data
        
        updated_images = await asyncio.gather(*[process_with_semaphore(img) for img in images])
        return updated_images
    
    def _build_prompt(self, context: str = "") -> str:
        """Build prompt for vision model"""
        prompt = """You are an expert legal document analyst. Analyze this image from a legal document and provide:

1. **Type of Content**: (e.g., chart, diagram, table, signature block, evidence photo, etc.)
2. **Key Information**: Extract and summarize all text and data visible in the image
3. **Visual Elements**: Describe charts, graphs, tables with numbers and trends
4. **Legal Relevance**: How this relates to legal proceedings or contractual terms
5. **Quality Notes**: Any legibility issues or important observations

Be concise but thorough. Focus on information that would be important to a legal professional."""
        
        if context:
            prompt += f"\n\nDocument Context: {context}"
        
        return prompt


class ImageAnalyzer:
    """Analyze extracted images for legal relevance"""
    
    def __init__(self):
        self.vision_processor = VisionProcessor()
    
    def needs_description(self, image_data: Dict[str, Any]) -> bool:
        """
        Determine if image needs vision model description
        (not pure text content that should be OCR'd)
        """
        # Always describe if has no text detected
        # Or if it's complex visual content
        return not image_data.get("has_text", False) or image_data.get("page", 0) % 5 == 0
    
    def enrich_images(
        self,
        images: List[Dict[str, Any]],
        document_context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Enrich images with descriptions and metadata
        """
        logger.info(f"Enriching {len(images)} images")
        
        images_to_process = [img for img in images if self.needs_description(img)]
        
        if images_to_process:
            images_to_process = self.vision_processor.describe_images_batch(
                images_to_process,
                document_context
            )
        
        # Merge back with original images
        processed_map = {img["page"]: img for img in images_to_process}
        return [processed_map.get(img["page"], img) for img in images]


def process_document_images(
    images: List[Dict[str, Any]],
    document_title: str = "",
    use_async: bool = False
) -> List[Dict[str, Any]]:
    """
    Main function to process all images in a document
    """
    if not images:
        logger.info("No images to process")
        return []
    
    processor = VisionProcessor()
    
    if use_async:
        # Run async processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                processor.describe_images_batch_async(images, document_title)
            )
        finally:
            loop.close()
    else:
        return processor.describe_images_batch(images, document_title)
