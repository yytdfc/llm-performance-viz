"""
Prompt generation module for LLM Test Tool.

This module provides functionality to generate prompts with both fixed and random components,
which can leverage prompt caching features in vLLM and similar systems.
"""

import random
import string
import uuid
import base64
from typing import Tuple, Dict, Union, List, Any
from PIL import Image, ImageDraw
import io

# Constant fixed part with single-digit numbers
# This will be the same for all requests to ensure consistent caching
FIXED_PROMPT = "5 3 5 1 1 7 7 7 1 2 9 4 3 3 8 6 6 4 9 2 9 7 2 9 1 2 9 2 6 5 8 8 3 3 1 3 7 6 9 5 1 2 8 0 4 9 5 0 9 8 6 0 8 8 9 7 5 5 5 8 1 2 9 0 5 6 8 9 4 2 2 2 1 0 7 0 7 4 5 4 0 5 4 7 7 2 8 4 2 4 9 2 9 0 1 7 5 8 2 1 "
FIXED_PROMPT_LENGTH = len(FIXED_PROMPT.split()) * 2
FINAL_PROMPT = "please repeat these number 10000 times"
FINAL_PROMPT_LENGTH = len(FINAL_PROMPT.split()) * 2

class PromptGenerator:
    """
    Handles generation of prompts with specified token lengths.
    
    The prompts are split into fixed and random parts to leverage prompt caching
    in systems like vLLM. The fixed part remains constant across requests,
    allowing for cache hits, while the random part ensures unique requests.
    """
    
    @staticmethod
    def generate_image(size: str) -> str:
        """
        Generate a simple test image with the specified dimensions.
        
        Args:
            size: Size of the image in the format "widthxheight"
        
        Returns:
            Base64-encoded image data with data URI prefix
        """
        # Parse dimensions from size string
        try:
            width, height = map(int, size.split('x'))
        except ValueError:
            # Default to 512x512 if parsing fails
            width, height = 512, 512
            
        # Create a simple test image with random colored rectangles
        image = Image.new('RGB', (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(image)
        
        # Draw a few random colored rectangles
        for i in range(5):
            x1 = random.randint(0, width - 1)
            y1 = random.randint(0, height - 1)
            x2 = random.randint(x1, width)
            y2 = random.randint(y1, height)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
        # Add random ID text to make the image unique
        unique_id = str(uuid.uuid4())[:8]
        draw.text((10, 10), f"Test Image {unique_id}", fill=(0, 0, 0))
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Return with data URI prefix
        return f"data:image/png;base64,{img_str}"
    
    @staticmethod
    def generate(total_length: int, fixed_length: int, image_count: int = 0, image_size: str = "512x512") -> Union[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Generate a prompt with specified fixed and random token lengths, optionally including images.
        
        Args:
            total_length: Total approximate token length
            fixed_length: Number of fixed tokens (for caching)
            image_count: Number of images to include (0 for text-only)
            image_size: Size of images in the format "widthxheight"
        
        Returns:
            Either a string prompt or a dictionary with multimodal content
        """
        if total_length <= 0 and image_count <= 0:
            return ""
            
        # Generate the text part of the prompt
        fixed_length = min(fixed_length, total_length)
        random_length = total_length - fixed_length
        
        fixed_part = "".join(fixed_length // FIXED_PROMPT_LENGTH * [FIXED_PROMPT]) + FIXED_PROMPT[:fixed_length // 2 % FIXED_PROMPT_LENGTH * 2]
        random_part = "".join([f"{random.randint(0, 9)} " for _ in range((random_length - FINAL_PROMPT_LENGTH) // 2)])
        text_prompt = fixed_part + random_part + FINAL_PROMPT
        
        # If no images requested, return text-only prompt
        if image_count <= 0:
            return text_prompt
            
        # Otherwise, create a multimodal prompt with images
        content_parts = []
        
        # Add text part
        if text_prompt:
            content_parts.append({"type": "text", "text": text_prompt})
            
        # Add image parts
        for _ in range(image_count):
            image_data = PromptGenerator.generate_image(image_size)
            content_parts.append({"type": "image_url", "image_url": {"url": image_data}})
            
        return {"content": content_parts}