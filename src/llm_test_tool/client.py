"""
API client module for LLM Test Tool.
"""

import json
import time
import requests
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from .prompt import PromptGenerator


@dataclass
class RequestResult:
    """Results from a single API request"""
    request_id: int
    success: bool
    first_token_latency: float = None
    end_to_end_latency: float = None
    response_length: int = 0
    error: str = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LlmApiClient:
    """Client for interacting with LLM APIs"""
    
    @staticmethod
    def send_request(config: Tuple[str, int, int, int, str, int, int, str]) -> Dict[str, Any]:
        """Send a request to the LLM API and record metrics"""
        model_id, input_tokens, output_tokens, request_id, url, random_tokens, image_count, image_size = config
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Generate prompt based on specified token length with fixed and random parts
        # Calculate fixed_length based on input_tokens and random_tokens
        fixed_length = max(0, input_tokens - random_tokens)
        user_prompt = PromptGenerator.generate(input_tokens, fixed_length, image_count, image_size)
        
        if image_count > 0:
            # Format for messages with image content
            messages = [
                {
                    "role": "user",
                    "content": user_prompt["content"]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
        
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": output_tokens,
            "stream": True,
            "ignore_eos": True,
            "stream_options": {"include_usage": True},
        }
        
        start_time = time.time()
        first_token_time = None
        response_text = []
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        # print(user_prompt)
        try:
            response = requests.post(url, json=payload, headers=headers, stream=True)
            
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    try:
                        out = json.loads(line[6:])
                        
                        
                            
                        # Collect response content
                        if "choices" in out and out["choices"] and "delta" in out["choices"][0]:
                            delta = out["choices"][0]["delta"]
                            if "reasoning_content" in delta and delta["reasoning_content"]:
                                response_text.append(delta["reasoning_content"])
                            if "content" in delta and delta["content"]:
                                response_text.append(delta["content"])
                            # Record time of first token
                            if first_token_time is None and response_text:
                                first_token_time = time.time()
                            # print(response_text[-1], end="", flush=True)
                                
                        # Record token usage if available
                        if "usage" in out:
                            usage = out["usage"]
                            # Update token usage with the latest information
                            if "prompt_tokens" in usage:
                                token_usage["prompt_tokens"] = usage["prompt_tokens"]
                            if "completion_tokens" in usage:
                                token_usage["completion_tokens"] = usage["completion_tokens"]
                            if "total_tokens" in usage:
                                token_usage["total_tokens"] = usage["total_tokens"]
                            
                            # If we're missing some values but have others, try to calculate them
                            if "prompt_tokens" not in usage and "completion_tokens" in usage and "total_tokens" in usage:
                                token_usage["prompt_tokens"] = usage["total_tokens"] - usage["completion_tokens"]
                            elif "completion_tokens" not in usage and "prompt_tokens" in usage and "total_tokens" in usage:
                                token_usage["completion_tokens"] = usage["total_tokens"] - usage["prompt_tokens"]
                            elif "total_tokens" not in usage and "prompt_tokens" in usage and "completion_tokens" in usage:
                                token_usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    
                    except Exception as e:
                        # Silently continue on parsing errors
                        continue
            # print()
        except Exception as e:
            print(f"Request error: {e}")
            return RequestResult(
                request_id=request_id,
                success=False,
                error=str(e)
            ).__dict__

        end_time = time.time()
        
        # Calculate latencies
        first_token_latency = (first_token_time - start_time) if first_token_time else None
        end_to_end_latency = end_time - start_time
        
        return RequestResult(
            request_id=request_id,
            success=True,
            first_token_latency=first_token_latency,
            end_to_end_latency=end_to_end_latency,
            response_length=len("".join(response_text)),
            prompt_tokens=token_usage["prompt_tokens"],
            completion_tokens=token_usage["completion_tokens"],
            total_tokens=token_usage["total_tokens"]
        ).__dict__