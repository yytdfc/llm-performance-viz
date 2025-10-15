"""
API client module for LLM Test Tool.
"""

import json
import time
import re
import requests
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from urllib.parse import urlparse

from .prompt import PromptGenerator

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


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
        
        # Check if this is a SageMaker endpoint URL
        parsed_url = urlparse(url)
        if parsed_url.scheme == 'sagemaker':
            return LlmApiClient._send_sagemaker_request(config)
        else:
            return LlmApiClient._send_http_request(config)
    
    @staticmethod
    def _send_http_request(config: Tuple[str, int, int, int, str, int, int, str]) -> Dict[str, Any]:
        """Send a request to HTTP-based LLM API"""
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
        
        try:
            response = requests.post(url, json=payload, headers=headers, stream=True)
            
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    try:
                        out = json.loads(line[6:])
                            
                        # Collect response content
                        if "choices" in out and out["choices"] and "delta" in out["choices"][0]:
                            delta = out["choices"][0]["delta"]
                            if "content" in delta and delta["content"]:
                                response_text.append(delta["content"])
                            elif "reasoning_content" in delta and delta["reasoning_content"]:
                                response_text.append(delta["reasoning_content"])
                            elif "reasoning" in delta and delta["reasoning"]:
                                response_text.append(delta["reasoning"])
                            # Record time of first token
                            if first_token_time is None and response_text:
                                first_token_time = time.time()
                                
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
    
    @staticmethod
    def _send_sagemaker_request(config: Tuple[str, int, int, int, str, int, int, str]) -> Dict[str, Any]:
        """Send a request to SageMaker endpoint"""
        model_id, input_tokens, output_tokens, request_id, url, random_tokens, image_count, image_size = config
        
        if not BOTO3_AVAILABLE:
            return RequestResult(
                request_id=request_id,
                success=False,
                error="boto3 is required for SageMaker endpoints. Please install it with: pip install boto3"
            ).__dict__
        
        # Parse SageMaker URL to extract region and endpoint name
        # Supports formats:
        # sagemaker://endpoint_name (uses default region)
        # sagemaker://region/endpoint_name (uses specified region)
        parsed_url = urlparse(url)
        region_name = None
        endpoint_name = None
        
        if parsed_url.netloc and not parsed_url.path.strip('/'):
            # Format: sagemaker://endpoint_name
            endpoint_name = parsed_url.netloc
        elif parsed_url.netloc and parsed_url.path.strip('/'):
            # Format: sagemaker://region/endpoint_name
            region_name = parsed_url.netloc
            endpoint_name = parsed_url.path.strip('/')
        
        if not endpoint_name:
            return RequestResult(
                request_id=request_id,
                success=False,
                error="Invalid SageMaker URL format. Use: sagemaker://endpoint_name or sagemaker://region/endpoint_name"
            ).__dict__
        
        # Generate prompt based on specified token length with fixed and random parts
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
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        
        try:
            # Create SageMaker runtime client with optional region
            if region_name:
                sagemaker_runtime = boto3.client('runtime.sagemaker', region_name=region_name)
            else:
                sagemaker_runtime = boto3.client('runtime.sagemaker')
            
            start_time = time.time()
            first_token_time = None
            response_text = []
            token_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            
            response = sagemaker_runtime.invoke_endpoint_with_response_stream(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            buffer = ""
            bytes_buffer = b""
            
            for event in response['Body']:
                try:
                    buffer += (bytes_buffer + event["PayloadPart"]["Bytes"]).decode()
                    bytes_buffer = b""
                except UnicodeDecodeError:
                    bytes_buffer += event["PayloadPart"]["Bytes"]
                    continue
                
                last_idx = 0
                for match in re.finditer(r'(^|\n)data:\s*(\{.+?\})\n', buffer):
                    try:
                        data = json.loads(match.group(2).strip())
                        last_idx = match.span()[1]
                        
                        # Handle token usage
                        if "usage" in data and data["usage"] is not None:
                            usage = data["usage"]
                            if "prompt_tokens" in usage:
                                token_usage["prompt_tokens"] = usage["prompt_tokens"]
                            if "completion_tokens" in usage:
                                token_usage["completion_tokens"] = usage["completion_tokens"]
                            if "total_tokens" in usage:
                                token_usage["total_tokens"] = usage["total_tokens"]
                        
                        # Handle content streaming
                        elif "choices" in data and data["choices"] and "delta" in data["choices"][0]:
                            delta = data["choices"][0]["delta"]
                            content = None
                            
                            if delta.get("reasoning"):
                                content = delta["reasoning"]
                            elif delta.get("reasoning_content"):
                                content = delta["reasoning_content"]
                            elif delta.get("content"):
                                content = delta["content"]
                            
                            if content:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                response_text.append(content)
                                
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # Silently continue on parsing errors
                        pass
                
                buffer = buffer[last_idx:]
            
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
            
        except Exception as e:
            print(f"SageMaker request error: {e}")
            return RequestResult(
                request_id=request_id,
                success=False,
                error=str(e)
            ).__dict__
