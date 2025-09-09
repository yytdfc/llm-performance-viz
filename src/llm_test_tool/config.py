"""
Configuration module for LLM Test Tool.
"""

import argparse
from dataclasses import dataclass


@dataclass
class TestConfig:
    """Configuration for the LLM API test"""
    processes: int
    requests_per_process: int
    model_id: str
    input_tokens: int
    random_tokens: int
    output_tokens: int
    url: str
    output_file: str
    image_count: int = 0
    image_size: str = "512x512"


def parse_arguments() -> TestConfig:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Simple multiprocess LLM API testing tool")
    parser.add_argument("--processes", type=int, default=4, 
                        help="Number of parallel processes")
    parser.add_argument("--requests", type=int, default=5, 
                        help="Number of requests per process")
    parser.add_argument("--model_id", type=str, default="gpt-3.5-turbo", 
                        help="Model ID to test")
    parser.add_argument("--input_tokens", type=int, default=1000, 
                        help="Total approximate input token length")
    parser.add_argument("--random_tokens", type=int, default=500, 
                        help="Number of random tokens to add to the prompt")
    parser.add_argument("--output_tokens", type=int, default=100, 
                        help="Maximum output tokens to generate")
    parser.add_argument("--url", type=str, default="http://localhost:8080/v1/chat/completions", 
                        help="API endpoint URL")
    parser.add_argument("--output", type=str, default="test_results.json", 
                        help="Results output file")
    parser.add_argument("--image_count", type=int, default=0, 
                        help="Number of images to include in the prompt")
    parser.add_argument("--image_size", type=str, default="512x512", 
                        help="Size of images in the prompt (e.g., 128x128, 512x512)")
    
    args = parser.parse_args()
    
    return TestConfig(
        processes=args.processes,
        requests_per_process=args.requests,
        model_id=args.model_id,
        input_tokens=args.input_tokens,
        random_tokens=args.random_tokens,
        output_tokens=args.output_tokens,
        url=args.url,
        output_file=args.output,
        image_count=args.image_count,
        image_size=args.image_size
    )


def print_test_config(config: TestConfig) -> None:
    """Print the test configuration"""
    print(f"Starting LLM API test:")
    print(f"- Processes: {config.processes}")
    print(f"- Requests per process: {config.requests_per_process}")
    print(f"- Total requests: {config.processes * config.requests_per_process}")
    print(f"- Model ID: {config.model_id}")
    print(f"- Total input tokens: {config.input_tokens}")
    print(f"- Random tokens: {config.random_tokens}")
    print(f"- Output tokens: {config.output_tokens}")
    print(f"- API endpoint: {config.url}")
    if config.image_count > 0:
        print(f"- Image count: {config.image_count}")
        print(f"- Image size: {config.image_size}")
    print("-" * 50)