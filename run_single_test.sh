#!/bin/bash

# Single test runner script
# Usage: ./run_single_test.sh <config_path> [--api-endpoint <url>] [additional_args]

set -e

# Function to run test for a specific config
run_test() {
    local config_path="$1"
    shift  # Remove first argument
    local additional_args="$@"  # Get all remaining arguments
    
    # Remove "model_configs/" prefix if present
    local clean_path="${config_path#model_configs/}"
    
    # Extract vllm version, instance type, and model name from clean path
    local vllm_version=$(echo "$clean_path" | cut -d'/' -f1)
    local instance_type=$(echo "$clean_path" | cut -d'/' -f2)
    local model_name=$(echo "$clean_path" | cut -d'/' -f3)
    
    # Strip .yaml suffix from model name
    model_name="${model_name%.yaml}"
    
    # Generate output directory in your format
    local output_dir="archive_results/${vllm_version}--${instance_type}--${model_name}"
    
    echo "=========================================="
    echo "Testing: $model_name"
    echo "Config: $config_path"
    echo "Output: $output_dir"
    echo "=========================================="
    
    # Run the test
    uv run run_auto_test.py --config "$config_path" --output-dir "$output_dir" $additional_args
    
    echo "✓ Completed: $model_name"
    echo ""
}

# Check if config path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_path> [--api-endpoint <url>] [additional_args]"
    echo "Examples:"
    echo "  $0 model_configs/sglang-v0.4.9.post4/g6e.4xlarge/Qwen3-30B-A3B-FP8.yaml"
    echo "  $0 model_configs/vllm-v0.9.2/g6e.4xlarge/config.yaml --api-endpoint http://localhost:8000/v1/chat/completions"
    echo "  $0 model_configs/vllm-v0.9.2/g6e.4xlarge/config.yaml --api-endpoint http://remote-server:8000/v1/chat/completions --skip-deployment"
    exit 1
fi

# Create output directory
mkdir -p archive_results

# Run the test
run_test "$@"