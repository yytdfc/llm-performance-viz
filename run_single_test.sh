#!/bin/bash

# Single test runner script
# Usage: ./run_single_test.sh <config_path> [--api-endpoint <url>] [--sysinfo] [additional_args]

set -e

# Function to get system information
get_sysinfo() {
    echo "=========================================="
    echo "System Information"
    echo "=========================================="
    
    # OS Version
    echo "--- OS Info ---"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            cat /etc/os-release | grep -E "^(NAME|VERSION|ID)="
        fi
        echo "Kernel: $(uname -r)"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS: $(sw_vers -productVersion)"
        echo "Kernel: $(uname -r)"
    fi
    echo ""
    
    # CUDA Version
    echo "--- CUDA Info ---"
    if command -v nvcc &> /dev/null; then
        echo "CUDA Version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)"
    elif [ -f /usr/local/cuda/version.txt ]; then
        cat /usr/local/cuda/version.txt
    elif [ -d /usr/local/cuda ]; then
        ls -la /usr/local/cuda 2>/dev/null | head -1
    else
        echo "CUDA: Not found"
    fi
    echo ""
    
    # NVIDIA Driver Version
    echo "--- NVIDIA Driver Info ---"
    if command -v nvidia-smi &> /dev/null; then
        echo "Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        echo ""
        echo "--- GPU Info ---"
        nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
    else
        echo "nvidia-smi: Not found"
    fi
    echo ""
    
    # Memory Info
    echo "--- Memory Info ---"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        free -h | head -2
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Total Memory: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')"
    fi
    echo ""
    
    # CPU Info
    echo "--- CPU Info ---"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        lscpu | grep -E "^(Model name|CPU\(s\)|Thread|Core)"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        sysctl -n machdep.cpu.brand_string
        echo "CPU Cores: $(sysctl -n hw.ncpu)"
    fi
    
    echo "=========================================="
}

# Function to get sysinfo as JSON (for saving to results)
get_sysinfo_json() {
    local os_name=""
    local os_version=""
    local kernel=""
    local cuda_version=""
    local driver_version=""
    local gpu_name=""
    local gpu_memory=""
    local cpu_model=""
    local cpu_cores=""
    local total_memory=""
    
    # OS Info
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        os_name=$(grep "^NAME=" /etc/os-release 2>/dev/null | cut -d'"' -f2 || echo "Linux")
        os_version=$(grep "^VERSION=" /etc/os-release 2>/dev/null | cut -d'"' -f2 || echo "")
        kernel=$(uname -r)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        os_name="macOS"
        os_version=$(sw_vers -productVersion 2>/dev/null || echo "")
        kernel=$(uname -r)
    fi
    
    # CUDA Version
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d',' -f1 || echo "")
    fi
    
    # NVIDIA Driver & GPU
    if command -v nvidia-smi &> /dev/null; then
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
        gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "")
    fi
    
    # CPU & Memory
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        cpu_model=$(lscpu 2>/dev/null | grep "Model name" | cut -d':' -f2 | xargs || echo "")
        cpu_cores=$(nproc 2>/dev/null || echo "")
        total_memory=$(free -g 2>/dev/null | awk '/^Mem:/{print $2 " GB"}' || echo "")
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        cpu_model=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
        cpu_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo "")
        total_memory=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB", $1/1024/1024/1024}' || echo "")
    fi
    
    # Output JSON
    cat <<EOF
{
  "benchmark_tool": "llm-performance-viz",
  "os": {
    "name": "$os_name",
    "version": "$os_version",
    "kernel": "$kernel"
  },
  "cuda_version": "$cuda_version",
  "nvidia_driver_version": "$driver_version",
  "gpu": {
    "name": "$gpu_name",
    "memory": "$gpu_memory"
  },
  "cpu": {
    "model": "$cpu_model",
    "cores": "$cpu_cores"
  },
  "total_memory": "$total_memory",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
}

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

# Check for --sysinfo flag first
if [ "$1" == "--sysinfo" ]; then
    get_sysinfo
    exit 0
fi

# Check for --sysinfo-json flag (outputs JSON format)
if [ "$1" == "--sysinfo-json" ]; then
    get_sysinfo_json
    exit 0
fi

# Check if config path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_path> [--api-endpoint <url>] [additional_args]"
    echo "       $0 --sysinfo          # Print system info and exit"
    echo "       $0 --sysinfo-json     # Print system info as JSON and exit"
    echo ""
    echo "Examples:"
    echo "  $0 --sysinfo"
    echo "  $0 model_configs/sglang-v0.4.9.post4/g6e.4xlarge/Qwen3-30B-A3B-FP8.yaml"
    echo "  $0 model_configs/vllm-v0.9.2/g6e.4xlarge/config.yaml --api-endpoint http://localhost:8000/v1/chat/completions"
    exit 1
fi

# Create output directory
mkdir -p archive_results

# Extract output dir from the config path for sysinfo saving
config_path="$1"
clean_path="${config_path#model_configs/}"
framework_version=$(echo "$clean_path" | cut -d'/' -f1)
instance_type=$(echo "$clean_path" | cut -d'/' -f2)
model_name=$(echo "$clean_path" | cut -d'/' -f3)
model_name="${model_name%.yaml}"
output_dir="archive_results/${framework_version}--${instance_type}--${model_name}"

# Save sysinfo first before running test
mkdir -p "$output_dir"
echo "Saving system info to $output_dir/sysinfo.json"
get_sysinfo_json > "$output_dir/sysinfo.json"

# Run the test
run_test "$@"