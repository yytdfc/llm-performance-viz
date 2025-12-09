# LLM Test Tool

A comprehensive tool for testing LLM API performance with automated deployment and extensive test matrices.

[中文版本 / Chinese Version](README_CN.md)

![Web Interface Screenshot](./assets/web_screenshot.webp)



## Quick Start Guide

### 0. Installation

```bash
git clone https://github.com/yytdfc/llm-performance-viz.git
cd llm-performance-viz/

uv sync  # optional
```

### 1. Prerequisites (Optional)

This step is optional but recommended for better deployment preparation. Ensure your model weights are available via Hugging Face Hub or pre-downloaded locally. The tool supports models like `Qwen/Qwen3-235B-A22B-FP8`, `deepseek-ai/DeepSeek-R1-0528`, etc.

#### Pre-download Model Weights
```bash
# Using Hugging Face CLI
uv run hf download Qwen/Qwen3-235B-A22B-FP8 --local-dir /opt/dlami/nvme/Qwen/Qwen3-235B-A22B-FP8
```

#### Pre-pull Docker Images
```bash
# vLLM images
docker pull vllm/vllm-openai:v0.9.2

# SGLang images
docker pull lmsysorg/sglang:v0.4.9.post4-cu126
```

The specific model paths and Docker images should be configured in your model configuration files (see `model_configs/` directory).

### 2. Create Configuration Files

Configuration files are stored in `model_configs/` organized by framework version and instance type:

```
model_configs/
└── [runtime_framework]/      # Runtime framework version (e.g., vllm-v0.9.2, sglang-v0.4.9.post4)
    └── [instance_type]/      # AWS instance type (e.g., p5.48xlarge, g6e.4xlarge)
        └── [model_config]/   # Model configuration YAML files
```

#### Configuration File Structure

The configuration file contains three main sections:

##### 1. Deployment Section

Deployment parameters map directly to Docker commands:

- `docker_image`: Docker image to use
- `container_name`: Name for the container
- `port`: Port to expose
- `command`: (Optional) Custom startup command, mainly used for SGLang
- `docker_params`: Maps to `docker run` parameters (gpus, shm-size, ipc, network, volume, environment)
- `app_args`: Maps to application startup parameters

##### 2. Test Matrix Section

```yaml
test_matrix:
  input_tokens: [1600, 6400, 12800]     # Input token lengths
  output_tokens: [100, 400, 1000]       # Output token lengths
  processing_num: [1, 16, 32, 64, 128]  # Concurrent request counts
  random_tokens: [100, 1600, 6400]      # Random token counts (for cache testing)
  image_count: [0, 1, 4]                # Number of images per prompt (optional)
  image_size: ["128x128", "512x512"]    # Image sizes to test (optional)
```

##### 3. Test Configuration Section

```yaml
test_config:
  requests_per_process: 5    # Number of requests each process sends
  warmup_requests: 1         # Number of warmup requests
  cooldown_seconds: 5        # Wait time between tests
```

#### Configuration Examples

**vLLM Configuration Example:**

```yaml
deployment:
  docker_image: "vllm/vllm-openai:v0.9.2"
  container_name: "vllm"
  port: 8080
  # Universal Docker parameters
  docker_params:
    gpus: "all"
    shm-size: "1000g"
    ipc: "host"
    network: "host"
    volume:
      - "/opt/dlami/nvme/:/vllm-workspace/"
  # Universal application arguments
  app_args:
    model: "deepseek-ai/DeepSeek-R1-0528"
    trust-remote-code: true
    max-model-len: 32768
    gpu-memory-utilization: 0.90
    tensor-parallel-size: 8
    enable-reasoning: true
    reasoning-parser: "deepseek_r1"
    tool-call-parser: "deepseek_v3" 
    enable-auto-tool-choice: true

test_matrix:
  input_tokens: [1600, 6400, 12800]
  output_tokens: [100, 400, 1000]
  processing_num: [1, 16, 32, 64, 128]
  random_tokens: [100, 1600, 6400]
  image_count: [0, 1, 4]       # Testing with 0, 1, or 4 images (multimodal models only)
  image_size: ["512x512"]     # Using 512x512 resolution images

test_config:
  requests_per_process: 5
  warmup_requests: 1
  cooldown_seconds: 5
```

**SGLang Configuration Example:**

```yaml
deployment:
  docker_image: "lmsysorg/sglang:v0.4.9.post4-cu126"
  container_name: "sglang"
  port: 8080
  
  # Universal Docker parameters
  docker_params:
    gpus: "all"
    shm-size: "1000g"
    ipc: "host"
    network: "host"
    volume:
      - "/opt/dlami/nvme/:/sgl-workspace/sglang/model"
  command: "python3 -m sglang.launch_server"
  # Universal application arguments
  app_args:
    host: "0.0.0.0"
    model-path: "model/deepseek-ai/DeepSeek-R1-0528"
    trust-remote-code: true
    tp-size: 8
    mem-fraction-static: 0.90
    tool-call-parser: "deepseekv3"
    reasoning-parser: "deepseek-r1"

test_matrix:
  input_tokens: [1600, 6400, 12800]
  output_tokens: [100, 400, 1000]
  processing_num: [1, 16, 32, 64, 128]
  random_tokens: [100, 1600, 6400]
  image_count: [0, 1, 4]       # Testing with 0, 1, or 4 images (multimodal models only)
  image_size: ["512x512"]     # Using 512x512 resolution images

test_config:
  requests_per_process: 5
  warmup_requests: 1
  cooldown_seconds: 5
```


### 3. Run Tests

#### Deploy Only (Without Testing)
```bash
uv run deploy_server.py --config your_config.yaml
```

You can add the `--show-command` parameter to dry-run and show the Docker deployment command to verify it's correct.

#### Single Test

Run a single test configuration with automatic deployment and result archiving:

```bash
# Run test with deployment
./run_single_test.sh model_configs/vllm-v0.9.2/g6e.48xlarge/Qwen3-235B-A22B-FP8-tp8ep.yaml

# Test with existing server (skip deployment)
./run_single_test.sh model_configs/vllm-v0.9.2/g6e.48xlarge/Qwen3-235B-A22B-FP8-tp8ep.yaml --skip-deployment

# Test with custom API endpoint
./run_single_test.sh model_configs/vllm-v0.9.2/g6e.4xlarge/config.yaml --api-endpoint http://localhost:8000/v1/chat/completions

# Print system info
./run_single_test.sh --sysinfo
```

Results are saved to `archive_results/` in a structured format for web visualization.

#### Results Directory Structure

Test results are organized in `archive_results/` with the following structure:

```
archive_results/
└── {framework_version}--{instance_type}--{model_name}/
    ├── sysinfo.json                                    # System information
    ├── comprehensive_results.json                      # Aggregated results
    └── test_in:{input}_out:{output}_proc:{proc}_rand:{rand}.json  # Individual test results
```

Example:
```
archive_results/
├── vllm-v0.9.2--g6e.4xlarge--Qwen3-30B-A3B-FP8/
│   ├── sysinfo.json
│   ├── comprehensive_results.json
│   ├── test_in:1600_out:400_proc:1_rand:100.json
│   ├── test_in:1600_out:400_proc:16_rand:100.json
│   └── ...
└── sglang-v0.4.9.post4--p5en.48xlarge--DeepSeek-R1-0528/
    └── ...
```

**sysinfo.json** - System hardware information:
```json
{
  "os": { "name": "Ubuntu", "version": "22.04", "kernel": "5.15.0" },
  "cuda_version": "12.4",
  "nvidia_driver_version": "550.54.15",
  "gpu": { "name": "NVIDIA H100 80GB HBM3", "memory": "81559 MiB" },
  "cpu": { "model": "Intel Xeon Platinum 8488C", "cores": "192" },
  "total_memory": "2048 GB",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**test_*.json** - Individual test result:
```json
{
  "metadata": {
    "processes": 16,
    "requests_per_process": 5,
    "total_requests": 80,
    "input_tokens": 1600,
    "random_tokens": 100,
    "output_tokens": 400,
    "total_test_duration": 45.23,
    "requests_per_second": 1.77
  },
  "statistics": {
    "successful_requests": 80,
    "success_rate": 1.0,
    "first_token_latency": { "min": 0.12, "max": 0.45, "mean": 0.25, "p50": 0.23, "p90": 0.38 },
    "end_to_end_latency": { "min": 3.2, "max": 5.1, "mean": 4.0, "p50": 3.9, "p90": 4.8 },
    "output_tokens_per_second": { "min": 78.5, "max": 125.3, "mean": 100.2, "p50": 102.1, "p90": 118.5 }
  }
}
```

#### Batch Testing

This executes all test configurations found in `./run_model_tests.sh`. Results are automatically organized by framework version, instance type, and model configuration, maintaining a consistent directory structure that enables the web server to provide comprehensive performance comparisons and visualizations across different setups.

```bash
# Run all configured tests
./run_model_tests.sh
```

### 4. Visualize Results

```bash
# Start visualization server
uv run start_viz_server.py

# Access web interface at: http://localhost:8000
```


## Manual Testing Usage

```bash
uv run llm-test --processes 4 --requests 10 --model_id "Qwen/Qwen3-30B-A3B-FP8" --input_tokens 1000 --random_tokens 500 --output_tokens 100 --url "http://localhost:8080/v1/chat/completions"
```

### Parameters

- `--processes`: Number of parallel processes (default: 4)
- `--requests`: Number of requests per process (default: 5)
- `--model_id`: Model ID to test (default: "gpt-3.5-turbo")
- `--input_tokens`: Total approximate input token length (default: 1000)
- `--random_tokens`: Number of random tokens to add to the prompt (default: 500)
- `--output_tokens`: Maximum output tokens to generate (default: 100)
- `--url`: API endpoint URL (default: "http://localhost:8080/v1/chat/completions")
- `--output`: Results output file (default: "test_results.json")
- `--image_count`: Number of images to include in the prompt (default: 0)
- `--image_size`: Size of images in the format "widthxheight" (default: "512x512")

## Example Output

```
Starting LLM API test:
- Processes: 4
- Requests per process: 10
- Total requests: 40
- Model ID: Qwen/Qwen3-30B-A3B-FP8
- Total input tokens: 1000
- Random tokens: 500
- Output tokens: 100
- API endpoint: http://localhost:8080/v1/chat/completions
--------------------------------------------------

Test completed!
Total duration: 12.45 seconds
Success rate: 100.00%
Throughput: 3.21 requests/second

First Token Latency (seconds):
- Min: 0.4521
- Max: 0.8976
- Mean: 0.6234

Percentiles:
- p25: 0.5123
- p50: 0.5987
- p75: 0.7234
- p90: 0.8456

End-to-End Latency (seconds):
- Min: 1.2345
- Max: 2.3456
- Mean: 1.7890

Percentiles:
- p25: 1.4567
- p50: 1.6789
- p75: 2.0123
- p90: 2.2345

Token Usage Statistics:

Prompt Tokens:
- Min: 998
- Max: 1002
- Mean: 1000.15

Percentiles:
- p25: 999
- p50: 1000
- p75: 1001
- p90: 1002

Completion Tokens:
- Min: 95
- Max: 105
- Mean: 99.8

Percentiles:
- p25: 97
- p50: 100
- p75: 102
- p90: 104

Output Tokens Per Second:
- Min: 45.23
- Max: 78.91
- Mean: 62.45

Percentiles:
- p25: 55.67
- p50: 61.23
- p75: 68.89
- p90: 74.56

Detailed results saved to: test_results.json
```


## Acknowledgements

This project is mainly coded by [Kiro](https://kiro.dev/).


## License

This project is licensed under the [MIT-0 License](./MIT-0).
