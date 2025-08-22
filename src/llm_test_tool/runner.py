"""
Test runner module for LLM Test Tool.
"""

import multiprocessing as mp
import time
from typing import Dict, List, Any

from .config import TestConfig
from .client import LlmApiClient


class TestRunner:
    """Handles execution of the LLM API test"""
    
    @staticmethod
    def run(config: TestConfig) -> List[Dict[str, Any]]:
        """Run the test with the specified configuration"""
        total_requests = config.processes * config.requests_per_process
        
        # Create unique IDs for each request
        request_ids = range(total_requests)
        
        # Create argument list for each request
        request_configs = [(config.model_id, config.input_tokens, config.output_tokens, 
                           req_id, config.url, config.random_tokens) for req_id in request_ids]
        
        print(f"\nStarting {total_requests} requests...")
        print("Progress:")
        
        results = []
        completed = 0
        successful = 0
        failed = 0
        latencies = []
        tps_values = []
        start_time = time.time()
        
        # Execute requests in parallel using imap_unordered
        with mp.Pool(processes=config.processes) as pool:
            # Use imap_unordered to get results as they complete
            for result in pool.imap_unordered(LlmApiClient.send_request, request_configs):
                results.append(result)
                completed += 1
                
                # Update statistics
                if result.get('success', False):
                    successful += 1
                    if result.get('end_to_end_latency'):
                        latencies.append(result['end_to_end_latency'])
                        # Keep only recent values to avoid memory growth
                        if len(latencies) > 1000:
                            latencies = latencies[-500:]
                    
                    if result.get('completion_tokens', 0) > 0 and result.get('end_to_end_latency'):
                        tps = result['completion_tokens'] / result['end_to_end_latency']
                        tps_values.append(tps)
                        # Keep only recent values to avoid memory growth
                        if len(tps_values) > 1000:
                            tps_values = tps_values[-500:]
                else:
                    failed += 1
                
                # Calculate and display progress
                elapsed = time.time() - start_time
                avg_latency = sum(latencies) / len(latencies) if latencies else 0
                avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0
                rps = completed / elapsed if elapsed > 0 else 0
                progress_pct = (completed / total_requests) * 100
                
                # Create progress bar
                bar_width = 30
                filled = int(bar_width * progress_pct / 100)
                bar = '█' * filled + '░' * (bar_width - filled)
                
                # Format and display progress line
                progress_line = (
                    f"\r[{bar}] {completed}/{total_requests} "
                    f"({progress_pct:.1f}%) | "
                    f"✓ {successful} | "
                    f"✗ {failed} | "
                    f"Latency: {avg_latency:.3f}s | "
                    f"TPS: {avg_tps:.1f} | "
                    f"RPS: {rps:.1f}"
                )
                
                print(progress_line, end='', flush=True)
        
        print()  # New line after progress bar
        return results