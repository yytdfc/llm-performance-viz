"""
Automated testing tool for vLLM deployments with comprehensive test matrix.
"""

import json
import yaml
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from .deployment import VllmDeployment
from .config import TestConfig
from .runner import TestRunner
from .analyzer import ResultAnalyzer


@dataclass
class TestCase:
    """Individual test case configuration"""
    input_tokens: int
    output_tokens: int
    processing_num: int
    random_tokens: int
    image_count: int = 0
    image_size: str = "512x512"
    
    def __str__(self):
        img_str = f"_img:{self.image_count}_{self.image_size}" if self.image_count > 0 else ""
        return f"in:{self.input_tokens}_out:{self.output_tokens}_proc:{self.processing_num}_rand:{self.random_tokens}{img_str}"


class AutoTestRunner:
    """Automated test runner for vLLM deployments"""
    
    def __init__(self, config_path: str, output_dir: str = None, api_endpoint: str = None):
        """Initialize with deployment configuration"""
        self.deployment = VllmDeployment(config_path)
        self.config_path = Path(config_path)
        self.custom_api_endpoint = api_endpoint
        
        # Load config file (support both YAML and JSON)
        with open(config_path, 'r') as f:
            if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                self.full_config = yaml.safe_load(f)
            else:
                self.full_config = json.load(f)
        
        self.test_matrix = self.full_config['test_matrix']
        self.test_config = self.full_config['test_config']
        
        # Generate output directory with timestamp and model ID if not specified
        if output_dir is None:
            output_dir = self._generate_output_dir()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate test cases from matrix
        self.test_cases = self._generate_test_cases()
    
    def _generate_output_dir(self) -> str:
        """Generate output directory name with current timestamp"""
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory name under test_results
        return f"test_results/{timestamp}"
        
    def _generate_test_cases(self) -> List[TestCase]:
        """Generate all test case combinations from the test matrix"""
        test_cases = []
        
        # Get image parameters if they exist in the test matrix
        image_counts = self.test_matrix.get('image_count', [0])
        image_sizes = self.test_matrix.get('image_size', ["512x512"])
        
        for input_tokens in self.test_matrix['input_tokens']:
            for output_tokens in self.test_matrix['output_tokens']:
                for processing_num in self.test_matrix['processing_num']:
                    for random_tokens in self.test_matrix['random_tokens']:
                        # Skip test cases where random_tokens > input_tokens
                        if random_tokens > input_tokens:
                            print(f"Skipping test case: input_tokens={input_tokens}, random_tokens={random_tokens} (random > input)")
                            continue
                        
                        # Add image parameters to test cases if they exist
                        if 'image_count' in self.test_matrix and 'image_size' in self.test_matrix:
                            for image_count in image_counts:
                                for image_size in image_sizes:
                                    test_cases.append(TestCase(
                                        input_tokens=input_tokens,
                                        output_tokens=output_tokens,
                                        processing_num=processing_num,
                                        random_tokens=random_tokens,
                                        image_count=image_count,
                                        image_size=image_size
                                    ))
                        else:
                            test_cases.append(TestCase(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                processing_num=processing_num,
                                random_tokens=random_tokens
                            ))
        return test_cases
    
    def _create_test_config(self, test_case: TestCase) -> TestConfig:
        """Create a TestConfig for a specific test case"""
        # Use custom API endpoint if provided, otherwise use deployment's API URL
        api_url = self.custom_api_endpoint if self.custom_api_endpoint else self.deployment.get_api_url()
        
        return TestConfig(
            processes=test_case.processing_num,
            requests_per_process=self.test_config['requests_per_process'],
            model_id=self.deployment.get_model_id(),
            input_tokens=test_case.input_tokens,
            random_tokens=test_case.random_tokens,
            output_tokens=test_case.output_tokens,
            url=api_url,
            output_file=str(self.output_dir / f"test_{test_case}.json"),
            image_count=test_case.image_count,
            image_size=test_case.image_size
        )
    
    def _run_warmup(self, test_case: TestCase) -> None:
        """Run warmup requests before the actual test"""
        print(f"Running {self.test_config['warmup_requests']} warmup requests...")
        
        # Use custom API endpoint if provided, otherwise use deployment's API URL
        api_url = self.custom_api_endpoint if self.custom_api_endpoint else self.deployment.get_api_url()
        
        warmup_config = TestConfig(
            processes=1,
            requests_per_process=self.test_config['warmup_requests'],
            model_id=self.deployment.get_model_id(),
            input_tokens=test_case.input_tokens,
            random_tokens=test_case.random_tokens,
            output_tokens=test_case.output_tokens,
            url=api_url,
            output_file=str(self.output_dir / f"warmup_{test_case}.json"),
            image_count=test_case.image_count,
            image_size=test_case.image_size
        )
        
        TestRunner.run(warmup_config)
        print("Warmup completed")
    
    def _load_existing_result(self, test_case: TestCase) -> Dict[str, Any]:
        """Load existing test result if it exists"""
        config = self._create_test_config(test_case)
        result_file = Path(config.output_file)
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    existing_result = json.load(f)
                print(f"Found existing result for test case: {test_case}")
                return existing_result
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load existing result for {test_case}: {e}")
                return None
        return None
    
    def run_single_test(self, test_case: TestCase, skip_existing: bool = True) -> Dict[str, Any]:
        """Run a single test case"""
        print(f"\n{'='*60}")
        print(f"Test case: {test_case}")
        print(f"Input tokens: {test_case.input_tokens}")
        print(f"Output tokens: {test_case.output_tokens}")
        print(f"Concurrent processes: {test_case.processing_num}")
        print(f"Random tokens: {test_case.random_tokens}")
        if test_case.image_count > 0:
            print(f"Image count: {test_case.image_count}")
            print(f"Image size: {test_case.image_size}")
        print(f"{'='*60}")
        
        # Check for existing results
        if skip_existing:
            existing_result = self._load_existing_result(test_case)
            if existing_result is not None:
                print("✓ Using existing test result (skipping execution)")
                return existing_result
        
        print("Running new test...")
        
        # Run warmup
        self._run_warmup(test_case)
        
        # Wait for cooldown
        if self.test_config['cooldown_seconds'] > 0:
            print(f"Cooling down for {self.test_config['cooldown_seconds']} seconds...")
            time.sleep(self.test_config['cooldown_seconds'])
        
        # Create test configuration
        config = self._create_test_config(test_case)
        
        # Run the actual test
        start_time = time.time()
        results = TestRunner.run(config)
        total_time = time.time() - start_time
        
        # Analyze results
        analysis = ResultAnalyzer.analyze(results, total_time, config)
        
        # Save results
        ResultAnalyzer.save_results(analysis, config.output_file)
        
        # Print summary
        ResultAnalyzer.print_summary(analysis)
        
        return analysis
    
    def run_all_tests(self, skip_existing: bool = True, skip_deployment: bool = False) -> Dict[str, Any]:
        """Run all test cases in the matrix"""
        print(f"Starting automated test suite with {len(self.test_cases)} test cases")
        print(f"Output directory: {self.output_dir}")
        print(f"Skip existing results: {'yes' if skip_existing else 'no'}")
        print(f"Skip deployment: {'yes' if skip_deployment else 'no'}")
        
        # Check for existing results
        remaining_tests = len(self.test_cases)
        if skip_existing:
            existing_count = 0
            for test_case in self.test_cases:
                if self._load_existing_result(test_case) is not None:
                    existing_count += 1
            
            remaining_tests = len(self.test_cases) - existing_count
            
            if existing_count > 0:
                print(f"Found {existing_count} existing test results that will be reused")
                print(f"Will run {remaining_tests} new tests")
        
        # Skip deployment if no new tests need to be run or if custom API endpoint is provided
        need_deployment = remaining_tests > 0 and not skip_deployment and not self.custom_api_endpoint
        
        if remaining_tests == 0:
            print("\n✓ All test results already exist - skipping server deployment")
            print("✓ No new tests to run - proceeding directly to results compilation")
            skip_deployment = True
        elif self.custom_api_endpoint:
            print(f"\nUsing custom API endpoint: {self.custom_api_endpoint}")
            print("Skipping deployment - using provided endpoint")
            skip_deployment = True
        elif not skip_deployment:
            print("\nDeploying vLLM server...")
            if not self.deployment.deploy():
                raise RuntimeError("Failed to deploy vLLM server")
        else:
            print("\nSkipping deployment - assuming server is already running")
        
        all_results = {}
        failed_tests = []
        skipped_tests = []
        
        try:
            for i, test_case in enumerate(self.test_cases, 1):
                print(f"\nProgress: {i}/{len(self.test_cases)}")
                
                try:
                    # Check if we should skip this test
                    if skip_existing:
                        existing_result = self._load_existing_result(test_case)
                        if existing_result is not None:
                            all_results[str(test_case)] = existing_result
                            skipped_tests.append(str(test_case))
                            continue
                    
                    result = self.run_single_test(test_case, skip_existing=False)  # Don't double-check
                    all_results[str(test_case)] = result
                except Exception as e:
                    print(f"Test case {test_case} failed: {e}")
                    failed_tests.append((str(test_case), str(e)))
                    continue
        
        finally:
            # Cleanup deployment if we deployed it
            if need_deployment:
                print("\nCleaning up deployment...")
                self.deployment.cleanup()
        
        # Save comprehensive results
        comprehensive_results = {
            "test_matrix": self.test_matrix,
            "test_config": self.test_config,
            "deployment_config": self.full_config['deployment'],
            "results": all_results,
            "failed_tests": failed_tests,
            "skipped_tests": skipped_tests,
            "summary": self._generate_summary(all_results)
        }
        
        summary_file = self.output_dir / "comprehensive_results.json"
        with open(summary_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"\nComprehensive results saved to: {summary_file}")
        self._print_final_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _generate_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all test cases"""
        if not all_results:
            return {}
        
        summary = {
            "total_tests": len(all_results),
            "avg_throughput": 0,
            "avg_first_token_latency": 0,
            "avg_end_to_end_latency": 0,
            "best_throughput": {"value": 0, "test_case": ""},
            "worst_throughput": {"value": float('inf'), "test_case": ""},
            "best_latency": {"value": float('inf'), "test_case": ""},
            "worst_latency": {"value": 0, "test_case": ""}
        }
        
        throughputs = []
        first_token_latencies = []
        end_to_end_latencies = []
        
        for test_case, result in all_results.items():
            if 'performance_metrics' in result:
                metrics = result['performance_metrics']
                
                throughput = metrics.get('throughput_tokens_per_second', 0)
                first_token_lat = metrics.get('avg_first_token_latency', 0)
                end_to_end_lat = metrics.get('avg_end_to_end_latency', 0)
                
                throughputs.append(throughput)
                first_token_latencies.append(first_token_lat)
                end_to_end_latencies.append(end_to_end_lat)
                
                # Track best/worst throughput
                if throughput > summary["best_throughput"]["value"]:
                    summary["best_throughput"] = {"value": throughput, "test_case": test_case}
                if throughput < summary["worst_throughput"]["value"]:
                    summary["worst_throughput"] = {"value": throughput, "test_case": test_case}
                
                # Track best/worst latency (lower is better)
                if end_to_end_lat < summary["best_latency"]["value"]:
                    summary["best_latency"] = {"value": end_to_end_lat, "test_case": test_case}
                if end_to_end_lat > summary["worst_latency"]["value"]:
                    summary["worst_latency"] = {"value": end_to_end_lat, "test_case": test_case}
        
        if throughputs:
            summary["avg_throughput"] = sum(throughputs) / len(throughputs)
        if first_token_latencies:
            summary["avg_first_token_latency"] = sum(first_token_latencies) / len(first_token_latencies)
        if end_to_end_latencies:
            summary["avg_end_to_end_latency"] = sum(end_to_end_latencies) / len(end_to_end_latencies)
        
        return summary
    
    def _print_final_summary(self, comprehensive_results: Dict[str, Any]) -> None:
        """Print final summary of all test results"""
        summary = comprehensive_results.get('summary', {})
        failed_tests = comprehensive_results.get('failed_tests', [])
        skipped_tests = comprehensive_results.get('skipped_tests', [])
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total test cases: {summary.get('total_tests', 0)}")
        print(f"Skipped tests (existing results): {len(skipped_tests)}")
        print(f"Failed tests: {len(failed_tests)}")
        print(f"Success rate: {((summary.get('total_tests', 0) - len(failed_tests)) / max(summary.get('total_tests', 1), 1) * 100):.1f}%")
        print()
        print(f"Average throughput: {summary.get('avg_throughput', 0):.2f} tokens/sec")
        print(f"Average first token latency: {summary.get('avg_first_token_latency', 0):.3f}s")
        print(f"Average end-to-end latency: {summary.get('avg_end_to_end_latency', 0):.3f}s")
        print()
        
        best_throughput = summary.get('best_throughput', {})
        if best_throughput.get('value', 0) > 0:
            print(f"Best throughput: {best_throughput['value']:.2f} tokens/sec ({best_throughput['test_case']})")
        
        best_latency = summary.get('best_latency', {})
        if best_latency.get('value', float('inf')) < float('inf'):
            print(f"Best latency: {best_latency['value']:.3f}s ({best_latency['test_case']})")
        
        if skipped_tests:
            print(f"\nSkipped tests (reused existing results):")
            for test_case in skipped_tests[:5]:  # Show first 5
                print(f"  - {test_case}")
            if len(skipped_tests) > 5:
                print(f"  ... and {len(skipped_tests) - 5} more")
        
        if failed_tests:
            print(f"\nFailed tests:")
            for test_case, error in failed_tests:
                print(f"  - {test_case}: {error}")
        
        print(f"{'='*80}")


def main():
    """Main entry point for automated testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated vLLM testing tool")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to deployment configuration file")
    parser.add_argument("--output-dir", type=str, default="test_results",
                        help="Output directory for test results")
    
    args = parser.parse_args()
    
    runner = AutoTestRunner(args.config, args.output_dir)
    runner.run_all_tests()


if __name__ == "__main__":
    main()