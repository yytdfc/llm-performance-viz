#!/usr/bin/env python3
"""
Convenience script to run automated vLLM testing.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_test_tool.auto_test import AutoTestRunner


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Automated vLLM testing tool with Docker deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default output directory
  python run_auto_test.py --config model_configs/vllm-v0.9.2/g6e.4xlarge/config.yaml
  
  # Run with custom output directory
  python run_auto_test.py --config config.yaml --output-dir my_results
  
  # Run with verbose output
  python run_auto_test.py --config config.yaml --verbose
  
  # Skip deployment (use existing running server)
  python run_auto_test.py --config config.yaml --skip-deployment
  
  # Force rerun all tests (ignore existing results)
  python run_auto_test.py --config config.yaml --force-rerun
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML or JSON configuration file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for test results (default: auto-generated with timestamp under test_results/)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--skip-deployment",
        action="store_true",
        help="Skip Docker deployment (assume server is already running)"
    )
    
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun all tests, ignoring existing results"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be tested without actually running tests"
    )
    
    parser.add_argument(
        "--api-endpoint",
        type=str,
        default=None,
        help="Custom API endpoint URL (e.g., http://localhost:8000/v1/chat/completions). If not specified, uses deployment config."
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Print configuration
    print(f"Starting automated vLLM testing")
    print(f"Config file: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Verbose mode: {'enabled' if args.verbose else 'disabled'}")
    print(f"Skip deployment: {'yes' if args.skip_deployment else 'no'}")
    print(f"Force rerun: {'yes' if args.force_rerun else 'no'}")
    print(f"Dry run: {'yes' if args.dry_run else 'no'}")
    if args.api_endpoint:
        print(f"Custom API endpoint: {args.api_endpoint}")
    print("-" * 60)
    
    try:
        runner = AutoTestRunner(args.config, args.output_dir, api_endpoint=args.api_endpoint)
        
        # Show the actual output directory being used
        if args.output_dir is None:
            print(f"Auto-generated output directory: {runner.output_dir}")
        else:
            print(f"Using specified output directory: {runner.output_dir}")
        
        if args.dry_run:
            print("DRY RUN MODE - Showing test cases that would be executed:")
            print(f"Total test cases: {len(runner.test_cases)}")
            
            # Check for existing results if not forcing rerun
            remaining_tests = len(runner.test_cases)
            if not args.force_rerun:
                existing_count = 0
                for test_case in runner.test_cases:
                    if runner._load_existing_result(test_case) is not None:
                        existing_count += 1
                
                remaining_tests = len(runner.test_cases) - existing_count
                
                if existing_count > 0:
                    print(f"Found {existing_count} existing results that would be reused")
                    print(f"Would run {remaining_tests} new tests")
                    
                    if remaining_tests == 0:
                        print("✓ All results exist - would skip server deployment")
                    elif not args.skip_deployment:
                        print("→ Would deploy server for new tests")
                    else:
                        print("→ Would assume server is already running")
            
            for i, test_case in enumerate(runner.test_cases, 1):
                existing = ""
                if not args.force_rerun and runner._load_existing_result(test_case) is not None:
                    existing = " (existing result)"
                print(f"  {i:3d}. {test_case}{existing}")
            print(f"\nWould save results to: {args.output_dir}")
            return
        
        # Run the tests
        skip_existing = not args.force_rerun
        runner.run_all_tests(skip_existing=skip_existing, skip_deployment=args.skip_deployment)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()