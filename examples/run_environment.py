"""
Example script for running the FactQA environment.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Any

import requests

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import directly from factqa package
from factqa.environment import FactQAEnvironment, FactQAConfig


def run_environment(config_path: Optional[str] = None):
    """
    Run the FactQA environment as a service.
    
    Args:
        config_path: Path to a JSON configuration file (optional)
    """
    # Load configuration if provided
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            config = FactQAConfig(**config_dict)
    
    # Create and run the environment
    env = FactQAEnvironment(config=config)
    
    print("FactQA environment created successfully!")
    print("System prompt:", env.get_system_prompt())
    
    # Get a sample training item
    sample = env.get_train_item()
    print("\nSample question:", sample["prompt"])
    print("Reference answer:", sample["metadata"]["answer"])
    
    # Get a sample evaluation batch
    eval_batch = env.get_eval_batch(batch_size=2)
    print(f"\nGenerated evaluation batch with {len(eval_batch)} items")
    
    print("\nTo integrate with Atropos:")
    print("1. Clone the Atropos repository: git clone https://github.com/NousResearch/atropos.git")
    print("2. Install it in development mode: cd atropos && pip install -e .")
    print("3. Copy this environment to the environments directory")
    print("4. Follow the Atropos documentation for running environments")


def test_environment_standalone():
    """
    Test the FactQA environment in standalone mode.
    """
    # Create environment
    env = FactQAEnvironment()
    
    print("Testing FactQA environment in standalone mode...")
    
    # Get a training item
    item = env.get_train_item()
    print(f"\nQuestion: {item['prompt']}")
    print(f"Reference answer: {item['metadata']['answer']}")
    
    # Test scoring
    test_completion = f"Let me think about this... <answer>{item['metadata']['answer']}</answer>"
    score = env.score_response(item['prompt'], test_completion, item['metadata'])
    print(f"\nTest completion: {test_completion}")
    print(f"Score: {score:.4f}")
    
    # Test batch scoring
    batch = env.get_eval_batch(batch_size=3)
    prompts = [item['prompt'] for item in batch]
    metadatas = [item['metadata'] for item in batch]
    completions = [f"<answer>{item['metadata']['answer']}</answer>" for item in batch]
    
    scores = env.batch_score(prompts, completions, metadatas)
    print(f"\nBatch scores: {[f'{s:.4f}' for s in scores]}")
    
    print("\nTest completed successfully!")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run or test the FactQA environment")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run the environment as a service")
    serve_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test the environment")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        run_environment(config_path=args.config)
    elif args.command == "test":
        test_environment_standalone()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
