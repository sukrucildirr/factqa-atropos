#!/usr/bin/env python3
"""
FactQA environment server for Atropos integration.
"""
import os
import sys
import logging
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to find the factqa module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FactQA environment
from factqa.environment import FactQAEnvironment, FactQAConfig

# Create a minimal base class that implements the required interface
class FactQAAtroposEnvironment:
    """FactQA environment for Atropos integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Convert dict config to FactQAConfig if provided
        factqa_config = None
        if config is not None:
            factqa_config = FactQAConfig(**config)
        
        # Create the underlying environment
        self.env = FactQAEnvironment(config=factqa_config)
        
        # Set up logging
        self.logger = logging.getLogger("factqa_atropos")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info("FactQA Atropos environment initialized")
    
    def get_system_prompt(self) -> str:
        return self.env.get_system_prompt()
    
    def get_train_item(self) -> Dict[str, Any]:
        return self.env.get_train_item()
    
    def get_eval_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        return self.env.get_eval_batch(batch_size=batch_size)
    
    def score_response(self, prompt: str, completion: str, metadata: Dict[str, Any]) -> float:
        return self.env.score_response(prompt, completion, metadata)
    
    def batch_score(self, prompts: List[str], completions: List[str], 
                   metadatas: List[Dict[str, Any]]) -> List[float]:
        return self.env.batch_score(prompts, completions, metadatas)
    
    # Add methods that might be required by the Atropos CLI
    def serve(self):
        self.logger.info("Starting FactQA environment server")
        # This would normally start a server, but we'll just print info
        self.logger.info(f"System prompt: {self.get_system_prompt()}")
        sample = self.get_train_item()
        self.logger.info(f"Sample question: {sample['prompt']}")
        self.logger.info(f"Sample answer: {sample['metadata']['answer']}")
        self.logger.info("Environment is ready to use")

if __name__ == "__main__":
    # Check if we're being called with 'serve'
    if len(sys.argv) > 1 and sys.argv[1] == 'serve':
        env = FactQAAtroposEnvironment()
        env.serve()
    else:
        # Try to use the Atropos CLI if available
        try:
            from atroposlib.cli import main
            sys.exit(main(FactQAAtroposEnvironment))
        except ImportError:
            # Fallback to direct execution
            print("Atropos CLI not found, running directly")
            env = FactQAAtroposEnvironment()
            env.serve()