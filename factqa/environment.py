"""
Core environment implementation for the FactQA Atropos integration.
"""
import json
import os
import random
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
from pydantic import BaseModel, Field

from .reward import FactQAReward
from .utils import load_dataset, extract_answer


class FactQAConfig(BaseModel):
    """Configuration for the FactQA environment."""
    
    dataset_path: str = Field(
        default="data/factqa_dataset.json",
        description="Path to the dataset file"
    )
    test_set_ratio: float = Field(
        default=0.2,
        description="Ratio of dataset to use for testing"
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum number of tokens for model responses"
    )
    keyword_weight: float = Field(
        default=0.4,
        description="Weight for keyword matching in evaluation"
    )
    semantic_weight: float = Field(
        default=0.6,
        description="Weight for semantic similarity in evaluation"
    )
    length_threshold_ratio: float = Field(
        default=0.5,
        description="Ratio of max length below which no penalty is applied"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Name of the sentence transformer model to use"
    )
    system_prompt: str = Field(
        default=(
            "You are a helpful AI assistant that answers factual questions accurately and concisely. "
            "Think carefully about the question before answering. "
            "Provide your final answer between <answer> and </answer> tags."
        ),
        description="System prompt for the model"
    )


class FactQAEnvironment:
    """
    FactQA environment for question answering evaluation.
    
    This environment evaluates language model responses to factual questions
    based on accuracy and conciseness.
    
    Note: This is a standalone version that can be used without Atropos.
    To integrate with Atropos, extend the BaseEnvironment class from atroposlib.
    """
    
    def __init__(self, config: Optional[FactQAConfig] = None, **kwargs):
        """
        Initialize the FactQA environment.
        
        Args:
            config: Configuration for the environment
            **kwargs: Additional arguments
        """
        # Initialize configuration
        self.config = config or FactQAConfig()
        
        # Set up logger first to avoid AttributeError
        self.logger = self._get_logger()
        
        # Initialize reward function
        self.reward_fn = FactQAReward(
            keyword_weight=self.config.keyword_weight,
            semantic_weight=self.config.semantic_weight,
            length_threshold_ratio=self.config.length_threshold_ratio,
            embedding_model=self.config.embedding_model
        )
        
        # Initialize dataset
        self.train_data = []
        self.test_data = []
        self.setup()
    
    def _get_logger(self):
        """Get a simple logger."""
        import logging
        logger = logging.getLogger("factqa")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def setup(self):
        """Set up the environment by loading and splitting the dataset."""
        # Resolve dataset path
        dataset_path = self.config.dataset_path
        if not os.path.isabs(dataset_path):
            # Try to find the dataset relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(current_dir, dataset_path),
                os.path.join(current_dir, "..", dataset_path),
                os.path.join(os.getcwd(), dataset_path)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    dataset_path = path
                    break
        
        # Load dataset
        try:
            data = load_dataset(dataset_path)
        except FileNotFoundError:
            # Fallback to a small dummy dataset
            self.logger.warning(f"Dataset not found at {dataset_path}. Using dummy data.")
            data = [
                {"question": "What is the capital of France?", "answer": "Paris"},
                {"question": "Who wrote Pride and Prejudice?", "answer": "Jane Austen"},
                {"question": "What is the chemical symbol for gold?", "answer": "Au"}
            ]
        
        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(data)
        
        # Split into train and test sets
        test_size = int(len(data) * self.config.test_set_ratio)
        self.test_data = data[:test_size]
        self.train_data = data[test_size:]
        
        self.logger.info(f"Loaded {len(self.train_data)} training examples and {len(self.test_data)} test examples")
    
    def format_prompt(self, item: Dict[str, str]) -> str:
        """
        Format a prompt for the model.
        
        Args:
            item: Dictionary containing question and answer
            
        Returns:
            Formatted prompt string
        """
        return item["question"]
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the model.
        
        Returns:
            System prompt string
        """
        return self.config.system_prompt
    
    def get_train_item(self) -> Dict[str, Any]:
        """
        Get a random training item.
        
        Returns:
            Dictionary with prompt and metadata
        """
        item = random.choice(self.train_data)
        prompt = self.format_prompt(item)
        
        return {
            "prompt": prompt,
            "metadata": {
                "question": item["question"],
                "answer": item["answer"]
            }
        }
    
    def get_eval_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Get a batch of evaluation items.
        
        Args:
            batch_size: Number of items to include in the batch
            
        Returns:
            List of dictionaries with prompts and metadata
        """
        # Sample from test data, with replacement if necessary
        if batch_size > len(self.test_data):
            eval_items = random.choices(self.test_data, k=batch_size)
        else:
            eval_items = random.sample(self.test_data, k=batch_size)
        
        batch = []
        for item in eval_items:
            prompt = self.format_prompt(item)
            batch.append({
                "prompt": prompt,
                "metadata": {
                    "question": item["question"],
                    "answer": item["answer"]
                }
            })
        
        return batch
    
    def score_response(self, prompt: str, completion: str, metadata: Dict[str, Any]) -> float:
        """
        Score a model response.
        
        Args:
            prompt: The prompt given to the model
            completion: The model's completion
            metadata: Additional metadata including the reference answer
            
        Returns:
            Score between 0.0 and 1.0
        """
        reference = metadata["answer"]
        completion_length = len(completion.split())  # Simple word count as proxy for tokens
        
        return self.reward_fn.calculate_reward(
            completion=completion,
            reference=reference,
            completion_length=completion_length,
            max_length=self.config.max_tokens
        )
    
    def batch_score(self, prompts: List[str], completions: List[str], 
                   metadatas: List[Dict[str, Any]]) -> List[float]:
        """
        Score a batch of model responses.
        
        Args:
            prompts: List of prompts given to the model
            completions: List of model completions
            metadatas: List of metadata dictionaries
            
        Returns:
            List of scores between 0.0 and 1.0
        """
        references = [metadata["answer"] for metadata in metadatas]
        completion_lengths = [len(completion.split()) for completion in completions]
        
        return self.reward_fn.batch_rewards(
            completions=completions,
            references=references,
            completion_lengths=completion_lengths,
            max_length=self.config.max_tokens
        )
    
    def serve(self):
        """
        Placeholder for Atropos integration.
        In a full Atropos integration, this would start the environment service.
        """
        self.logger.info("This is a standalone environment. To integrate with Atropos:")
        self.logger.info("1. Clone the Atropos repository: git clone https://github.com/NousResearch/atropos.git")
        self.logger.info("2. Install it in development mode: cd atropos && pip install -e .")
        self.logger.info("3. Extend BaseEnvironment from atroposlib.envs.base")
        self.logger.info("4. Implement the required methods for Atropos integration")
        
        # Print some information about the environment
        self.logger.info(f"FactQA environment initialized with {len(self.train_data)} training examples")
        self.logger.info(f"System prompt: {self.get_system_prompt()}")
        
        # Get a sample item
        sample = self.get_train_item()
        self.logger.info(f"Sample question: {sample['prompt']}")
        self.logger.info(f"Sample answer: {sample['metadata']['answer']}")


# For Atropos integration, uncomment and modify this code:
"""
# This requires atroposlib to be installed from the Atropos repository
try:
    from atroposlib.envs.base import BaseEnvironment
    
    class FactQAAtroposEnvironment(BaseEnvironment):
        def __init__(self, config: Optional[FactQAConfig] = None, **kwargs):
            super().__init__(**kwargs)
            self.env = FactQAEnvironment(config=config)
            
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
except ImportError:
    # atroposlib not available, use standalone version
    pass
"""
