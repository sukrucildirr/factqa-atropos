"""
Reward function implementation for the FactQA environment.
"""
from typing import Dict, List, Optional, Union, Any

from .utils import AnswerEvaluator, extract_answer, calculate_length_penalty


class FactQAReward:
    """
    Reward function for the FactQA environment.
    Evaluates model responses based on answer accuracy and conciseness.
    """
    
    def __init__(self, 
                 keyword_weight: float = 0.4,
                 semantic_weight: float = 0.6,
                 length_threshold_ratio: float = 0.5,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the reward function.
        
        Args:
            keyword_weight: Weight for keyword matching in evaluation
            semantic_weight: Weight for semantic similarity in evaluation
            length_threshold_ratio: Ratio of max length below which no penalty is applied
            embedding_model: Name of the sentence transformer model to use
        """
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.length_threshold_ratio = length_threshold_ratio
        self.evaluator = AnswerEvaluator(model_name=embedding_model)
    
    def calculate_reward(self, 
                        completion: str, 
                        reference: str, 
                        completion_length: int,
                        max_length: int) -> float:
        """
        Calculate the reward for a model completion.
        
        Args:
            completion: The model's completion text
            reference: The reference answer
            completion_length: Length of the completion in tokens
            max_length: Maximum allowed length in tokens
            
        Returns:
            float: Reward score between 0.0 and 1.0
        """
        # Extract the answer from the completion
        extracted_answer = extract_answer(completion)
        
        # If no valid answer format, return 0
        if extracted_answer is None:
            return 0.0
        
        # Calculate accuracy score
        accuracy_score = self.evaluator.evaluate(
            extracted_answer, 
            reference,
            keyword_weight=self.keyword_weight,
            semantic_weight=self.semantic_weight
        )
        
        # Calculate length penalty
        length_penalty = calculate_length_penalty(
            completion_length, 
            max_length,
            threshold_ratio=self.length_threshold_ratio
        )
        
        # Apply length penalty to accuracy score
        final_score = accuracy_score * length_penalty
        
        return final_score
    
    def batch_rewards(self, 
                     completions: List[str], 
                     references: List[str],
                     completion_lengths: List[int],
                     max_length: int) -> List[float]:
        """
        Calculate rewards for a batch of completions.
        
        Args:
            completions: List of model completion texts
            references: List of reference answers
            completion_lengths: List of completion lengths in tokens
            max_length: Maximum allowed length in tokens
            
        Returns:
            List of reward scores between 0.0 and 1.0
        """
        rewards = []
        
        for completion, reference, length in zip(completions, references, completion_lengths):
            reward = self.calculate_reward(
                completion, 
                reference, 
                length,
                max_length
            )
            rewards.append(reward)
        
        return rewards
    
    def __call__(self, 
                completions: List[str], 
                references: List[str],
                completion_lengths: List[int],
                max_length: int) -> List[float]:
        """
        Callable interface for the reward function.
        
        Args:
            completions: List of model completion texts
            references: List of reference answers
            completion_lengths: List of completion lengths in tokens
            max_length: Maximum allowed length in tokens
            
        Returns:
            List of reward scores between 0.0 and 1.0
        """
        return self.batch_rewards(completions, references, completion_lengths, max_length)
