"""
Utility functions for the FactQA environment.
"""
import json
import os
import re
from typing import Dict, List, Any, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AnswerEvaluator:
    """
    Evaluates the similarity between model answers and reference answers.
    Uses both keyword matching and semantic similarity with sentence embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the answer evaluator with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
    
    def keyword_match_score(self, prediction: str, reference: str) -> float:
        """
        Calculate a simple keyword match score between prediction and reference.
        
        Args:
            prediction: The model's predicted answer
            reference: The reference answer
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        # Normalize text
        pred_norm = re.sub(r'[^\w\s]', ' ', prediction.lower())
        ref_norm = re.sub(r'[^\w\s]', ' ', reference.lower())
        
        # Get unique words
        pred_words = set(pred_norm.split())
        ref_words = set(ref_norm.split())
        
        if not ref_words:
            return 0.0
            
        # Calculate intersection
        intersection = pred_words.intersection(ref_words)
        
        # Calculate Jaccard similarity
        return len(intersection) / len(ref_words)
    
    def semantic_similarity(self, prediction: str, reference: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Args:
            prediction: The model's predicted answer
            reference: The reference answer
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        # Get embeddings
        pred_embedding = self.model.encode([prediction])
        ref_embedding = self.model.encode([reference])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(pred_embedding, ref_embedding)[0][0]
        
        # Normalize to 0-1 range (cosine similarity is between -1 and 1)
        return (similarity + 1) / 2
    
    def evaluate(self, prediction: str, reference: str, 
                keyword_weight: float = 0.4, 
                semantic_weight: float = 0.6) -> float:
        """
        Evaluate the prediction against the reference using a weighted combination
        of keyword matching and semantic similarity.
        
        Args:
            prediction: The model's predicted answer
            reference: The reference answer
            keyword_weight: Weight for keyword matching score
            semantic_weight: Weight for semantic similarity score
            
        Returns:
            float: Combined score between 0.0 and 1.0
        """
        keyword_score = self.keyword_match_score(prediction, reference)
        semantic_score = self.semantic_similarity(prediction, reference)
        
        # Combine scores with weights
        return (keyword_weight * keyword_score) + (semantic_weight * semantic_score)


def load_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Load the FactQA dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON dataset file
        
    Returns:
        List of question-answer pairs
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data


def extract_answer(completion: str) -> Optional[str]:
    """
    Extract the answer from a model completion.
    Looks for text between <answer> and </answer> tags.
    
    Args:
        completion: The model's completion text
        
    Returns:
        The extracted answer or None if no valid answer found
    """
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    match = answer_pattern.search(completion)
    
    if match:
        return match.group(1).strip()
    
    return None


def calculate_length_penalty(completion_length: int, max_length: int, 
                           threshold_ratio: float = 0.5) -> float:
    """
    Calculate a length penalty for the completion.
    No penalty for responses under threshold_ratio * max_length.
    Linear penalty scaling from 1.0 down to 0.0 for responses between
    threshold_ratio * max_length and max_length.
    
    Args:
        completion_length: Length of the completion in tokens
        max_length: Maximum allowed length in tokens
        threshold_ratio: Ratio of max_length below which no penalty is applied
        
    Returns:
        float: Penalty factor between 0.0 and 1.0
    """
    threshold = threshold_ratio * max_length
    
    if completion_length <= threshold:
        return 1.0
    
    if completion_length >= max_length:
        return 0.0
    
    # Linear scaling between threshold and max_length
    return 1.0 - ((completion_length - threshold) / (max_length - threshold))
