"""
Basic test for the FactQA environment.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from factqa.environment import FactQAEnvironment, FactQAConfig
from factqa.reward import FactQAReward
from factqa.utils import extract_answer, calculate_length_penalty


class TestFactQAEnvironment(unittest.TestCase):
    """Test cases for the FactQA environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FactQAConfig(
            dataset_path="../data/factqa_dataset.json",
            test_set_ratio=0.2,
            max_tokens=1024
        )
        
        # Mock the AnswerEvaluator to avoid loading models during tests
        self.patcher = patch('factqa.utils.AnswerEvaluator')
        self.mock_evaluator = self.patcher.start()
        self.mock_evaluator_instance = MagicMock()
        self.mock_evaluator.return_value = self.mock_evaluator_instance
        self.mock_evaluator_instance.evaluate.return_value = 0.8
        
        self.env = FactQAEnvironment(config=self.config)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.patcher.stop()
    
    def test_environment_initialization(self):
        """Test that the environment initializes correctly."""
        self.assertIsNotNone(self.env)
        self.assertIsNotNone(self.env.train_data)
        self.assertIsNotNone(self.env.test_data)
    
    def test_get_system_prompt(self):
        """Test that the system prompt is returned correctly."""
        system_prompt = self.env.get_system_prompt()
        self.assertIsInstance(system_prompt, str)
        self.assertGreater(len(system_prompt), 0)
    
    def test_get_train_item(self):
        """Test that a training item is returned correctly."""
        item = self.env.get_train_item()
        self.assertIsInstance(item, dict)
        self.assertIn("prompt", item)
        self.assertIn("metadata", item)
        self.assertIn("question", item["metadata"])
        self.assertIn("answer", item["metadata"])
    
    def test_get_eval_batch(self):
        """Test that an evaluation batch is returned correctly."""
        batch = self.env.get_eval_batch(batch_size=3)
        self.assertIsInstance(batch, list)
        self.assertEqual(len(batch), 3)
        for item in batch:
            self.assertIn("prompt", item)
            self.assertIn("metadata", item)
    
    def test_score_response(self):
        """Test that responses are scored correctly."""
        prompt = "What is the capital of France?"
        completion = "The capital of France is <answer>Paris</answer>."
        metadata = {"question": prompt, "answer": "Paris"}
        
        score = self.env.score_response(prompt, completion, metadata)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_batch_score(self):
        """Test that batches of responses are scored correctly."""
        prompts = ["What is the capital of France?", "Who wrote Pride and Prejudice?"]
        completions = [
            "The capital of France is <answer>Paris</answer>.",
            "Pride and Prejudice was written by <answer>Jane Austen</answer>."
        ]
        metadatas = [
            {"question": prompts[0], "answer": "Paris"},
            {"question": prompts[1], "answer": "Jane Austen"}
        ]
        
        scores = self.env.batch_score(prompts, completions, metadatas)
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 2)
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestFactQAUtils(unittest.TestCase):
    """Test cases for the FactQA utility functions."""
    
    def test_extract_answer(self):
        """Test that answers are correctly extracted from completions."""
        # Test valid format
        completion = "The answer is <answer>42</answer>."
        self.assertEqual(extract_answer(completion), "42")
        
        # Test with newlines
        completion = "Let me think...\n<answer>Paris</answer>\nThat's it."
        self.assertEqual(extract_answer(completion), "Paris")
        
        # Test with no tags
        completion = "The answer is 42."
        self.assertIsNone(extract_answer(completion))
        
        # Test with incomplete tags
        completion = "The answer is <answer>42."
        self.assertIsNone(extract_answer(completion))
    
    def test_calculate_length_penalty(self):
        """Test that length penalties are calculated correctly."""
        max_length = 100
        threshold_ratio = 0.5
        
        # Test below threshold (no penalty)
        self.assertEqual(calculate_length_penalty(40, max_length, threshold_ratio), 1.0)
        
        # Test at threshold (no penalty)
        self.assertEqual(calculate_length_penalty(50, max_length, threshold_ratio), 1.0)
        
        # Test above threshold (partial penalty)
        penalty = calculate_length_penalty(75, max_length, threshold_ratio)
        self.assertGreater(penalty, 0.0)
        self.assertLess(penalty, 1.0)
        
        # Test at max length (full penalty)
        self.assertEqual(calculate_length_penalty(100, max_length, threshold_ratio), 0.0)
        
        # Test above max length (full penalty)
        self.assertEqual(calculate_length_penalty(120, max_length, threshold_ratio), 0.0)


if __name__ == '__main__':
    unittest.main()
