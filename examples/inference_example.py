"""
Example script for inference using the FactQA environment.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from factqa.environment import FactQAEnvironment, FactQAConfig
from factqa.utils import load_dataset, extract_answer


def run_inference(
    model_name: str,
    dataset_path: str,
    num_examples: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    config_path: Optional[str] = None
):
    """
    Run inference using a specified model on the FactQA dataset.
    
    Args:
        model_name: Name or path of the model to use
        dataset_path: Path to the dataset file
        num_examples: Number of examples to process
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        config_path: Path to a JSON configuration file (optional)
    """
    # Load configuration if provided
    config = None
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            config = FactQAConfig(**config_dict)
    else:
        config = FactQAConfig()
    
    # Override dataset path
    config.dataset_path = dataset_path
    
    # Create environment
    env = FactQAEnvironment(config=config)
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Get examples from the environment
    examples = env.get_eval_batch(batch_size=num_examples)
    
    results = []
    
    for i, example in enumerate(examples):
        prompt = example["prompt"]
        system_prompt = env.get_system_prompt()
        reference = example["metadata"]["answer"]
        
        # Format input for the model
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            # Use chat template if available
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            model_input = tokenizer.apply_chat_template(messages, return_tensors="pt")
        else:
            # Fallback to simple concatenation
            full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
            model_input = tokenizer(full_prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            model_input = {k: v.cuda() for k, v in model_input.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **model_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        # Decode response
        completion = tokenizer.decode(outputs[0][model_input["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Score the response
        score = env.score_response(prompt, completion, example["metadata"])
        
        # Extract answer
        extracted_answer = extract_answer(completion)
        
        # Store results
        result = {
            "question": prompt,
            "reference_answer": reference,
            "model_completion": completion,
            "extracted_answer": extracted_answer,
            "score": score
        }
        
        results.append(result)
        
        # Print progress
        print(f"\nExample {i+1}/{num_examples}:")
        print(f"Question: {prompt}")
        print(f"Reference: {reference}")
        print(f"Model answer: {extracted_answer if extracted_answer else 'No valid answer format'}")
        print(f"Score: {score:.4f}")
    
    # Calculate average score
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nAverage score: {avg_score:.4f}")
    
    # Save results
    output_path = "inference_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run inference with the FactQA environment")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="../data/factqa_dataset.json", help="Path to dataset file")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of examples to process")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    run_inference(
        model_name=args.model,
        dataset_path=args.dataset,
        num_examples=args.num_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
