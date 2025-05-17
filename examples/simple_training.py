"""
Example script for a simple training loop using the FactQA environment.
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler

from factqa.environment import FactQAEnvironment, FactQAConfig


def simple_training(
    model_name: str,
    output_dir: str = "trained_model",
    num_steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = 1,
    save_every: int = 20
):
    """
    Run a simple training loop using the FactQA environment.
    
    Args:
        model_name: Name or path of the model to fine-tune
        output_dir: Directory to save the trained model
        num_steps: Number of training steps
        batch_size: Batch size for training
        learning_rate: Learning rate
        gradient_accumulation_steps: Number of steps to accumulate gradients
        save_every: Save model every N steps
    """
    # Create environment
    env = FactQAEnvironment()
    
    print(f"Created FactQA environment for training")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_steps
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    model.train()
    global_step = 0
    
    while global_step < num_steps:
        # Get batch from environment
        batch_data = []
        for _ in range(batch_size):
            batch_data.append(env.get_train_item())
        
        # Process each example
        for i, example in enumerate(batch_data):
            prompt = example["prompt"]
            system_prompt = env.get_system_prompt()
            
            # Format input for the model
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                # Use chat template if available
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
            else:
                # Fallback to simple concatenation
                full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\n\nAnswer:"
                inputs = tokenizer(full_prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Log progress
            print(f"Step {global_step+1}/{num_steps}, Example {i+1}/{len(batch_data)}, Loss: {loss.item()*gradient_accumulation_steps:.4f}")
            
            # Update weights if gradient accumulation is complete
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Save checkpoint
                if global_step % save_every == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    print(f"Saved checkpoint to {checkpoint_dir}")
                
                if global_step >= num_steps:
                    break
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete! Model saved to {output_dir}")
    
    print("\nNote: This is a standalone training example.")
    print("To integrate with the full Atropos framework:")
    print("1. Clone the Atropos repository: git clone https://github.com/NousResearch/atropos.git")
    print("2. Install it in development mode: cd atropos && pip install -e .")
    print("3. Copy the FactQA environment to the environments directory")
    print("4. Use the example_trainer from Atropos for a more sophisticated training loop")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Simple training loop for FactQA")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="trained_model", help="Directory to save the trained model")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--save-every", type=int, default=20, help="Save model every N steps")
    
    args = parser.parse_args()
    
    simple_training(
        model_name=args.model,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_every=args.save_every
    )


if __name__ == "__main__":
    main()
