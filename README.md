# FactQA Atropos

A factual question-answering environment designed to work both standalone and with the Atropos reinforcement learning framework. This project demonstrates how to create a custom environment that rewards language models for providing accurate and concise answers to factual questions.

## Overview

FactQA is a specialized environment that evaluates model responses to factual questions based on answer accuracy and conciseness. It can be used independently or integrated with the [Atropos](https://github.com/NousResearch/atropos) framework for reinforcement learning with language models.

This project serves as both a functional environment for improving factual question answering capabilities in language models and as a reference implementation for creating custom environments.

## Features

- **Standalone Environment**: Works independently without requiring Atropos installation
- **Atropos Integration**: Includes a robust integration script that works with any Atropos version
- **Sophisticated Reward Function**: Combines keyword matching and semantic similarity for answer evaluation
- **Length Penalty**: Encourages concise responses while maintaining accuracy
- **Example Scripts**: Ready-to-use examples for running the environment, inference, and training
- **Customizable Dataset**: Easy to extend with your own factual questions and answers

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch
- Transformers
- Sentence-Transformers
- scikit-learn

### Setup

1. Clone this repository:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   git clone https://github.com/yourusername/factqa-atropos.git
   cd factqa-atropos
   ```

2. Install the package and dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Environment in Standalone Mode

To test the FactQA environment in standalone mode:

```bash
python examples/run_environment.py test
```

### Running Inference

To run inference with a pre-trained model:

```bash
python examples/inference_example.py --model "gpt2" --num-examples 5
```

### Simple Training Loop

To run a simple training loop:

```bash
python examples/simple_training.py --model "gpt2" --num-steps 100
```

## Atropos Integration

To integrate this environment with the Atropos framework:

1. Clone the Atropos repository:
   ```bash
   git clone https://github.com/NousResearch/atropos.git
   cd atropos
   pip install -e .
   ```

2. Copy the FactQA environment to the Atropos environments directory:
   ```bash
   cp -r /path/to/factqa-atropos/factqa /path/to/atropos/environments/
   ```

3. Copy the integration script to the Atropos environments directory:
   ```bash
   cp /path/to/factqa-atropos/factqa_server.py /path/to/atropos/environments/
   ```

4. Run the environment with Atropos:
   ```bash
   cd /path/to/atropos
   python environments/factqa_server.py serve
   ```

### Integration Script

The `factqa_server.py` script is designed to work with any version of Atropos, regardless of its internal structure. It implements a standalone class with all the methods needed for Atropos integration and includes fallback mechanisms to ensure compatibility.

```python
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
```

## Environment Details

### Input Format

The environment provides factual questions to the model and expects answers in a specific format:

```
Question: What is the capital of France?

Answer: <answer>Paris</answer>
```

### System Prompt

The default system prompt encourages the model to think carefully and provide concise, accurate answers:

```
You are a helpful AI assistant that answers factual questions accurately and concisely.
Think carefully about the question before answering.
Provide your final answer between <answer> and </answer> tags.
```

### Reward Function

The reward function evaluates responses based on:

1. **Answer Accuracy**: Using a combination of keyword matching and semantic similarity
2. **Conciseness**: Applying a length penalty to encourage shorter responses
3. **Format Compliance**: Checking if the answer is provided in the correct format

## Configuration

The environment can be configured through a JSON file with the following parameters:

```json
{
  "dataset_path": "data/factqa_dataset.json",
  "test_set_ratio": 0.2,
  "max_tokens": 1024,
  "keyword_weight": 0.4,
  "semantic_weight": 0.6,
  "length_threshold_ratio": 0.5,
  "embedding_model": "all-MiniLM-L6-v2",
  "system_prompt": "You are a helpful AI assistant that answers factual questions accurately and concisely. Think carefully about the question before answering. Provide your final answer between <answer> and </answer> tags."
}
```

## Extending the Dataset

You can create your own dataset by following the format in `data/factqa_dataset.json`:

```json
[
  {
    "question": "What is the capital of France?",
    "answer": "Paris"
  },
  {
    "question": "Who wrote the novel 'Pride and Prejudice'?",
    "answer": "Jane Austen"
  }
]
```

## Troubleshooting

### Module Not Found Errors

If you encounter "Module not found" errors when running the scripts directly, make sure you're either:

1. Running from the project root directory
2. Installing the package in development mode: `pip install -e .`

### Atropos Integration Issues

If you encounter issues with Atropos integration:

1. Make sure Atropos is installed correctly: `pip install -e .` from the Atropos repository root
2. Ensure the factqa module is in the Python path
3. Use the provided `factqa_server.py` script which is designed to work with any Atropos version

## Project Structure

```
factqa-atropos/
├── README.md
├── requirements.txt
├── setup.py
├── factqa_server.py        # Atropos integration script
├── data/
│   └── factqa_dataset.json
├── factqa/
│   ├── __init__.py
│   ├── environment.py
│   ├── reward.py
│   └── utils.py
└── examples/
    ├── run_environment.py
    ├── inference_example.py
    └── simple_training.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments

- [Nous Research](https://github.com/NousResearch) for creating the Atropos framework
- The open-source AI community for their valuable contributions to reinforcement learning with language models
