# DAPO AI Agent Implementation

This repository contains an implementation of the Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) algorithm for reinforcement learning with language models. DAPO is designed to fine-tune language models for complex reasoning tasks.

## Features

- **Group Relative Policy Optimization (GRPO)** - Samples G responses per prompt using the current policy and normalizes each response's reward relative to the group's mean and standard deviation.

- **Clip-Higher (Asymmetric Clipping)** - Uses separate lower and upper clipping thresholds for the policy update ratio, allowing for better exploration.

- **Dynamic Sampling** - Mitigates the "gradient deadzone" by dynamically oversampling and skipping prompts that yield 100% or 0% success.

- **Token-Level Policy Gradient Loss** - Calculates the policy gradient loss across all tokens in all samples, rather than averaging per sample.

- **Overlong Reward Shaping** - Handles excessively long or truncated outputs with a soft length penalty.

## Installation

```bash
# Clone the repository
git clone https://github.com/ai-in-pm/DAPO.git
cd DAPO

# Setup on Windows
activate.bat

# Setup on Linux/MacOS
python -m venv venv
source venv/bin/activate  # Linux/MacOS
pip install -r requirements.txt
```

## Project Structure

```
├── config/              # Configuration files
├── data/                # Data storage and databases
├── logs/                # Log files
├── models/              # Model definitions
│   └── checkpoints/     # Saved model checkpoints
├── scripts/             # Utility scripts
│   └── generate_sample_data.py  # Sample data generator
├── utils/               # Utility functions
│   ├── config.py        # Configuration manager
│   ├── data.py          # Dataset handling
│   ├── database.py      # SQLite database integration
│   └── logging.py       # Logging setup
├── main.py              # Main entry point and interactive mode
├── train.py             # Training script
├── eval.py              # Evaluation script
```

## Quick Start (Windows)

```bash
# Generate sample training data
run_generate_data.bat

# Initialize the database
run_init_database.bat

# Train the model
run_train.bat

# Run interactive mode
run_interactive.bat
```

## Quick Start (Linux/MacOS)

```bash
# Generate sample training data
python scripts/generate_sample_data.py --num-samples 200 --output data/dataset.jsonl

# Initialize database
python scripts/init_database.py --db-path data/dapo.db

# Run training
python train.py --config config/default.yaml

# Run interactive mode
python main.py interactive --model models/checkpoints/dapo_final
```

## Sample Data

The repository includes scripts to generate synthetic training data across multiple task types:

- **Math Problems** - Basic arithmetic operations
- **Reasoning Tasks** - Logical reasoning problems
- **Classification** - Sentiment analysis and categorization
- **Code Generation** - Simple Python function implementation
- **Creative Writing** - Short story generation with diverse themes

## Database Integration

The implementation includes SQLite database integration for:

1. **Training Data Management** - Import/export training samples
2. **Interaction Storage** - Record user-agent interactions
3. **Metrics Tracking** - Store evaluation metrics

## Development

For more detailed information on the DAPO algorithm and its implementation, refer to the [DAPO_IMPLEMENTATION_GUIDE.md](DAPO_IMPLEMENTATION_GUIDE.md) document.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
