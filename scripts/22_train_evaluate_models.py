#!/usr/bin/env python3
# bmipred/scripts/22_train_evaluate_models.py

import sys
import os
import logging
import yaml


# add src/ to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bmipred.modeling.pipeline import BMIPredPipeline

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def setup_global_logging() -> None:
    """Configure global logging for scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_config(path: str = "scripts/configs/22_train_evaluate_models.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = os.path.join(REPO_ROOT, path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for BMI prediction experiments."""
    # Setup logging first
    setup_global_logging()
    
    # Load configuration
    config = load_config()
    
    # Initialize and run pipeline
    pipeline = BMIPredPipeline(config)
    pipeline.run_full_experiment()


if __name__ == "__main__":
    main()