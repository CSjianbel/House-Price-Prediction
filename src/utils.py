import os
import argparse

def parse_cli_args() -> tuple[str]:
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='House price prediction')

    # Add arguments
    parser.add_argument('dataset', help='Path to the dataset')
    parser.add_argument('feature', help='Feature for prediction')

    # Parse arguments from command line
    args = parser.parse_args()

    return (args.dataset, args.feature)


def validate(dataset: str) -> bool:
    valid = True
    if not os.path.exists(dataset):
        valid = False
        print("Invalid dataset provided")

    return valid