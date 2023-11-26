import os

import pandas as pd

from utils import parse_cli_args, validate

def main():
    dataset, feature = parse_cli_args()

    if not validate(dataset, feature):
        return

    df = pd.read_csv(dataset)
    



if __name__ == '__main__':
    main()
