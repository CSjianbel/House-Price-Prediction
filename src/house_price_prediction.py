import os

import pandas as pd

from linear_regression import LinearRegression
from utils import parse_cli_args, validate

def main():
    dataset, feature = parse_cli_args()

    if not validate(dataset):
        return

    df = pd.read_csv(dataset)

    try: 
        X = df[feature].values
    except Exception: 
        print('Invalid feature provided.')
        return

    y = df['house_price_of_unit_area'].values

    model = LinearRegression()
    model.fit(X, y)
    model.visualize(X, y, feature, 'House Price')


if __name__ == '__main__':
    main()
