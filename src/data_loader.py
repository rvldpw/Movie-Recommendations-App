import pandas as pd


def load_data(path):
    df = pd.read_csv(path)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df['datetime'] = pd.to_datetime(
        df['timestamp'],
        unit='s'
    )

    return df
