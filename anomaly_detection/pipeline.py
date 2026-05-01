import pandas as pd

def load_data(filepath):
    df = pd.read_csv(
        filepath,
        sep=' ',
        header=None,
        names=['date', 'time', 'sensor', 'state']
    )

    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='mixed')
    df = df.drop(columns=['date', 'time'])

    # keep raw state too (useful later)
    df['state'] = df['state'].map({'ON': 1, 'OFF': 0})

    print(f"✔ Loaded {len(df)} rows from {filepath}")
    return df