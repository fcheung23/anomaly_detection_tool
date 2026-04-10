import pandas as pd

def load_data(filepath):
    df = pd.read_csv(
        filepath,
        sep=' ',
        header=None,
        names=['date', 'time', 'sensor', 'state']
    )
    
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='mixed') # combine date + time to timestamp and convert to datetime object
    df = df.drop(columns=['date', 'time']) # delete date and time columns
    df['state'] = df['state'].map({'ON': 1, 'OFF': 0}) # convert ON/OFF to 1/0
    
    print(f"Loaded {len(df)} rows from {filepath}")
    print(df.head(5))
    return df