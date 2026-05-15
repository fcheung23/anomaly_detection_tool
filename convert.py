import pandas as pd

df = pd.read_csv(
    "weekendaway.csv",
    sep='\t',
    header=None,
    names=['timestamp', 'sensor', 'state', 'extra'],
    usecols=['timestamp', 'sensor', 'state']
)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
df['time'] = df['timestamp'].dt.strftime('%H:%M:%S.%f')
df['state'] = df['state'].str.strip()

df[['date', 'time', 'sensor', 'state']].to_csv('weekendaway_fixed.csv', sep=' ', header=False, index=False)