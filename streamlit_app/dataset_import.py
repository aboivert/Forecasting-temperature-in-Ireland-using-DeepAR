import pandas as pd


def data_import(name):
    df = pd.read_csv(name).dropna()
    df.date = pd.to_datetime(df.date)
    # df = df.asfreq('d')
    df = df[["date", "temp", "wetb", "dewpt", "vappr", "rhum"]]
    df = df[df['date'].dt.hour == 12]
    df = df[-1000:]
    df = df.set_index(df.date)
    return df
