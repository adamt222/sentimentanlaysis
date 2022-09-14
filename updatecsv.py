import pandas as pd

df = pd.read_csv('data/IMDBDataset.csv')

df['sentiment'] = df['sentiment'].replace({'positive': 1})

df['sentiment'] = df['sentiment'].replace({'negative': 0})

df.to_csv("NewIMDBReviews.csv", index=False)
