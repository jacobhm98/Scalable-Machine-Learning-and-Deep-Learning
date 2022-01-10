import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

file_list = [
    "../data/raw/2020/stocks/2020-stocks-submissions.csv",
    "../data/raw/2020/wallstreetbets/2020-wallstreetbets-submissions.csv",
    "../data/raw/2021/stocks/2021-stocks-submissions.csv",
    "../data/raw/2021/wallstreetbets/2021-wallstreetbets-submissions.csv"
]
for file in file_list:
    df = pd.read_csv(file)
    sentiment_column = []
    for title in df["title"]:
        popularity = sia.polarity_scores(title)
        sentiment_column.append(popularity["compound"])
    df["sentiment"] = sentiment_column
    df.to_csv(file)

popularity = sia.polarity_scores("EV automakers are doing great!")
print(popularity)