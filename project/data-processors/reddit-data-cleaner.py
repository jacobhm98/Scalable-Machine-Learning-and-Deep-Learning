import pandas as pd

file_list = [
    "../data/raw/2020/stocks/2020-stocks-submissions.csv",
    "../data/raw/2020/wallstreetbets/2020-wallstreetbets-submissions.csv",
    "../data/raw/2021/stocks/2021-stocks-submissions.csv",
    "../data/raw/2021/wallstreetbets/2021-wallstreetbets-submissions.csv"
]

cleaned_file = "../data/cleaned/reddit-data.csv"
cleaned_df = pd.DataFrame(columns=['score', 'num_comments', 'created_utc', 'sentiment', 'subreddit'])
for file in file_list:
    df = pd.read_csv(file)
    # drop text since we are analysing the sentiment in the title for simplicity
    df = df.drop(columns=["selftext", "id", "title"])
    df['subreddit'] = 'stocks' if "stocks" in file else 'wallstreetbets'
    cleaned_df = cleaned_df.append(df)

# sort by timestamp and convert from unix timestamp to date
cleaned_df = cleaned_df.drop("Unnamed: 0", axis=1)
cleaned_df = cleaned_df.sort_values(by='created_utc', ascending=True)
cleaned_df['created_utc'] = pd.to_datetime(cleaned_df['created_utc'], unit='s').dt.date
cleaned_df.to_csv(cleaned_file, index=False)