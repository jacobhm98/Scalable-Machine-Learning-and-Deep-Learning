import csv

import pandas as pd
import random
import numpy as np

post_file = "../data/cleaned/reddit-data.csv"

stock_file = "../data/raw/price-movements/tesla-stock-movement.csv"

subreddit_to_int = {"stocks": 0, "wallstreetbets": 1}

#amount of posts that go into one datapoint
K = 5

def main():
    post_dfs, stock_dfs = get_dataframes(post_file, stock_file)
    schema, data_set = merge_dfs(post_dfs, stock_dfs)
    write_data_set(schema, data_set, "../data/data_set.csv")

def write_data_set(schema, data_set, filename):
    schema.insert(0, 'date')
    schema.append('price_movement')
    with open(filename, 'w') as f:
        write = csv.writer(f)
        write.writerow(schema)
        write.writerows(data_set)


def merge_dfs(post_dfs, stock_dfs):
    data_set = []
    schema = None
    schema_has_been_retrieved = False
    for index, row in stock_dfs.iterrows():
        date = row['Date']
        posts = post_dfs.loc[post_dfs['created_utc'] == date]
        try:
            entries = split_posts_into_K_chunks(date, posts, [])
            for entry in entries:
                label = row['Price_Change']
                if not schema_has_been_retrieved:
                    data_point, schema = create_data_point(row['Volume'], entry, return_schema=True)
                    process_data_point(schema, data_point)
                    schema_has_been_retrieved = True
                else:
                    data_point = create_data_point(row['Volume'], entry)
                    process_data_point(schema, data_point)
                data_set.append((date, data_point, label))
        except:
            print("Not enough entries")
            pass
    return schema, data_set

def process_data_point(schema, data_point):
    assert len(schema) == len(data_point)
    for i in range(len(data_point)):
        if schema[i] == 'subreddit':
            data_point[i] = subreddit_to_int[data_point[i]]
    return data_point

def create_data_point(volume, posts, return_schema=False):
    posts = posts.drop(columns="created_utc")
    data_point = posts.to_numpy().flatten().tolist()
    data_point.append(volume)
    if return_schema:
        return data_point, get_schema(posts)
    return data_point

def get_schema(posts):
    schema = []
    for i in range(K):
        schema_k = [str(x) for x in posts.columns]
        schema.append(schema_k)
    schema = [item for sublist in schema for item in sublist]
    schema.append("volume")
    return schema

def split_posts_into_K_chunks(date, posts, entries):
    num_posts = posts.shape[0]
    select_list = [False] * num_posts
    if num_posts < K:
        raise ValueError("not enough posts to create a datapoint")
    samples = random.sample(range(num_posts), K)
    for sample in samples:
        select_list[sample] = True
    entries.append(posts.iloc[select_list])
    if num_posts - K > K:
        select_list = [not elem for elem in select_list]
        split_posts_into_K_chunks(date, posts.iloc[select_list], entries)
    return entries


def get_dataframes(post_file, stock_file):
    post_df = pd.read_csv(post_file)
    stock_df = pd.read_csv(stock_file)
    return post_df, stock_df


main()
