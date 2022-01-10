import praw
from psaw import PushshiftAPI
import time
import pandas as pd
import datetime as dt
import os

def log_action(action):
    print(action)
    return

# PSAW
push_shift_api = PushshiftAPI()

# PRAW
reddit = praw.Reddit(
    client_id = "<reddit-client-id>",
    client_secret = "<reddit-client-secret>",
    username = "<reddit-username>",
    password = "<reddit-password>",
    user_agent = "some agent"
)

subreddits = ['stocks']
start_year = 2020
end_year = 2021
# directory on which to store the data
base_directory = './data/raw/'

### create directories to store data ###

for year in range(start_year, end_year+1):
    action = "[Year] " + str(year)
    log_action(action)

    dirpath = base_directory + str(year)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # timestamps that define window of posts
    ts_after = int(dt.datetime(year, 1, 1).timestamp())
    ts_before = int(dt.datetime(year+1, 1, 1).timestamp())

    for subreddit in subreddits:
        start_time = time.time()

        action = "\t[Subreddit] " + subreddit
        log_action(action)

        subredditdirpath = dirpath + '/' + subreddit
        if os.path.exists(subredditdirpath):
            continue
        else:
            os.makedirs(subredditdirpath)

        submissions_csv_path = str(year) + '-' + subreddit + '-submissions.csv'
        
### BLOCK 3 ###


### collect submission ids with PSAW ###

        gen = push_shift_api.search_submissions(
            after=ts_after,
            before=ts_before,
            filter=['id'],
            q='Tesla',
            subreddit=subreddit,
            limit=5000
        )

        action = f"\t\t[Info] Finished pushshift fetch with time: {time.time() - start_time: .2f}s"
        log_action(action)

### collect submission information with PRAW ###

        submissions_dict = {
            "id": [],
            "title": [],
            "score": [],
            "num_comments": [],
            "created_utc": [],
            "selftext": [],
        }

        for submission_psaw in gen:
            # psaw
            submission_id = submission_psaw.d_['id']
            # praw
            submission_praw = reddit.submission(id=submission_id)

            submissions_dict["id"].append(submission_praw.id)
            submissions_dict["title"].append(submission_praw.title)
            submissions_dict["score"].append(submission_praw.score)
            submissions_dict["num_comments"].append(submission_praw.num_comments)
            submissions_dict["created_utc"].append(submission_praw.created_utc)
            submissions_dict["selftext"].append(submission_praw.selftext)

        action = f"\t\t[Info] Finished fetching time: {time.time() - start_time: .2f}s"
        log_action(action)

    ### save data ###

        pd.DataFrame(submissions_dict).to_csv(subredditdirpath + '/' + submissions_csv_path, index=False)

        action = f"\t\t[Info] Found submissions: {pd.DataFrame(submissions_dict).shape[0]}"
        log_action(action)

        action = f"\t\t[Info] Elapsed time: {time.time() - start_time: .2f}s"
        log_action(action)