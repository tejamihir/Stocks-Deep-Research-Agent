import praw
import pandas as pd
from datetime import datetime
from typing import Optional
import re
from pathlib import Path
from transformers import pipeline

# -------------------- CONFIG --------------------
REDDIT_CLIENT_ID = "CAakGtb6JoXA5PQhoh_EBg"
REDDIT_CLIENT_SECRET = "_sFuQ8Fyk5xN2v8PmIFtrvJ3hNBMBQ"
REDDIT_USER_AGENT = "reddit-feedback-scraper by u/tejamihir"

SUBREDDITS = ["wallstreetbets", "investing", 'stocks']
LIMIT = 1000  # number of posts to fetch per subreddit
# Max comments to collect per post (None for all)
MAX_COMMENTS_PER_POST = 200
# ------------------------------------------------

def get_reddit_instance():
    """Authenticate and return a Reddit instance."""
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

def fetch_posts(reddit, subreddits, limit=100):
    """Fetch latest posts from given subreddits."""
    posts = []
    for sub in subreddits:
        print(f"Fetching from r/{sub}...")
        subreddit = reddit.subreddit(sub)
        for post in subreddit.hot(limit=limit):
            posts.append({
                "subreddit": sub,
                "title": post.title,
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": datetime.utcfromtimestamp(post.created_utc),
                "url": post.url,
                "text": post.selftext,
                "id": post.id
            })
    return pd.DataFrame(posts)

def fetch_comments_for_posts(reddit, posts_df: pd.DataFrame, max_per_post: Optional[int] = None) -> pd.DataFrame:
    """Fetch comments for each post in posts_df.

    posts_df must contain columns: ['subreddit', 'id', 'title']
    """
    comments: list[dict] = []
    if posts_df.empty:
        return pd.DataFrame(comments)
    print("Fetching comments")
    for _, row in posts_df.iterrows():
        post_id = row["id"]
        post_title = row["title"]
        subreddit = row["subreddit"]
        submission = reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)
        collected = 0
        for c in submission.comments.list():
            comments.append({
                "subreddit": subreddit,
                "post_id": post_id,
                "post_title": post_title,
                "comment_id": c.id,
                "parent_id": c.parent_id,
                "author": getattr(c.author, "name", None),
                "body": c.body,
                "score": c.score,
                "created_utc": datetime.utcfromtimestamp(c.created_utc),
                "is_submitter": getattr(c, "is_submitter", False)
            })
            collected += 1
            if max_per_post is not None and collected >= max_per_post:
                break
    return pd.DataFrame(comments)

# -------------------- SCRIPT EXECUTION --------------------
reddit = get_reddit_instance()
# df = fetch_posts(reddit, SUBREDDITS, LIMIT)

# Save under repo_root/data/reddit_output
repo_root = Path(__file__).resolve().parents[1]
out_path = repo_root / "data" / "reddit_output" / "reddit_feedbacks.csv"
# df.to_csv(out_path, index=False)
# print(f"✅ Saved {len(df)} posts to {out_path}")

# -------------------- TICKER EXTRACTION --------------------
# If df isn't defined above, try loading from the saved posts CSV
try:
    _df_defined = isinstance(df, pd.DataFrame)
except NameError:
    _df_defined = False

if not _df_defined:
    if out_path.exists():
        df = pd.read_csv(out_path)
    else:
        df = pd.DataFrame(columns=["title", "text"])  # empty fallback

if df.empty:
    print("No posts available to extract tickers.")
    result_df = pd.DataFrame(columns=["TICKER", "combined_text"])
else:
    # Combine title and text
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df['combined_text'] = df['title'] + ' ' + df['text']

    # Regex for tickers (1-5 uppercase letters)
    ticker_regex = r"\b[A-Z]{1,5}\b"

    def get_ticker(row):
        blacklist = {'AND', 'THE', 'FOR', 'WITH', 'ARE', 'BUT', 'HERE', 'DUE', 'BULL'}
        tickers = re.findall(ticker_regex, row['combined_text'])
        tickers = [t for t in tickers if t not in blacklist]
        return ','.join(tickers) if tickers else None

    df['TICKER'] = df.apply(get_ticker, axis=1)
    result_df = df[df['TICKER'].notnull()][['TICKER', 'combined_text']]

    tickers_out_path = repo_root / "data" / "reddit_output" / "filtered_reddit_ticker_posts.csv"
    result_df.to_csv(tickers_out_path, index=False)
    print(f"✅ Saved {len(result_df)} ticker-tagged rows to {tickers_out_path}")



# result_df should have a column 'combined_text' with the post text

# Use CPU and truncate long inputs to model max length (512 tokens)
if result_df.empty:
    print("No ticker-tagged posts to analyze for sentiment.")
else:
    sentiment_analyzer = pipeline(
        'sentiment-analysis',
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )
    texts = result_df['combined_text'].astype(str).tolist()
    sentiments = sentiment_analyzer(texts, truncation=True, max_length=512)
    result_df['sentiment'] = [s['label'].lower() for s in sentiments]
    sentiments_out_path = repo_root / "data" / "reddit_output" / "reddit_post_with_sentiment.csv"

    # Saving or reviewing the result
    result_df.to_csv(sentiments_out_path, index=False)
    print(result_df.head())



# Fetch and save comments
# comments_df = fetch_comments_for_posts(reddit, df, MAX_COMMENTS_PER_POST)
# comments_out_path = repo_root / "data" / "reddit_output" / "reddit_comments.csv"
# comments_df.to_csv(comments_out_path, index=False)
# print(f"✅ Saved {len(comments_df)} comments to {comments_out_path}")
