import streamlit as st
import praw
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# --- Configure Reddit API ---
# You must create a Reddit app (https://www.reddit.com/prefs/apps) to get these credentials.
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',  # Replace with your client id
                     client_secret='YOUR_CLIENT_SECRET',  # Replace with your client secret
                     user_agent='YOUR_USER_AGENT')  # Replace with a descriptive user agent


# --- Functions for scraping and sentiment analysis ---
def fetch_reddit_thread(url):
    """Fetches the thread title and all comments from a Reddit thread URL."""
    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=0)  # Remove 'MoreComments' objects
    comments = [comment.body for comment in submission.comments.list()]
    return submission.title, comments


def analyze_sentiment(text):
    """Returns the sentiment scores for a given text."""
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def analyze_comments(comments):
    """Analyzes each comment and returns a list of sentiment scores."""
    sentiments = []
    for comment in comments:
        scores = analyze_sentiment(comment)
        sentiments.append(scores)
    return sentiments


# --- Streamlit Frontend ---
def main():
    st.title("Reddit Thread Sentiment Analyzer")
    st.markdown(
        "Enter the URL of a Reddit thread below and the app will scrape the comments and analyze their sentiment.")

    reddit_url = st.text_input("Reddit Thread URL", "")

    if reddit_url:
        try:
            title, comments = fetch_reddit_thread(reddit_url)
            st.subheader("Thread Title")
            st.write(title)
            st.write(f"**Comments Scraped:** {len(comments)}")

            # Perform sentiment analysis on the comments
            sentiments = analyze_comments(comments)
            df = pd.DataFrame(sentiments)

            st.subheader("Sentiment Scores for Comments")
            st.dataframe(df)

            # Compute average sentiment scores
            avg_scores = df.mean()
            st.subheader("Average Sentiment Scores")
            st.write(avg_scores)
            st.bar_chart(avg_scores)
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
