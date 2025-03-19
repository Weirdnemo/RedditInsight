import streamlit as st
import praw
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Streamlit Page Config
st.set_page_config(page_title="Reddit Sentiment Analyzer", layout="wide")

# Custom CSS for better UI
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stTextInput, .stTextArea {
        background-color: #1e1e1e;
        color: white;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Reddit API Authentication
reddit = praw.Reddit(
    client_id="PX4BjrFMX5ixQ1IDS73Eeg",
    client_secret="TNcR3UdwSFugluwOHgOO3tVeIr-15A",
    user_agent="my-reddit-scraper",
)

# App Title
st.title("📊 Reddit Thread Sentiment Analyzer")
st.markdown("Analyze the sentiment of Reddit threads in real-time.")

# Input Reddit Thread URL
url = st.text_input("🔗 Enter Reddit Thread URL", "")


# Function to extract thread ID
def extract_thread_id(url):
    try:
        parts = url.split("/")
        thread_id = parts[6]
        return thread_id
    except IndexError:
        return None


# Sentiment Analysis Function
def analyze_sentiment(comments):
    sentiments = []
    for comment in comments:
        blob = TextBlob(comment.body)
        polarity = blob.sentiment.polarity
        sentiments.append({"comment": comment.body, "sentiment": polarity})
    return pd.DataFrame(sentiments)


# Scrape & Analyze Sentiment
if url:
    thread_id = extract_thread_id(url)

    if thread_id:
        try:
            with st.spinner("Fetching comments and analyzing sentiment..."):
                submission = reddit.submission(id=thread_id)
                submission.comments.replace_more(limit=0)
                comments = submission.comments.list()

                df = analyze_sentiment(comments)

                # Display Data
                st.subheader("📝 Top Comments with Sentiment")
                st.dataframe(df.head(10))

                # Plot Sentiment
                st.subheader("📊 Sentiment Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(df["sentiment"], bins=20, kde=True, color="red", ax=ax)
                ax.set_xlabel("Sentiment Score")
                ax.set_ylabel("Frequency")
                ax.set_title("Sentiment Analysis of Comments")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

    else:
        st.warning("⚠️ Invalid Reddit URL. Please enter a valid thread URL.")
