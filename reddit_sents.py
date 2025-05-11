import streamlit as st
import praw
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Streamlit Page Config
st.set_page_config(
    page_title="Sentilytics for reddit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown(
    """
    <style>
    /* Main background and text colors */
    body {
        background-color: #0e1117;
        color: white;
    }
    
    /* Input fields styling */
    .stTextInput, .stTextArea {
        background-color: #1e1e1e;
        color: white;
        border-radius: 8px;
        border: 1px solid #2e2e2e;
        padding: 8px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #ff6b6b;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 75, 75, 0.2);
    }
    
    /* Headers styling */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ff4b4b;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Plot styling */
    .stPlot {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Spinner styling */
    .stSpinner {
        color: #ff4b4b;
    }
    
    /* Error and warning messages */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
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

# App Title with enhanced styling
st.markdown(
    """
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>📊 Sentilytics for reddit</h1>
        <p style='font-size: 1.2rem; color: #888;'>Analyze the sentiment of Reddit threads in real-time</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Input Reddit Thread URL with enhanced styling
    url = st.text_input(
        "🔗 Enter Reddit Thread URL",
        "",
        help="Paste a Reddit thread URL to analyze its comments"
    )

with col2:
    st.markdown("### ℹ️ How to use")
    st.markdown("""
    1. Find a Reddit thread you want to analyze
    2. Copy its URL
    3. Paste it in the input field
    4. Wait for the analysis to complete
    """)

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
            with st.spinner("🔍 Fetching comments and analyzing sentiment..."):
                submission = reddit.submission(id=thread_id)
                submission.comments.replace_more(limit=0)
                comments = submission.comments.list()

                df = analyze_sentiment(comments)

                # Create two columns for results
                col1, col2 = st.columns(2)

                with col1:
                    # Display Data with enhanced styling
                    st.markdown("### 📝 Top Comments with Sentiment")
                    st.dataframe(
                        df.head(10),
                        use_container_width=True,
                        height=400
                    )

                with col2:
                    # Plot Sentiment with enhanced styling
                    st.markdown("### 📊 Sentiment Distribution")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.set_style("darkgrid")
                    sns.histplot(df["sentiment"], bins=20, kde=True, color="#ff4b4b", ax=ax)
                    ax.set_xlabel("Sentiment Score")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Sentiment Analysis of Comments")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                # Add summary statistics
                st.markdown("### 📈 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Sentiment", f"{df['sentiment'].mean():.2f}")
                with col2:
                    st.metric("Most Positive", f"{df['sentiment'].max():.2f}")
                with col3:
                    st.metric("Most Negative", f"{df['sentiment'].min():.2f}")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

    else:
        st.warning("⚠️ Invalid Reddit URL. Please enter a valid thread URL.")
