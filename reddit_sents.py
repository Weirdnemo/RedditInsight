import streamlit as st
import praw
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')  # Required for TextBlob
    nltk.download('wordnet')  # Required for TextBlob

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


# Function to analyze sentiment with timestamps
def analyze_sentiment(comments):
    sentiments = []
    for comment in comments:
        blob = TextBlob(comment.body)
        polarity = blob.sentiment.polarity
        sentiments.append({
            "comment": comment.body,
            "sentiment": polarity,
            "created_utc": datetime.fromtimestamp(comment.created_utc),
            "score": comment.score,
            "author": str(comment.author)
        })
    return pd.DataFrame(sentiments)

# Function to clean text for word cloud
def clean_text_for_wordcloud(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    # Add custom stopwords
    custom_stopwords = {'http', 'https', 'www', 'com', 'org', 'net', 'imgur', 'jpg', 'png', 'gif', 'webp', 'amp', 'reddit', 'redd', 'edit', 'deleted', 'removed'}
    stop_words.update(custom_stopwords)
    
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(filtered_tokens)

# Function to generate word cloud with cleaned text
def generate_wordcloud(text):
    # Clean the text
    cleaned_text = clean_text_for_wordcloud(text)
    
    wordcloud = WordCloud(
        background_color='#0e1117',
        colormap='Reds',
        width=800,
        height=400,
        max_words=100,
        min_font_size=10,
        max_font_size=100,
        random_state=42
    ).generate(cleaned_text)
    return wordcloud

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

                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment Analysis", "📈 Timeline", "☁️ Word Cloud", "📝 Comments"])

                with tab1:
                    # Sentiment Distribution with Plotly
                    fig_dist = px.histogram(
                        df,
                        x="sentiment",
                        nbins=30,
                        color_discrete_sequence=['#ff4b4b'],
                        title="Sentiment Distribution of Comments"
                    )
                    fig_dist.update_layout(
                        template="plotly_dark",
                        showlegend=False,
                        xaxis_title="Sentiment Score",
                        yaxis_title="Number of Comments"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    # Summary Statistics in a more visual way
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fig_avg = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=df['sentiment'].mean(),
                            title={'text': "Average Sentiment"},
                            gauge={'axis': {'range': [-1, 1]},
                                  'bar': {'color': "#ff4b4b"}}
                        ))
                        fig_avg.update_layout(height=200)
                        st.plotly_chart(fig_avg, use_container_width=True)

                    with col2:
                        fig_max = go.Figure(go.Indicator(
                            mode="number",
                            value=df['sentiment'].max(),
                            title={'text': "Most Positive"},
                            number={'font': {'color': "#00ff00"}}
                        ))
                        fig_max.update_layout(height=200)
                        st.plotly_chart(fig_max, use_container_width=True)

                    with col3:
                        fig_min = go.Figure(go.Indicator(
                            mode="number",
                            value=df['sentiment'].min(),
                            title={'text': "Most Negative"},
                            number={'font': {'color': "#ff0000"}}
                        ))
                        fig_min.update_layout(height=200)
                        st.plotly_chart(fig_min, use_container_width=True)

                with tab2:
                    # Sentiment Timeline
                    df['hour'] = df['created_utc'].dt.hour
                    timeline_data = df.groupby('hour')['sentiment'].mean().reset_index()
                    
                    fig_timeline = px.line(
                        timeline_data,
                        x='hour',
                        y='sentiment',
                        title='Sentiment Trend Over Time',
                        markers=True
                    )
                    fig_timeline.update_layout(
                        template="plotly_dark",
                        xaxis_title="Hour of Day",
                        yaxis_title="Average Sentiment",
                        showlegend=False
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

                    # Sentiment vs Score Scatter Plot
                    fig_scatter = px.scatter(
                        df,
                        x='sentiment',
                        y='score',
                        color='sentiment',
                        title='Comment Score vs Sentiment',
                        color_continuous_scale='RdBu'
                    )
                    fig_scatter.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_scatter, use_container_width=True)

                with tab3:
                    # Generate and display word cloud
                    st.markdown("### ☁️ Word Cloud Analysis")
                    st.markdown("Most common words in the comments (excluding common words, URLs, and special characters)")
                    
                    all_text = ' '.join(df['comment'].astype(str))
                    wordcloud = generate_wordcloud(all_text)
                    
                    fig_wc, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig_wc)

                    # Top words by sentiment
                    st.markdown("### 🔝 Most Common Words by Sentiment")
                    
                    # Get positive and negative comments
                    positive_comments = df[df['sentiment'] > 0.3]['comment'].astype(str)
                    negative_comments = df[df['sentiment'] < -0.3]['comment'].astype(str)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Positive Comments")
                        if not positive_comments.empty:
                            positive_words = ' '.join(positive_comments)
                            wordcloud_pos = generate_wordcloud(positive_words)
                            fig_pos, ax = plt.subplots(figsize=(5, 3))
                            ax.imshow(wordcloud_pos, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig_pos)
                        else:
                            st.info("No strongly positive comments found")
                    
                    with col2:
                        st.markdown("#### Negative Comments")
                        if not negative_comments.empty:
                            negative_words = ' '.join(negative_comments)
                            wordcloud_neg = generate_wordcloud(negative_words)
                            fig_neg, ax = plt.subplots(figsize=(5, 3))
                            ax.imshow(wordcloud_neg, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig_neg)
                        else:
                            st.info("No strongly negative comments found")

                with tab4:
                    # Interactive comments table with sentiment coloring
                    st.markdown("### 📝 Comments Analysis")
                    
                    # Add sentiment category
                    df['sentiment_category'] = pd.cut(
                        df['sentiment'],
                        bins=[-1, -0.3, 0.3, 1],
                        labels=['Negative', 'Neutral', 'Positive']
                    )
                    
                    # Display interactive dataframe
                    st.dataframe(
                        df[['comment', 'sentiment', 'sentiment_category', 'score', 'author']]
                        .sort_values('score', ascending=False)
                        .head(20),
                        use_container_width=True,
                        height=400
                    )

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

    else:
        st.warning("⚠️ Invalid Reddit URL. Please enter a valid thread URL.")
