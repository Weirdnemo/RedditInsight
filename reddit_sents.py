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
import json
from io import StringIO

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
    
    # Remove common words and short words
    common_words = {'http', 'https', 'www', 'com', 'org', 'net', 'imgur', 'jpg', 'png', 'gif', 'webp', 'amp', 'reddit', 'redd', 'edit', 'deleted', 'removed', 'the', 'and', 'that', 'this', 'but', 'they', 'have', 'from', 'what', 'when', 'where', 'which', 'who', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'}
    words = text.split()
    filtered_words = [word for word in words if word not in common_words and len(word) > 2]
    
    return ' '.join(filtered_words)

# Function to generate word cloud
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

# Function to export data
def export_data(df, format='csv'):
    if format == 'csv':
        return df.to_csv(index=False)
    elif format == 'json':
        return df.to_json(orient='records', indent=2)
    elif format == 'excel':
        output = StringIO()
        df.to_excel(output, index=False)
        return output.getvalue()

# Streamlit Page Config
st.set_page_config(
    page_title="Sentilytics for reddit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Theme Selection
    theme = st.radio(
        "Choose Theme",
        ["Dark", "Light"],
        index=0
    )
    
    # Analysis Settings
    st.subheader("Analysis Settings")
    min_comments = st.slider(
        "Minimum Comments to Analyze",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )
    
    sentiment_threshold = st.slider(
        "Sentiment Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Threshold for categorizing comments as positive/negative"
    )
    
    # Export Options
    st.subheader("Export Data")
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "JSON", "Excel"]
    )
    
    if st.button("Export Analysis"):
        if 'df' in locals():
            data = export_data(df, export_format.lower())
            st.download_button(
                label=f"Download {export_format}",
                data=data,
                file_name=f"sentiment_analysis.{export_format.lower()}",
                mime=f"text/{export_format.lower()}"
            )

# Custom CSS based on theme
if theme == "Dark":
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
            border-radius: 8px;
            border: 1px solid #2e2e2e;
            padding: 8px;
        }
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
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #ff4b4b;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .dataframe {
            background-color: #1e1e1e;
            border-radius: 8px;
            padding: 1rem;
        }
        .stPlot {
            background-color: #1e1e1e;
            border-radius: 8px;
            padding: 1rem;
        }
        .stSpinner {
            color: #ff4b4b;
        }
        .stAlert {
            border-radius: 8px;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: #ffffff;
            color: #262730;
        }
        .stTextInput, .stTextArea {
            background-color: #f0f2f6;
            color: #262730;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            padding: 8px;
        }
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
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #ff4b4b;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        .dataframe {
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 1rem;
        }
        .stPlot {
            background-color: #f0f2f6;
            border-radius: 8px;
            padding: 1rem;
        }
        .stSpinner {
            color: #ff4b4b;
        }
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
            "author": str(comment.author),
            "length": len(comment.body)
        })
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

                if len(comments) < min_comments:
                    st.warning(f"⚠️ This thread has fewer comments than the minimum threshold ({min_comments}). Analysis might not be representative.")
                
                df = analyze_sentiment(comments)

                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Sentiment Analysis", 
                    "📈 Timeline", 
                    "☁️ Word Cloud", 
                    "📝 Comments",
                    "📈 Statistics"
                ])

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
                        template="plotly_dark" if theme == "Dark" else "plotly_white",
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
                        template="plotly_dark" if theme == "Dark" else "plotly_white",
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
                    fig_scatter.update_layout(
                        template="plotly_dark" if theme == "Dark" else "plotly_white"
                    )
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
                    positive_comments = df[df['sentiment'] > sentiment_threshold]['comment'].astype(str)
                    negative_comments = df[df['sentiment'] < -sentiment_threshold]['comment'].astype(str)
                    
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
                        bins=[-1, -sentiment_threshold, sentiment_threshold, 1],
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

                with tab5:
                    st.markdown("### 📊 Comment Statistics")
                    
                    # Basic Statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Comments", len(df))
                        st.metric("Average Comment Length", f"{df['length'].mean():.0f} characters")
                        st.metric("Most Active Hour", df['created_utc'].dt.hour.mode().iloc[0])
                    
                    with col2:
                        st.metric("Average Score", f"{df['score'].mean():.1f}")
                        st.metric("Highest Scoring Comment", df['score'].max())
                        st.metric("Unique Authors", df['author'].nunique())
                    
                    # Comment Length Distribution
                    fig_length = px.histogram(
                        df,
                        x="length",
                        title="Comment Length Distribution",
                        nbins=30
                    )
                    fig_length.update_layout(
                        template="plotly_dark" if theme == "Dark" else "plotly_white",
                        xaxis_title="Comment Length (characters)",
                        yaxis_title="Number of Comments"
                    )
                    st.plotly_chart(fig_length, use_container_width=True)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

    else:
        st.warning("⚠️ Invalid Reddit URL. Please enter a valid thread URL.")
