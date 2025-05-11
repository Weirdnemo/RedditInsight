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
import networkx as nx
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
vader = SentimentIntensityAnalyzer()

# Function to clean text for word cloud
@st.cache_data
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
@st.cache_data
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
@st.cache_data
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
    page_title="RedditInsight",
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
try:
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID", "PX4BjrFMX5ixQ1IDS73Eeg"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET", "TNcR3UdwSFugluwOHgOO3tVeIr-15A"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "my-reddit-scraper"),
    )
    # Test API connection
    reddit.user.me()
except Exception as e:
    st.error("❌ Error connecting to Reddit API. Please check your credentials and try again.")
    st.stop()

# Function to handle API rate limits
def handle_rate_limit(func):
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e):  # Rate limit error
                    if attempt < max_retries - 1:
                        st.warning(f"⚠️ Rate limit reached. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        st.error("❌ Rate limit exceeded. Please try again later.")
                        st.stop()
                else:
                    raise e
    return wrapper

@handle_rate_limit
def fetch_submission(thread_id):
    return reddit.submission(id=thread_id)

@handle_rate_limit
def fetch_comments(submission):
    submission.comments.replace_more(limit=0)
    return submission.comments.list()

# App Title with enhanced styling
st.markdown(
    """
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>📊 RedditInsight</h1>
        <p style='font-size: 1.2rem; color: #888;'>Deep dive into Reddit discussions with advanced sentiment analysis</p>
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

# Function to analyze emotions
@st.cache_data
def analyze_emotions(text):
    # Basic emotion keywords
    emotions = {
        'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'loved', 'best'],
        'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'terrible', 'awful', 'hate', 'hated', 'worst'],
        'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'rage', 'hate', 'hated'],
        'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'nervous'],
        'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected']
    }
    
    text = text.lower()
    emotion_scores = {emotion: 0 for emotion in emotions}
    
    for emotion, keywords in emotions.items():
        for keyword in keywords:
            if keyword in text:
                emotion_scores[emotion] += 1
    
    # Get the dominant emotion
    if sum(emotion_scores.values()) > 0:
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
    else:
        dominant_emotion = 'neutral'
    
    return emotion_scores, dominant_emotion

# Function to analyze sentiment with enhanced features
@st.cache_data
def analyze_sentiment(comments):
    sentiments = []
    for comment in comments:
        # TextBlob analysis
        blob = TextBlob(comment.body)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # VADER analysis
        vader_scores = vader.polarity_scores(comment.body)
        
        # Emotion analysis
        emotion_scores, dominant_emotion = analyze_emotions(comment.body)
        
        sentiments.append({
            "comment": comment.body,
            "sentiment": polarity,
            "subjectivity": subjectivity,
            "vader_compound": vader_scores['compound'],
            "vader_pos": vader_scores['pos'],
            "vader_neg": vader_scores['neg'],
            "vader_neu": vader_scores['neu'],
            "dominant_emotion": dominant_emotion,
            "emotion_scores": emotion_scores,
            "created_utc": datetime.fromtimestamp(comment.created_utc),
            "score": comment.score,
            "author": str(comment.author),
            "length": len(comment.body)
        })
    return pd.DataFrame(sentiments)

# Function to create comment network
@st.cache_data
def create_comment_network(df):
    G = nx.Graph()
    
    # Add nodes (authors)
    authors = df['author'].unique()
    for author in authors:
        G.add_node(author)
    
    # Add edges based on similar sentiment and time proximity
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if (abs(df.iloc[i]['created_utc'] - df.iloc[j]['created_utc']).total_seconds() < 3600 and  # Within 1 hour
                abs(df.iloc[i]['sentiment'] - df.iloc[j]['sentiment']) < 0.3):  # Similar sentiment
                G.add_edge(df.iloc[i]['author'], df.iloc[j]['author'])
    
    return G

# Scrape & Analyze Sentiment
if url:
    thread_id = extract_thread_id(url)

    if thread_id:
        try:
            with st.spinner("🔍 Fetching comments and analyzing sentiment..."):
                # Add progress bar
                progress_bar = st.progress(0)
                
                # Fetch submission with rate limit handling
                submission = fetch_submission(thread_id)
                progress_bar.progress(20)
                
                # Get comments with rate limit handling
                comments = fetch_comments(submission)
                progress_bar.progress(60)

                if len(comments) < min_comments:
                    st.warning(f"⚠️ This thread has fewer comments than the minimum threshold ({min_comments}). Analysis might not be representative.")
                
                # Analyze sentiment
                df = analyze_sentiment(comments)
                progress_bar.progress(80)

                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Sentiment Analysis", 
                    "📈 Timeline", 
                    "☁️ Word Cloud", 
                    "📝 Comments",
                    "📈 Statistics"
                ])
                
                progress_bar.progress(100)
                st.success("✅ Analysis complete!")

                with tab1:
                    # Enhanced Sentiment Analysis
                    st.markdown("### 📊 Advanced Sentiment Analysis")
                    
                    try:
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
                    except Exception as e:
                        st.error(f"Error creating sentiment distribution: {str(e)}")

                    try:
                        # Subjectivity vs Polarity Scatter Plot
                        fig_subj = px.scatter(
                            df,
                            x="sentiment",
                            y="subjectivity",
                            color="dominant_emotion",
                            title="Subjectivity vs Polarity by Emotion",
                            size="score",
                            hover_data=["comment"]
                        )
                        fig_subj.update_layout(
                            template="plotly_dark" if theme == "Dark" else "plotly_white"
                        )
                        st.plotly_chart(fig_subj, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating scatter plot: {str(e)}")

                    try:
                        # Emotion Distribution
                        emotion_counts = df['dominant_emotion'].value_counts()
                        fig_emotion = px.pie(
                            values=emotion_counts.values,
                            names=emotion_counts.index,
                            title="Distribution of Emotions"
                        )
                        fig_emotion.update_layout(
                            template="plotly_dark" if theme == "Dark" else "plotly_white"
                        )
                        st.plotly_chart(fig_emotion, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating emotion distribution: {str(e)}")

                    # Summary Statistics
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
                        fig_subj_avg = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=df['subjectivity'].mean(),
                            title={'text': "Average Subjectivity"},
                            gauge={'axis': {'range': [0, 1]},
                                  'bar': {'color': "#4b7bff"}}
                        ))
                        fig_subj_avg.update_layout(height=200)
                        st.plotly_chart(fig_subj_avg, use_container_width=True)

                    with col3:
                        fig_vader = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=df['vader_compound'].mean(),
                            title={'text': "VADER Sentiment"},
                            gauge={'axis': {'range': [-1, 1]},
                                  'bar': {'color': "#4bff4b"}}
                        ))
                        fig_vader.update_layout(height=200)
                        st.plotly_chart(fig_vader, use_container_width=True)

                with tab2:
                    # Enhanced Timeline Analysis
                    st.markdown("### 📈 Advanced Timeline Analysis")
                    
                    # Create time-based features
                    df['hour'] = df['created_utc'].dt.hour
                    df['day'] = df['created_utc'].dt.day_name()
                    
                    # Sentiment Heatmap
                    pivot_data = df.pivot_table(
                        values='sentiment',
                        index='day',
                        columns='hour',
                        aggfunc='mean'
                    )
                    
                    fig_heatmap = px.imshow(
                        pivot_data,
                        title="Sentiment Heatmap by Day and Hour",
                        color_continuous_scale='RdBu'
                    )
                    fig_heatmap.update_layout(
                        template="plotly_dark" if theme == "Dark" else "plotly_white",
                        xaxis_title="Hour of Day",
                        yaxis_title="Day of Week"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Emotion Timeline
                    emotion_timeline = df.groupby('hour')['dominant_emotion'].apply(
                        lambda x: x.value_counts().index[0]
                    ).reset_index()
                    
                    fig_emotion_timeline = px.line(
                        emotion_timeline,
                        x='hour',
                        y='dominant_emotion',
                        title='Dominant Emotion Over Time',
                        markers=True
                    )
                    fig_emotion_timeline.update_layout(
                        template="plotly_dark" if theme == "Dark" else "plotly_white",
                        xaxis_title="Hour of Day",
                        yaxis_title="Dominant Emotion"
                    )
                    st.plotly_chart(fig_emotion_timeline, use_container_width=True)

                    # Comment Network
                    st.markdown("### 🤝 Comment Interaction Network")
                    G = create_comment_network(df)
                    
                    # Create network visualization
                    pos = nx.spring_layout(G)
                    fig_network = go.Figure()
                    
                    # Add edges
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    fig_network.add_trace(go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    ))
                    
                    # Add nodes
                    node_x = []
                    node_y = []
                    node_text = []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(node)
                    
                    fig_network.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers',
                        hoverinfo='text',
                        text=node_text,
                        marker=dict(
                            showscale=True,
                            colorscale='YlOrRd',
                            size=10,
                            colorbar=dict(
                                thickness=15,
                                title='Node Connections',
                                xanchor='left'
                            )
                        )
                    ))
                    
                    fig_network.update_layout(
                        title='Comment Interaction Network',
                        template="plotly_dark" if theme == "Dark" else "plotly_white",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40)
                    )
                    st.plotly_chart(fig_network, use_container_width=True)

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
