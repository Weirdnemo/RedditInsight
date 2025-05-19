import streamlit as st
from typing import Optional, Dict, Any

# Define color scheme
colors = {
    'primary': '#FF4B4B',  # Red accent
    'secondary': '#262730',  # Dark background
    'success': '#00C851',
    'warning': '#FFAB00',
    'error': '#D32F2F',
    'text': '#FFFFFF',
    'text_secondary': '#BDBDBD',
    'background': '#0E1117'
}

def setup_page():
    """Set up the page configuration and theme"""
    st.set_page_config(
        page_title="RedditInsight",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="📊"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .reportview-container {
            background-color: #0E1117;
            color: #FFFFFF;
        }
        .sidebar .sidebar-content {
            background-color: #262730;
            color: #FFFFFF;
        }
        .stTextInput label {
            color: #FFFFFF;
        }
        .stTextInput input {
            background-color: #424242;
            border: 1px solid #666;
            color: #FFFFFF;
        }
        .stButton button {
            background-color: #FF4B4B;
            color: #FFFFFF;
        }
        .stButton button:hover {
            background-color: #FF1744;
        }
        .stMarkdown {
            color: #FFFFFF;
        }
        .stProgress {
            background-color: #424242;
        }
        .stProgress .stProgress__value {
            background-color: #FF4B4B;
        }
        </style>
    """, unsafe_allow_html=True)

def create_header(title: str, subtitle: str) -> None:
    """Create a styled header section"""
    st.markdown(
        f"""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3rem; margin-bottom: 1rem; color: {colors['primary']};'>📊 {title}</h1>
            <p style='font-size: 1.2rem; color: {colors['text_secondary']};'>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_input_section() -> Optional[str]:
    """Create a styled input section with better layout"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url = st.text_input(
            "🔗 Enter Reddit Thread URL",
            "",
            help="Paste a Reddit thread URL to analyze its comments",
            placeholder="https://www.reddit.com/r/..."
        )
    
    with col2:
        st.markdown("### ℹ️ How to use")
        st.markdown("""
        1. Find a Reddit thread you want to analyze
        2. Copy its URL
        3. Paste it in the input field
        4. Wait for the analysis to complete
        """)
    
    return url

def create_analysis_tabs(df: pd.DataFrame) -> None:
    """Create styled analysis tabs with better organization"""
    tab1, tab2, tab3 = st.tabs(["📊 Sentiment Analysis", "📈 Timeline Analysis", "☁️ Word Cloud"])
    
    with tab1:
        st.markdown("""
            <style>
            .sentiment-chart {
                background-color: #424242;
                padding: 1rem;
                border-radius: 8px;
            }
            </style>
            <div class='sentiment-chart'>
                <h3>Sentiment Distribution</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Add sentiment charts here
        
    with tab2:
        st.markdown("""
            <style>
            .timeline-chart {
                background-color: #424242;
                padding: 1rem;
                border-radius: 8px;
            }
            </style>
            <div class='timeline-chart'>
                <h3>Comment Timeline</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Add timeline charts here
        
    with tab3:
        st.markdown("""
            <style>
            .wordcloud-container {
                background-color: #424242;
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
            }
            </style>
            <div class='wordcloud-container'>
                <h3>Word Cloud Analysis</h3>
                <p style='color: #BDBDBD;'>Most common words in the comments (excluding common words, URLs, and special characters)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add word cloud here

def show_loading_state(message: str) -> None:
    """Show a styled loading state"""
    st.markdown(
        f"""
        <div style='text-align: center; padding: 2rem;'>
            <div style='background-color: #424242; padding: 1rem; border-radius: 8px;'>
                <p style='color: {colors['text']};'>{message}</p>
                <div style='margin-top: 1rem;'>
                    <div class='progress-bar' style='background-color: #666; height: 4px; border-radius: 2px;'>
                        <div class='progress' style='background-color: {colors['primary']}; height: 100%; width: 0%;'></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_error(message: str) -> None:
    """Show a styled error message"""
    st.markdown(
        f"""
        <div style='text-align: center; padding: 2rem;'>
            <div style='background-color: #424242; padding: 1rem; border-radius: 8px;'>
                <p style='color: {colors['error']}; font-size: 1.2rem;'>❌ {message}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_warning(message: str) -> None:
    """Show a styled warning message"""
    st.markdown(
        f"""
        <div style='text-align: center; padding: 2rem;'>
            <div style='background-color: #424242; padding: 1rem; border-radius: 8px;'>
                <p style='color: {colors['warning']}; font-size: 1.2rem;'>⚠️ {message}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
