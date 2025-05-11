# 📊 Sentilytics for Reddit

A powerful sentiment analysis tool for Reddit threads that provides real-time insights into comment sentiments using natural language processing and interactive visualizations.

## ✨ Features

- **Real-time Sentiment Analysis**: Analyze the sentiment of any Reddit thread's comments
- **Interactive Visualizations**:
  - Sentiment distribution charts
  - Timeline analysis
  - Word clouds for positive and negative comments
  - Interactive comment explorer
- **Advanced Text Processing**:
  - Clean text analysis
  - Removal of common words and technical artifacts
  - Sentiment categorization
- **User-friendly Interface**:
  - Modern dark theme
  - Responsive design
  - Easy-to-use input system

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Reddit API credentials (Client ID and Client Secret)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reddit-sents.git
cd reddit-sents
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data (this will happen automatically on first run, but you can also do it manually):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

Note: If you encounter any NLTK data download issues, you can try downloading all NLTK data (this will take more space but ensures all required resources are available):
```python
import nltk
nltk.download('all')
```

### Configuration

1. Get your Reddit API credentials:
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Fill in the required information
   - Note down your Client ID and Client Secret

2. Update the Reddit API credentials in `reddit_sents.py`:
```python
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="my-reddit-scraper"
)
```

## 🎮 Usage

1. Start the Streamlit app:
```bash
streamlit run reddit_sents.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Enter a Reddit thread URL in the input field

4. Explore the analysis:
   - View sentiment distribution
   - Check the timeline of comments
   - Explore word clouds
   - Browse through individual comments

## 📊 Visualization Features

### Sentiment Analysis Tab
- Interactive histogram of sentiment distribution
- Gauge chart showing average sentiment
- Color-coded indicators for most positive and negative sentiments

### Timeline Tab
- Sentiment trend over time
- Interactive scatter plot of comment scores vs. sentiment

### Word Cloud Tab
- Overall word cloud of comments
- Separate word clouds for positive and negative comments
- Cleaned text analysis (no URLs, common words, or technical artifacts)

### Comments Tab
- Interactive table of comments
- Sentiment categorization
- Score and author information

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - Web application framework
- [PRAW](https://praw.readthedocs.io/) - Reddit API wrapper
- [TextBlob](https://textblob.readthedocs.io/) - Natural language processing
- [Plotly](https://plotly.com/) - Interactive visualizations
- [WordCloud](https://github.com/amueller/word_cloud) - Word cloud generation
- [NLTK](https://www.nltk.org/) - Natural Language Toolkit

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Reddit API for providing access to thread data
- All the amazing open-source libraries that made this project possible

