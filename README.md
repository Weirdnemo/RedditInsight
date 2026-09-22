# RedditInsight

A Streamlit app that pulls a Reddit thread's comments and runs sentiment,
emotion, and discussion-network analysis on them — sentiment distribution,
polarity vs. subjectivity, emotion breakdown, an activity heatmap, a
commenter interaction network, word clouds, and a sortable comment table.

![status](https://img.shields.io/badge/status-active-brightgreen)

## Features

- **Sentiment analysis** — TextBlob polarity/subjectivity plus VADER compound scores
- **Emotion detection** — keyword-based joy / sadness / anger / fear / surprise classification
- **Discussion network** — graphs commenters whose comments land close together in time with similar sentiment
- **Word clouds** — overall, and split by positive/negative comments
- **Timeline heatmap** — sentiment by day of week and hour
- **CSV / JSON / Excel export**
- **Offline demo mode** — analyze a bundled sample thread with zero API calls, for presentations where you can't rely on live internet or Reddit access

## Quick start

```bash
pip install -r requirements.txt
streamlit run reddit_insight.py
```

Open the app and switch **Data source → Sample dataset (offline demo)** in
the sidebar to try it immediately, with no Reddit account or credentials
needed. Two sample threads ship with the project.

To analyze real threads, you'll need Reddit API credentials — see
**[SETUP.md](SETUP.md)** for the full walkthrough (creating a Reddit app,
storing credentials safely, and capturing your own sample data for demos).

## Project structure

```
reddit_insight.py              the Streamlit app
reddit_core.py                 shared sentiment/emotion logic + credential lookup
fetch_sample_data.py           capture a real thread into sample_data/ (needs credentials)
generate_synthetic_samples.py  regenerate the bundled placeholder sample threads
sample_data/                   bundled sample threads used by offline demo mode
.streamlit/secrets.toml.example  template for your Reddit API credentials
requirements.txt
SETUP.md                       detailed setup + credentials guide
```

`sample_data/` and `.streamlit/` must stay as subfolders next to
`reddit_insight.py` — if they get flattened when downloading/copying files
individually, the app won't find them and "Sample dataset" won't appear as
an option in the sidebar.

## Requirements

Python 3.10+ and the packages in `requirements.txt` (Streamlit, PRAW,
pandas, Plotly, NLTK, TextBlob, WordCloud, NetworkX, openpyxl).

## Notes

- Live mode calls the official Reddit Data API via PRAW in read-only mode — no username/password needed, just a client ID and secret from a "script" app.
- Credentials are read from `.streamlit/secrets.toml` or environment variables only — never hardcode them in source files.
- Reddit's API access policy has tightened significantly over 2025–2026 (self-service app registration is now gated behind manual approval for new apps). Demo mode exists partly because of this — see SETUP.md for context.
