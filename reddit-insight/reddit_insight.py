"""
RedditInsight — Reddit thread sentiment & discussion analyzer.

Setup:
    1. Create a Reddit "script" app at https://www.reddit.com/prefs/apps
    2. Provide credentials via .streamlit/secrets.toml (recommended) or
       environment variables REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT
    3. streamlit run reddit_insight.py

See SETUP.md for the full walkthrough.
"""

import json
import os
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import praw
import prawcore
import streamlit as st
from wordcloud import WordCloud

from reddit_core import EMOTION_KEYWORDS, STOPWORDS_EXTRA, analyze_comment, get_credential

# --------------------------------------------------------------------------
# One-time setup
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="RedditInsight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Theme / styling
# --------------------------------------------------------------------------

ACCENT = "#FF4500"  # Reddit orange, used sparingly as the single brand accent


def inject_css(dark: bool) -> None:
    bg = "#0b0c10" if dark else "#f7f7f8"
    surface = "#15171c" if dark else "#ffffff"
    surface_alt = "#1c1f26" if dark else "#f0f1f3"
    text = "#e7e7ea" if dark else "#1a1a1e"
    subtext = "#9a9ba3" if dark else "#63646c"
    border = "#262933" if dark else "#e4e4e7"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}
        .stApp {{
            background-color: {bg};
            color: {text};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {surface};
            border-right: 1px solid {border};
        }}
        .hero {{
            text-align: center;
            padding: 2.5rem 0 2rem 0;
        }}
        .hero h1 {{
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.4rem;
            color: {text};
        }}
        .hero h1 span {{ color: {ACCENT}; }}
        .hero p {{
            font-size: 1.05rem;
            color: {subtext};
            margin: 0;
        }}
        .card {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
        }}
        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
        }}
        .status-ok {{ background: rgba(46, 204, 113, 0.15); color: #2ecc71; }}
        .status-bad {{ background: rgba(231, 76, 60, 0.15); color: #e74c3c; }}

        .stTextInput input, .stTextArea textarea {{
            background-color: {surface_alt};
            color: {text};
            border-radius: 10px;
            border: 1px solid {border};
        }}
        .stButton>button {{
            background-color: {ACCENT};
            color: white;
            border-radius: 10px;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
            border: none;
            transition: all 0.15s ease;
        }}
        .stButton>button:hover {{
            filter: brightness(1.08);
            transform: translateY(-1px);
        }}
        .stMarkdown h3 {{
            font-weight: 700;
            letter-spacing: -0.01em;
        }}
        div[data-testid="stMetric"] {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }}
        code {{ color: {ACCENT}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Reddit client
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_reddit_client():
    """Build a read-only PRAW client. Returns (client, error_message)."""
    client_id = get_credential("REDDIT_CLIENT_ID")
    client_secret = get_credential("REDDIT_CLIENT_SECRET")
    user_agent = get_credential("REDDIT_USER_AGENT", "reddit-insight/1.0")

    if not client_id or not client_secret:
        return None, "missing_credentials"

    try:
        client = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            check_for_updates=False,
        )
        client.read_only = True
        # Lightweight call that actually validates app-only credentials
        # (reddit.user.me() does NOT work for script apps without a user login).
        _ = client.subreddit("announcements").id
        return client, None
    except prawcore.exceptions.ResponseException as e:
        return None, f"Reddit rejected the credentials ({e})."
    except Exception as e:
        return None, f"Could not connect to Reddit: {e}"


def show_setup_guide() -> None:
    with st.expander("🔑 Reddit API isn't configured yet — click for setup steps", expanded=True):
        st.markdown(
            """
1. Go to **[reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)** and click **"create app"** (bottom left).
2. Fill it in:
   - **name**: anything, e.g. `reddit-insight`
   - **type**: select **script**
   - **redirect uri**: `http://localhost:8080` (required, unused for script apps)
3. After creating it, copy:
   - the string under the app name (that's your **client_id**)
   - the **secret** field (that's your **client_secret**)
4. Add them to `.streamlit/secrets.toml` in this project (see `secrets.toml.example`):
```toml
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "reddit-insight/1.0 by u/your_username"
```
   or export them as environment variables before running the app.
5. Never commit `secrets.toml` or paste real credentials into source files —
   `.gitignore` already excludes it in this project.
            """
        )


# --------------------------------------------------------------------------
# Reddit fetching (rate-limit aware, cached)
# --------------------------------------------------------------------------

def looks_like_reddit_url(url: str) -> bool:
    return bool(re.search(r"reddit\.com/", url, re.IGNORECASE))


def with_rate_limit_retry(func, *args, max_retries: int = 3, **kwargs):
    delay = 2
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except prawcore.exceptions.TooManyRequests:
            if attempt == max_retries - 1:
                raise
            st.toast(f"Rate limited — retrying in {delay}s…")
            time.sleep(delay)
            delay *= 2


@st.cache_data(ttl=600, show_spinner=False)
def fetch_thread_dataframe(_reddit, url: str) -> pd.DataFrame:
    """Fetch a submission's comments and run sentiment/emotion analysis.

    Cached for 10 minutes per URL so slider/UI tweaks don't re-hit the API.
    The leading underscore on `_reddit` tells Streamlit not to hash the client.
    Uses PRAW's own URL parsing (reddit.submission(url=...)) rather than a
    hand-rolled regex, since it correctly handles old.reddit.com, np.reddit.com,
    and other permalink variants that a simple pattern would miss.
    """
    submission = with_rate_limit_retry(_reddit.submission, url=url)
    submission.comments.replace_more(limit=0)
    comments = submission.comments.list()

    rows = []
    for c in comments:
        body = c.body or ""
        rows.append(
            {
                "comment": body,
                **analyze_comment(body),
                "created_utc": datetime.fromtimestamp(c.created_utc),
                "score": c.score,
                "author": str(c.author) if c.author else "[deleted]",
                "length": len(body),
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["title"] = submission.title
    df.attrs["subreddit"] = str(submission.subreddit)
    df.attrs["permalink"] = f"https://reddit.com{submission.permalink}"
    return df


# --------------------------------------------------------------------------
# Derived visuals
# --------------------------------------------------------------------------


def clean_text_for_wordcloud(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    words = [w for w in text.split() if w not in STOPWORDS_EXTRA and len(w) > 2]
    return " ".join(words)


def generate_wordcloud(text: str, dark: bool):
    cleaned = clean_text_for_wordcloud(text)
    if not cleaned.strip():
        return None
    return WordCloud(
        background_color="#0b0c10" if dark else "#ffffff",
        colormap="Oranges" if dark else "Reds",
        width=800,
        height=400,
        max_words=100,
        random_state=42,
    ).generate(cleaned)


def build_comment_network(df: pd.DataFrame, top_n: int = 150) -> nx.Graph:
    """Vectorized edge-building: connect authors whose comments landed within
    an hour of each other with similar sentiment. Capped to the top-N
    highest-scoring comments so this stays fast and the graph stays readable.
    """
    sub = df.nlargest(top_n, "score").reset_index(drop=True) if len(df) > top_n else df.reset_index(drop=True)

    times = sub["created_utc"].astype("int64").to_numpy() // 10**9
    sentiments = sub["sentiment"].to_numpy()
    authors = sub["author"].to_numpy()

    time_diff = np.abs(times[:, None] - times[None, :])
    sent_diff = np.abs(sentiments[:, None] - sentiments[None, :])
    mask = (time_diff < 3600) & (sent_diff < 0.3)
    np.fill_diagonal(mask, False)
    i_idx, j_idx = np.where(np.triu(mask))

    G = nx.Graph()
    G.add_nodes_from(authors)
    for i, j in zip(i_idx, j_idx):
        if authors[i] != authors[j]:
            G.add_edge(authors[i], authors[j])
    return G


def export_bytes(df: pd.DataFrame, fmt: str) -> tuple[bytes, str]:
    export_cols = [
        "comment", "sentiment", "subjectivity", "vader_compound", "dominant_emotion",
        "score", "author", "created_utc",
    ]
    slim = df[export_cols]
    if fmt == "csv":
        return slim.to_csv(index=False).encode("utf-8"), "text/csv"
    if fmt == "json":
        return slim.to_json(orient="records", indent=2, date_format="iso").encode("utf-8"), "application/json"
    if fmt == "excel":
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            slim.to_excel(writer, index=False, sheet_name="comments")
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    raise ValueError(fmt)


# --------------------------------------------------------------------------
# Offline sample datasets (demo mode)
# --------------------------------------------------------------------------
#
# For live showcases/demos, "Sample dataset" mode reads pre-fetched thread
# data from sample_data/*.json instead of calling Reddit at all — so a rate
# limit, an auth hiccup, or Reddit's API being Reddit's API mid-presentation
# can't take the demo down. Regenerate these files anytime with a real thread
# by running fetch_sample_data.py once while you have working credentials.

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"


def list_sample_datasets() -> dict:
    """Return {label: filepath} for every bundled sample thread."""
    if not SAMPLE_DIR.exists():
        return {}
    datasets = {}
    for path in sorted(SAMPLE_DIR.glob("*.json")):
        try:
            meta = json.loads(path.read_text()).get("meta", {})
            label = meta.get("title", path.stem)
            if meta.get("subreddit"):
                label = f"{label}  (r/{meta['subreddit']})"
        except Exception:
            label = path.stem
        datasets[label] = path
    return datasets


def load_sample_dataframe(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    df = pd.DataFrame(payload["comments"])
    df["created_utc"] = pd.to_datetime(df["created_utc"])
    meta = payload.get("meta", {})
    df.attrs["title"] = meta.get("title", path.stem)
    df.attrs["subreddit"] = meta.get("subreddit", "unknown")
    df.attrs["permalink"] = meta.get("permalink", "")
    df.attrs["is_sample"] = True
    return df


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

SAMPLE_DATASETS = list_sample_datasets()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    dark_mode = st.toggle("Dark theme", value=True)

    st.markdown("---")
    st.markdown("### 🔌 Data source")
    mode_options = ["Live Reddit thread"]
    if SAMPLE_DATASETS:
        mode_options.append("Sample dataset (offline demo)")
    data_mode = st.radio(
        "Mode", mode_options, label_visibility="collapsed",
        help="Sample mode reads a pre-fetched thread from disk — no Reddit API calls, so it works even if credentials or Reddit itself are having a bad day. Handy for live presentations.",
    )

    st.markdown("---")
    st.markdown("### 📥 Analysis")
    min_comments = st.slider("Flag threads with fewer comments than", 10, 1000, 100, 10)
    sentiment_threshold = st.slider(
        "Positive / negative cutoff", 0.0, 1.0, 0.3, 0.05,
        help="Comments beyond ±this polarity are labeled positive/negative; everything else is neutral.",
    )

    st.markdown("---")
    st.markdown("### 📤 Export")
    export_format = st.selectbox("Format", ["CSV", "JSON", "Excel"])
    export_slot = st.empty()  # filled in after analysis runs, once df exists

inject_css(dark_mode)
plotly_template = "plotly_dark" if dark_mode else "plotly_white"

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📊 Reddit<span>Insight</span></h1>
        <p>Sentiment, emotion, and discussion-network analysis for any Reddit thread</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = None

if data_mode == "Live Reddit thread":
    reddit_client, conn_error = get_reddit_client()

    status_col, _ = st.columns([1, 3])
    with status_col:
        if reddit_client:
            st.markdown('<span class="status-pill status-ok">● Reddit API connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill status-bad">● Reddit API not connected</span>', unsafe_allow_html=True)

    if not reddit_client:
        show_setup_guide()
        if conn_error and conn_error != "missing_credentials":
            st.error(conn_error)
        if SAMPLE_DATASETS:
            st.info("👈 No working credentials yet? Switch to **Sample dataset** mode in the sidebar to keep exploring the app offline.")
        st.stop()

    # --- URL input ---
    col1, col2 = st.columns([2, 1])
    with col1:
        url = st.text_input("🔗 Reddit thread URL", placeholder="https://www.reddit.com/r/.../comments/abc123/...")
        go_clicked = st.button("Analyze thread", type="primary")
    with col2:
        st.markdown(
            """
            <div class="card">
            <b>How to use</b><br>
            1. Copy a Reddit thread URL<br>
            2. Paste it and click Analyze<br>
            3. Results are cached 10 min — re-click Analyze for fresh data
            </div>
            """,
            unsafe_allow_html=True,
        )

    if go_clicked and url:
        if not looks_like_reddit_url(url):
            st.warning("⚠️ That doesn't look like a Reddit thread URL.")
        else:
            st.session_state["thread_url"] = url

    thread_url = st.session_state.get("thread_url")

    if thread_url:
        try:
            with st.spinner("Fetching comments and analyzing sentiment…"):
                df = fetch_thread_dataframe(reddit_client, thread_url)
        except prawcore.exceptions.NotFound:
            st.error("❌ That thread couldn't be found — it may have been deleted or the URL is off.")
            st.stop()
        except praw.exceptions.InvalidURL:
            st.error("❌ PRAW couldn't parse that as a submission URL. Try copying the link directly from the thread's share button.")
            st.stop()
        except Exception as e:
            st.error(f"❌ An error occurred while fetching this thread: {e}")
            st.stop()

else:
    # --- Sample dataset mode: no network calls, reads bundled JSON ---
    st.markdown('<span class="status-pill status-ok">● Offline demo mode — no Reddit API calls</span>', unsafe_allow_html=True)
    label = st.selectbox("📁 Sample thread", list(SAMPLE_DATASETS.keys()))
    df = load_sample_dataframe(SAMPLE_DATASETS[label])
    st.caption("Loaded from a local file. Swap in your own captured threads by running `fetch_sample_data.py` — see SETUP.md.")

# --------------------------------------------------------------------------
# Analysis (shared by both modes once df is populated)
# --------------------------------------------------------------------------

if df is not None:
    if df.empty:
        st.info("This thread has no comments to analyze.")
        st.stop()

    st.caption(f"**{df.attrs.get('title', '')}** — r/{df.attrs.get('subreddit', '')} · {len(df)} comments")

    if data_mode == "Live Reddit thread" and len(df) < min_comments:
        st.warning(f"⚠️ This thread has fewer comments ({len(df)}) than your threshold ({min_comments}); results may be noisy.")

    df["sentiment_category"] = pd.cut(
        df["sentiment"],
        bins=[-1, -sentiment_threshold, sentiment_threshold, 1],
        labels=["Negative", "Neutral", "Positive"],
        include_lowest=True,
    )

    # Export button now that df exists
    with export_slot:
        data_bytes, mime = export_bytes(df, export_format.lower())
        st.download_button(
            f"Download {export_format}",
            data=data_bytes,
            file_name=f"reddit_insight_{df.attrs.get('subreddit', 'thread')}.{export_format.lower()}",
            mime=mime,
            width="stretch",
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Sentiment", "📈 Timeline", "☁️ Word cloud", "📝 Comments", "📈 Stats"]
    )

    with tab1:
        st.markdown("### Sentiment overview")

        fig_dist = px.histogram(
            df, x="sentiment", nbins=30, color_discrete_sequence=[ACCENT],
            title="Sentiment distribution",
        )
        fig_dist.update_layout(template=plotly_template, showlegend=False)
        st.plotly_chart(fig_dist, width="stretch")

        # size can't be negative in Plotly, but comment scores can be — clip just for marker sizing
        plot_df = df.assign(marker_size=df["score"].clip(lower=1))
        fig_subj = px.scatter(
            plot_df, x="sentiment", y="subjectivity", color="dominant_emotion", size="marker_size",
            hover_data=["comment", "score"], title="Subjectivity vs. polarity by emotion",
        )
        fig_subj.update_layout(template=plotly_template)
        st.plotly_chart(fig_subj, width="stretch")

        emotion_counts = df["dominant_emotion"].value_counts()
        fig_emotion = px.pie(values=emotion_counts.values, names=emotion_counts.index, title="Emotion breakdown")
        fig_emotion.update_layout(template=plotly_template)
        st.plotly_chart(fig_emotion, width="stretch")

        m1, m2, m3 = st.columns(3)
        for col, label, value, color in [
            (m1, "Average sentiment", df["sentiment"].mean(), ACCENT),
            (m2, "Average subjectivity", df["subjectivity"].mean(), "#4b7bff"),
            (m3, "Average VADER score", df["vader_compound"].mean(), "#2ecc71"),
        ]:
            with col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=value, title={"text": label},
                    gauge={"axis": {"range": [-1, 1]}, "bar": {"color": color}},
                ))
                fig.update_layout(height=200, margin=dict(t=40, b=10))
                st.plotly_chart(fig, width="stretch")

    with tab2:
        st.markdown("### Timeline")

        tdf = df.copy()
        tdf["hour"] = tdf["created_utc"].dt.hour
        tdf["day"] = tdf["created_utc"].dt.day_name()

        pivot = tdf.pivot_table(values="sentiment", index="day", columns="hour", aggfunc="mean")
        fig_heat = px.imshow(pivot, title="Sentiment by day & hour", color_continuous_scale="RdBu")
        fig_heat.update_layout(template=plotly_template, xaxis_title="Hour", yaxis_title="Day")
        st.plotly_chart(fig_heat, width="stretch")

        st.markdown("### Discussion network")
        st.caption("Authors linked when their comments land within an hour of each other with similar sentiment (top 150 comments by score).")
        G = build_comment_network(df)
        if G.number_of_edges() == 0:
            st.info("Not enough overlapping activity to draw a network for this thread.")
        else:
            pos = nx.spring_layout(G, seed=42)
            edge_x, edge_y = [], []
            for a, b in G.edges():
                x0, y0 = pos[a]; x1, y1 = pos[b]
                edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

            node_x, node_y, node_text, node_size = [], [], [], []
            for n in G.nodes():
                x, y = pos[n]
                node_x.append(x); node_y.append(y); node_text.append(n)
                node_size.append(8 + 2 * G.degree(n))

            fig_net = go.Figure()
            fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                          line=dict(width=0.5, color="#888"), hoverinfo="none"))
            fig_net.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
                marker=dict(size=node_size, color=ACCENT, showscale=False),
            ))
            fig_net.update_layout(
                template=plotly_template, showlegend=False, hovermode="closest",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                margin=dict(b=10, l=10, r=10, t=10),
            )
            st.plotly_chart(fig_net, width="stretch")

    with tab3:
        st.markdown("### Word cloud")
        all_text = " ".join(df["comment"].astype(str))
        wc = generate_wordcloud(all_text, dark_mode)
        if wc:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        c1, c2 = st.columns(2)
        pos_comments = df[df["sentiment"] > sentiment_threshold]["comment"].astype(str)
        neg_comments = df[df["sentiment"] < -sentiment_threshold]["comment"].astype(str)
        with c1:
            st.markdown("#### Positive")
            wc_pos = generate_wordcloud(" ".join(pos_comments), dark_mode) if not pos_comments.empty else None
            if wc_pos:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.imshow(wc_pos, interpolation="bilinear"); ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No strongly positive comments at this threshold.")
        with c2:
            st.markdown("#### Negative")
            wc_neg = generate_wordcloud(" ".join(neg_comments), dark_mode) if not neg_comments.empty else None
            if wc_neg:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.imshow(wc_neg, interpolation="bilinear"); ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("No strongly negative comments at this threshold.")

    with tab4:
        st.markdown("### Comments")
        st.dataframe(
            df[["comment", "sentiment", "sentiment_category", "score", "author"]]
            .sort_values("score", ascending=False)
            .head(50),
            width="stretch", height=420,
        )

    with tab5:
        st.markdown("### Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total comments", len(df))
            st.metric("Avg. comment length", f"{df['length'].mean():.0f} chars")
            st.metric("Most active hour", int(df["created_utc"].dt.hour.mode().iloc[0]))
        with c2:
            st.metric("Average score", f"{df['score'].mean():.1f}")
            st.metric("Top comment score", int(df["score"].max()))
            st.metric("Unique authors", df["author"].nunique())

        fig_len = px.histogram(df, x="length", nbins=30, title="Comment length distribution")
        fig_len.update_layout(template=plotly_template, xaxis_title="Characters", yaxis_title="Comments")
        st.plotly_chart(fig_len, width="stretch")
elif data_mode == "Live Reddit thread":
    export_slot.empty()
