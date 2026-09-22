# Setup

## 1. Rotate your old credentials

The previous version of this script had a Reddit client ID and secret hardcoded
as fallback defaults. If that file was ever committed, shared, or deployed,
treat those credentials as compromised: go to
[reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) and delete that app
(or regenerate its secret) before doing anything else.

## 2. Create a fresh Reddit app

1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) → **create app**.
2. Type: **script**. Redirect URI: `http://localhost:8080` (unused, but required).
3. Copy the string under the app name (**client_id**) and the **secret** field.

## 3. Store credentials (pick one)

**Option A — Streamlit secrets (recommended for local + Streamlit Cloud):**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml with your real values
```

**Option B — environment variables:**

```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export REDDIT_USER_AGENT="reddit-insight/1.0 by u/your_username"
```

Either way, `.streamlit/secrets.toml` and `.env` are already in `.gitignore` —
don't remove them from it.

## 4. Install and run

```bash
pip install -r requirements.txt
streamlit run reddit_insight.py
```

The app will show a green "Reddit API connected" pill once credentials are
found and valid; otherwise it shows the setup guide inline.

## 5. Offline demo mode (for presentations / showcases)

The sidebar has a **Data source** toggle: "Live Reddit thread" or "Sample
dataset (offline demo)". Sample mode reads a pre-fetched thread from
`sample_data/*.json` and makes zero network calls — nothing about the
showcase depends on Reddit's API, your credentials, or your wifi working on
the day.

Two synthetic sample threads ship out of the box so this works immediately
with no setup. To swap in a real thread you captured yourself:

```bash
python fetch_sample_data.py "https://www.reddit.com/r/.../comments/abc123/..." my_thread
```

This writes `sample_data/my_thread.json` in the same schema, and it shows up
in the sidebar's sample picker right away. Run it once while your credentials
are working — after that, the file is just static data the app reads locally.

(`generate_synthetic_samples.py` is how the two bundled placeholder threads
were made, in case you want to regenerate them with different topics.)
