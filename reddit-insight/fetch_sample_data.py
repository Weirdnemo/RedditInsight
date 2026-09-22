"""
Run this once, while your Reddit credentials work, to capture a real thread
into sample_data/ for use in the app's offline "Sample dataset" demo mode.

    python fetch_sample_data.py "https://www.reddit.com/r/.../comments/abc123/..." my_thread

Writes sample_data/my_thread.json in the same schema the app already ships
with (see laptop_recommendation_thread.json / pineapple_pizza_debate.json),
so it shows up in the sidebar's "Sample thread" picker immediately — replace
the bundled synthetic ones or just add alongside them.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import praw

from reddit_core import analyze_comment, get_credential


def main(url: str, out_name: str) -> None:
    client_id = get_credential("REDDIT_CLIENT_ID")
    client_secret = get_credential("REDDIT_CLIENT_SECRET")
    user_agent = get_credential("REDDIT_USER_AGENT", "reddit-insight-sample-fetcher/1.0")

    if not client_id or not client_secret:
        sys.exit(
            "No credentials found. Set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET "
            "via .streamlit/secrets.toml or env vars first — see SETUP.md."
        )

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    reddit.read_only = True

    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=0)

    comments = []
    for c in submission.comments.list():
        body = c.body or ""
        comments.append(
            {
                "comment": body,
                **analyze_comment(body),
                "created_utc": datetime.fromtimestamp(c.created_utc).isoformat(),
                "score": c.score,
                "author": str(c.author) if c.author else "[deleted]",
                "length": len(body),
            }
        )

    payload = {
        "meta": {
            "title": submission.title,
            "subreddit": str(submission.subreddit),
            "permalink": f"https://reddit.com{submission.permalink}",
        },
        "comments": comments,
    }

    out_path = Path("sample_data") / f"{out_name}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path} with {len(comments)} comments.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: python {sys.argv[0]} <reddit_thread_url> <output_name>")
    main(sys.argv[1], sys.argv[2])
