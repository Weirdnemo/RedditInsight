"""One-off script (not part of the shipped app) that builds the two bundled
sample_data/*.json files with synthetic-but-realistic comment threads, run
through the exact same TextBlob/VADER/emotion pipeline the app uses, so the
demo-mode visuals look like real analysis output. Not real Reddit content —
replace with fetch_sample_data.py output whenever you have working credentials.
"""

import json
import random
from datetime import datetime, timedelta

from reddit_core import analyze_comment

random.seed(7)

AUTHORS = [
    "throwaway_9182", "quiet_forest", "mkzhang", "dev_sarah", "budget_baller",
    "gpu_goblin", "night_owl_22", "campus_carl", "penny_pincher", "retro_kid",
    "coffeeholic", "linux_lena", "reply_guy_99", "sunday_scroller", "dr_debug",
    "moderately_online", "[deleted]", "used_to_lurk", "tabs_not_spaces", "cardboard_box_pc",
]

POSITIVE = [
    "This is genuinely great advice, thank you for taking the time to write it out.",
    "I did exactly this last year and it's been amazing, zero regrets.",
    "Honestly the best value pick right now, I love mine.",
    "Wonderful writeup, saved me hours of research.",
    "Can confirm, had the best experience with this exact setup.",
    "This made my day, thank you so much for the detailed comment.",
    "Excited to try this out, sounds like a great plan.",
]

NEGATIVE = [
    "This is terrible advice honestly, wouldn't recommend it to anyone.",
    "I tried this and it was an awful experience, total waste of money.",
    "Worst purchase I've made in years, deeply regret it.",
    "This take is kind of annoying, feels like nobody actually tested it.",
    "I'm furious this is still being recommended, it broke within a week.",
    "Miserable experience, customer support was useless too.",
    "Hated every second of setting this up, would not do it again.",
]

NEUTRAL = [
    "Depends on your budget honestly, there's no single right answer here.",
    "I've seen mixed results, some people love it and some don't.",
    "Worth checking a few reviews before deciding either way.",
    "It's fine for basic use but probably not for anything heavier.",
    "There are a few options in that price range, hard to say which is best.",
    "I'd wait for a sale rather than buying right now.",
    "Not sure this thread has a consensus yet, still reading through it.",
]

SURPRISED = [
    "Wait, I did not expect that price at all, shocked honestly.",
    "Huh, surprised nobody's mentioned the obvious alternative yet.",
    "That's an unexpected twist, wasn't anticipating this outcome.",
]

FEARFUL = [
    "Kind of worried this is going to break down after the warranty ends.",
    "Honestly a little anxious about committing to this without more reviews.",
    "I'm nervous this won't hold up under heavier use, anyone tested that?",
]


def make_comment_pool(topic_positive_extra, topic_negative_extra):
    pool = []
    pool += [(t, "joy") for t in POSITIVE + topic_positive_extra]
    pool += [(t, "sadness") for t in NEGATIVE + topic_negative_extra]
    pool += [(t, "neutral") for t in NEUTRAL]
    pool += [(t, "surprise") for t in SURPRISED]
    pool += [(t, "fear") for t in FEARFUL]
    return pool


def build_thread(title, subreddit, permalink, pool, n_comments, start, hours_span):
    comments = []
    for _ in range(n_comments):
        body, _ = random.choice(pool)
        author = random.choice(AUTHORS)
        created = start + timedelta(hours=random.uniform(0, hours_span))
        score = max(-8, int(random.gauss(12, 20)))
        analysis = analyze_comment(body)
        comments.append(
            {
                "comment": body,
                **analysis,
                "created_utc": created.isoformat(),
                "score": score,
                "author": author,
                "length": len(body),
            }
        )
    comments.sort(key=lambda c: c["created_utc"])
    return {
        "meta": {"title": title, "subreddit": subreddit, "permalink": permalink},
        "comments": comments,
    }


if __name__ == "__main__":
    start = datetime(2026, 8, 14, 9, 0, 0)

    laptop_pool = make_comment_pool(
        topic_positive_extra=[
            "This laptop handled four years of CS assignments without a hiccup.",
            "Battery life alone makes this worth it for campus all day.",
        ],
        topic_negative_extra=[
            "The fan noise on that model is unbearable during compiles.",
            "Mine had a screen defect within the first month, avoid it.",
        ],
    )
    thread1 = build_thread(
        title="Best budget laptop for a CS major starting this fall?",
        subreddit="StudentLife",
        permalink="/r/StudentLife/comments/demo1/best_budget_laptop_for_a_cs_major/",
        pool=laptop_pool,
        n_comments=140,
        start=start,
        hours_span=36,
    )

    pizza_pool = make_comment_pool(
        topic_positive_extra=[
            "Sweet and salty is a wonderful combination, don't @ me.",
            "Pineapple pizza is genuinely amazing when the ham is smoked.",
        ],
        topic_negative_extra=[
            "Pineapple on pizza is a crime against Italian cuisine, full stop.",
            "This take is awful, fruit does not belong anywhere near cheese.",
        ],
    )
    thread2 = build_thread(
        title="Unpopular opinion: pineapple absolutely belongs on pizza",
        subreddit="unpopularopinion",
        permalink="/r/unpopularopinion/comments/demo2/unpopular_opinion_pineapple_belongs_on_pizza/",
        pool=pizza_pool,
        n_comments=160,
        start=start,
        hours_span=48,
    )

    with open("sample_data/laptop_recommendation_thread.json", "w") as f:
        json.dump(thread1, f, indent=2)
    with open("sample_data/pineapple_pizza_debate.json", "w") as f:
        json.dump(thread2, f, indent=2)

    print("Wrote sample_data/laptop_recommendation_thread.json:", len(thread1["comments"]), "comments")
    print("Wrote sample_data/pineapple_pizza_debate.json:", len(thread2["comments"]), "comments")
