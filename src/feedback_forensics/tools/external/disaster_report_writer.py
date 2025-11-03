#!/usr/bin/env python3
"""
Re-creation of the prompting pipeline from
“Harnessing Prompt-based Large Language Models for Disaster Monitoring
and Automated Reporting from Social Media Feedback” (Cantini et al., 2024)

Example usage:

```bash
python disaster_report_writer.py --input tweets.tsv --model openrouter/google/gemini-2.5-flash --limit 100
```
"""

import asyncio
import argparse
import csv
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

from tqdm.asyncio import tqdm
import inverse_cai.models

# ========== PROMPTS (same as before, unchanged for clarity) ==========

CLASS_LABELS = {
    "caution and advice": "notices issued or revoked",
    "sympathy and support": "tweets containing prayers, thoughts, and emotional support",
    "requests or urgent needs": "reports of urgent needs or supplies such as food, water, clothing, money, medicine, or blood",
    "infrastructure and utility damage": "reports of damage to buildings, roads, bridges, power lines, communication poles, or vehicles",
    "rescue volunteering or donation effort": "reports of any rescue, volunteering, or donation efforts, including safe transport, evacuation, medical or food assistance, shelters, monetary or service donations",
    "not humanitarian": "the tweet does not convey information related to humanitarian aid or is off-topic for the disaster",
    "displaced people and evacuations": "people who have changed residence or were evacuated due to the crisis, even temporarily",
    "injured or dead people": "reports of people injured or killed as a result of the disaster",
    "missing or found people": "reports of missing or found people after a catastrophic event",
}

CLASSIFICATION_FEWSHOT_EXAMPLES = {
    "caution and advice": [
        "Avoid driving near flooded areas in Houston right now.",
        "Officials ask residents to stay indoors until the storm passes.",
    ],
    "sympathy and support": [
        "Praying for everyone affected by this hurricane.",
        "Sending love and strength to Texas right now.",
    ],
    "requests or urgent needs": [
        "We need bottled water and blankets at the community center.",
        "Family with kids needs food and diapers near downtown.",
    ],
    "infrastructure and utility damage": [
        "Power lines are down on Main St.",
        "Bridge near the bayou is partially collapsed.",
    ],
    "rescue volunteering or donation effort": [
        "Volunteers with boats heading to West Houston.",
        "Donations being collected at the high school gym.",
    ],
    "not humanitarian": [
        "This football game is unbelievable!",
        "Check out my new video.",
    ],
    "displaced people and evacuations": [
        "We evacuated to Austin for safety.",
        "People from the coastal area are being moved to shelters.",
    ],
    "injured or dead people": [
        "Two people reported injured after the building collapse.",
        "Paramedics confirm one fatality near the river.",
    ],
    "missing or found people": [
        "Looking for my cousin last seen near Rockport.",
        "Child found at the shelter, parents unknown.",
    ],
}


def build_classification_prompt(tweets: List[Tuple[str, str]]) -> str:
    parts = [
        "Act as a helpful data annotator. Classify each tweet into one of the following categories:",
    ]
    for label, desc in CLASS_LABELS.items():
        parts.append(f"- {label}: {desc}")
    parts.append("Correct examples:")
    for label, examples in CLASSIFICATION_FEWSHOT_EXAMPLES.items():
        for ex in examples:
            parts.append(f"[{ex}] -> {label}")
    parts.append("Now classify the following tweets (format: tweet_id,label):")
    for tid, text in tweets:
        parts.append(f"{tid}: {text}")
    return "\n".join(parts)


NER_PROMPT_TEMPLATE = """You are an advanced NER module specialized in LOCATION MENTIONS in disaster-related tweets.
Extract ONLY location-like entities. Output CSV: tweet_id,location_entities (no header).
Tweets:
{tweets}
"""

GEO_PROMPT_TEMPLATE = """Extract the following geographical information from tweets and their NER-based entities:
tweet_id,state,city,zip_code,other_geo (CSV, no header)
Tweets and entities:
{rows}
"""

REPORT_SYSTEM_PROMPT = (
    "Act as an adept report writer creating reports based on a set of tweets."
)
TITLE_PROMPT_TEMPLATE = """Generate a short, captivating title (max 10 words) for the city "{city}" during the disaster "{disaster}".
Tweets:
{tweets}
"""
INTRO_PROMPT_TEMPLATE = """Title: "{title}"
Generate an engaging one-paragraph introduction for a report on {city} during {disaster}.
Tweets:
{tweets}
"""
CONTENT_PROMPT_TEMPLATE = """Title: "{title}"
Introduction: "{introduction}"
Generate a concise section analyzing issues reported in {city} during {disaster}, referencing tweet_ids.
Tweets:
{tweets}
"""

# ========== DATA STRUCTURES ==========


@dataclass
class Tweet:
    tweet_id: str
    text: str
    label: Optional[str] = None
    is_sub_event: bool = False
    ner_locations: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None


SUBEVENT_CLASSES = {
    "infrastructure and utility damage",
    "displaced people and evacuations",
    "injured or dead people",
    "missing or found people",
}


def agg_to_binary(label: str) -> bool:
    return label in SUBEVENT_CLASSES


# ========== SIMPLE TSV HELPERS ==========


def read_tweets_from_tsv(
    path: Path, limit: Optional[int] = None, include_labels: bool = False
) -> List[Tweet]:
    tweets = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        overall_num_tweets = 0

        def load_tweet(row: Dict[str, str]) -> Tweet:
            text = row.get("text", row.get("tweet_text", ""))
            if not include_labels:
                return Tweet(tweet_id=row["tweet_id"], text=text)
            else:
                return Tweet(
                    tweet_id=row["tweet_id"],
                    text=text,
                    label=row.get("label", None),
                    is_sub_event=row.get("is_sub_event", None),
                    ner_locations=row.get("ner_locations", None),
                    state=row.get("state", None),
                    city=row.get("city", None),
                )

        for i, row in enumerate(reader):
            overall_num_tweets += 1
            if limit and i <= limit - 1:
                tweets.append(load_tweet(row))
            elif not limit:
                tweets.append(load_tweet(row))
            else:
                continue

    print(f"Loaded {len(tweets)} tweets out of {overall_num_tweets}")
    return tweets


def write_tweets_to_tsv(path: Path, tweets: List[Tweet]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        tweets_dict = [t.__dict__ for t in tweets]
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tweet_id",
                "text",
                "label",
                "is_sub_event",
                "ner_locations",
                "state",
                "city",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(tweets_dict)


def write_reports_to_tsv(path: Path, reports: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "city",
                "state",
                "title",
                "introduction",
                "content",
                "tweet_ids",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(reports)


# ========== MINIMAL CACHE WRAPPER ==========


class PromptCache:
    def __init__(self, cache_path: Optional[Path]):
        self.path = cache_path
        self.data = {}
        self._lock = asyncio.Lock()  # <-- add this
        if cache_path and cache_path.exists():
            try:
                self.data = json.loads(cache_path.read_text())
            except Exception:
                self.data = {}

    def key(self, prompt: str, model: str) -> str:
        return hashlib.sha1((prompt + model).encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        return self.data.get(self.key(prompt, model))

    async def set(self, prompt: str, model: str, response: str):
        if not self.path:
            return
        async with self._lock:  # <-- prevent concurrent writes
            self.data[self.key(prompt, model)] = response
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=0))


# ========== ASSUMED LLM CALL ==========


async def get_model_response(prompt: str, model_name: str):
    model = inverse_cai.models.get_model(model_name)

    response = None
    response_content = None

    max_retries = 3
    for i in range(max_retries):
        try:
            response = await model.ainvoke(
                prompt,
            )
            response_content = response.content
            break
        except Exception as e:
            print(f"Error getting model response: {e}")
            if i < max_retries - 1:
                print(
                    f"Retrying ({i+1}/{max_retries} tries). Sleeping for 3 seconds..."
                )
                await asyncio.sleep(0.1 * i)
            continue

    return response_content


async def call_llm(
    prompt: str, model: str, sem: asyncio.Semaphore, cache: PromptCache
) -> str:
    cached = cache.get(prompt, model)
    if cached is not None:
        return cached
    async with sem:
        result = await get_model_response(prompt, model)
        await cache.set(prompt, model, result)
        return result


# ========== CLASSIFICATION ==========


def chunked(seq, size):
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def parse_class_output(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "," in line:
            tid, label = line.split(",", 1)
            out[tid.strip()] = label.strip().lower()
        elif ":" in line:
            tid, label = line.split(":", 1)
            out[tid.strip()] = label.strip().lower()
        else:
            print(f"Warning: No label found for tweet {tid}")
            continue
    return out


async def classify_tweets(tweets: List[Tweet], model: str, sem, cache, batch=20):
    batches = chunked([(t.tweet_id, t.text) for t in tweets], batch)
    tasks = [
        call_llm(build_classification_prompt(b), model, sem, cache) for b in batches
    ]
    results = await tqdm.gather(*tasks, desc="Classifying tweets")
    all_map = {}
    for r in results:
        all_map.update(parse_class_output(r))
    for t in tweets:
        t.label = all_map.get(t.tweet_id)
        t.is_sub_event = agg_to_binary(t.label or "")


# ========== NER + GEO ==========


def parse_ner_output(text):
    out = {}
    for l in text.splitlines():
        if "," in l:
            tid, locs = l.split(",", 1)
            out[tid.strip()] = locs.strip()
    return out


def parse_geo_output(text):
    out = {}
    for l in text.splitlines():
        parts = [p.strip() for p in l.split(",", 4)]
        if len(parts) == 5:
            tid, state, city, *_ = parts
            out[tid] = {"state": state, "city": city}
    return out


async def ner_and_geo(tweets: List[Tweet], model: str, sem, cache, batch=30):
    sub = [t for t in tweets if t.is_sub_event]
    if not sub:
        return
    # NER
    ner_prompts = [
        NER_PROMPT_TEMPLATE.format(
            tweets="\n".join(f"{t.tweet_id},{t.text}" for t in b)
        )
        for b in chunked(sub, batch)
    ]
    ner_results = await tqdm.gather(
        *[call_llm(p, model, sem, cache) for p in ner_prompts], desc="Extracting NER"
    )
    ner_map = {}
    for r in ner_results:
        ner_map.update(parse_ner_output(r))
    for t in sub:
        t.ner_locations = ner_map.get(t.tweet_id)
    # GEO
    geo_prompts = [
        GEO_PROMPT_TEMPLATE.format(
            rows="\n".join(f"{t.tweet_id},{t.text},{t.ner_locations}" for t in b)
        )
        for b in chunked(sub, batch)
    ]
    geo_results = await tqdm.gather(
        *[call_llm(p, model, sem, cache) for p in geo_prompts], desc="Extracting GEO"
    )
    geo_map = {}
    for r in geo_results:
        geo_map.update(parse_geo_output(r))
    for t in sub:
        g = geo_map.get(t.tweet_id, {})
        t.state, t.city = g.get("state"), g.get("city")
        if t.state:
            t.state = t.state.lower()
        if t.city:
            t.city = t.city.lower()


# ========== REPORTS ==========


def group_by_city_state(tweets: List[Tweet]) -> Dict[Tuple[str, str], List[Tweet]]:
    d = defaultdict(list)
    for t in tweets:
        if t.is_sub_event:
            d[(t.city or "unknown").lower(), (t.state or "unknown").lower()].append(t)
    return d


async def make_report(
    city, state, tweets, disaster, model, sem, cache, tweets_limit=250
):
    if tweets_limit and len(tweets) > tweets_limit:
        print(
            f"Warning: {len(tweets)} tweets found for {city}, {state}, truncating to {tweets_limit}"
        )
        tweets = tweets[:tweets_limit]

    block = "\n".join(f"{t.tweet_id}: {t.text}" for t in tweets)
    title = await call_llm(
        TITLE_PROMPT_TEMPLATE.format(city=city, disaster=disaster, tweets=block),
        model,
        sem,
        cache,
    )
    intro = await call_llm(
        INTRO_PROMPT_TEMPLATE.format(
            city=city, disaster=disaster, tweets=block, title=title.strip()
        ),
        model,
        sem,
        cache,
    )
    content = await call_llm(
        CONTENT_PROMPT_TEMPLATE.format(
            city=city,
            disaster=disaster,
            tweets=block,
            title=title.strip(),
            introduction=intro.strip(),
        ),
        model,
        sem,
        cache,
    )
    return {
        "city": city,
        "state": state,
        "title": title.strip(),
        "introduction": intro.strip(),
        "content": content.strip(),
        "tweet_ids": ",".join(t.tweet_id for t in tweets),
    }


async def make_all_reports(tweets, model, sem, cache, disaster):
    groups = group_by_city_state(tweets)
    tasks = [
        make_report(c, s, g, disaster, model, sem, cache)
        for (c, s), g in groups.items()
    ]
    return await tqdm.gather(*tasks, desc="Generating reports")


# ========== MAIN ==========


async def main_async(args):
    print("Starting disaster report writer with args")
    sem = asyncio.Semaphore(args.max_concurrent)

    if args.output_dir is None:
        model_name = args.model.split("/")[-1].replace("-", "_")
        output_dir = Path(f"exp/disaster_report_writer/{model_name}")
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = PromptCache(output_dir / "cache.json")

    # saving config
    with open(output_dir / "config.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    if not args.reuse_labels:
        tweets = read_tweets_from_tsv(Path(args.input), limit=args.limit)
        await classify_tweets(tweets, args.model, sem, cache, batch=args.class_batch)
        await ner_and_geo(tweets, args.model, sem, cache, batch=args.geo_batch)
    else:
        tweets = read_tweets_from_tsv(
            Path(args.input), limit=args.limit, include_labels=True
        )
        print(
            f"Skipping classification and NER, reusing labels from input file {Path(args.input)}"
        )

    # save tweets to tsv
    write_tweets_to_tsv(output_dir / "tweets.tsv", tweets)

    reports = await make_all_reports(
        tweets, args.model, sem, cache, disaster=args.disaster
    )
    write_reports_to_tsv(output_dir / "reports.tsv", reports)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument(
        "--reuse-labels", action="store_true", help="Reuse labels from input file"
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--disaster", default="Hurricane Harvey")
    p.add_argument(
        "--limit", type=int, default=None, help="Process only first N tweets"
    )
    p.add_argument("--max-concurrent", type=int, default=5)
    p.add_argument("--class-batch", type=int, default=20)
    p.add_argument("--geo-batch", type=int, default=30)
    return p.parse_args()


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
