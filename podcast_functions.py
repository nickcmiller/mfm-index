import feedparser
import requests
import urllib
import logging
import traceback

import json

def parse_feed(feed_url: str):
    try:
        feed = feedparser.parse(feed_url)
        if feed is None:
            raise ValueError("Feed is None")
        return feed
    except Exception as e:
        logging.error(f"Failed to parse feed: {e}")
        traceback.print_exc()
        return None

def extract_entry_metadata(entry: dict):
    entry_id = entry.id
    title = entry.title
    published = entry.published
    summary = entry.summary
    url = entry.links[0]['href']
    return {
        "entry_id": entry_id,
        "title": title,
        "published": published,
        "summary": summary,
        "url": url
    }

def download_podcast_audio(audio_url: str, title: str, file_path: str=None):
    if file_path is None:
        file_path = os.getcwd() + "/"
    
    file_name = file_path+title+".mp3"
    
    response = requests.get(audio_url)
    
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
        logging.info(f"File downloaded: {title}")
    else:
        logging.error(f"Failed to download the file: {title}")

    return file_name

if __name__ == "__main__":
    feed_url = "https://feeds.megaphone.fm/HS2300184645"
    feed = parse_feed(feed_url)
    first_entry = feed.entries[0]
    keys = first_entry.keys()
    entry_metadata = extract_entry_metadata(first_entry)
    print(json.dumps(entry_metadata, indent=4))
    print(keys)
