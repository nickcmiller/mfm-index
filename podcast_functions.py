import feedparser
import requests
import urllib
import logging
import traceback

import json

def parse_feed(
    feed_url: str
) -> feedparser.FeedParserDict:
    """
    Parses a podcast feed and returns the feed object.

    Arguments:
        feed_url: str - The URL of the podcast feed.

    Returns:
        feedparser.FeedParserDict - The feed object.
    """
    try:
        feed = feedparser.parse(feed_url)
        if feed is None:
            raise ValueError("Feed is None")
        return feed
    except Exception as e:
        logging.error(f"Failed to parse feed: {e}")
        traceback.print_exc()
        return None

def extract_entry_metadata(
    entry: dict
) -> dict:
    """
    Extracts metadata from an entry in a podcast feed.

    Arguments:
        entry: dict - The entry to extract metadata from.

    Returns:
        dict - The extracted metadata.
    """
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

def extract_metadata_from_feed(
    feed: feedparser.FeedParserDict
) -> dict:
    """
    Extracts metadata from a podcast feed.
    """
    entries = []
    
    for entry in feed.entries:
        entry_metadata = extract_entry_metadata(entry)
        entries.append(entry_metadata)
    
    return entries

def download_podcast_audio(
    audio_url: str, 
    title: str, 
    file_path: str=None
) -> str:
    """
    Downloads a podcast audio file from a URL and saves it to a file.

    Arguments:
        audio_url: str - The URL of the podcast audio file.
        title: str - The title of the podcast episode.
        file_path: str - The path to save the podcast audio file to.

    Returns:
        str - The path to the saved podcast audio file.
    """
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
    entries = extract_metadata_from_feed(feed)
    print(json.dumps(entries, indent=4))
    print(f"Entries: {len(entries)}")
