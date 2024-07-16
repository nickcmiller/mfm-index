from genai_toolbox.text_prompting.model_calls import groq_text_response, openai_text_response
from genai_toolbox.helper_functions.datetime_helpers import get_date_with_timezone
import feedparser
import urllib

import requests
import logging
import traceback
import json
from typing import List, Optional
import os

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
        This function processes a single entry from a podcast feed, retrieving key information such as the entry ID, title, publication date, summary, and audio URL. 
        It ensures that all required attributes are present in the entry and handles the extraction of the audio URL from the entry's links.
        
        Arguments:
            entry: dict - The entry to extract metadata from.

        Returns:
            dict - The extracted metadata.

        Raises:
            AttributeError: If any required attribute is missing from the entry.
    """
    required_attributes = ['id', 'title', 'published', 'summary', 'links']
    
    entry_data = {}
    for attr in required_attributes:
        if not hasattr(entry, attr):
            raise AttributeError(f"Required attribute '{attr}' is missing from the entry")
        if attr != 'links':
            entry_data[attr] = getattr(entry, attr)
    
    entry_data['url'] = next((link['href'] for link in entry.links if link.get('rel') == 'enclosure'), None)
    if entry_data['url'] is None:
        raise AttributeError("No 'enclosure' link found in the entry")
    
    return entry_data

def extract_metadata_from_feed(
    feed: feedparser.FeedParserDict
) -> List[dict]:
    """
        Extracts metadata from a podcast feed by processing the entire feed object and iterating through all entries.
        It collects key details such as entry ID, title, publication date, summary, and audio URL for each episode, as well as a general summary of the entire podcast feed. 
        The function compiles this information into a structured format for easier programmatic use.

        Arguments:
            feed: feedparser.FeedParserDict - The feed object.

        Returns:
            List[dict] - A list of entries from the podcast feed.

        Example:
            >>> result = extract_metadata_from_feed(parse_feed("https://feeds.simplecast.com/3hnxp7yk"))
            >>> print(result)
            {'entry_id': 'd8029cde-4677-4ac9-bdbc-2f05fba1c1c5', 
            'title': '#351 The Founder of Rolex: Hans Wilsdorf', 
            'published': 'Tue, 4 Jun 2024 16:05:59 +0000', 
            'summary': 'What I learned from reading about Hans Wilsdorf and the founding of Rolex.', 
            'url': 'https://www.founderspodcast.com/', 
            'feed_summary': 'Learn from history\'s greatest entrepreneurs...'}, 
            {...}, {...}]
    """
    # List of possible fields for feed summary
    feed_summary_fields = ["summary", "description", "info", "about"]
    
    # Get feed summary
    feed_summary = ""
    for field in feed_summary_fields:
        if hasattr(feed.feed, field):
            feed_summary = getattr(feed.feed, field)
            break

    feed_title = feed.feed.title
    
    entries = []
    for entry in feed.entries:
        entry_metadata = extract_entry_metadata(entry)
        entry_metadata["feed_summary"] = feed_summary
        entry_metadata["feed_title"] = feed_title
        entries.append(entry_metadata)    
    
    return entries

def return_all_entries_from_feed(
    feed_url: str
) -> List[dict]:
    """
        Returns a list of entries from a podcast feed.

        Arguments:
            feed_url: str - The URL of the podcast feed.

        Returns:
            List[dict] - A list of entries from the podcast feed.

        Example:
            >>> result = return_entries_from_feed(
                feed_url="https://feeds.simplecast.com/3hnxp7yk"
            )
            >>> print(result)
            [{'entry_id': 'd8029cde-4677-4ac9-bdbc-2f05fba1c1c5', 
            'title': '#351 The Founder of Rolex: Hans Wilsdorf', 
            'published': 'Tue, 4 Jun 2024 16:05:59 +0000', 
            'summary': 'What I learned from reading about Hans Wilsdorf and the founding of Rolex.', 
            'url': 'https://www.founderspodcast.com/', 
            'feed_summary': 'Learn from history\'s greatest entrepreneurs...'}, 
            {...}, {...}]
    """
    feed = parse_feed(feed_url)
    return extract_metadata_from_feed(feed)

def return_entries_by_date(
    feed_url: str,
    start_date_str: str,
    end_date_str: Optional[str] = "today"
) -> List[dict]:
    """
        Retrieves podcast entries from a specified feed URL that are published within a given date range.

        Args:
            feed_url (str): The URL of the podcast feed.
            start_date_str (str): The start date as a string in YYYY-MM-DD format. Defaults to "1900-01-01".
            end_date_str (Optional[str]): The end date as a string in YYYY-MM-DD format. Defaults to today's date.

        Returns:
            List[dict]: A list of podcast entries that fall within the specified date range.

        Raises:
            ValueError: If the start_date_str or end_date_str cannot be parsed into a date.
    """
    try:
        start_date = get_date_with_timezone(start_date_str)
        end_date = get_date_with_timezone(end_date_str)
    except ValueError as e:
        raise ValueError(f"Error parsing date strings: {e}")

    entries = return_all_entries_from_feed(feed_url)
    filtered_entries = [
        entry for entry in entries
        if start_date <= get_date_with_timezone(entry['published']) <= end_date
    ]

    return filtered_entries

def download_podcast_audio(
    audio_url: str, 
    title: str, 
    download_dir_name: Optional[str] = None
) -> str:
    """
        Downloads a podcast audio file from a URL and saves it to a file.

        Arguments:
            audio_url: str - The URL of the podcast audio file.
            title: str - The title of the podcast episode.
            file_path: str - The path to save the podcast audio file to.

        Returns:
            str - The path to the saved podcast audio file.

        Example:
            >>> result = download_podcast_audio(
                audio_url="https://www.example.com/podcast.mp3",
                title="Podcast Episode Title",
                file_path="/path/to/save/podcast.mp3"
            )
            >>> print(result)
            "/path/to/save/podcast.mp3"
    """
    if download_dir_name is None:
        download_dir_path = os.getcwd()
    else:
        download_dir_path = os.path.join(os.getcwd(), download_dir_name)
        if not os.path.exists(download_dir_path):
            os.makedirs(download_dir_path)
    
    safe_title = ''.join(char for char in title if char.isalnum() or char in " -_")
    title_with_underscores = safe_title.replace(" ", "_")
    file_name = os.path.join(download_dir_path, title_with_underscores + ".mp3")
    
    response = requests.get(audio_url)
    
    if response.status_code == 200:
        logging.info(f"File downloaded: {title}")
        try:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            logging.info(f"File written: {file_name}")
        except Exception as e:
            logging.error(f"Failed to write file: {e}")
    else:
        logging.error(f"Failed to download the file: {title}")

    return file_name
        
def generate_episode_summary(
    summary: str,
    feed_summary: str
) -> str:
    summary_prompt = f"""
        Description of the podcast feed:\n {feed_summary} \n\n
        
        Description of this specific podcast episode:\n {summary} \n

        Describe the hosts and the guests expected in this specific episode.
    """
    
    try:
        response = groq_text_response(
            model_choice="llama3-70b",
            prompt=summary_prompt
        )
    except Exception as e:
        logging.error(f"Groq model call failed: {e}")
        response = openai_text_response(
            model_choice="text-davinci-003",
            prompt=summary_prompt
        )

    return response

if __name__ == "__main__":
    mfm_feed_url = "https://feeds.megaphone.fm/HS2300184645"
    dithering_feed_url = "https://dithering.passport.online/feed/podcast/KCHirQXM6YBNd6xFa1KkNJ"
    entries = return_all_entries_from_feed(dithering_feed_url)
    test_feed = return_entries_by_date(feed_url=dithering_feed_url, start_date_str="2024-06-17", end_date_str="2024-06-19")
    
    print(json.dumps(test_feed, indent=4))
    print(json.dumps(entries[:3], indent=4))
