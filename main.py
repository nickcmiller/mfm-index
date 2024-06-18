from genai_toolbox.download_sources.podcast_functions import return_entries_from_feed
from genai_toolbox.helper_functions.string_helpers import write_string_to_file

import json
import datetime
from dateutil import parser
import pytz

def get_date_with_timezone(
    date_input: str, 
    timezone_str: str = 'UTC'
) -> datetime.datetime:
    # Parse the date string
    naive_date = parser.parse(date_input)
    # Localize the date if it's naive
    timezone = pytz.timezone(timezone_str)
    if naive_date.tzinfo is None:
        return timezone.localize(naive_date)
    else:
        return naive_date

# Example usage
date_input = "June 8, 2024"
published_date = get_date_with_timezone(date_input, 'UTC')

feed_url = "https://feeds.megaphone.fm/HS2300184645"
feed_entries = return_entries_from_feed(feed_url)

filtered_entries = [entry for entry in feed_entries if get_date_with_timezone(entry['published']) > published_date]
"""
Sample Filter Entry:
[
    {
        "entry_id": "cdf26398-2ca3-11ef-a5ee-975153545b62",
        "title": "EXCLUSIVE: $3B Founder Reveals His Next Big Idea",
        "published": "Mon, 17 Jun 2024 14:00:00 -0000",
        "summary": "Episode 597: Sam Parr ( https://twitter.com/theSamParr ) talks to Brett Adcock ( https://x.com/adcock_brett ) about his next big idea, his checklist for entrepreneurs, and his framework for learning new things and moving fast.\u00a0\n\n\u2014\nShow Notes:\n(0:00) Solving school shootings\n(3:15) Cold calling NASA\n(6:14) Spotting the mega-trend\n(8:37) \"Thinking big is easier\"\n(12:42) Brett's philosophy on company names\n(16:22) Brett's ideas note app: genetics, super-sonic travel, synthetic foods\n(19:45) \"I just want to win\"\n(21:46) Brett's checklist for entrepreneurs\n(25:17) Being fast in hardware\n(30:15) Brett's framework for learning new things\n(33:00) Who does Brett admire\n\n\u2014\nLinks:\n\u2022 [Steal This] Get our proven writing frameworks that have made us millions https://clickhubspot.com/copy\n\u2022 Brett Adcock - https://www.brettadcock.com/\n\u2022 Cover - https://www.cover.ai/\n\u2022 Figure - https://figure.ai/\n\n\u2014\nCheck Out Shaan's Stuff:\nNeed to hire? You should use the same service Shaan uses to hire developers, designers, & Virtual Assistants \u2192 it\u2019s called Shepherd (tell \u2018em Shaan sent you): https://bit.ly/SupportShepherd\n\n\u2014\nCheck Out Sam's Stuff:\n\u2022 Hampton - https://www.joinhampton.com/\n\u2022 Ideation Bootcamp - https://www.ideationbootcamp.co/\n\u2022 Copy That - https://copythat.com\n\u2022 Hampton Wealth Survey - https://joinhampton.com/wealth\n\u2022 Sam\u2019s List - http://samslist.co/\n\n\nMy First Million is a HubSpot Original Podcast // Brought to you by The HubSpot Podcast Network // Production by Arie Desormeaux // Editing by Ezra Bakker Trupiano",
        "url": "https://pdst.fm/e/chrt.fm/track/28555/pdrl.fm/2a922f/traffic.megaphone.fm/HS9983733981.mp3?updated=1718638499",
        "feed_summary": "Sam Parr and Shaan Puri brainstorm new business ideas based on trends & opportunities they see in the market. Sometimes they bring on famous guests to brainstorm with them."
    },
]
"""
write_string_to_file("mfm_feed.txt", json.dumps(filtered_entries, indent=4))


