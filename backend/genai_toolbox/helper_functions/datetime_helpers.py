from datetime import datetime
from dateutil import parser
import pytz

import logging

logging.basicConfig(level=logging.INFO)

def convert_date_format(
    date_string: str
) -> str:
    """
        Converts a date string from one format to another.

        This function takes a date string in the format "Day, DD Mon YYYY HH:MM:SS +ZZZZ"
        and converts it to the format "Month DD, YYYY".

        Args:
            date_string (str): The input date string in the format "Day, DD Mon YYYY HH:MM:SS +ZZZZ".

        Returns:
            str: The formatted date string in the format "Month DD, YYYY".

        Raises:
            ValueError: If the input date string is not in the expected format.

        Example:
            >>> convert_date_format("Tue, 18 Jun 2024 09:00:36 +0000")
            "June 18, 2024"
    """
    dt = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %z")
    return dt.strftime("%B %d, %Y")

def get_date_with_timezone(
    date_str: str, 
    timezone_str: str = 'UTC'
) -> datetime:
    """
        Parses a date string and returns a datetime object localized to the specified timezone.

        Args:
            date_str (str): The date string to parse.
            timezone_str (str): The timezone to localize the date to. Defaults to 'UTC'.

        Returns:
            datetime: A datetime object localized to the specified timezone.

        Raises:
            ValueError: If the date string is invalid or the timezone string is not recognized.

        Examples:
            >>> get_date_with_timezone("2024-06-01")
            datetime.datetime(2024, 6, 1, 0, 0, tzinfo=<UTC>)
            >>> get_date_with_timezone("June 1st, 2024", "America/New_York")
            datetime.datetime(2024, 6, 1, 0, 0, tzinfo=<DstTzInfo 'America/New_York' EST-1 day, 19:00:00 STD>)
    """
    if isinstance(date_str, str) and date_str.lower() == "today":
        date_str = datetime.now().strftime('%Y-%m-%d')
        logging.debug(f"Date string is today, setting to {date_str}")

    try:
        logging.debug(f"Parsing date string: {date_str}")
        naive_date = parser.parse(date_str)
        logging.debug(f"Parsed date string: {naive_date}")
    except ValueError as e:
        raise ValueError(f"Invalid date string: {date_str}. Error: {e}")
    except TypeError as e:
        raise ValueError(f"Invalid date string: {date_str}. Error: {e}")

    try:
        timezone = pytz.timezone(timezone_str)
        logging.debug(f"Timezone: {timezone}")
    except pytz.UnknownTimeZoneError as e:
        raise ValueError(f"Unknown timezone: {timezone_str}. Error: {e}")

    if naive_date.tzinfo is None:
        localized_date = timezone.localize(naive_date)
        logging.debug(f"Localized date: {localized_date}")
    else:
        localized_date = naive_date.astimezone(timezone)
        logging.debug(f"Localized date: {localized_date}")

    return localized_date
