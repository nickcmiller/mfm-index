import yt_dlp
from pydub import AudioSegment
import os
import logging
import traceback

logging.basicConfig(level=logging.INFO)

def yt_dlp_download(yt_url:str, output_path:str = None) -> str:
    """
    Downloads the audio track from a specified YouTube video URL using the yt-dlp library, then converts it to an MP3 format file.

    This function configures yt-dlp to extract the best quality audio available and uses FFmpeg (via yt-dlp's postprocessors) to convert the audio to MP3 format. The resulting MP3 file is saved to the specified or default output directory with a filename derived from the video title.

    Args:
        yt_url (str): The URL of the YouTube video from which audio will be downloaded. This should be a valid YouTube video URL.

    Returns:
        str: The absolute file path of the downloaded and converted MP3 file. This path includes the filename which is derived from the original video title.

    Raises:
        yt_dlp.utils.DownloadError: If there is an issue with downloading the video's audio due to reasons such as video unavailability or restrictions.
        
        Exception: For handling unexpected errors during the download and conversion process.
    """
    if output_path is None:
        output_path = os.getcwd()

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(yt_url, download=True)
            file_name = ydl.prepare_filename(result)
            mp3_file_path = file_name.rsplit('.', 1)[0] + '.mp3'
            logging.info(f"yt_dlp_download saved YouTube video to file path: {mp3_file_path}")
            return mp3_file_path
    except yt_dlp.utils.DownloadError as e:
        logging.error(f"yt_dlp_download failed to download audio from URL {yt_url}: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred with yt_dlp_download: {e}")
        logging.error(traceback.format_exc())
        raise