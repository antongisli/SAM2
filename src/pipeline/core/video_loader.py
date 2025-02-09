"""
Video loader module supporting both local files and YouTube URLs
"""

import cv2
import yt_dlp
import logging
import os
from pathlib import Path
from typing import Tuple, Optional
import re

logger = logging.getLogger(__name__)

class VideoLoader:
    def __init__(self, source: str, cache_dir: str = 'cache'):
        """Initialize video loader with source (file path or YouTube URL)"""
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cap = None
        
    def _is_youtube_url(self, url: str) -> bool:
        """Check if the source is a YouTube URL"""
        youtube_regex = (
            r'(https?://)?(www\.)?'
            '(youtube|youtu|youtube-nocookie)\.(com|be)/'
            '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
        )
        return bool(re.match(youtube_regex, url))
        
    def _download_youtube_video(self) -> Optional[str]:
        """Download YouTube video and return local file path"""
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': str(self.cache_dir / '%(id)s.%(ext)s'),
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info("Downloading YouTube video...")
                info = ydl.extract_info(self.source, download=True)
                video_path = self.cache_dir / f"{info['id']}.mp4"
                logger.info(f"Video downloaded to: {video_path}")
                return str(video_path)
                
        except Exception as e:
            logger.error(f"Error downloading YouTube video: {str(e)}")
            return None
            
    def open(self) -> bool:
        """Open video source and return success status"""
        try:
            if self._is_youtube_url(self.source):
                video_path = self._download_youtube_video()
                if not video_path:
                    return False
            else:
                video_path = self.source
                
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error opening video: {str(e)}")
            return False
            
    def get_info(self) -> Tuple[int, int, float, int]:
        """Get video information: width, height, fps, frame_count"""
        if not self.cap:
            return 0, 0, 0, 0
            
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return width, height, fps, frame_count
        
    def read(self):
        """Read next frame from video"""
        if not self.cap:
            return False, None
        return self.cap.read()
        
    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
            self.cap = None
