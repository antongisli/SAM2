#!/usr/bin/env python3

"""
New entry point for the modular video analysis pipeline
"""

import os
import cv2
import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.pipeline import VideoPipeline, PipelineConfig
from src.pipeline.core.video_loader import VideoLoader

# Set up logging
def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '\033[1m%(asctime)s\033[0m - %(levelname)s - \033[92m%(message)s\033[0m'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set up file handler
    log_file = f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    # Set up console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    return log_file

def main():
    parser = argparse.ArgumentParser(description='Video Analysis Pipeline')
    parser.add_argument('input', help='Path to input video file or YouTube URL')
    parser.add_argument('--output-dir', default='output',
                      help='Directory to save results (default: output)')
    parser.add_argument('--cache-dir', default='cache',
                      help='Directory to cache downloaded videos (default: cache)')
    parser.add_argument('--skip-frames', type=int, default=10,
                      help='Number of frames to skip between analysis (default: 10)')
    parser.add_argument('--sam-type', default='vit_b',
                      choices=['vit_h', 'vit_l', 'vit_b'],
                      help='SAM model type (default: vit_b)')
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("="*50)
    logger.info("Video Analysis Pipeline Started")
    logger.info("="*50)
    logger.info(f"Log file: {log_file}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline with custom config
    config = PipelineConfig(sam_type=args.sam_type)
    pipeline = VideoPipeline(config)
    
    # Initialize video loader
    video = VideoLoader(args.input, cache_dir=args.cache_dir)
    if not video.open():
        logger.error(f"Error: Could not open video source '{args.input}'")
        return
    
    # Get video info
    width, height, fps, frame_count = video.get_info()
    logger.info(f"Video Info:")
    logger.info(f"- Resolution: {width}x{height}")
    logger.info(f"- FPS: {fps}")
    logger.info(f"- Total Frames: {frame_count}")
    logger.info(f"- Processing every {args.skip_frames} frames")
    
    # Process video
    frame_idx = 0
    all_interactions = []
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        if frame_idx % args.skip_frames == 0:
            logger.info(f"\nProcessing frame {frame_idx}")
            
            try:
                # Process frame
                interactions, annotated_frame = pipeline.process_frame(frame)
                
                # Save results
                if interactions:
                    all_interactions.extend([
                        {**interaction, 'frame': frame_idx}
                        for interaction in interactions
                    ])
                    
                    # Save annotated frame
                    output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(output_path), annotated_frame)
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                
        frame_idx += 1
    
    # Cleanup
    video.release()
    
    # Save all interactions
    if all_interactions:
        import json
        output_path = output_dir / "interactions.json"
        with open(output_path, 'w') as f:
            json.dump(all_interactions, f, indent=2)
            
    logger.info("\nProcessing complete!")
    logger.info(f"Found {len(all_interactions)} interactions")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
