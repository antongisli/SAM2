#!/usr/bin/env python3

"""
Simple CLIP-based analyzer for detecting fuel truck interactions with planes
"""

import torch
import clip
from PIL import Image
import cv2
import numpy as np
import logging
import argparse
from datetime import datetime
import os
from pathlib import Path
import json
from typing import List, Dict, Tuple

from src.pipeline.core.video_loader import VideoLoader

# Set up logging
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/clip_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

class CLIPAnalyzer:
    def __init__(self, device: str = None):
        """Initialize CLIP analyzer"""
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else \
                    "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load higher resolution CLIP model
        logging.info(f"Loading CLIP model (ViT-L/14@336px) on {device}...")
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device=device)
        
        # Define more specific prompts
        self.prompts = [
            # Positive prompts for refueling
            "a close-up view of an aviation fuel hose connected to an aircraft wing",
            "an airport fuel truck with its refueling hose attached to a plane's fuel port",
            "a commercial aircraft being refueled through a visible fuel line connection",
            "a clear view of an airplane's refueling point with a fuel hose attached",
            # Contextual prompts
            "airport ground crew performing aircraft refueling operations",
            "fuel truck positioned next to aircraft wing during refueling",
            # Negative prompts
            "a plane parked at the gate without any fuel trucks",
            "airport ground equipment not connected to aircraft",
            "passengers boarding or deplaning an aircraft",
            "baggage and cargo loading operations"
        ]
        
        # Pre-compute text features
        self.num_positive = 6  # First 6 prompts are positive
        self.text_features = self._encode_text(self.prompts)
        
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts into CLIP space"""
        text_tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
        
    def analyze_frame(self, frame: np.ndarray, threshold: float = 0.2) -> Tuple[bool, float]:
        """
        Analyze a frame to detect refueling activity
        Returns: (is_refueling, confidence_score)
        """
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create multiple crops for analysis
        crops = []
        
        # Center crop
        center_crop = frame
        
        # If frame is large enough, add zoomed crops of left and right sides
        if w > 1000:
            left_third = frame[:, :w//3]
            right_third = frame[:, -w//3:]
            crops.extend([left_third, center_crop, right_third])
        else:
            crops = [center_crop]
            
        max_score = -float('inf')
        best_result = (False, 0.0)
        
        # Analyze each crop
        for i, crop in enumerate(crops):
            # Convert frame to PIL Image
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            # Preprocess and encode image
            image_input = self.preprocess(crop_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                
            # Calculate scores
            positive_score = similarity[0, :self.num_positive].mean().item()
            negative_score = similarity[0, self.num_positive:].mean().item()
            score_diff = positive_score - negative_score
            
            # Log scores for this crop
            crop_name = ['left', 'center', 'right'][i] if len(crops) > 1 else 'full'
            logging.info(f"\nCLIP Scores for {crop_name} crop:")
            for prompt, score in zip(self.prompts, similarity[0]):
                logging.info(f"- {prompt}: {score:.3f}")
            logging.info(f"Positive avg: {positive_score:.3f}")
            logging.info(f"Negative avg: {negative_score:.3f}")
            logging.info(f"Score diff: {score_diff:.3f}")
            
            # Keep track of best score
            if score_diff > max_score:
                max_score = score_diff
                best_result = (score_diff > threshold, score_diff)
                
        return best_result

def main():
    parser = argparse.ArgumentParser(description='CLIP-based Refueling Detection')
    parser.add_argument('input', help='Path to input video file or YouTube URL')
    parser.add_argument('--output-dir', default='output',
                      help='Directory to save results (default: output)')
    parser.add_argument('--cache-dir', default='cache',
                      help='Directory to cache downloaded videos (default: cache)')
    parser.add_argument('--skip-frames', type=int, default=1000,
                      help='Number of frames to skip between analysis (default: 1000)')
    parser.add_argument('--threshold', type=float, default=0.2,
                      help='CLIP score threshold (default: 0.2)')
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging()
    logging.info("="*50)
    logging.info("CLIP Refueling Detection Started")
    logging.info("="*50)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize video loader and CLIP analyzer
    video = VideoLoader(args.input, cache_dir=args.cache_dir)
    if not video.open():
        logging.error(f"Error: Could not open video source '{args.input}'")
        return
        
    analyzer = CLIPAnalyzer()
    
    # Process video
    frame_idx = 0
    detections = []
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        if frame_idx % args.skip_frames == 0:
            logging.info(f"\nProcessing frame {frame_idx}")
            
            try:
                # Analyze frame
                is_refueling, score = analyzer.analyze_frame(frame, args.threshold)
                
                if is_refueling:
                    logging.info("✓ Detected refueling activity!")
                    detection = {
                        'frame': frame_idx,
                        'score': score,
                        'timestamp': frame_idx / video.get_info()[2]  # Convert to seconds
                    }
                    detections.append(detection)
                    
                    # Save frame
                    output_path = output_dir / f"refueling_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(output_path), frame)
                else:
                    logging.info("✗ No refueling activity detected")
                    
            except Exception as e:
                logging.error(f"Error processing frame {frame_idx}: {str(e)}")
                
        frame_idx += 1
        
    # Cleanup
    video.release()
    
    # Save detections
    if detections:
        output_path = output_dir / "refueling_detections.json"
        with open(output_path, 'w') as f:
            json.dump(detections, f, indent=2)
            
    logging.info("\nProcessing complete!")
    logging.info(f"Found {len(detections)} potential refueling events")
    logging.info(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
