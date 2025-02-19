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
from tqdm import tqdm
from ultralytics import YOLO
import contextlib

from src.pipeline.core.video_loader import VideoLoader

# Set up logging
def setup_logging(output_dir: str, level=logging.INFO) -> str:
    """Setup logging to both file and console"""
    # Create logs directory
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'detection_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

class CLIPAnalyzer:
    def __init__(self, device: str = 'mps', yolo_threshold: float = 0.9, clip_threshold: float = 0.31):
        """Initialize CLIP analyzer"""
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else \
                    "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load CLIP model
        logging.info(f"Loading CLIP model (ViT-L/14@336px) on {device}...")
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device=device)
        
        # Load YOLO model
        logging.info("Loading YOLO model...")
        self.yolo = YOLO('yolov8x.pt')
        self.yolo.verbose = False
        
        # Thresholds
        self.yolo_threshold = yolo_threshold  # Minimum confidence for YOLO detections
        self.clip_threshold = clip_threshold  # Minimum score difference for CLIP verification
        
        # Define prompts for fuel truck verification
        self.prompts = [
            # Positive prompts
            "a truck carrying fuel or liquid",
            "a fuel tanker truck",
            "an airport refueling truck",
            # Negative prompts
            "a regular delivery truck",
            "a box truck",
            "a flatbed truck",
            "a garbage truck",
            "a pickup truck"
        ]
        
        # Pre-compute text features
        self.num_positive = 3  # First 3 prompts are positive
        self.text_features = self._encode_text(self.prompts)
        
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts with CLIP"""
        with torch.no_grad():
            text_tokens = clip.tokenize(prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
        
    def verify_truck(self, frame: np.ndarray, box: List[float]) -> Tuple[bool, float]:
        """Verify if a detected truck is a fuel truck"""
        x1, y1, x2, y2 = map(int, box)
        truck_img = frame[y1:y2, x1:x2]
        
        # Convert to PIL and preprocess
        truck_pil = Image.fromarray(cv2.cvtColor(truck_img, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(truck_pil).unsqueeze(0).to(self.device)
        
        # Get CLIP features and similarity
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            
        # Calculate scores
        positive_score = similarity[0, :self.num_positive].mean().item()
        negative_score = similarity[0, self.num_positive:].mean().item()
        score_diff = positive_score - negative_score
        
        # Log scores for debugging
        logging.debug("\nCLIP Scores for truck:")
        for i, (prompt, score) in enumerate(zip(self.prompts, similarity[0])):
            logging.debug(f"- {prompt}: {score:.3f}")
        logging.debug(f"Positive avg: {positive_score:.3f}")
        logging.debug(f"Negative avg: {negative_score:.3f}")
        logging.debug(f"Score diff: {score_diff:.3f}")
        
        return score_diff > self.clip_threshold, score_diff
        
    def analyze_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Analyze a frame to detect fuel trucks and planes
        Returns: (detections, annotated_frame)
        """
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        # Detect trucks and planes with YOLO (silently)
        with contextlib.redirect_stdout(None):
            results = self.yolo(frame, classes=[4, 7], verbose=False)  # class 4=airplane, 7=truck
        
        detections = []
        
        # Process each detection
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates directly as list
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = box.cls[0].item()  # Get class ID
                conf = box.conf[0].item()  # Get confidence
                
                # Log all detections
                logging.debug(f"\nYOLO Detection:")
                logging.debug(f"- Class: {'airplane' if cls == 4 else 'truck' if cls == 7 else 'unknown'}")
                logging.debug(f"- Confidence: {conf:.3f}")
                logging.debug(f"- Threshold: {self.yolo_threshold}")
                
                if conf > self.yolo_threshold:  # Confidence threshold
                    logging.debug("  Passed threshold!")
                    # Check for truck
                    if cls == 7:  # Truck
                        # Verify if it's a fuel truck using CLIP
                        is_fuel_truck, clip_score = self.verify_truck(frame, [x1, y1, x2, y2])
                        
                        if is_fuel_truck:
                            # Add detection
                            detection = {
                                'type': 'fuel_truck',
                                'box': [x1, y1, x2, y2],
                                'yolo_conf': conf,
                                'clip_score': clip_score
                            }
                            detections.append(detection)
                            
                            # Draw fuel truck box in green
                            cv2.rectangle(annotated_frame, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        (0, 255, 0), 2)
                            text = f"Fuel Truck: {clip_score:.2f}"
                            cv2.putText(annotated_frame, text, 
                                      (int(x1), int(y1)-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Check for airplane (independent of truck check)
                    if cls == 4:  # Airplane
                        logging.debug("  Drawing airplane box!")
                        # Add detection
                        detection = {
                            'type': 'airplane',
                            'box': [x1, y1, x2, y2],
                            'yolo_conf': conf
                        }
                        detections.append(detection)
                        
                        # Draw airplane box in blue
                        cv2.rectangle(annotated_frame, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (255, 0, 0), 2)
                        text = f"Airplane: {conf:.2f}"
                        cv2.putText(annotated_frame, text, 
                                  (int(x1), int(y1)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return detections, annotated_frame

def main():
    parser = argparse.ArgumentParser(description='CLIP-based Refueling Detection')
    parser.add_argument('input_path', help='Path to video file or YouTube URL')
    parser.add_argument('--skip-frames', type=int, default=1000, help='Number of frames to skip between analyses')
    parser.add_argument('--threshold', type=float, default=0.2, help='Confidence threshold for detection')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory for downloaded videos')
    parser.add_argument('--yolo-threshold', type=float, default=0.3, help='Minimum confidence for YOLO detections')
    parser.add_argument('--clip-threshold', type=float, default=0.2, help='Minimum score difference for CLIP verification')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    log_file = setup_logging(args.output_dir, level=log_level)
    
    # Initialize analyzer
    analyzer = CLIPAnalyzer(yolo_threshold=args.yolo_threshold, clip_threshold=args.clip_threshold)
    
    # Initialize video loader
    video_loader = VideoLoader(args.input_path, args.cache_dir)
    if not video_loader.open():
        logging.error("Error opening video stream or file")
        return
    cap = video_loader.cap
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    output_path = os.path.join(args.output_dir, 'output.mp4')
    codecs_to_try = ['avc1', 'h264', 'mp4v']
    out = None
    
    for codec in codecs_to_try:
        try:
            logging.info(f"Trying {codec} codec...")
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                logging.info(f"Successfully opened video writer with {codec} codec")
                break
            out.release()
        except Exception as e:
            logging.warning(f"Failed to use {codec} codec: {e}")
    
    if out is None or not out.isOpened():
        raise Exception("Failed to create video writer with any codec")
    
    try:
        # Process video frames
        frame_number = 0
        detections = []
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = video_loader.read()
                if not ret:
                    break
                    
                # Update progress bar
                pbar.update(1)
                
                if frame_number % args.skip_frames == 0:
                    # Analyze frame
                    frame_detections, annotated_frame = analyzer.analyze_frame(frame)
                    
                    # Add detections
                    for detection in frame_detections:
                        detection['frame_number'] = frame_number
                        detection['timestamp'] = frame_number / fps
                        detections.append(detection)
                        
                    # Write annotated frame to video
                    out.write(annotated_frame)
                # else:
                #     # Write original frame to video
                #     out.write(frame)
                    
                frame_number += 1
                
    finally:
        video_loader.release()
        out.release()
        
    # Save detection results
    results_path = os.path.join(args.output_dir, 'detections.json')
    with open(results_path, 'w') as f:
        json.dump({'detections': detections}, f, indent=2)
        
    logging.info(f"\nAnalysis complete!")
    logging.info(f"Found {len(detections)} potential fuel trucks")
    logging.info(f"Results saved to {args.output_dir}")
    logging.info(f"Output video saved to {output_path}")

if __name__ == '__main__':
    main()
