#!/usr/bin/env python3

import os
import cv2
import json
import torch
import logging
import argparse
import contextlib
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from typing import List, Tuple, Dict
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
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

class FlorenceAnalyzer:
    def __init__(self, device: str = 'mps', yolo_threshold: float = 0.3, florence_threshold: float = 0.2):
        """
        Initialize the analyzer
        Args:
            device: Device to run models on ('cuda', 'cpu', 'mps')
            yolo_threshold: Minimum confidence for YOLO detections (0-1)
            florence_threshold: Minimum score for Florence verification (0-1)
        """
        self.device = device
        
        # Load Florence-2
        logging.info(f"Loading Florence-2 model on {device}...")
        model_id = "microsoft/florence-2-large"
        
        # Use token from environment variable if available
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token is None:
            logging.warning("HUGGINGFACE_TOKEN environment variable not set. You may encounter authentication issues.")
            
        self.processor = AutoProcessor.from_pretrained(model_id, token=token, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device).eval()
        
        # Load YOLO
        logging.info(f"Loading YOLO model...")
        self.yolo = YOLO('yolov8x.pt')
        self.yolo.verbose = False
        
        # Thresholds
        self.yolo_threshold = yolo_threshold
        self.florence_threshold = florence_threshold
        
        # Define the task prompt for fuel truck detection
        self.task_prompt = "Is there a fuel truck or refueling vehicle in this image? Answer yes or no and explain why."

    def verify_truck(self, frame: np.ndarray, box: List[float]) -> Tuple[bool, float]:
        """Verify if a detected truck is a fuel truck"""
        x1, y1, x2, y2 = map(int, box)
        truck_img = frame[y1:y2, x1:x2]
        
        # Convert to PIL and preprocess
        if truck_img.size == 0:
            return False, 0.0
            
        pil_image = Image.fromarray(cv2.cvtColor(truck_img, cv2.COLOR_BGR2RGB))
        
        with torch.no_grad():
            # Process image with Florence-2
            inputs = self.processor(
                text=self.task_prompt,
                images=pil_image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                early_stopping=True,
                do_sample=False,
                num_beams=1
            )
            
            # Decode and post-process
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            answer = self.processor.post_process_generation(
                generated_text,
                task=self.task_prompt,
                image_size=(pil_image.width, pil_image.height)
            )
            
            # Parse the answer
            is_fuel_truck = answer.lower().startswith("yes")
            # Use a confidence of 1.0 if yes, 0.0 if no since Florence-2 doesn't provide scores
            score = 1.0 if is_fuel_truck else 0.0
            
            logging.debug(f"Florence-2 response: {answer}")
            
            return bool(is_fuel_truck), float(score)
    
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
                        # Verify if it's a fuel truck using Florence
                        is_fuel_truck, florence_score = self.verify_truck(frame, [x1, y1, x2, y2])
                        
                        if is_fuel_truck:
                            # Add detection
                            detection = {
                                'type': 'fuel_truck',
                                'box': [x1, y1, x2, y2],
                                'yolo_conf': conf,
                                'florence_score': florence_score
                            }
                            detections.append(detection)
                            
                            # Draw fuel truck box in green
                            cv2.rectangle(annotated_frame, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        (0, 255, 0), 2)
                            text = f"Fuel Truck: {florence_score:.2f}"
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
    parser = argparse.ArgumentParser(description='Detect fuel trucks and airplanes in video')
    parser.add_argument('input_path', help='Path to input video file or YouTube URL')
    parser.add_argument('--skip-frames', type=int, default=30, help='Number of frames to skip')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--cache-dir', default='cache', help='Cache directory for downloaded videos')
    parser.add_argument('--yolo-threshold', type=float, default=0.3, help='Minimum confidence for YOLO detections')
    parser.add_argument('--florence-threshold', type=float, default=0.2, help='Minimum score for Florence verification')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    log_file = setup_logging(args.output_dir, level=log_level)
    
    # Initialize analyzer
    analyzer = FlorenceAnalyzer(yolo_threshold=args.yolo_threshold, florence_threshold=args.florence_threshold)
    
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
                
                if frame_number % args.skip_frames == 0:
                    # Analyze frame
                    frame_detections, annotated_frame = analyzer.analyze_frame(frame)
                    
                    # Add detections
                    for detection in frame_detections:
                        detection['frame'] = frame_number
                        detections.append(detection)
                        
                    # Write annotated frame to video
                    out.write(annotated_frame)
                    
                frame_number += 1
                pbar.update(1)
                
    finally:
        video_loader.release()
        out.release()
        
    # Save detection results
    results_file = os.path.join(args.output_dir, 'detections.json')
    with open(results_file, 'w') as f:
        json.dump(detections, f, indent=2)
        
    logging.info(f"\nAnalysis complete!")
    logging.info(f"Found {len(detections)} potential fuel trucks")
    logging.info(f"Results saved to {args.output_dir}")
    logging.info(f"Output video saved to {output_path}")

if __name__ == '__main__':
    main()
