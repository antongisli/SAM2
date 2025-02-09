#!/usr/bin/env python3

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'

import cv2
import numpy as np
import torch
import supervision as sv
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import open_clip
import yt_dlp
import os
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import logging
from datetime import datetime
from PIL import Image
import urllib.parse
import re
import time
import platform
import sys
import json
import clip
from scipy.spatial import cKDTree

# Set up logging
import logging
import sys
from datetime import datetime

# Create logs directory if it doesn't exist
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
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log system info
logger.info("="*50)
logger.info("Video Analysis Pipeline Started")
logger.info("="*50)
logger.info(f"Log file: {log_file}")
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"OpenCV version: {cv2.__version__}")
logger.info(f"NumPy version: {np.__version__}")
logger.info("="*50)

class VideoAnalysisPipeline:
    def __init__(self, sam_type='vit_b'):
        """Initialize the pipeline with specified SAM model type"""
        self.sam_type = sam_type
        
        # Set environment variables for MPS
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.5'
        
        # Device setup
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device.upper()} acceleration")
        
        logger.info("\n=== Setting up models ===")
        
        # Initialize models
        sam_checkpoint = self.download_sam_model()
        self.setup_models(sam_checkpoint)
        
        # Initialize counters and logs
        self.frame_count = 0
        self.interaction_log = []
        
        # Define text prompts for CLIP
        self.text_inputs = [
            "an airport fuel truck refueling an aircraft",
            "a fuel tanker truck at an airport",
            "an aviation fuel truck",
            "a regular delivery truck",
            "a cargo truck",
            "a maintenance vehicle",
            "a baggage truck",
            "a catering truck"
        ]
        
        # Get text features for fuel truck classification
        text_tokens = clip.tokenize(self.text_inputs).to(self.device)
        with torch.no_grad():
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            
        # First 3 prompts are positive (fuel trucks), rest are negative
        self.num_positive_prompts = 3

    def download_sam_model(self) -> str:
        """Download SAM model checkpoint if not present"""
        checkpoints = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_b': 'sam_vit_b_01ec64.pth'
        }
        
        checkpoint_file = checkpoints[self.sam_type]
        checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            logger.info(f"Downloading {checkpoint_file}...")
            url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_file}"
            
            try:
                import urllib.request
                urllib.request.urlretrieve(url, checkpoint_path)
                logger.info(f"Downloaded {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to download checkpoint: {str(e)}")
                raise
        
        return checkpoint_path

    def setup_models(self, sam_checkpoint):
        # Initialize SAM
        logger.info(f"Loading SAM model ({self.sam_type})...")
        t0 = time.time()
        sam = sam_model_registry[self.sam_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        logger.info(f"SAM model loaded in {time.time()-t0:.2f}s")

        # Initialize YOLO
        logger.info("Loading YOLO model (will download if not found)...")
        t0 = time.time()
        self.detector = YOLO('yolov8x.pt')
        self.detector.to(self.device)
        logger.info(f"YOLO model loaded in {time.time()-t0:.2f}s")

        # Initialize CLIP
        logger.info("Loading CLIP model...")
        t0 = time.time()
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        logger.info(f"CLIP model loaded in {time.time()-t0:.2f}s")

        # Other settings
        self.input_size = 1024

    def mask_min_distance(self, mask1: np.ndarray, mask2: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        """Calculate minimum distance between SAM masks using spatial partitioning."""
        t0 = time.time()
        
        # Get coordinates with subsampling (every 4th point)
        y1, x1 = np.where(mask1[::4, ::4])
        y1 = y1 * 4  # Scale back to original coordinates
        x1 = x1 * 4
        
        y2, x2 = np.where(mask2[::4, ::4])
        y2 = y2 * 4
        x2 = x2 * 4
        
        t_sampling = time.time() - t0
        
        if len(x1) == 0 or len(x2) == 0:
            logger.debug(f"Empty mask detected. Sampling time: {t_sampling*1000:.1f}ms")
            return float('inf'), float('inf'), (0, 0)

        # Build spatial index for mask2 points
        t_tree_start = time.time()
        mask2_points = np.column_stack((x2, y2))
        tree = cKDTree(mask2_points)
        t_tree = time.time() - t_tree_start
        
        # Find nearest neighbor for each mask1 point using parallel processing
        t_query_start = time.time()
        distances, _ = tree.query(np.column_stack((x1, y1)), workers=-1)
        t_query = time.time() - t_query_start
        
        if len(distances) == 0:
            return float('inf'), float('inf'), (0, 0)
        
        min_distance = np.min(distances)
        
        # Calculate plane dimensions from original mask (not subsampled)
        t_dims_start = time.time()
        y2_full, x2_full = np.where(mask2)
        plane_height = np.max(y2_full) - np.min(y2_full) if len(y2_full) > 0 else 0
        plane_width = np.max(x2_full) - np.min(x2_full) if len(x2_full) > 0 else 0
        plane_length = max(plane_height, plane_width)
        t_dims = time.time() - t_dims_start
        
        # Normalize distance by plane length
        norm_distance = min_distance / plane_length if plane_length > 0 else float('inf')
        
        total_time = time.time() - t0
        logger.info(f"\nDistance calculation timing:")
        logger.info(f"- Point sampling: {t_sampling*1000:.1f}ms")
        logger.info(f"- Tree building: {t_tree*1000:.1f}ms")
        logger.info(f"- Nearest neighbor query: {t_query*1000:.1f}ms")
        logger.info(f"- Dimension calculation: {t_dims*1000:.1f}ms")
        logger.info(f"- Total time: {total_time*1000:.1f}ms")
        logger.info(f"- Points processed: {len(x1)} truck, {len(x2)} plane")
        
        return min_distance, norm_distance, (plane_height, plane_width)

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Process a single frame through the detection pipeline"""
        timing = {}
        frame_time = self.frame_count / self.fps if hasattr(self, 'fps') else 0
        
        # Resize frame if needed
        t0 = time.time()
        h, w = frame.shape[:2]
        if max(h, w) > self.input_size:
            scale = self.input_size / max(h, w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        
        # Convert frame to RGB for YOLO and CLIP
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timing['preprocessing'] = time.time() - t0
        
        # Step 1: YOLO Detection (coarse-grained)
        t0 = time.time()
        results = self.detector(rgb_frame)
        timing['yolo'] = time.time() - t0
        
        # Extract vehicles and planes
        vehicles = [
            {
                'box': box.xyxy[0].cpu().numpy(),
                'score': box.conf[0].cpu().numpy(),
                'class': box.cls[0].cpu().numpy()
            }
            for box in results[0].boxes
            if box.cls[0].cpu().numpy() in [2, 7]  # cars/trucks
        ]
        planes = [
            {
                'box': box.xyxy[0].cpu().numpy(),
                'score': box.conf[0].cpu().numpy()
            }
            for box in results[0].boxes
            if box.cls[0].cpu().numpy() == 5  # planes
        ]
        
        # Step 2: CLIP Verification for vehicles (batch processing)
        t0 = time.time()
        fuel_trucks = []
        if vehicles:
            # Prepare batch of vehicle crops
            vehicle_crops = []
            for vehicle in vehicles:
                x1, y1, x2, y2 = map(int, vehicle['box'])
                if x1 >= x2 or y1 >= y2:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_pil = Image.fromarray(crop)
                vehicle_crops.append(self.clip_preprocess(crop_pil))
            
            if vehicle_crops:
                # Batch process all vehicles
                vehicle_batch = torch.stack(vehicle_crops).to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(vehicle_batch)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                
                # Process results
                for i, (vehicle, scores) in enumerate(zip(vehicles, similarity)):
                    positive_score = scores[:self.num_positive_prompts].mean().item()
                    negative_score = scores[self.num_positive_prompts:].mean().item()
                    score_diff = positive_score - negative_score
                    
                    logger.info(f"\nCLIP Scores for vehicle {i+1}:")
                    for prompt, score in zip(self.text_inputs, scores):
                        logger.info(f"- {prompt}: {score.item():.3f}")
                    logger.info(f"Positive avg: {positive_score:.3f}, Negative avg: {negative_score:.3f}")
                    
                    if score_diff > 0.3:
                        vehicle['clip_score'] = score_diff
                        fuel_trucks.append(vehicle)
        
        timing['clip'] = time.time() - t0
        
        # Step 3: SAM-based proximity analysis (batch processing)
        t0 = time.time()
        interactions = []
        
        # Log detections for debugging
        if vehicles:
            logger.info(f"\nFrame {self.frame_count}: Found {len(vehicles)} vehicles and {len(planes)} planes")
            for v in vehicles:
                logger.info(f"Vehicle: YOLO score={v['score']:.2f}")
        
        if fuel_trucks and planes:
            # Convert frame to RGB for SAM
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(rgb_frame)
            
            # Get all masks in one batch
            truck_boxes = [truck['box'] for truck in fuel_trucks]
            plane_boxes = [plane['box'] for plane in planes]
            
            # Process trucks
            truck_masks = []
            for truck_box in truck_boxes:
                mask, _, _ = self.sam_predictor.predict(box=truck_box)
                truck_masks.append(mask[0].astype(bool))
            
            # Process planes
            plane_masks = []
            for plane_box in plane_boxes:
                mask, _, _ = self.sam_predictor.predict(box=plane_box)
                plane_masks.append(mask[0].astype(bool))
            
            # Check all interactions
            for truck_idx, (truck, truck_mask) in enumerate(zip(fuel_trucks, truck_masks)):
                for plane_idx, (plane, plane_mask) in enumerate(zip(planes, plane_masks)):
                    # Calculate SAM-based distances
                    min_dist, norm_dist, plane_dims = self.mask_min_distance(truck_mask, plane_mask)
                    
                    # Always log spatial metrics
                    logger.info(
                        f"\nSpatial Analysis (Truck {truck_idx+1} - Plane {plane_idx+1}):"
                        f"\n- Truck-Plane Distance: {min_dist:.1f}px"
                        f"\n- Normalized Distance: {norm_dist:.2f}x plane length"
                        f"\n- Plane Size: {plane_dims[0]}x{plane_dims[1]}px"
                    )
                    
                    if norm_dist < 0.5:  # Within half a plane length
                        interaction = {
                            'frame': self.frame_count,
                            'time': frame_time,
                            'fuel_truck': {
                                'position': truck['box'].tolist(),
                                'yolo_score': float(truck['score']),
                                'clip_score': float(truck['clip_score']),
                                'mask_area': int(np.sum(truck_mask))
                            },
                            'plane': {
                                'position': plane['box'].tolist(),
                                'yolo_score': float(plane['score']),
                                'mask_area': int(np.sum(plane_mask))
                            },
                            'spatial_metrics': {
                                'min_px_distance': float(min_dist),
                                'normalized_distance': float(norm_dist),
                                'plane_dimensions': [int(d) for d in plane_dims],
                                'threshold': 0.5
                            }
                        }
                        interactions.append(interaction)
                        
                        # Log interaction details
                        logger.info(f"\nInteraction detected!")
                        logger.info(f"- Time: {frame_time:.1f}s")
                        logger.info(f"- Fuel truck: YOLO={truck['score']:.2f}, CLIP={truck['clip_score']:.2f}")
                        logger.info(f"- Distance: {min_dist:.1f}px ({norm_dist:.2f}x plane length)")
        
        timing['sam'] = time.time() - t0
        
        # Log timing information
        total_time = sum(timing.values())
        timing_str = "\n=== Frame Processing Times ===\n"
        for step, t in timing.items():
            pct = (t / total_time) * 100
            timing_str += f"{step}: {t:.3f}s ({pct:.1f}%)\n"
        timing_str += f"Total: {total_time:.3f}s"
        logger.info(timing_str)
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame, fuel_trucks, planes, interactions)
        
        self.frame_count += 1
        return interactions, annotated_frame

    def verify_vehicle_type(self, vehicle: Dict, frame: np.ndarray) -> float:
        """Verify if a vehicle is a fuel truck using CLIP with multiple prompts"""
        # Extract vehicle crop
        x1, y1, x2, y2 = map(int, vehicle['box'])
        if x1 >= x2 or y1 >= y2:
            return 0.0
            
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
            
        # Convert to PIL and preprocess for CLIP
        crop_pil = Image.fromarray(crop)
        crop_tensor = self.clip_preprocess(crop_pil).unsqueeze(0).to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(crop_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity with all concepts
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            
            # Average score for positive prompts (first num_positive_prompts)
            positive_score = similarity[0, :self.num_positive_prompts].mean().item()
            
            # Average score for negative prompts
            negative_score = similarity[0, self.num_positive_prompts:].mean().item()
            
            # Log the scores for debugging
            logger.info(f"\nCLIP Scores for vehicle:")
            for i, (prompt, score) in enumerate(zip(self.text_inputs, similarity[0])):
                logger.info(f"- {prompt}: {score.item():.3f}")
            logger.info(f"Positive avg: {positive_score:.3f}, Negative avg: {negative_score:.3f}")
            
            # Return positive score only if it's significantly higher than negative score
            score_diff = positive_score - negative_score
            return score_diff if score_diff > 0 else 0.0

    def verify_with_gpt_stub(self, interaction: dict) -> dict:
        """Stub for GPT-4 verification system"""
        return {
            **interaction,
            'gpt_verification': {
                'called': interaction['needs_gpt_verification'],
                'verified': False,  # Would be set by actual GPT-4 call
                'analysis': "Stub: Would send to GPT-4 for verification" if interaction['needs_gpt_verification'] else "Not required",
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }

    def process_video(self, input_path: str, output_dir: str, skip_frames: int = 10):
        """Process a video file and save results"""
        logger.info("\n" + "="*50)
        logger.info("Starting video processing")
        logger.info(f"Input video: {input_path}")
        logger.info("="*50 + "\n")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info("\n=== Video Properties ===")
        logger.info(f"Resolution: {width}x{height}")
        logger.info(f"FPS: {fps:.2f}")
        logger.info(f"Frame count: {frame_count}")
        logger.info(f"Duration: {duration:.2f}s ({duration/60:.1f}min)")
        logger.info(f"Processing every {skip_frames}th frame")
        logger.info("="*22 + "\n")
        
        # Store fps as class attribute for frame time calculation
        self.fps = fps
        
        # Create output video writer
        output_file = os.path.join(output_dir, 'output.mp4')
        
        # Use x264 codec for better compatibility
        if platform.system() == 'Darwin':  # macOS
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        out = cv2.VideoWriter(output_file, 
                            fourcc,
                            fps/skip_frames,  # Adjust output FPS based on frame skipping
                            (width, height))
        
        if not out.isOpened():
            logger.error("Failed to create output video writer")
            raise RuntimeError("Could not initialize video writer")
            
        logger.info(f"Created output video writer: {output_file}")
        
        frame_idx = 0
        processed_frames = 0
        start_time = time.time()
        processing_times = []
        last_log_time = time.time()
        log_interval = 2.0  # Log every 2 seconds
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames based on skip_frames parameter
                if frame_idx % skip_frames != 0:
                    frame_idx += 1
                    continue
                    
                # Calculate current video time
                current_time = frame_idx / fps
                frame_start_time = time.time()
                
                # Only log every log_interval seconds to avoid spam
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    progress = (frame_idx / frame_count) * 100
                    remaining_frames = (frame_count - frame_idx) / skip_frames  # Account for frame skipping
                    avg_time = np.mean(processing_times) if processing_times else 0
                    eta = remaining_frames * avg_time
                    logger.info(
                        f"\n=== Progress: {progress:.1f}% ===\n"
                        f"Frame: {frame_idx}/{frame_count}\n"
                        f"Video time: {frame_idx/fps:.1f}s/{duration:.1f}s\n"
                        f"Processing speed: {1/avg_time:.1f} FPS\n"
                        f"ETA: {eta:.1f}s ({eta/60:.1f}min)\n"
                    )
                    last_log_time = current_time
                
                # Process frame
                interactions, annotated_frame = self.process_frame(frame)
                
                # Calculate and log processing speed
                frame_processing_time = time.time() - frame_start_time
                processing_times.append(frame_processing_time)
                avg_processing_time = np.mean(processing_times[-100:])  # Average over last 100 frames
                
                processed_frames += 1
                elapsed_time = time.time() - start_time
                effective_fps = processed_frames / elapsed_time
                
                if len(interactions) > 0:
                    logger.info(
                        f"Found {len(interactions)} interactions: " + 
                        ", ".join([f"Fuel truck at {i['fuel_truck']['position']} with plane at {i['plane']['position']}" for i in interactions])
                    )
                
                # Write frame to output video
                out.write(annotated_frame)
                frame_idx += 1
                
        except KeyboardInterrupt:
            logger.info("\nProcessing interrupted by user")
        finally:
            # Clean up
            cap.release()
            out.release()
            
            # Log final statistics
            total_time = time.time() - start_time
            logger.info("\n" + "="*50)
            logger.info("Processing complete!")
            logger.info(f"Total frames processed: {processed_frames}")
            logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f}min)")
            logger.info(f"Average processing time per frame: {np.mean(processing_times):.3f}s")
            logger.info(f"Average FPS: {processed_frames/total_time:.2f}")
            logger.info(f"Output saved to: {output_file}")
            logger.info("="*50 + "\n")
            
            # Save interaction log with verification stub
            if self.interaction_log:
                log_file = os.path.join(output_dir, 'analysis.json')
                try:
                    metadata = {
                        'video_path': input_path,
                        'duration': duration,
                        'total_frames': frame_count,
                        'processed_frames': self.frame_count,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    with open(log_file, 'w') as f:
                        json.dump({
                            'metadata': metadata,
                            'interactions': [
                                self.verify_with_gpt_stub(interaction) 
                                for interaction in self.interaction_log
                            ],
                            'stats': {
                                'total_frames': frame_count,
                                'frames_processed': self.frame_count,
                                'interactions_found': len(self.interaction_log),
                                'gpt_verification_needed': sum(
                                    1 for i in self.interaction_log 
                                    if i['needs_gpt_verification']
                                )
                            }
                        }, f, indent=2)
                    logger.info(f"\nSaved analysis to {log_file}")
                except Exception as e:
                    logger.error(f"Failed to save analysis: {str(e)}")
                    
        return output_file

    def annotate_frame(self, frame: np.ndarray, fuel_trucks: List[Dict], planes: List[Dict], interactions: List[Dict]) -> np.ndarray:
        """Annotate frame with bounding boxes and labels"""
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and labels for fuel trucks
        for truck in fuel_trucks:
            x1, y1, x2, y2 = map(int, truck['box'])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Fuel Truck ({truck['score']:.2f})"
            cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw bounding boxes and labels for planes
        for plane in planes:
            x1, y1, x2, y2 = map(int, plane['box'])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Plane ({plane['score']:.2f})"
            cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw interaction lines
        for interaction in interactions:
            truck_box = interaction['fuel_truck']['position']
            plane_box = interaction['plane']['position']
            cv2.line(annotated_frame, (int((truck_box[0] + truck_box[2])/2), int((truck_box[1] + truck_box[3])/2)), 
                     (int((plane_box[0] + plane_box[2])/2), int((plane_box[1] + plane_box[3])/2)), 
                     (255, 255, 0), 2)
        
        return annotated_frame

    def check_proximity(self, box1: List[float], box2: List[float], threshold: float = 0.3) -> bool:
        """Check if two bounding boxes are close to each other"""
        def box_center(box):
            return [(box[0] + box[2])/2, (box[1] + box[3])/2]
        
        center1 = box_center(box1)
        center2 = box_center(box2)
        
        # Calculate distance between centers
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Calculate diagonal of first box as reference
        box1_diag = np.sqrt((box1[2] - box1[0])**2 + (box1[3] - box1[1])**2)
        
        return distance < box1_diag * threshold

    def calculate_distance(self, box1: List[float], box2: List[float]) -> float:
        """Calculate normalized distance between two boxes"""
        def box_center(box):
            return [(box[0] + box[2])/2, (box[1] + box[3])/2]
        
        center1 = box_center(box1)
        center2 = box_center(box2)
        
        # Calculate distance between centers
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Normalize by the diagonal of the first box
        box1_diag = np.sqrt((box1[2] - box1[0])**2 + (box1[3] - box1[1])**2)
        
        return distance / box1_diag

def main():
    parser = argparse.ArgumentParser(description='Video Analysis Pipeline')
    parser.add_argument('--youtube_url', type=str, required=True, help='YouTube video URL')
    parser.add_argument('--skip_frames', type=int, default=10, help='Process every Nth frame')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--model_type', type=str, default='vit_b', choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = VideoAnalysisPipeline(sam_type=args.model_type)
    
    # Download video
    logger.info("\nDownloading video from YouTube...")
    try:
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]',  # Best quality MP4
            'outtmpl': 'downloads/input.mp4',  # Output template
            'quiet': True,
            'no_warnings': True
        }
        
        # Create downloads directory
        os.makedirs('downloads', exist_ok=True)
        
        # Download video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([args.youtube_url])
        
        video_path = 'downloads/input.mp4'
        logger.info(f"Video downloaded to: {video_path}")
    except Exception as e:
        logger.error(f"Failed to download video: {str(e)}")
        return

    # Process video
    try:
        output_file = pipeline.process_video(video_path, args.output_dir, args.skip_frames)
        logger.info(f"\nProcessing complete! Output saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

if __name__ == '__main__':
    main()
