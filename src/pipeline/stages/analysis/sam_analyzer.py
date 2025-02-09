"""
SAM-based proximity analysis
"""

import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from scipy.spatial import cKDTree
import cv2
from typing import Dict, List, Tuple
import logging
import time

logger = logging.getLogger(__name__)

class SAMAnalyzer:
    def __init__(self, config):
        """Initialize SAM analyzer with configuration"""
        self.config = config
        logger.info(f"Loading SAM model ({config.sam_type})...")
        sam = sam_model_registry[config.sam_type](checkpoint=self._get_checkpoint())
        sam.to(device=config.device)
        self.predictor = SamPredictor(sam)
        
    def _get_checkpoint(self) -> str:
        """Get SAM checkpoint path"""
        checkpoints = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_b': 'sam_vit_b_01ec64.pth'
        }
        return checkpoints[self.config.sam_type]
        
    def mask_min_distance(self, mask1: np.ndarray, mask2: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        """Calculate minimum distance between two masks"""
        t0 = time.time()
        
        # Get coordinates with subsampling
        y1, x1 = np.where(mask1[::4, ::4])
        y1 = y1 * 4
        x1 = x1 * 4
        
        y2, x2 = np.where(mask2[::4, ::4])
        y2 = y2 * 4
        x2 = x2 * 4
        
        if len(x1) == 0 or len(x2) == 0:
            return float('inf'), float('inf'), (0, 0)
            
        # Build spatial index
        mask2_points = np.column_stack((x2, y2))
        tree = cKDTree(mask2_points)
        
        # Find nearest neighbor for each mask1 point
        distances, _ = tree.query(np.column_stack((x1, y1)), workers=-1)
        
        if len(distances) == 0:
            return float('inf'), float('inf'), (0, 0)
            
        min_distance = np.min(distances)
        
        # Calculate plane dimensions
        y2_full, x2_full = np.where(mask2)
        plane_height = np.max(y2_full) - np.min(y2_full) if len(y2_full) > 0 else 0
        plane_width = np.max(x2_full) - np.min(x2_full) if len(x2_full) > 0 else 0
        plane_length = max(plane_height, plane_width)
        
        # Normalize distance
        norm_distance = min_distance / plane_length if plane_length > 0 else float('inf')
        
        return min_distance, norm_distance, (plane_height, plane_width)
        
    def analyze_proximity(self, frame: np.ndarray, 
                        fuel_trucks: List[Dict], 
                        planes: List[Dict]) -> List[Dict]:
        """Analyze proximity between fuel trucks and planes"""
        if not fuel_trucks or not planes:
            return []
            
        # Set image for SAM
        self.predictor.set_image(frame)
        
        # Get masks for trucks and planes
        truck_masks = []
        for truck in fuel_trucks:
            mask, _, _ = self.predictor.predict(box=truck['box'])
            truck_masks.append(mask[0].astype(bool))
            
        plane_masks = []
        for plane in planes:
            mask, _, _ = self.predictor.predict(box=plane['box'])
            plane_masks.append(mask[0].astype(bool))
            
        # Analyze interactions
        interactions = []
        for truck_idx, (truck, truck_mask) in enumerate(zip(fuel_trucks, truck_masks)):
            for plane_idx, (plane, plane_mask) in enumerate(zip(planes, plane_masks)):
                min_dist, norm_dist, plane_dims = self.mask_min_distance(truck_mask, plane_mask)
                
                logger.info(
                    f"\nSpatial Analysis (Truck {truck_idx+1} - Plane {plane_idx+1}):"
                    f"\n- Distance: {min_dist:.1f}px ({norm_dist:.2f}x plane length)"
                )
                
                if norm_dist < self.config.proximity_threshold:
                    interaction = {
                        'fuel_truck': {
                            'position': truck['box'].tolist(),
                            'yolo_score': float(truck['score']),
                            'clip_score': float(truck['clip_score']),
                            'mask_area': int(np.sum(truck_mask))
                        },
                        'plane': {
                            'position': plane['box'].tolist(),
                            'yolo_score': float(plane['score']),
                            'clip_score': float(plane['clip_score']),
                            'mask_area': int(np.sum(plane_mask))
                        },
                        'spatial_metrics': {
                            'min_px_distance': float(min_dist),
                            'normalized_distance': float(norm_dist),
                            'plane_dimensions': [int(d) for d in plane_dims],
                            'threshold': self.config.proximity_threshold
                        }
                    }
                    interactions.append(interaction)
                    
        return interactions
