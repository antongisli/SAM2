"""
CLIP-based object verification
"""

import torch
import clip
import numpy as np
from PIL import Image
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class CLIPVerifier:
    def __init__(self, config):
        """Initialize CLIP verifier with configuration"""
        self.config = config
        logger.info(f"Loading CLIP model ({config.clip_model})...")
        self.model, self.preprocess = clip.load(config.clip_model)
        self.model.to(config.device)
        
        # For planes: first 4 prompts are positive, rest are negative
        self.num_plane_positive = 4
        plane_prompts = self.config.plane_prompts
        self.plane_text_features = self._encode_text(plane_prompts)
        
        # For fuel trucks: first 3 prompts are positive, rest are negative
        self.num_fuel_truck_positive = 3
        fuel_truck_prompts = self.config.fuel_truck_prompts
        self.fuel_truck_text_features = self._encode_text(fuel_truck_prompts)
        
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts into CLIP space"""
        text_tokens = clip.tokenize(prompts).to(self.config.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def _verify_objects(self, frame: np.ndarray, 
                       detections: List[Dict], 
                       text_features: torch.Tensor,
                       object_type: str, num_positive_prompts: int) -> List[Dict]:
        """Verify objects using CLIP"""
        if not detections:
            logger.info(f"No {object_type} candidates to verify")
            return []
            
        logger.info(f"\nVerifying {len(detections)} {object_type} candidates with CLIP")
            
        # Prepare batch of crops
        crops = []
        valid_detections = []
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = map(int, detection['box'])
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid box dimensions for {object_type} {i+1}")
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                logger.warning(f"Empty crop for {object_type} {i+1}")
                continue
            crop_pil = Image.fromarray(crop)
            crops.append(self.preprocess(crop_pil))
            valid_detections.append(detection)
            
        if not crops:
            logger.warning(f"No valid crops for {object_type} verification")
            return []
            
        # Batch process
        image_batch = torch.stack(crops).to(self.config.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
        verified_objects = []
        for i, (detection, scores) in enumerate(zip(valid_detections, similarity)):
            positive_score = scores[:num_positive_prompts].mean().item()
            negative_score = scores[num_positive_prompts:].mean().item()
            score_diff = positive_score - negative_score
            
            logger.info(f"\nCLIP Scores for {object_type} candidate {i+1}:")
            logger.info(f"Box: {[int(x) for x in detection['box']]}")
            logger.info(f"YOLO confidence: {detection['score']:.3f}")
            if object_type == "plane":
                for j, (prompt, score) in enumerate(zip(self.config.plane_prompts, scores)):
                    logger.info(f"- {prompt}: {score.item():.3f}")
            else:
                for j, (prompt, score) in enumerate(zip(self.config.fuel_truck_prompts, scores)):
                    logger.info(f"- {prompt}: {score.item():.3f}")
            logger.info(f"Positive avg: {positive_score:.3f}, Negative avg: {negative_score:.3f}")
            logger.info(f"Score diff: {score_diff:.3f} (threshold: {self.config.clip_threshold})")
            
            if score_diff > self.config.clip_threshold:
                detection['clip_score'] = score_diff
                verified_objects.append(detection)
                logger.info(f"✓ Verified as {object_type}")
            else:
                logger.info(f"✗ Not verified as {object_type}")
                
        logger.info(f"\nVerified {len(verified_objects)}/{len(detections)} {object_type}s")
        return verified_objects
    
    def verify_planes(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Verify planes using CLIP"""
        return self._verify_objects(frame, detections, self.plane_text_features, "plane", self.num_plane_positive)
    
    def verify_fuel_trucks(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Verify fuel trucks using CLIP"""
        return self._verify_objects(frame, detections, self.fuel_truck_text_features, "fuel_truck", self.num_fuel_truck_positive)
