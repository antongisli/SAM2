#!/usr/bin/env python3

import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
import random
import utils
from utils import TaskType
import time

def generate_color():
    """Generate a random bright color"""
    return (
        random.randint(100, 255),
        random.randint(100, 255),
        random.randint(100, 255)
    )

def draw_detections(image, boxes, labels):
    """Draw bounding boxes and labels on the image"""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    class_colors = {}
    
    for box, label in zip(boxes, labels):
        if label not in class_colors:
            class_colors[label] = generate_color()
        color = class_colors[label]
        
        # Draw box
        draw.rectangle(box, outline=color, width=3)
        
        # Draw label background
        text = label.capitalize()
        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # Draw label text
        draw.text((box[0], box[1]), text, fill='white', font=font)
    
    return draw_image

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    # Calculate areas
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def analyze_image(image_path, search_term=None, device="mps"):
    print("\nChecking available devices:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {device}")
    
    print("\nLoading Florence-2 model...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/florence-2-large",
        trust_remote_code=True
    ).to(device)
    model_load_time = time.time() - start_time
    print(f"Model loaded successfully in {model_load_time:.2f} seconds")
    print(f"Model device: {next(model.parameters()).device}")
    
    processor = AutoProcessor.from_pretrained(
        "microsoft/florence-2-large",
        trust_remote_code=True
    )
    print("Processor loaded successfully")
    
    # Set model info in utils
    utils.set_model_info(model, processor)
    
    # Load image
    print(f"\nLoading image: {image_path}")
    start_time = time.time()
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    print(f"Image loaded in {time.time() - start_time:.2f} seconds")
    
    # Analyze image
    print("\nAnalyzing image...")
    start_time = time.time()
    if search_term:
        print(f"Searching for: {search_term}")
        # Try both open vocab detection and phrase grounding
        task_prompt = TaskType.OPEN_VOCAB_DETECTION
        results_od = utils.run_example(task_prompt, image, text_input=search_term, device=device)
        print("\nOpen Vocabulary Detection Results (Raw):")
        print(results_od)
        
        task_prompt = TaskType.PHRASE_GROUNDING
        results_pg = utils.run_example(task_prompt, image, text_input=f"Find all {search_term}s in the image", device=device)
        print("\nPhrase Grounding Results (Raw):")
        print(results_pg)
        
        # Combine results from both methods
        boxes = []
        labels = []
        scores = []  # Add scores list
        
        # Add Open Vocabulary Detection results (higher confidence)
        if TaskType.OPEN_VOCAB_DETECTION in results_od:
            od_results = results_od[TaskType.OPEN_VOCAB_DETECTION]
            if 'bboxes' in od_results and 'bboxes_labels' in od_results:
                boxes.extend(od_results['bboxes'])
                labels.extend(od_results['bboxes_labels'])
                scores.extend([1.0] * len(od_results['bboxes']))  # Higher confidence for specific detection
        
        # Add Phrase Grounding results (lower confidence)
        if TaskType.PHRASE_GROUNDING in results_pg:
            pg_results = results_pg[TaskType.PHRASE_GROUNDING]
            if 'bboxes' in pg_results and 'labels' in pg_results:
                boxes.extend(pg_results['bboxes'])
                labels.extend([search_term] * len(pg_results['bboxes']))
                scores.extend([0.9] * len(pg_results['bboxes']))  # Lower confidence for general detection
        
        # Only keep the detection with highest confidence
        if boxes:
            max_score_idx = scores.index(max(scores))
            results = {
                'bboxes': [boxes[max_score_idx]],
                'labels': [labels[max_score_idx]],
                'scores': [scores[max_score_idx]]
            }
        else:
            results = {
                'bboxes': [],
                'labels': [],
                'scores': []
            }
    else:
        print("Performing general object detection")
        task_prompt = TaskType.DENSE_REGION_CAPTION
        results = utils.run_example(task_prompt, image, device=device)
        print("\nDense Region Caption Results (Raw):")
        print(results)
        results = utils.convert_to_od_format(results[TaskType.DENSE_REGION_CAPTION])
    inference_time = time.time() - start_time
    print(f"Analysis completed in {inference_time:.2f} seconds")
    
    return image, results

def main():
    parser = argparse.ArgumentParser(description='Analyze image using Florence-2')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--device', default='mps', choices=['cuda', 'mps', 'cpu'],
                      help='Device to run the model on')
    parser.add_argument('--output', default='output.jpg',
                      help='Path to save the annotated image')
    parser.add_argument('--search', 
                      help='Search for specific objects (e.g., "a fuel truck", "aircraft on the ground")')
    args = parser.parse_args()
    
    try:
        # Analyze image
        image, results = analyze_image(args.image_path, search_term=args.search, device=args.device)
        
        print("\nDetected objects:")
        if 'bboxes' in results and 'labels' in results:
            boxes = results['bboxes']
            labels = results['labels']
            scores = results['scores']
            print(f"\nFound {len(labels)} objects:")
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                print(f"{i+1}. {label.capitalize()} (Score: {score:.2f})")
                print(f"   Location: {box}")
            
            # Draw detections on image
            print(f"\nDrawing detections on image...")
            annotated_image = draw_detections(image, boxes, labels)
            
            # Save annotated image
            print(f"Saving annotated image to: {args.output}")
            annotated_image.save(args.output, quality=95)
        else:
            print("No objects detected")
    except Exception as e:
        print(f"Error analyzing image: {e}")

if __name__ == "__main__":
    main()
