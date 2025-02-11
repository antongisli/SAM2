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
    # Create a copy of the image for drawing
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Use different colors for different objects
    colors = utils.colormap
    
    # Draw each detection
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # Get coordinates
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are in correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Get color for this object
        color = colors[i % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        font_size = 20
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Draw label background
        label_text = label.capitalize()
        text_bbox = draw.textbbox((x1, y1-font_size-5), label_text, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                      fill=color)
        
        # Draw label text
        draw.text((x1, y1-font_size-5), label_text, fill='white', font=font)
    
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

def analyze_image(image_path, search_term=None, mode="grounding", device="mps"):
    print("\nChecking available devices:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {device}")
    print(f"Mode: {mode}")
    
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
    
    if mode == "caption":
        print("Using Dense Region Caption mode")
        task_prompt = TaskType.DENSE_REGION_CAPTION
        results = utils.run_example(task_prompt, image, device=device)
        print("\nDense Region Caption Results:")
        print(results)
        if TaskType.DENSE_REGION_CAPTION in results:
            dense_results = results[TaskType.DENSE_REGION_CAPTION]
            results = {
                'bboxes': dense_results.get('bboxes', []),
                'labels': dense_results.get('labels', []),
                'scores': [1.0] * len(dense_results.get('bboxes', []))
            }
        else:
            results = {
                'bboxes': [],
                'labels': [],
                'scores': []
            }
    elif mode == "vocab":
        if not search_term:
            print("Error: Search term required for vocab mode")
            return None, None
            
        print(f"Using Open Vocabulary Detection mode, searching for: {search_term}")
        task_prompt = TaskType.OPEN_VOCAB_DETECTION
        results_od = utils.run_example(task_prompt, image, text_input=search_term, device=device)
        print("\nOpen Vocabulary Detection Results:")
        print(results_od)
        
        if TaskType.OPEN_VOCAB_DETECTION in results_od:
            od_results = results_od[TaskType.OPEN_VOCAB_DETECTION]
            results = {
                'bboxes': od_results.get('bboxes', []),
                'labels': od_results.get('bboxes_labels', []),
                'scores': [1.0] * len(od_results.get('bboxes', []))
            }
        else:
            results = {
                'bboxes': [],
                'labels': [],
                'scores': []
            }
    elif mode == "grounding":
        if not search_term:
            print("Error: Search term required for grounding mode")
            return None, None
            
        print(f"Using Phrase Grounding mode, searching for: {search_term}")
        task_prompt = TaskType.PHRASE_GROUNDING
        results_pg = utils.run_example(task_prompt, image, text_input=f"Find {search_term}", device=device)
        print("\nPhrase Grounding Results:")
        print(results_pg)
        
        if TaskType.PHRASE_GROUNDING in results_pg:
            pg_results = results_pg[TaskType.PHRASE_GROUNDING]
            results = {
                'bboxes': pg_results.get('bboxes', []),
                'labels': pg_results.get('labels', []),
                'scores': [1.0] * len(pg_results.get('bboxes', []))
            }
        else:
            results = {
                'bboxes': [],
                'labels': [],
                'scores': []
            }
    elif mode == "objects":
        print("Using Object Detection mode")
        task_prompt = TaskType.OBJECT_DETECTION
        results_od = utils.run_example(task_prompt, image, device=device)
        print("\nObject Detection Results:")
        print(results_od)
        
        if TaskType.OBJECT_DETECTION in results_od:
            od_results = results_od[TaskType.OBJECT_DETECTION]
            results = {
                'bboxes': od_results.get('bboxes', []),
                'labels': od_results.get('labels', []),
                'scores': [1.0] * len(od_results.get('bboxes', []))
            }
        else:
            results = {
                'bboxes': [],
                'labels': [],
                'scores': []
            }
    elif mode == "regions":
        print("Using Region Proposal mode")
        task_prompt = TaskType.REGION_PROPOSAL
        results_rp = utils.run_example(task_prompt, image, device=device)
        print("\nRegion Proposal Results:")
        print(results_rp)
        
        if TaskType.REGION_PROPOSAL in results_rp:
            rp_results = results_rp[TaskType.REGION_PROPOSAL]
            results = {
                'bboxes': rp_results.get('bboxes', []),
                'labels': ['region'] * len(rp_results.get('bboxes', [])),
                'scores': [1.0] * len(rp_results.get('bboxes', []))
            }
        else:
            results = {
                'bboxes': [],
                'labels': [],
                'scores': []
            }
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
    parser.add_argument('--mode', default='grounding', 
                      choices=['grounding', 'caption', 'vocab', 'objects', 'regions'],
                      help='''Detection mode:
                           "grounding": phrase grounding (natural language search)
                           "vocab": open vocabulary detection (precise search)
                           "caption": dense region caption (detailed descriptions)
                           "objects": general object detection
                           "regions": region proposals only''')
    args = parser.parse_args()
    
    try:
        # Analyze image
        image, results = analyze_image(args.image_path, search_term=args.search, mode=args.mode, device=args.device)
        
        if image is None or results is None:
            return
        
        print("\nDetected objects:")
        if 'bboxes' in results and 'labels' in results:
            boxes = results['bboxes']
            labels = results['labels']
            scores = results.get('scores', [1.0] * len(labels))
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
