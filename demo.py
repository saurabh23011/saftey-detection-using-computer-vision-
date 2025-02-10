from flask import Flask, render_template, request, Response, jsonify
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import base64
from io import BytesIO
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = '192b9bdd22ab9ed4d12e236c78afcb9a393ec15f71bbf5dc987d54727823bcbf'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
camera = None

def load_model():
    """Load custom trained YOLOv11 model for PPE detection"""
    global model
    if model is None:
        model = YOLO('best (1).pt')
    return model

def process_image(image, conf_threshold=0.25):
    """Process image with custom YOLO model for PPE detection"""
    model = load_model()
    results = model(image, conf=conf_threshold)
    return results

def analyze_ppe_compliance(results):
    """Analyze PPE compliance from detection results"""
    # Updated required PPE list to include additional items
    required_ppe = {
        # 'dust masks', 'eyewear', 'gloves', 'jackets', 
        # 'protective boots', 'protective helmets', 'shields', 
        # New items added
        'Airforce hat', 'Para hat', 'Marcos Commando', 'Gun','Airforces',
        'Para Commando'
    }
    detected_ppe = set()
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            # Normalize class names to match required PPE
            normalized_name = cls_name.lower()
            
        #     # Add specific handling for new items
        #     if normalized_name in ['Airforce hat', 'Para hat', 'Marcos Commando', 'Gun','Airforces',
        # # 'Para Commando']:
        #         detected_ppe.add(cls_name)
            
            # Existing PPE detection logic
            if normalized_name in [item.lower() for item in required_ppe]:
                detected_ppe.add(cls_name)
    
    missing_ppe = required_ppe - detected_ppe
    compliance_status = {
        'compliant': len(missing_ppe) == 0,
        'missing_equipment': list(missing_ppe),
        'detected_equipment': list(detected_ppe)
    }
    
    return compliance_status

def draw_boxes(image, results):
    """Draw bounding boxes and labels on the image with PPE-specific colors"""
    # Expanded color mapping for different PPE types
    color_map = {
        'helmet': (255, 255, 0),
        'vest': (255, 165, 0),
        'Airforces': (0, 255, 255),
        'Gun': (255, 0, 0),
        # New items with custom colors
        'airforce hat': (0, 128, 255),  # Orange-blue
        'para hat': (255, 0, 128),      # Magenta
        'marcos commando': (128, 0, 255),  # Purple
        'para commando': (0, 255, 128)  # Lime green
          
    }
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            
            # Get color, default to white if not in color map
            color = color_map.get(cls_name.lower(), (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f'{cls_name} {conf:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1-20), (x1+w, y1), color, -1)
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 1)
    
    compliance = analyze_ppe_compliance(results)
    status_text = "COMPLIANT" if compliance['compliant'] else "NON-COMPLIANT"
    status_color = (0, 255, 0) if compliance['compliant'] else (0, 0, 255)
    
    cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, status_color, 2)
    
    return image, compliance

# Rest of the code remains the same as in the original script
# ... (all other functions and routes remain unchanged)

if __name__ == '__main__':
    app.run(debug=True)