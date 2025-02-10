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
        model = YOLO('best.pt')
    return model

def process_image(image, conf_threshold=0.25):
    """Process image with custom YOLO model for PPE detection"""
    model = load_model()
    results = model(image, conf=conf_threshold)
    return results

def analyze_ppe_compliance(results):
    """Analyze PPE compliance from detection results"""
    required_ppe = {'dust masks', 'eyewear', 'gloves', 'jackets', 'protective boots', 'protective helmets', 'shields'}
    detected_ppe = set()
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
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
    # Color mapping for different PPE types
    color_map = {
        'helmet': (0, 255, 0),
        'vest': (255, 165, 0),
        'gloves': (0, 255, 255),
        'boots': (255, 0, 0)
    }
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            
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

def generate_frames():
    """Generate frames from webcam with PPE detection"""
    global camera
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        try:
            # Convert frame to RGB for YOLO processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with YOLO
            results = process_image(frame_rgb)
            
            # Draw detection boxes and get compliance info
            processed_frame, compliance = draw_boxes(frame, results)
            
            # Add compliance information to frame
            y_offset = 60  # Start below the compliance status
            for equipment in compliance.get('detected_equipment', []):
                cv2.putText(processed_frame, f"Detected: {equipment}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)
                y_offset += 25
                
            for equipment in compliance.get('missing_equipment', []):
                cv2.putText(processed_frame, f"Missing: {equipment}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 0, 255), 2)
                y_offset += 25
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue

def process_video_file(video_path, output_path, confidence=0.25):
    """Process video file and save output with PPE detection"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file"

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    compliance_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = process_image(frame_rgb, confidence)
        processed_frame, compliance = draw_boxes(frame, results)
        
        # Write frame
        out.write(processed_frame)
        
        # Store compliance data
        compliance_history.append(compliance)
        frame_count += 1

    cap.release()
    out.release()

    # Calculate overall compliance statistics
    total_compliant = sum(1 for c in compliance_history if c['compliant'])
    compliance_rate = (total_compliant / frame_count) * 100 if frame_count > 0 else 0

    return {
        'output_path': output_path,
        'total_frames': frame_count,
        'compliant_frames': total_compliant,
        'compliance_rate': compliance_rate,
        'fps': fps
    }, None

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/process_image', methods=['POST'])
def process_uploaded_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    confidence = float(request.form.get('confidence', 0.25))
    
    image_stream = BytesIO(file.read())
    image = Image.open(image_stream)
    image_np = np.array(image)
    
    results = process_image(image_np, confidence)
    output_image = image_np.copy()
    output_image, compliance = draw_boxes(output_image, results)
    
    output_buffer = BytesIO()
    output_image_pil = Image.fromarray(output_image)
    output_image_pil.save(output_buffer, format='JPEG')
    output_base64 = base64.b64encode(output_buffer.getvalue()).decode()
    
    return jsonify({
        'processed_image': f'data:image/jpeg;base64,{output_base64}',
        'compliance': compliance
    })

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    confidence = float(request.form.get('confidence', 0.25))
    
    # Create unique filename for both input and output videos
    timestamp = int(time.time())
    input_filename = f'input_video_{timestamp}.mp4'
    output_filename = f'output_video_{timestamp}.mp4'
    
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    # Save uploaded video
    file.save(input_path)
    
    # Process video
    try:
        results, error = process_video_file(input_path, output_path, confidence)
        if error:
            return jsonify({'error': error}), 400
        
        # Generate video URL
        video_url = f'/static/uploads/{output_filename}'
        
        return jsonify({
            'video_url': video_url,
            'statistics': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up input video
        if os.path.exists(input_path):
            os.remove(input_path)

@app.route('/video_feed')
def video_feed():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam')
def start_webcam():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return jsonify({'status': 'success'})

@app.route('/stop_webcam')
def stop_webcam():
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)