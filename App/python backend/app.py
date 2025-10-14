from flask import Flask, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import onnxruntime as ort
import base64
import io
from PIL import Image
import time
import logging
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get secret key from environment or generate a temporary one for development
SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    if os.getenv('FLASK_ENV') == 'production':
        raise ValueError("SECRET_KEY must be set in production")
    else:
        # Generate a temporary key for development
        import secrets
        SECRET_KEY = secrets.token_hex(32)
        logger.warning("Using temporary SECRET_KEY for development")

app.config['SECRET_KEY'] = SECRET_KEY

# Remove async_mode to fix compatibility issues
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   logger=True,
                   engineio_logger=True)

class RealTimePotholeDetector:
    def __init__(self, repo_id="subhodeepmoitra/pothole-detection-yolov8", filename="best.onnx"):
        try:
            logger.info("Downloading ONNX model from Hugging Face...")
            
            # Download model from Hugging Face Hub
            self.model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir="./huggingface_cache"
            )
            logger.info(f"Model downloaded to: {self.model_path}")
            
            # Initialize ONNX Runtime session
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_size = 640
            self.conf_threshold = 0.25
            logger.info("Model loaded successfully from Hugging Face!")
            
        except Exception as e:
            logger.error(f"Failed to load model from Hugging Face: {e}")
            # Fallback: Try local model if Hugging Face fails
            try:
                logger.info("Trying local model as fallback...")
                self.model_path = 'best.onnx'
                self.session = ort.InferenceSession(self.model_path)
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [output.name for output in self.session.get_outputs()]
                self.input_size = 640
                self.conf_threshold = 0.25
                logger.info("Local model loaded successfully!")
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise e

    def preprocess(self, frame):
        """Preprocess frame for YOLOv8 model"""
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)  # Add batch dimension
        return img

    def postprocess(self, outputs, original_shape):
        """Postprocess YOLOv8 outputs to get detections"""
        predictions = outputs[0]  # Output0: [1, 37, 8400]
        
        detections = []
        
        if predictions is not None and len(predictions) > 0:
            # Transpose to [8400, 37]
            predictions = predictions[0].T
            
            # Filter by confidence (objectness score is at index 4)
            scores = predictions[:, 4:5]
            keep = scores > self.conf_threshold
            
            if np.any(keep):
                # Get boxes (normalized coordinates)
                boxes = predictions[keep.squeeze()][:, :4]
                confidences = predictions[keep.squeeze()][:, 4]
                
                # Convert normalized coordinates to pixel coordinates
                h, w = original_shape[:2]
                scale_x = w / self.input_size
                scale_y = h / self.input_size
                
                for i, box in enumerate(boxes):
                    x_center, y_center, width, height = box
                    
                    # Convert from center format to corner format
                    x1 = int((x_center - width/2) * scale_x)
                    y1 = int((y_center - height/2) * scale_y)
                    x2 = int((x_center + width/2) * scale_x)
                    y2 = int((y_center + height/2) * scale_y)
                    
                    # Ensure coordinates are within frame bounds
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    # Only add if box has reasonable size
                    if (x2 - x1) > 10 and (y2 - y1) > 10:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidences[i]),
                            'class_id': 0,
                            'class_name': 'pothole'
                        })
        
        return detections

    def process_frame(self, frame_data):
        """Process base64 image frame and return detections"""
        try:
            # Decode base64 image
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            image_data = base64.b64decode(frame_data)
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process frame
            input_tensor = self.preprocess(frame)
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Get detections
            detections = self.postprocess(outputs, frame.shape)
            
            return {
                'success': True,
                'detections': detections,
                'timestamp': time.time(),
                'frame_size': frame.shape
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                'success': False,
                'error': str(e),
                'detections': []
            }

# Initialize detector
try:
    detector = RealTimePotholeDetector(
        repo_id="subhodeepmoitra/pothole-detection-yolov8",
        filename="best.onnx"
    )
    MODEL_LOADED = True
    logger.info("Pothole detector initialized successfully!")
except Exception as e:
    logger.error(f"Failed to initialize detector: {e}")
    MODEL_LOADED = False
    detector = None

# SocketIO handlers
@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    logger.info(f'Client connected: {client_id}')
    emit('status', {
        'message': 'Connected to pothole detection server',
        'model_loaded': MODEL_LOADED,
        'client_id': client_id,
        'model_source': 'Hugging Face' if MODEL_LOADED else 'None'
    })

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    logger.info(f'Client disconnected: {client_id}')

@socketio.on('frame_data')
def handle_frame(data):
    """Handle incoming frame data from client"""
    if not MODEL_LOADED or detector is None:
        emit('error', {'message': 'Model not loaded'})
        return
    
    try:
        start_time = time.time()
        
        # Process the frame
        result = detector.process_frame(data['image'])
        
        # Add processing time
        result['processing_time'] = time.time() - start_time
        result['frame_id'] = data.get('frame_id', 0)
        
        # Send detections back to client
        emit('detections', result)
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        emit('error', {'message': f'Processing error: {str(e)}'})

@socketio.on('health_check')
def handle_health_check():
    """Health check endpoint"""
    emit('health_response', {
        'status': 'healthy',
        'model_loaded': MODEL_LOADED,
        'timestamp': time.time(),
        'model_source': 'Hugging Face' if MODEL_LOADED else 'None'
    })

@app.route('/')
def index():
    return {
        'status': 'Pothole Detection API', 
        'model_loaded': MODEL_LOADED,
        'model_source': 'Hugging Face' if MODEL_LOADED else 'None',
        'repository': 'subhodeepmoitra/pothole-detection-yolov8'
    }

@app.route('/health')
def health():
    return {
        'status': 'healthy', 
        'model_loaded': MODEL_LOADED,
        'timestamp': time.time()
    }

# New endpoint to get model info
@app.route('/model-info')
def model_info():
    if MODEL_LOADED and detector:
        return {
            'model_loaded': True,
            'model_path': detector.model_path,
            'input_size': detector.input_size,
            'confidence_threshold': detector.conf_threshold,
            'repository': 'subhodeepmoitra/pothole-detection-yolov8'
        }
    else:
        return {
            'model_loaded': False,
            'error': 'Model not available'
        }

@app.route('/process-frame', methods=['POST'])
def process_frame_http():
    """HTTP endpoint for frame processing"""
    if not MODEL_LOADED or detector is None:
        return {'success': False, 'error': 'Model not loaded'}
    
    try:
        data = request.get_json()
        start_time = time.time()
        
        result = detector.process_frame(data['image'])
        result['processing_time'] = time.time() - start_time
        result['frame_id'] = data.get('frame_id', 0)
        
        return result
        
    except Exception as e:
        logger.error(f"HTTP frame processing error: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    logger.info("Starting Pothole Detection Server...")
    logger.info(f"Model repository: subhodeepmoitra/pothole-detection-yolov8")
    
    # Use this for development
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=True
    )
