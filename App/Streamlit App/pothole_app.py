import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import base64
import io
import time
import logging
from PIL import Image
import tempfile
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PotholeDetector:
    def __init__(self, repo_id="subhodeepmoitra/pothole-detection-yolov8", filename="best.onnx"):
        try:
            logger.info("Loading ONNX model from Hugging Face...")
            
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
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            st.error(f"Failed to load AI model: {e}")
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

    def process_frame(self, frame):
        """Process frame and return detections"""
        try:
            # Process frame
            input_tensor = self.preprocess(frame)
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Get detections
            detections = self.postprocess(outputs, frame.shape)
            
            return {
                'success': True,
                'detections': detections,
                'frame_size': frame.shape
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                'success': False,
                'error': str(e),
                'detections': []
            }

# Streamlit App
def main():
    st.set_page_config(
        page_title="Real-time Pothole Detection",
        page_icon="🚧",
        layout="wide"
    )
    
    # App title
    st.title("🚧 Real-time Pothole Detection")
    st.markdown("AI-powered pothole detection using YOLOv8")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        with st.spinner("Loading AI model..."):
            try:
                st.session_state.detector = PotholeDetector()
                st.session_state.model_loaded = True
                st.success("AI model loaded successfully!")
            except Exception as e:
                st.session_state.model_loaded = False
                st.error(f"Failed to load model: {e}")
                return
    
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {
            'last_processing_time': 0,
            'total_frames_processed': 0,
            'total_potholes_detected': 0
        }
    
    # Sidebar
    st.sidebar.title("Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Adjust detection sensitivity"
    )
    
    st.session_state.detector.conf_threshold = confidence_threshold
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed & Detection")
        
        # Camera input
        camera_option = st.radio(
            "Input Source:",
            ["Webcam", "Upload Image/Video"],
            horizontal=True
        )
        
        if camera_option == "Webcam":
            camera_image = st.camera_input("Take a picture for pothole detection")
            
            if camera_image is not None:
                # Process the image
                with st.spinner("Analyzing for potholes..."):
                    # Convert to OpenCV format
                    image = Image.open(camera_image)
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    start_time = time.time()
                    result = st.session_state.detector.process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    # Update stats
                    st.session_state.processing_stats['last_processing_time'] = processing_time
                    st.session_state.processing_stats['total_frames_processed'] += 1
                    
                    if result['success']:
                        # Draw detections on image
                        output_frame = frame.copy()
                        
                        for detection in result['detections']:
                            x1, y1, x2, y2 = detection['bbox']
                            confidence = detection['confidence']
                            
                            # Draw bounding box
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            # Draw label
                            label = f"Pothole: {confidence:.1%}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            
                            cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (0, 255, 0), -1)
                            cv2.putText(output_frame, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                        # Update pothole count
                        st.session_state.processing_stats['total_potholes_detected'] += len(result['detections'])
                        
                        # Convert back to RGB for display
                        output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display results
                        st.image(output_frame_rgb, caption="Detection Results", use_column_width=True)
                        
                        # Show detection info
                        if result['detections']:
                            st.success(f"Found {len(result['detections'])} pothole(s)!")
                            for i, det in enumerate(result['detections']):
                                st.info(f"Pothole {i+1}: {det['confidence']:.1%} confidence")
                        else:
                            st.info("🔍 No potholes detected in this image")
                    
                    else:
                        st.error(f"Processing error: {result['error']}")
        
        else:  # Upload Image/Video
            uploaded_file = st.file_uploader(
                "Upload an image or video",
                type=['jpg', 'jpeg', 'png', 'mp4', 'mov'],
                help="Upload an image or video file for pothole detection"
            )
            
            if uploaded_file is not None:
                if uploaded_file.type.startswith('image'):
                    # Process image
                    image = Image.open(uploaded_file)
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    with st.spinner("🔍 Analyzing image for potholes..."):
                        start_time = time.time()
                        result = st.session_state.detector.process_frame(frame)
                        processing_time = time.time() - start_time
                        
                        st.session_state.processing_stats['last_processing_time'] = processing_time
                        st.session_state.processing_stats['total_frames_processed'] += 1
                        
                        if result['success'] and result['detections']:
                            # Draw detections
                            output_frame = frame.copy()
                            for detection in result['detections']:
                                x1, y1, x2, y2 = detection['bbox']
                                confidence = detection['confidence']
                                
                                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                label = f"Pothole: {confidence:.1%}"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 10), 
                                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                                cv2.putText(output_frame, label, (x1, y1 - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            
                            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                            st.image(output_frame_rgb, caption="Detection Results", use_column_width=True)
                            st.success(f"Found {len(result['detections'])} pothole(s)!")
                            
                            st.session_state.processing_stats['total_potholes_detected'] += len(result['detections'])
                            
                        else:
                            st.image(uploaded_file, caption="Original Image", use_column_width=True)
                            st.info("🔍 No potholes detected in this image")
    
    with col2:
        st.subheader("Detection Statistics")
        
        # Stats cards
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                "Processing Time",
                f"{st.session_state.processing_stats['last_processing_time']*1000:.0f}ms"
            )
            st.metric(
                "Total Frames",
                st.session_state.processing_stats['total_frames_processed']
            )
        
        with metric_col2:
            st.metric(
                "Potholes Detected",
                st.session_state.processing_stats['total_potholes_detected']
            )
            st.metric(
                "Confidence Threshold",
                f"{confidence_threshold:.2f}"
            )
        
        # Model info
        st.subheader("Model Information")
        st.info(f"""
        - **Model**: YOLOv8 (ONNX)
        - **Input Size**: 640x640 pixels
        - **Repository**: subhodeepmoitra/pothole-detection-yolov8
        - **Status**: Loaded
        """)
        
        # Instructions
        st.subheader("How to Use")
        st.markdown("""
        1. **Webcam Mode**: Allow camera access and take pictures
        2. **Upload Mode**: Upload images or videos
        3. **Adjust Sensitivity**: Use slider to fine-tune detection
        4. **View Results**: Detections shown with bounding boxes
        """)
        
        # Performance note
        st.subheader("⚡ Performance")
        st.warning("""
        **Note**: Processing time depends on server load.
        - Expected: 800-1500ms per frame
        - Free tier may have slower performance
        """)

if __name__ == "__main__":
    main()