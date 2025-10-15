import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import av
import threading
import queue
import time
import logging
from PIL import Image
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WebRTC configuration for real-time streaming
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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
        """Process frame and return detections with visualization"""
        try:
            # Process frame
            input_tensor = self.preprocess(frame)
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            # Get detections
            detections = self.postprocess(outputs, frame.shape)
            
            # Draw detections on frame
            output_frame = frame.copy()
            for detection in detections:
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
            
            return {
                'success': True,
                'detections': detections,
                'processed_frame': output_frame,
                'frame_size': frame.shape
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'processed_frame': frame
            }

class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.detection_count = 0
        
    def recv(self, frame):
        """Process each video frame in real-time"""
        try:
            # Convert frame to OpenCV format
            img = frame.to_ndarray(format="bgr24")
            
            # Process frame
            result = self.detector.process_frame(img)
            
            # Update FPS counter
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time
            
            # Update detection count
            if result['success']:
                self.detection_count += len(result['detections'])
            
            # Convert back to video frame
            processed_frame = result['processed_frame']
            return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
            
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            return frame

# Streamlit App
def main():
    st.set_page_config(
        page_title="Real-time Pothole Detection",
        page_icon="🚧",
        layout="wide"
    )
    
    # App title
    st.title("Real-time Pothole Detection")
    st.markdown("AI-powered real-time pothole detection using YOLOv8 and live video streaming")
    st.markdown("Developed by Subhodeep Moitra, Dept. of Computer Applications, Techno College Hooghly")
    
    # Initialize session state
    if 'detector' not in st.session_state:
        with st.spinner("Loading AI model..."):
            try:
                st.session_state.detector = PotholeDetector()
                st.session_state.model_loaded = True
                st.session_state.realtime_stats = {
                    'total_frames': 0,
                    'total_potholes': 0,
                    'current_fps': 0,
                    'processing_times': []
                }
                # Initialize camera mode (environment = back camera, user = front camera)
                if 'camera_mode' not in st.session_state:
                    st.session_state.camera_mode = "environment"  # Default to back camera
                st.success("AI model loaded successfully!")
            except Exception as e:
                st.session_state.model_loaded = False
                st.error(f"Failed to load model: {e}")
                return
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Camera selection buttons
    st.sidebar.subheader("📷 Camera Selection")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📷 Back Camera", use_container_width=True):
            st.session_state.camera_mode = "environment"
            st.rerun()
    
    with col2:
        if st.button("📱 Front Camera", use_container_width=True):
            st.session_state.camera_mode = "user"
            st.rerun()
    
    # Show current camera status
    current_camera = "Back Camera" if st.session_state.camera_mode == "environment" else "Front Camera"
    st.sidebar.info(f"**Current:** {current_camera}")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Adjust detection sensitivity"
    )
    
    st.session_state.detector.conf_threshold = confidence_threshold
    
    # Processing mode selection
    processing_mode = st.sidebar.radio(
        "Processing Mode",
        ["Real-time Video", "Single Image", "Video File"],
        help="Choose between real-time streaming or file processing"
    )
    
    # Main content based on mode
    if processing_mode == "Real-time Video":
        show_realtime_detection()
    elif processing_mode == "Single Image":
        show_single_image_detection()
    else:
        show_video_file_detection()

def show_realtime_detection():
    st.header("Real-time Video Detection")
    
    # Show current camera status prominently
    current_camera = "Back Camera" if st.session_state.camera_mode == "environment" else "Front Camera"
    camera_icon = "📷" if st.session_state.camera_mode == "environment" else "📱"
    
    st.info(f"""
    **Instructions:**
    - Click 'START' to begin real-time detection
    - Allow camera permissions when prompted
    - **Current Camera:** {camera_icon} {current_camera}
    - Detection runs at 2-5 FPS depending on your device
    - Green bounding boxes show detected potholes
    - Switch cameras using buttons in sidebar
    """)
    
    # Camera constraints based on selection
    media_stream_constraints = {
        "video": {
            "facingMode": st.session_state.camera_mode,
            "width": {"ideal": 1280},
            "height": {"ideal": 720}
        }, 
        "audio": False
    }
    
    # Initialize video processor
    video_processor = VideoProcessor(st.session_state.detector)
    
    # Real-time WebRTC streamer with camera selection
    webrtc_ctx = webrtc_streamer(
        key=f"pothole-detection-{st.session_state.camera_mode}",  # Unique key for each camera mode
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: video_processor,
        media_stream_constraints=media_stream_constraints,
        async_processing=True,
    )
    
    # Camera switching tips
    if st.session_state.camera_mode == "environment":
        st.warning("💡 **Using Back Camera** - Best for road scanning. Switch to front camera for testing.")
    else:
        st.warning("💡 **Using Front Camera** - Good for testing. Switch to back camera for actual road scanning.")
    
    # Real-time statistics
    if webrtc_ctx.state.playing:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current FPS", f"{video_processor.fps}")
        with col2:
            st.metric("Potholes Detected", video_processor.detection_count)
        with col3:
            st.metric("Total Frames", video_processor.frame_count)
        with col4:
            st.metric("Confidence", f"{st.session_state.detector.conf_threshold:.2f}")
        
        # Camera status in main view
        st.success(f"🎥 {current_camera} - Live detection running...")
        
        # Performance warning
        st.warning("""
        **Performance Note:** 
        - Real-time processing may be slower on some devices
        - For better performance, try lowering video quality in your browser
        - Processing speed depends on your device capabilities
        - If camera doesn't switch, refresh the page and try again
        """)

def show_single_image_detection():
    st.header("📷 Single Image Detection")
    
    uploaded_file = st.file_uploader(
        "Upload an image for pothole detection",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image file to detect potholes"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Detection Results")
            with st.spinner("Analyzing image for potholes..."):
                # Convert to OpenCV format
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                start_time = time.time()
                result = st.session_state.detector.process_frame(frame)
                processing_time = time.time() - start_time
                
                if result['success']:
                    # Convert back to RGB for display
                    output_frame_rgb = cv2.cvtColor(result['processed_frame'], cv2.COLOR_BGR2RGB)
                    st.image(output_frame_rgb, use_column_width=True)
                    
                    # Show results
                    if result['detections']:
                        st.success(f"Found {len(result['detections'])} pothole(s) in {processing_time*1000:.0f}ms!")
                        for i, det in enumerate(result['detections']):
                            st.info(f"**Pothole {i+1}:** {det['confidence']:.1%} confidence")
                    else:
                        st.info("No potholes detected in this image")
                else:
                    st.error(f"Processing error: {result['error']}")

def show_video_file_detection():
    st.header("Video File Detection")
    
    uploaded_file = st.file_uploader(
        "Upload a video for pothole detection",
        type=['mp4', 'mov', 'avi'],
        help="Upload a video file to detect potholes frame by frame"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        st.info("Video processing will start soon...")
        
        # Process video
        if st.button("Start Video Processing"):
            process_video_file(video_path)
        
        # Clean up
        try:
            os.unlink(video_path)
        except:
            pass

def process_video_file(video_path):
    """Process uploaded video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Could not open video file")
        return
    
    # Video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info(f"Video Info: {fps:.1f} FPS, {total_frames} frames")
    
    # Create placeholder for video display
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    frame_count = 0
    pothole_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 3rd frame for performance
        if frame_count % 3 == 0:
            result = st.session_state.detector.process_frame(frame)
            
            if result['success']:
                pothole_count += len(result['detections'])
                
                # Display processed frame
                display_frame = cv2.cvtColor(result['processed_frame'], cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame, caption=f"Frame {frame_count}", use_column_width=True)
                
                # Update stats
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                with stats_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Processing FPS", f"{current_fps:.1f}")
                    with col2:
                        st.metric("Frames Processed", frame_count)
                    with col3:
                        st.metric("Potholes Found", pothole_count)
        
        frame_count += 1
    
    cap.release()
    st.success(f"Video processing complete! Processed {frame_count} frames, found {pothole_count} potholes.")

if __name__ == "__main__":
    main()
