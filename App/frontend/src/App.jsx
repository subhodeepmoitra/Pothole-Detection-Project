import React, { useRef, useEffect, useState } from 'react';
import io from 'socket.io-client';
import './App.css';

const PotholeDetector = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const socketRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detections, setDetections] = useState([]);
  const [stats, setStats] = useState({ fps: 0, processingTime: 0 });
  const [error, setError] = useState('');
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(0);

  // Socket connection
  useEffect(() => {
    // Connect to WebSocket server (Render backend URL)
    const backendUrl = import.meta.env.VITE_BACKEND_URL || 'https://pothole-detection-project.onrender.com';
    socketRef.current = io(backendUrl);

    socketRef.current.on('connect', () => {
      console.log('✅ Connected to detection server');
      setIsConnected(true);
      setError('');
    });

    socketRef.current.on('disconnect', () => {
      console.log('❌ Disconnected from server');
      setIsConnected(false);
    });

    socketRef.current.on('status', (data) => {
      console.log('Server status:', data);
      if (!data.model_loaded) {
        setError('AI model not loaded on server');
      }
    });

    socketRef.current.on('detections', (data) => {
      if (data.success) {
        setDetections(data.detections);
        setStats(prev => ({
          ...prev,
          processingTime: data.processing_time
        }));
      } else {
        console.error('Detection error:', data.error);
      }
    });

    socketRef.current.on('error', (data) => {
      setError(data.message);
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  // Start/stop webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 1280, 
          height: 720,
          facingMode: 'environment' 
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsDetecting(true);
        startDetectionLoop();
      }
    } catch (err) {
      console.error('Error accessing webcam:', err);
      setError('Cannot access webcam. Please check permissions.');
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsDetecting(false);
    }
  };

  // Capture frame and send to server
  const captureAndSendFrame = () => {
    if (!videoRef.current || !socketRef.current || !isConnected || !isDetecting) return;

    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Reduce quality for faster transmission
    const imageData = canvas.toDataURL('image/jpeg', 0.7);
    
    // Send frame to server
    socketRef.current.emit('frame_data', {
      image: imageData,
      frame_id: frameCountRef.current++,
      timestamp: Date.now()
    });

    // Update FPS
    const now = Date.now();
    if (now - lastFpsUpdateRef.current > 1000) {
      setStats(prev => ({ ...prev, fps: frameCountRef.current }));
      frameCountRef.current = 0;
      lastFpsUpdateRef.current = now;
    }
  };

  // Detection loop
  const startDetectionLoop = () => {
    const loop = () => {
      if (isDetecting && isConnected) {
        captureAndSendFrame();
        requestAnimationFrame(loop);
      }
    };
    loop();
  };

  // Draw detections on canvas
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const video = videoRef.current;

    if (!video) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw bounding boxes
    detections.forEach(detection => {
      const [x1, y1, x2, y2] = detection.bbox;
      const confidence = detection.confidence;

      // Draw bounding box
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Draw label background
      ctx.fillStyle = '#00FF00';
      ctx.font = '16px Arial';
      const text = `Pothole: ${(confidence * 100).toFixed(1)}%`;
      const textWidth = ctx.measureText(text).width;
      
      ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);

      // Draw label text
      ctx.fillStyle = '#000000';
      ctx.fillText(text, x1 + 5, y1 - 8);
    });

    // Draw stats on canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(10, 10, 200, 80);
    
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '14px Arial';
    ctx.fillText(`FPS: ${stats.fps}`, 20, 30);
    ctx.fillText(`Processing: ${(stats.processingTime * 1000).toFixed(1)}ms`, 20, 50);
    ctx.fillText(`Potholes: ${detections.length}`, 20, 70);

  }, [detections, stats]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>🚧 Real-time Pothole Detection</h1>
        <p>AI-powered detection with WebSocket backend</p>
      </header>

      <main className="main-content">
        {/* Connection Status */}
        <div className={`status-bar ${isConnected ? 'connected' : 'disconnected'}`}>
          <div className="status-indicator"></div>
          <span>
            {isConnected ? '✅ Connected to AI Server' : '❌ Disconnected'}
          </span>
        </div>

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        <div className="detection-section">
          {/* Camera Feed */}
          <div className="camera-container">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="video-element"
            />
            <canvas
              ref={canvasRef}
              className="canvas-overlay"
            />
            
            {!isDetecting && (
              <div className="camera-placeholder">
                <p>📷 Camera feed will appear here</p>
                <button 
                  onClick={startWebcam} 
                  className="start-button"
                  disabled={!isConnected}
                >
                  Start Camera
                </button>
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="controls">
            {isDetecting ? (
              <button onClick={stopWebcam} className="stop-button">
                🛑 Stop Detection
              </button>
            ) : (
              <button 
                onClick={startWebcam} 
                className="start-button"
                disabled={!isConnected}
              >
                🎬 Start Detection
              </button>
            )}
          </div>

          {/* Statistics */}
          <div className="stats-grid">
            <div className="stat-card">
              <h3>Potholes Detected</h3>
              <p className="stat-number">{detections.length}</p>
            </div>
            <div className="stat-card">
              <h3>FPS</h3>
              <p className="stat-number">{stats.fps}</p>
            </div>
            <div className="stat-card">
              <h3>Processing Time</h3>
              <p className="stat-number">{(stats.processingTime * 1000).toFixed(1)}ms</p>
            </div>
            <div className="stat-card">
              <h3>Connection</h3>
              <p className={`stat-status ${isConnected ? 'connected' : 'disconnected'}`}>
                {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
              </p>
            </div>
          </div>

          {/* Detection Details */}
          {detections.length > 0 && (
            <div className="detections-list">
              <h3>Detected Potholes:</h3>
              <div className="detections-grid">
                {detections.map((det, index) => (
                  <div key={index} className="detection-item">
                    <span>Pothole {index + 1}</span>
                    <span>{(det.confidence * 100).toFixed(1)}% confidence</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default PotholeDetector;
