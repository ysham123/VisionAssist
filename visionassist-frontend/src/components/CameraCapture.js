import React, { useRef, useEffect, useState } from 'react';
import './CameraCapture.css';

const CameraCapture = ({ isActive, cameraStream, onNewCaption, isLoading }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [captionInterval, setCaptionInterval] = useState(null);
  const [frameCount, setFrameCount] = useState(0);

  // Set up video stream
  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
      videoRef.current.play();
    }
  }, [cameraStream]);

  // Start/stop caption generation
  useEffect(() => {
    if (isActive && cameraStream) {
      // Generate captions every 3 seconds
      const interval = setInterval(() => {
        captureAndAnalyze();
      }, 3000);
      
      setCaptionInterval(interval);
      
      return () => {
        if (interval) {
          clearInterval(interval);
        }
      };
    } else {
      if (captionInterval) {
        clearInterval(captionInterval);
        setCaptionInterval(null);
      }
    }
  }, [isActive, cameraStream]);

  // Capture frame and generate caption
  const captureAndAnalyze = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw current frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert to blob for processing
      canvas.toBlob(async (blob) => {
        if (blob) {
          // In a real implementation, this would send to your Python backend
          // For now, we'll simulate with demo captions
          const demoCaption = await generateDemoCaption();
          onNewCaption(demoCaption, 0.85);
          setFrameCount(prev => prev + 1);
        }
      }, 'image/jpeg', 0.8);

    } catch (error) {
      console.error('Frame capture error:', error);
    }
  };

  // Demo caption generation (replace with actual API call)
  const generateDemoCaption = async () => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const demoCapptions = [
      "A person sitting at a desk with a laptop computer",
      "A wooden table with various objects on it",
      "A room with natural lighting from a window",
      "A person holding a smartphone in their hands",
      "A coffee cup on a wooden surface",
      "A bookshelf with books and decorative items",
      "A plant in a pot near a window",
      "A keyboard and mouse on a desk",
      "A person wearing glasses looking at the camera",
      "A wall with framed pictures hanging on it"
    ];
    
    return demoCapptions[Math.floor(Math.random() * demoCapptions.length)];
  };

  // Force capture for manual trigger
  const forceCaptureAndAnalyze = () => {
    if (isActive) {
      captureAndAnalyze();
    }
  };

  return (
    <div className="camera-capture">
      <div className="camera-container">
        <div className="camera-header">
          <h2 className="camera-title">
            <span className="camera-icon">📹</span>
            Live Camera Feed
          </h2>
          <div className="camera-status">
            {isActive ? (
              <span className="status-active">
                <span className="status-dot"></span>
                Live
              </span>
            ) : (
              <span className="status-inactive">Inactive</span>
            )}
          </div>
        </div>

        <div className="video-container">
          {isActive && cameraStream ? (
            <>
              <video
                ref={videoRef}
                className="camera-video"
                autoPlay
                muted
                playsInline
                aria-label="Live camera feed for VisionAssist"
              />
              <canvas
                ref={canvasRef}
                className="capture-canvas"
                style={{ display: 'none' }}
                aria-hidden="true"
              />
              
              {/* Video overlay */}
              <div className="video-overlay">
                <div className="frame-counter">
                  Frame: {frameCount}
                </div>
                
                {isLoading && (
                  <div className="processing-indicator">
                    <div className="spinner"></div>
                    <span>Processing...</span>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="camera-placeholder">
              <div className="placeholder-content">
                <span className="placeholder-icon">📷</span>
                <h3>Camera Not Active</h3>
                <p>Click "Start VisionAssist" to begin real-time captioning</p>
              </div>
            </div>
          )}
        </div>

        {/* Camera Controls */}
        <div className="camera-controls">
          <button
            className="btn btn-secondary"
            onClick={forceCaptureAndAnalyze}
            disabled={!isActive || isLoading}
            aria-label="Force capture and analyze current frame"
          >
            <span className="btn-icon">📸</span>
            Capture Now
          </button>
          
          <div className="camera-info">
            <span className="info-item">
              <span className="info-label">Resolution:</span>
              <span className="info-value">640x480</span>
            </span>
            <span className="info-item">
              <span className="info-label">FPS:</span>
              <span className="info-value">30</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraCapture;
