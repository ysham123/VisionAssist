import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import VoiceInput from './components/VoiceInput';

// API configuration
const API_BASE_URL = 'http://localhost:5000'; // Flask API backend
const CONVERSATIONAL_API_URL = 'http://localhost:5001'; // Ollama conversational backend

function App() {
  const [isActive, setIsActive] = useState(false);
  const [currentCaption, setCurrentCaption] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [error, setError] = useState(null);
  const [cameraStream, setCameraStream] = useState(null);
  const [frameRate, setFrameRate] = useState(3); // Frames per second
  const [isProcessing, setIsProcessing] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [showChat, setShowChat] = useState(true);
  
  // Conversational AI state
  const [isConversationMode, setIsConversationMode] = useState(false);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [isResponding, setIsResponding] = useState(false);
  const [ollamaStatus, setOllamaStatus] = useState({ available: false, models: [] });
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const canvasContextRef = useRef(null); // Cache canvas context
  const animationFrameRef = useRef(null);
  const lastProcessedTime = useRef(0);
  const speechSynthesis = useRef(window.speechSynthesis);
  const speechQueue = useRef([]);
  const speechTimeout = useRef(null);
  const processingLock = useRef(false);
  const lastSpokenCaption = useRef('');
  const isSpeaking = useRef(false);

  // Initialize and cleanup
  useEffect(() => {
    // Check for Web Speech API support
    if (!('speechSynthesis' in window)) {
      setAudioEnabled(false);
      console.warn('Web Speech API not supported, audio feedback disabled');
    }

    // Cleanup function
    return () => {
      stopCamera();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (speechTimeout.current) {
        clearTimeout(speechTimeout.current);
      }
      if (speechSynthesis.current) {
        speechSynthesis.current.cancel();
      }
    };
  }, []);

  // Advanced speech management to eliminate stuttering
  const speakCaption = useCallback((text) => {
    if (!audioEnabled || !speechSynthesis.current || !text) return;
    
    // Don't speak if it's the same caption as last time
    if (text === lastSpokenCaption.current) {
      return;
    }
    
    // Check similarity to avoid repetitive speech
    if (lastSpokenCaption.current) {
      const similarity = text.toLowerCase().split(' ').filter(word => 
        lastSpokenCaption.current?.toLowerCase().includes(word)
      ).length;
      if (similarity > text.split(' ').length * 0.8) {
        return; // Skip very similar captions
      }
    }
    
    // Clear any existing timeout
    if (speechTimeout.current) {
      clearTimeout(speechTimeout.current);
    }
    
    // Debounce speech - wait 500ms before speaking to avoid rapid fire
    speechTimeout.current = setTimeout(() => {
      speakNow(text);
    }, 500);
  }, [audioEnabled]);
  
  function speakNow(textToSpeak) {
    if (!audioEnabled) return;
    
    // Double-check if we should still speak
    if (isSpeaking.current || speechSynthesis.current.speaking) {
      return;
    }
    
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.rate = 0.9; // Slightly slower for better clarity
    utterance.pitch = 1.0;
    utterance.volume = 0.8;
    
    utterance.onstart = () => {
      isSpeaking.current = true;
      console.log('🔊 Started speaking:', textToSpeak.substring(0, 50) + '...');
    };
    
    utterance.onend = () => {
      isSpeaking.current = false;
      lastSpokenCaption.current = textToSpeak;
      console.log('✅ Finished speaking');
    };
    
    utterance.onerror = (event) => {
      console.warn('Speech synthesis error:', event.error);
      isSpeaking.current = false;
    };
    
    speechSynthesis.current.speak(utterance);
  }

  // Add caption to chat history
  const addToChat = useCallback((caption, timestamp = new Date()) => {
    const chatEntry = {
      id: Date.now(),
      caption,
      timestamp,
      type: 'caption'
    };
    
    setChatHistory(prev => {
      const newHistory = [chatEntry, ...prev];
      // Keep only last 50 entries to prevent memory issues
      return newHistory.slice(0, 50);
    });
  }, []);

  // Parse caption for objects and add visual pointers
  const parseObjectsFromCaption = useCallback((caption) => {
    // Simple object detection from caption text
    const commonObjects = [
      'person', 'man', 'woman', 'people', 'face', 'hand', 'hair',
      'laptop', 'computer', 'phone', 'smartphone', 'keyboard', 'mouse',
      'table', 'desk', 'chair', 'book', 'cup', 'coffee', 'bottle',
      'window', 'door', 'wall', 'plant', 'flower', 'tree',
      'car', 'bike', 'bicycle', 'building', 'house'
    ];
    
    const detectedInCaption = [];
    const lowerCaption = caption.toLowerCase();
    
    commonObjects.forEach(obj => {
      if (lowerCaption.includes(obj)) {
        detectedInCaption.push({
          id: `${obj}-${Date.now()}`,
          name: obj,
          // Random positions for demo - in real implementation would use object detection AI
          x: Math.random() * 80 + 10, // 10-90% from left
          y: Math.random() * 80 + 10, // 10-90% from top
          confidence: Math.random() * 0.3 + 0.7 // 70-100% confidence
        });
      }
    });
    
    setDetectedObjects(detectedInCaption);
    return detectedInCaption;
  }, []);

  // Send frame to backend for captioning
  const processFrame = async (imageData) => {
    if (processingLock.current) return null;
    processingLock.current = true;
    
    try {
      // Send image to Flask API backend
      const response = await fetch(`${API_BASE_URL}/caption`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ image: imageData })
      });
      
      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      // Update detected objects from backend
      if (result.objects && Array.isArray(result.objects)) {
        setDetectedObjects(result.objects);
      }
      
      return result.caption;
      
    } catch (error) {
      console.error('Error processing frame:', error);
      
      // Fallback to demo captions if API fails
      const demoCaptions = [
        "A person sitting at a desk with a laptop computer",
        "A wooden table with various objects on it",
        "A room with natural lighting from a window",
        "A person holding a smartphone in their hands",
        "A coffee cup on a wooden surface",
        "A bookshelf with books and decorative items",
        "A plant in a pot near a window",
        "A keyboard and mouse on a desk"
      ];
      return demoCaptions[Math.floor(Math.random() * demoCaptions.length)];
    } finally {
      processingLock.current = false;
    }
  };

  // Check Ollama status
  const checkOllamaStatus = useCallback(async () => {
    try {
      const response = await fetch(`${CONVERSATIONAL_API_URL}/ollama-status`);
      if (response.ok) {
        const status = await response.json();
        setOllamaStatus(status);
        return status.available;
      }
    } catch (error) {
      console.warn('Ollama status check failed:', error);
      setOllamaStatus({ available: false, models: [] });
    }
    return false;
  }, []);

  // Handle voice commands from speech recognition
  const handleVoiceCommand = useCallback(async (voiceText) => {
    if (!voiceText.trim() || isResponding) return;
    
    console.log('🎤 Voice command received:', voiceText);
    setIsResponding(true);
    
    try {
      // Capture current frame for context
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      if (!video || !canvas || !video.srcObject || video.readyState < 2) {
        speakNow('Camera is not ready. Please make sure the camera is active.');
        return;
      }
      
      // Capture frame
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0);
      const imageData = canvas.toDataURL('image/jpeg', 0.8);
      
      // Send to conversational API
      const response = await fetch(`${CONVERSATIONAL_API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData,
          question: voiceText
        })
      });
      
      if (!response.ok) {
        throw new Error(`Conversational API failed: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        // Add to conversation history
        const conversationEntry = {
          id: Date.now(),
          timestamp: new Date(),
          question: voiceText,
          response: result.response,
          caption: result.caption
        };
        
        setConversationHistory(prev => [conversationEntry, ...prev.slice(0, 9)]);
        
        // Speak the response
        speakNow(result.response);
        
        console.log('🤖 AI Response:', result.response);
      } else {
        throw new Error(result.error || 'Unknown error');
      }
      
    } catch (error) {
      console.error('Error processing voice command:', error);
      const errorMessage = 'Sorry, I had trouble processing your request. Please try again.';
      speakNow(errorMessage);
    } finally {
      setIsResponding(false);
    }
  }, [isResponding, speakNow]);

  // Toggle conversation mode
  const toggleConversationMode = useCallback(() => {
    setIsConversationMode(prev => {
      const newMode = !prev;
      if (newMode) {
        speakNow('Conversation mode activated. You can now ask me questions about what I see.');
        checkOllamaStatus();
      } else {
        speakNow('Conversation mode deactivated. Returning to automatic captioning.');
      }
      return newMode;
    });
  }, [speakNow, checkOllamaStatus]);

  // Check Ollama status on component mount
  useEffect(() => {
    checkOllamaStatus();
    const interval = setInterval(checkOllamaStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, [checkOllamaStatus]);

  // Process video frames for captioning
  const processVideoFrame = useCallback(async (timestamp) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    console.log('processVideoFrame called', { 
      isActive, 
      videoRef: !!video, 
      canvasRef: !!canvas,
      videoSrcObject: !!video?.srcObject,
      videoReadyState: video?.readyState
    });
    
    // Use video.srcObject instead of isActive to check if camera is active
    if (!video || !canvas || !video.srcObject || video.readyState < 2) {
      console.log('Skipping frame processing - conditions not met');
      return;
    }

    const now = Date.now();
    const elapsed = now - lastProcessedTime.current;
    const interval = 1000 / frameRate; // Convert FPS to milliseconds

    console.log('Frame timing:', { elapsed, interval, shouldProcess: elapsed > interval });

    if (elapsed > interval) {
      lastProcessedTime.current = now - (elapsed % interval);
      
      // Only process if not already processing
      if (!isProcessing && !processingLock.current) {
        try {
          setIsProcessing(true);
          
          // Capture current frame with cached context
          const canvas = canvasRef.current;
          const video = videoRef.current;
          
          // Initialize or get cached context
          if (!canvasContextRef.current) {
            canvasContextRef.current = canvas.getContext('2d');
          }
          const context = canvasContextRef.current;
          
          // Optimize canvas size for faster processing
          const targetWidth = Math.min(video.videoWidth, 640);
          const targetHeight = (video.videoHeight * targetWidth) / video.videoWidth;
          
          // Only resize canvas if dimensions changed
          if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
            canvas.width = targetWidth;
            canvas.height = targetHeight;
          }
          
          // Draw video frame to canvas (scaled for performance)
          context.drawImage(video, 0, 0, targetWidth, targetHeight);
          
          // Get image data with optimized quality
          const imageData = canvas.toDataURL('image/jpeg', 0.7);
          
          // Process the frame
          const caption = await processFrame(imageData);
          
          if (caption) {
            setCurrentCaption(caption);
            speakCaption(caption);
            addToChat(caption);
            // Object detection is now handled by backend API
          }
        } catch (error) {
          console.error('Error processing video frame:', error);
          setError('Error processing video frame. Please try again.');
        } finally {
          setIsProcessing(false);
        }
      }
    }
    
    // Continue the processing loop
    animationFrameRef.current = requestAnimationFrame(processVideoFrame);
  }, [isActive, frameRate, isProcessing, speakCaption]);

  // Helper function for camera error messages
  const getCameraErrorMessage = (error) => {
    if (error.name === 'NotAllowedError') {
      return '🚫 Camera access denied. Please allow camera permissions and try again.';
    } else if (error.name === 'NotFoundError') {
      return '📷 No camera found. Please connect a camera and try again.';
    } else if (error.name === 'NotSupportedError') {
      return '❌ Camera not supported in this browser. Try Chrome, Firefox, or Safari.';
    } else if (error.name === 'NotReadableError') {
      return '⚠️ Camera is being used by another application. Please close other apps and try again.';
    } else if (error.message.includes('playback')) {
      return '❌ Could not start video playback. Please try refreshing the page.';
    }
    return 'Unable to access camera. Please try again.';
  };

  // Start camera feed with real-time captioning
  const startCamera = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Check if getUserMedia is supported
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera not supported in this browser');
      }
      
      // Request camera access with better error handling
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280, min: 640 },
          height: { ideal: 720, min: 480 },
          facingMode: 'environment', // Prefer back camera on mobile
          frameRate: { ideal: 30, max: 30 } // Limit frame rate for performance
        },
        audio: false
      });
      
      // Set up video element
      const video = videoRef.current;
      if (!video) {
        throw new Error('Video element not found');
      }
      
      console.log('Setting up video stream...', stream);
      
      // Set up video stream
      video.srcObject = stream;
      video.playsInline = true;
      video.muted = true;
      video.autoplay = true;
      
      // Wait for video to be ready with timeout
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Video loading timeout'));
        }, 10000); // 10 second timeout
        
        const onLoadedMetadata = () => {
          console.log('Video metadata loaded, starting playback...');
          video.play()
            .then(() => {
              console.log('Video playback started successfully');
              clearTimeout(timeout);
              resolve();
            })
            .catch(err => {
              console.error('Error playing video:', err);
              clearTimeout(timeout);
              reject(new Error(`Could not start video playback: ${err.message}`));
            });
        };
        
        const onError = (err) => {
          console.error('Video error:', err);
          clearTimeout(timeout);
          reject(new Error(`Video error: ${err.message || 'Unknown video error'}`));
        };
        
        video.onloadedmetadata = onLoadedMetadata;
        video.onerror = onError;
        
        // Fallback: try to play immediately if metadata is already loaded
        if (video.readyState >= 1) {
          onLoadedMetadata();
        }
      });
      
      // Update state first
      setCameraStream(stream);
      setIsActive(true);
      setCurrentCaption('VisionAssist is active. Analyzing scene...');
      
      // Start the processing loop immediately
      console.log('Starting processing loop...');
      lastProcessedTime.current = Date.now();
      processVideoFrame();
      
    } catch (err) {
      console.error('Camera access error:', err);
      console.error('Error name:', err.name);
      console.error('Error message:', err.message);
      console.error('Error stack:', err.stack);
      
      // Log additional debugging info
      console.log('Navigator mediaDevices available:', !!navigator.mediaDevices);
      console.log('getUserMedia available:', !!navigator.mediaDevices?.getUserMedia);
      console.log('Video element ref:', videoRef.current);
      
      setError(getCameraErrorMessage(err));
      stopCamera();
    } finally {
      setIsLoading(false);
    }
  };

  // Stop camera and clean up
  const stopCamera = useCallback(() => {
    // Stop all tracks in the stream
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => {
        track.stop();
      });
      setCameraStream(null);
    }
    
    // Stop any ongoing animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    // Reset states
    setIsActive(false);
    setIsProcessing(false);
    setCurrentCaption('');
    
    // Clear video element
    const video = videoRef.current;
    if (video) {
      video.srcObject = null;
    }
    
    // Stop any ongoing speech
    if (speechSynthesis.current) {
      speechSynthesis.current.cancel();
    }
  }, [cameraStream]);

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">📸 CaptureNow</div>
        <h1>Real-Time Image Capture & Captions</h1>
        <nav className="nav">
          <a href="#">Home</a>
          <a href="#">Gallery</a>
          <a href="#">Settings</a>
          <a href="#">About</a>
        </nav>
      </header>

      {/* Error Display */}
      {error && (
        <div className="error" role="alert">
          <div className="error-content">
            <div className="error-message">{error}</div>
            {error.includes('denied') && (
              <div className="error-help">
                <strong>How to fix:</strong>
                <ul>
                  <li>Click the camera icon in your browser's address bar</li>
                  <li>Select "Allow" for camera access</li>
                  <li>Refresh the page and try again</li>
                  <li>Make sure no other apps are using your camera</li>
                </ul>
              </div>
            )}
            {error.includes('not supported') && (
              <div className="error-help">
                <strong>Supported browsers:</strong> Chrome, Firefox, Safari, Edge
              </div>
            )}
          </div>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* Main Content */}
      <main className="main">
        <div className="main-layout">
          {/* Camera Section */}
          <div className="camera-section">
            {/* Camera Container */}
            <div className="camera-container">
          {/* Video element - always present but hidden when not active */}
          <video 
            ref={videoRef}
            className={`live-feed ${isActive ? 'active' : 'hidden'}`}
            autoPlay 
            playsInline 
            muted
            aria-label="Live camera feed"
          />
          
          {isActive && (
            <>
              <div className="camera-overlay"></div>
              
              {/* Object Detection Pointers */}
              {detectedObjects.map(obj => (
                <div 
                  key={obj.id}
                  className="object-pointer"
                  style={{
                    position: 'absolute',
                    left: `${obj.x}%`,
                    top: `${obj.y}%`,
                    transform: 'translate(-50%, -50%)'
                  }}
                >
                  <div className="pointer-dot"></div>
                  <div className="pointer-label">
                    {obj.name}
                    <span className="confidence">({Math.round(obj.confidence * 100)}%)</span>
                  </div>
                </div>
              ))}
              
              {/* Real-time Caption Overlay */}
              <div className="live-caption-overlay">
                <div className="live-caption">
                  {isProcessing ? (
                    <div className="caption-loading">
                      <div className="spinner"></div>
                      Analyzing scene...
                    </div>
                  ) : (
                    currentCaption || 'Real-time captioning active'
                  )}
                </div>
              </div>
            </>
          )}
          
          {!isActive && (
            <div className="camera-placeholder">
              <div className="placeholder-content">
                <span className="placeholder-icon">📹</span>
                <p>Start camera to begin real-time captioning</p>
              </div>
            </div>
          )}
          
          {/* Capture Controls */}
          <div className="capture-controls">
            {!isActive ? (
              <button 
                className="btn btn-start" 
                onClick={startCamera}
                disabled={isLoading}
                aria-label="Start real-time captioning"
              >
                {isLoading ? (
                  <>
                    <div className="spinner"></div>
                    Starting Camera...
                  </>
                ) : (
                  <>
                    📹 Start Real-Time Captioning
                  </>
                )}
              </button>
            ) : (
              <>
                <div className="frame-rate-control">
                  <label htmlFor="frameRate">Caption Updates:</label>
                  <select 
                    id="frameRate"
                    value={frameRate}
                    onChange={(e) => setFrameRate(Number(e.target.value))}
                    disabled={isProcessing}
                    aria-label="Select caption update frequency"
                  >
                    <option value={1}>1/sec (Low CPU)</option>
                    <option value={3}>3/sec (Balanced)</option>
                    <option value={5}>5/sec (Smooth)</option>
                  </select>
                </div>
                
                <button 
                  className={`btn btn-audio ${audioEnabled ? 'active' : ''}`}
                  onClick={() => setAudioEnabled(!audioEnabled)}
                  aria-label={audioEnabled ? 'Disable audio feedback' : 'Enable audio feedback'}
                >
                  {audioEnabled ? '🔊 Audio On' : '🔇 Audio Off'}
                </button>
                
                <button 
                  className={`btn btn-conversation ${isConversationMode ? 'active' : ''}`}
                  onClick={toggleConversationMode}
                  disabled={!ollamaStatus.available}
                  aria-label={isConversationMode ? 'Disable conversation mode' : 'Enable conversation mode'}
                  title={!ollamaStatus.available ? 'Ollama not available' : ''}
                >
                  {isConversationMode ? '🤖 Chat Mode On' : '💬 Chat Mode'}
                </button>
                
                <button 
                  className="btn btn-stop" 
                  onClick={stopCamera}
                  aria-label="Stop camera"
                >
                  ⏹️ Stop Camera
                </button>
              </>
            )}
          </div>
          
          {/* Instructions */}
          <p className="instruction-text">
            {!isActive 
              ? "Start your camera to begin real-time captioning" 
              : isConversationMode 
                ? "Conversation mode active - ask me questions about what I see!"
                : "Real-time AI captioning active - objects detected will be highlighted"
            }
          </p>
          
          {/* Voice Input Component - only show when conversation mode is active */}
          {isConversationMode && isActive && (
            <VoiceInput 
              onVoiceCommand={handleVoiceCommand}
              isActive={isConversationMode && isActive}
              audioEnabled={audioEnabled}
            />
          )}
          
            </div>
            {/* Hidden canvas for image capture */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>
          
          {/* Chat Sidebar */}
          <div className={`chat-sidebar ${showChat ? 'open' : 'closed'}`}>
            <div className="chat-header">
              <h3>{isConversationMode ? 'Conversation' : 'Caption History'}</h3>
              <button 
                className="chat-toggle"
                onClick={() => setShowChat(!showChat)}
                aria-label={showChat ? 'Hide chat' : 'Show chat'}
              >
                {showChat ? '→' : '←'}
              </button>
            </div>
            
            <div className="chat-content">
              {/* Conversation History - when in conversation mode */}
              {isConversationMode && conversationHistory.length > 0 && (
                <div className="conversation-section">
                  <h4>Recent Conversations</h4>
                  <div className="conversation-messages">
                    {conversationHistory.map(entry => (
                      <div key={entry.id} className="conversation-entry">
                        <div className="conversation-time">
                          {entry.timestamp.toLocaleTimeString()}
                        </div>
                        <div className="user-question">
                          <span className="speaker">You:</span> {entry.question}
                        </div>
                        <div className="ai-response">
                          <span className="speaker">AI:</span> {entry.response}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Regular Caption History */}
              {!isConversationMode && (
                <>
                  {chatHistory.length === 0 ? (
                    <div className="chat-empty">
                      <p>No captions yet. Start the camera to begin!</p>
                    </div>
                  ) : (
                    <div className="chat-messages">
                      {chatHistory.map(entry => (
                        <div key={entry.id} className="chat-message">
                          <div className="message-time">
                            {entry.timestamp.toLocaleTimeString()}
                          </div>
                          <div className="message-text">
                            {entry.caption}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
              
              {/* Empty state for conversation mode */}
              {isConversationMode && conversationHistory.length === 0 && (
                <div className="chat-empty">
                  <p>No conversations yet. Use voice input to ask questions!</p>
                  {!ollamaStatus.available && (
                    <p className="ollama-status">⚠️ Ollama not available</p>
                  )}
                </div>
              )}
            </div>
            
            <div className="chat-footer">
              <button 
                className="btn btn-clear"
                onClick={() => setChatHistory([])}
                disabled={chatHistory.length === 0}
              >
                Clear History
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>&copy; 2025 CaptureNow | All Rights Reserved</p>
      </footer>
    </div>
  );
}

export default App;
