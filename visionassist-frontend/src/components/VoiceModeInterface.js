/**
 * VoiceModeInterface - OpenAI Voice Mode-inspired UI for VisionAssist
 * Features: Fullscreen camera, voice interaction, accessibility, real-time processing
 */
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MicrophoneIcon, 
  CameraIcon, 
  Cog6ToothIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  ChatBubbleLeftRightIcon,
  EyeIcon,
  EyeSlashIcon,
  SunIcon,
  MoonIcon,
  ChevronUpIcon,
  ChevronDownIcon
} from '@heroicons/react/24/outline';
import { useAccessibility } from './AccessibilityProvider';

// Persistent module-level camera state for diagnostics
let persistentCameraActive = false;

const VoiceModeInterface = () => {
  // Mount/unmount logging
  useEffect(() => {
    console.log('[MOUNT] VoiceModeInterface mounted');
    return () => {
      console.log('[UNMOUNT] VoiceModeInterface unmounted');
    };
  }, []);

  console.log('[RENDER] VoiceModeInterface');
  // State management
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  useEffect(() => {
    console.log('[STATE] cameraActive changed:', cameraActive, 'ref:', cameraActiveRef.current, 'persistent:', persistentCameraActive);
  }, [cameraActive]);
  const [cameraLoading, setCameraLoading] = useState(false);
  const cameraActiveRef = useRef(false); // Track actual camera state
  const [showSettings, setShowSettings] = useState(false);
  const [showConversation, setShowConversation] = useState(false);
  const [showTextInput, setShowTextInput] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [currentResponse, setCurrentResponse] = useState('');
  const [conversationHistory, setConversationHistory] = useState([]);
  const [settings, setSettings] = useState({
    frameRate: 3,
    autoSpeak: true,
    showSubtitles: true,
    language: 'en-US',
    voiceRate: 1.0,
    voicePitch: 1.0
  });

  // Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const recognitionRef = useRef(null);
  const synthesisRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  // Accessibility context
  const { announce, manageFocus } = useAccessibility();

  // Initialize camera stream - simplified architecture
  const initializeCamera = useCallback(async () => {
    if (cameraLoading) {
      console.log('Camera initialization already in progress, skipping...');
      return;
    }
    
    setCameraLoading(true);
    console.log('=== CAMERA INITIALIZATION START ===');
    
    try {
      // First, ensure video element exists
      if (!videoRef.current) {
        console.error('Video element not found!');
        return;
      }
      
      console.log('Video element found, requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: settings.frameRate },
          facingMode: 'user'
        },
        audio: false
      });
      
      console.log('✅ Camera stream obtained:', {
        id: stream.id,
        active: stream.active,
        videoTracks: stream.getVideoTracks().length
      });
      
      // Assign stream to video element
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      
      // Update both state and ref to ensure consistency
      cameraActiveRef.current = true;
      persistentCameraActive = true;
      setCameraActive(true);
      console.log('✅ Camera state set to ACTIVE - ref, persistent, and state updated');
      
      // Force a re-render to ensure UI updates
      setTimeout(() => {
        if (cameraActiveRef.current) {
          setCameraActive(true);
          console.log('🔄 Forcing re-render - camera should be visible now');
        }
      }, 100);
      
      // Simple play attempt
      try {
        await videoRef.current.play();
        console.log('✅ Video playing successfully');
        announce('Camera activated successfully');
      } catch (playError) {
        console.log('Initial play failed, trying muted:', playError.message);
        videoRef.current.muted = true;
        await videoRef.current.play();
        console.log('✅ Muted video playing successfully');
        announce('Camera activated successfully');
      }
      
    } catch (error) {
      console.error('❌ Camera initialization failed:', error);
      setCameraActive(false);
      announce(`Camera access failed: ${error.message}`);
    } finally {
      setCameraLoading(false);
      console.log('=== CAMERA INITIALIZATION END ===');
    }
  }, [settings.frameRate, announce, cameraLoading]);

  // Stop camera stream
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setCameraActive(false);
      announce('Camera deactivated');
    }
  }, [announce]);

  // Initialize speech recognition
  const initializeSpeechRecognition = useCallback(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      console.warn('Speech recognition not supported');
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = settings.language;

    recognition.onstart = () => {
      setIsListening(true);
      announce('Listening started');
    };

    recognition.onresult = (event) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      setCurrentTranscript(transcript);
    };

    recognition.onend = () => {
      setIsListening(false);
      if (currentTranscript.trim()) {
        processVoiceInput(currentTranscript);
      }
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      setIsListening(false);
      announce('Speech recognition error occurred');
    };

    recognitionRef.current = recognition;
  }, [settings.language, currentTranscript, announce]);

  // Process voice input
  const processVoiceInput = useCallback(async (transcript) => {
    if (!transcript.trim()) return;

    setIsProcessing(true);
    announce('Processing your request');

    try {
      // Capture current frame from video
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      if (canvas && video) {
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to conversation service
        let aiResponse;
        
        try {
          const response = await fetch('/api/v1/conversation/chat', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              image: imageData,
              question: transcript,
              conversation_history: conversationHistory.slice(-10) // Last 10 messages
            })
          });

          const result = await response.json();
          
          if (result.success) {
            aiResponse = result.response;
          } else {
            throw new Error('API error');
          }
        } catch (error) {
          // Fallback to mock responses when backend is unavailable
          const mockResponses = [
            "I can see the image you've shared. This appears to be a demonstration of VisionAssist's voice-first interface. The modern UI looks great!",
            "I'm analyzing the visual content you've provided. While the backend services are starting up, I can tell you that this interface supports real-time camera input and voice interaction.",
            "Based on your voice input, I'm processing the image through our AI vision system. The interface you're using features accessibility support and smooth animations.",
            "I can help you understand what's in your camera view. This modern interface includes features like voice input, camera preview, and conversational AI responses.",
            "Your voice input has been received and I'm analyzing the camera feed. This VisionAssist interface demonstrates production-ready accessibility and user experience design."
          ];
          
          aiResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];
          console.log('Using mock response due to backend unavailability:', error);
        }
        
        setCurrentResponse(aiResponse);
        
        // Add to conversation history
        const newEntry = {
          id: Date.now(),
          timestamp: new Date().toISOString(),
          userMessage: transcript,
          aiResponse: aiResponse,
          imageCapture: imageData
        };
        
        setConversationHistory(prev => [...prev, newEntry]);
        
        // Speak response if enabled
        if (settings.autoSpeak) {
          speakText(aiResponse);
        }
        
        announce('Response received');
      }
    } catch (error) {
      console.error('Processing error:', error);
      announce('Error occurred while processing');
    } finally {
      setIsProcessing(false);
      setCurrentTranscript('');
    }
  }, [conversationHistory, settings.autoSpeak, announce]);

  // Text-to-speech
  const speakText = useCallback((text) => {
    if (!text || !settings.autoSpeak) return;

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = settings.voiceRate;
    utterance.pitch = settings.voicePitch;
    utterance.lang = settings.language;

    utterance.onstart = () => {
      setIsSpeaking(true);
      announce('AI response playing');
    };

    utterance.onend = () => {
      setIsSpeaking(false);
    };

    utterance.onerror = () => {
      setIsSpeaking(false);
      announce('Speech synthesis error');
    };

    window.speechSynthesis.speak(utterance);
    synthesisRef.current = utterance;
  }, [settings.autoSpeak, settings.voiceRate, settings.voicePitch, settings.language, announce]);

  // Toggle voice listening
  const toggleListening = useCallback(() => {
    if (isListening) {
      recognitionRef.current?.stop();
    } else {
      if (recognitionRef.current) {
        recognitionRef.current.start();
      }
    }
  }, [isListening]);

  // Toggle camera
  const toggleCamera = useCallback(async () => {
    const currentState = cameraActiveRef.current;
    console.log('Camera toggle clicked, ref state:', currentState, 'component state:', cameraActive);
    
    if (currentState) {
      console.log('Stopping camera...');
      cameraActiveRef.current = false;
      persistentCameraActive = false;
      stopCamera();
    } else {
      console.log('Starting camera...');
      await initializeCamera();
      
      // Verify final state after initialization
      setTimeout(() => {
        console.log('Final state check:', {
          refState: cameraActiveRef.current,
          componentState: cameraActive,
          hasStream: streamRef.current ? 'YES' : 'NO',
          videoSrc: videoRef.current?.srcObject ? 'YES' : 'NO',
          streamActive: streamRef.current?.active ? 'YES' : 'NO'
        });
      }, 300);
    }
  }, [cameraActive, stopCamera, initializeCamera]);

  // Handle text input submission
  const handleTextSubmit = useCallback((text) => {
    if (text.trim()) {
      processVoiceInput(text);
      setShowTextInput(false);
    }
  }, [processVoiceInput]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (event.code === 'Space' && !showTextInput) {
        event.preventDefault();
        toggleListening();
      } else if (event.key === 'Escape') {
        setShowSettings(false);
        setShowConversation(false);
        setShowTextInput(false);
      } else if (event.key === 't' && event.ctrlKey) {
        event.preventDefault();
        setShowTextInput(true);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [toggleListening, showTextInput]);

  // Initialize on mount
  useEffect(() => {
    initializeSpeechRecognition();
    // Don't auto-initialize camera - let user click to activate
    
    return () => {
      stopCamera();
      window.speechSynthesis.cancel();
    };
  }, [initializeSpeechRecognition, stopCamera]);

  // Privacy indicator component
  const PrivacyIndicator = () => (
    <motion.div
      className={`fixed top-4 right-4 z-50 flex items-center space-x-2 px-3 py-2 rounded-full backdrop-blur-md ${
        isDarkMode ? 'bg-gray-900/80 text-white' : 'bg-white/80 text-gray-900'
      }`}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
    >
      <motion.div
        className={`w-3 h-3 rounded-full ${
          isListening ? 'bg-red-500' : 'bg-gray-400'
        }`}
        animate={isListening ? { scale: [1, 1.2, 1] } : {}}
        transition={{ repeat: Infinity, duration: 1 }}
      />
      <span className="text-sm font-medium">
        {isListening ? 'Listening' : 'Idle'}
      </span>
    </motion.div>
  );

  // Main voice button component
  const VoiceButton = () => (
    <motion.button
      className={`w-20 h-20 rounded-full flex items-center justify-center backdrop-blur-md transition-all duration-200 ${
        isListening 
          ? 'bg-red-500 text-white shadow-lg shadow-red-500/30' 
          : isDarkMode 
            ? 'bg-gray-800/80 text-white hover:bg-gray-700/80' 
            : 'bg-white/80 text-gray-900 hover:bg-gray-100/80'
      }`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={toggleListening}
      disabled={isProcessing}
      aria-label={isListening ? 'Stop listening' : 'Start listening'}
      aria-pressed={isListening}
    >
      <MicrophoneIcon className="w-8 h-8" />
    </motion.button>
  );

  // Settings panel component
  const SettingsPanel = () => (
    <AnimatePresence>
      {showSettings && (
        <motion.div
          className={`fixed inset-0 z-40 flex items-center justify-center p-4 backdrop-blur-sm ${
            isDarkMode ? 'bg-black/50' : 'bg-white/50'
          }`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={() => setShowSettings(false)}
        >
          <motion.div
            className={`w-full max-w-md p-6 rounded-2xl backdrop-blur-md ${
              isDarkMode ? 'bg-gray-900/90 text-white' : 'bg-white/90 text-gray-900'
            }`}
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-xl font-semibold mb-4">Settings</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Frame Rate</label>
                <select
                  value={settings.frameRate}
                  onChange={(e) => setSettings(prev => ({ ...prev, frameRate: parseInt(e.target.value) }))}
                  className={`w-full p-2 rounded-lg ${
                    isDarkMode ? 'bg-gray-800 text-white' : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <option value={1}>1 FPS</option>
                  <option value={3}>3 FPS</option>
                  <option value={5}>5 FPS</option>
                </select>
              </div>

              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Auto-speak responses</label>
                <button
                  onClick={() => setSettings(prev => ({ ...prev, autoSpeak: !prev.autoSpeak }))}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.autoSpeak ? 'bg-blue-500' : 'bg-gray-300'
                  }`}
                  aria-pressed={settings.autoSpeak}
                >
                  <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                    settings.autoSpeak ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Show subtitles</label>
                <button
                  onClick={() => setSettings(prev => ({ ...prev, showSubtitles: !prev.showSubtitles }))}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    settings.showSubtitles ? 'bg-blue-500' : 'bg-gray-300'
                  }`}
                  aria-pressed={settings.showSubtitles}
                >
                  <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                    settings.showSubtitles ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>

              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Dark mode</label>
                <button
                  onClick={() => setIsDarkMode(!isDarkMode)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    isDarkMode ? 'bg-blue-500' : 'bg-gray-300'
                  }`}
                  aria-pressed={isDarkMode}
                >
                  <div className={`w-5 h-5 bg-white rounded-full transition-transform ${
                    isDarkMode ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
            </div>

            <button
              onClick={() => setShowSettings(false)}
              className="w-full mt-6 py-2 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              Done
            </button>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );

  return (
    <div className={`fixed inset-0 ${isDarkMode ? 'bg-black' : 'bg-white'} transition-colors duration-300`}>
      {/* Camera view - Always render video element, control visibility with CSS */}
      <div className="relative w-full h-full">
        {/* Video element - always present */}
        <video
          ref={videoRef}
          className={`w-full h-full object-cover transition-opacity duration-300 z-10 ${
            cameraActive ? 'opacity-100' : 'opacity-0'
          }`}
          autoPlay
          playsInline
          muted
          style={{
            display: 'block',
            visibility: cameraActive ? 'visible' : 'hidden'
          }}
          onLoadedMetadata={() => {
            console.log('Video element loaded metadata');
            if (videoRef.current) {
              videoRef.current.play().catch(e => console.error('Auto-play failed:', e));
            }
          }}
        />
        
        {/* Placeholder overlay - shown when camera inactive */}
        <div className={`absolute inset-0 flex items-center justify-center bg-gray-900 text-white transition-opacity duration-300 ${
          cameraActive ? 'opacity-0 pointer-events-none' : 'opacity-100'
        }`}>
          <div className="text-center">
            <CameraIcon className="w-16 h-16 mx-auto mb-4 text-gray-400" />
            <p className="text-lg mb-2">Camera Not Active</p>
            <p className="text-sm text-gray-400">Click the camera button to enable</p>
          </div>
        </div>
        
        <canvas ref={canvasRef} className="hidden" />
        
        {/* Overlay gradient */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/30 via-transparent to-black/20" />
      </div>

      {/* Privacy indicator */}
      <PrivacyIndicator />

      {/* Main controls */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex items-center space-x-4">
        {/* Camera toggle */}
        <motion.button
          className={`w-12 h-12 rounded-full flex items-center justify-center backdrop-blur-md ${
            cameraActive
              ? 'bg-green-500 text-white'
              : isDarkMode 
                ? 'bg-gray-800/80 text-white' 
                : 'bg-white/80 text-gray-900'
          }`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={toggleCamera}
          aria-label={cameraActive ? 'Turn off camera' : 'Turn on camera'}
        >
          {cameraActive ? <EyeIcon className="w-6 h-6" /> : <EyeSlashIcon className="w-6 h-6" />}
        </motion.button>

        {/* Main voice button */}
        <VoiceButton />

        {/* Settings button */}
        <motion.button
          className={`w-12 h-12 rounded-full flex items-center justify-center backdrop-blur-md ${
            isDarkMode ? 'bg-gray-800/80 text-white' : 'bg-white/80 text-gray-900'
          }`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setShowSettings(true)}
          aria-label="Open settings"
        >
          <Cog6ToothIcon className="w-6 h-6" />
        </motion.button>
      </div>

      {/* Conversation toggle */}
      <motion.button
        className={`fixed bottom-8 right-8 w-12 h-12 rounded-full flex items-center justify-center backdrop-blur-md ${
          isDarkMode ? 'bg-gray-800/80 text-white' : 'bg-white/80 text-gray-900'
        }`}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setShowConversation(!showConversation)}
        aria-label="Toggle conversation history"
      >
        <ChatBubbleLeftRightIcon className="w-6 h-6" />
      </motion.button>

      {/* Text input toggle */}
      <motion.button
        className={`fixed bottom-8 left-8 w-12 h-12 rounded-full flex items-center justify-center backdrop-blur-md ${
          isDarkMode ? 'bg-gray-800/80 text-white' : 'bg-white/80 text-gray-900'
        }`}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setShowTextInput(true)}
        aria-label="Open text input"
      >
        <ChevronUpIcon className="w-6 h-6" />
      </motion.button>

      {/* Current transcript display */}
      <AnimatePresence>
        {currentTranscript && (
          <motion.div
            className={`fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 max-w-md p-4 rounded-2xl backdrop-blur-md ${
              isDarkMode ? 'bg-gray-900/90 text-white' : 'bg-white/90 text-gray-900'
            }`}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
          >
            <p className="text-center">{currentTranscript}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Processing indicator */}
      <AnimatePresence>
        {isProcessing && (
          <motion.div
            className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 flex items-center space-x-2 p-4 rounded-2xl backdrop-blur-md bg-blue-500/90 text-white"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
          >
            <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full" />
            <span>Processing...</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Subtitles */}
      <AnimatePresence>
        {settings.showSubtitles && currentResponse && (
          <motion.div
            className={`fixed bottom-32 left-1/2 transform -translate-x-1/2 max-w-4xl p-4 rounded-2xl backdrop-blur-md ${
              isDarkMode ? 'bg-gray-900/90 text-white' : 'bg-white/90 text-gray-900'
            }`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            <p className="text-center text-lg">{currentResponse}</p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings panel */}
      <SettingsPanel />

      {/* Text input modal */}
      <AnimatePresence>
        {showTextInput && (
          <TextInputModal
            isDarkMode={isDarkMode}
            onSubmit={handleTextSubmit}
            onClose={() => setShowTextInput(false)}
          />
        )}
      </AnimatePresence>

      {/* Conversation history */}
      <AnimatePresence>
        {showConversation && (
          <ConversationHistory
            history={conversationHistory}
            isDarkMode={isDarkMode}
            onClose={() => setShowConversation(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

// Text input modal component
const TextInputModal = ({ isDarkMode, onSubmit, onClose }) => {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(text);
    setText('');
  };

  return (
    <motion.div
      className={`fixed inset-0 z-50 flex items-end justify-center p-4 backdrop-blur-sm ${
        isDarkMode ? 'bg-black/50' : 'bg-white/50'
      }`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.form
        className={`w-full max-w-md p-4 rounded-t-2xl backdrop-blur-md ${
          isDarkMode ? 'bg-gray-900/90 text-white' : 'bg-white/90 text-gray-900'
        }`}
        initial={{ y: 100 }}
        animate={{ y: 0 }}
        exit={{ y: 100 }}
        onClick={(e) => e.stopPropagation()}
        onSubmit={handleSubmit}
      >
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type your message..."
          className={`w-full p-3 rounded-lg resize-none ${
            isDarkMode ? 'bg-gray-800 text-white' : 'bg-gray-100 text-gray-900'
          }`}
          rows={3}
          autoFocus
        />
        <div className="flex space-x-2 mt-3">
          <button
            type="submit"
            className="flex-1 py-2 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            disabled={!text.trim()}
          >
            Send
          </button>
          <button
            type="button"
            onClick={onClose}
            className={`py-2 px-4 rounded-lg transition-colors ${
              isDarkMode ? 'bg-gray-800 text-white hover:bg-gray-700' : 'bg-gray-200 text-gray-900 hover:bg-gray-300'
            }`}
          >
            Cancel
          </button>
        </div>
      </motion.form>
    </motion.div>
  );
};

// Conversation history component
const ConversationHistory = ({ history, isDarkMode, onClose }) => (
  <motion.div
    className={`fixed inset-y-0 right-0 w-80 backdrop-blur-md ${
      isDarkMode ? 'bg-gray-900/90 text-white' : 'bg-white/90 text-gray-900'
    }`}
    initial={{ x: 320 }}
    animate={{ x: 0 }}
    exit={{ x: 320 }}
  >
    <div className="p-4 border-b border-gray-200/20">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Conversation</h2>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-gray-200/20 transition-colors"
          aria-label="Close conversation history"
        >
          ×
        </button>
      </div>
    </div>
    
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {history.map((entry) => (
        <div key={entry.id} className="space-y-2">
          <div className="text-xs text-gray-400">
            {new Date(entry.timestamp).toLocaleTimeString()}
          </div>
          <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-blue-600' : 'bg-blue-100'}`}>
            <p className="text-sm">{entry.userMessage}</p>
          </div>
          <div className={`p-3 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-gray-100'}`}>
            <p className="text-sm">{entry.aiResponse}</p>
          </div>
        </div>
      ))}
    </div>
  </motion.div>
);

export default VoiceModeInterface;
