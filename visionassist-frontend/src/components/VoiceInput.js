import React, { useState, useRef, useEffect } from 'react';
import './VoiceInput.css';

const VoiceInput = ({ onVoiceCommand, isActive, audioEnabled }) => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isSupported, setIsSupported] = useState(false);
  const recognitionRef = useRef(null);
  const timeoutRef = useRef(null);

  useEffect(() => {
    // Check if speech recognition is supported
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      setIsSupported(true);
      
      // Initialize speech recognition
      recognitionRef.current = new SpeechRecognition();
      const recognition = recognitionRef.current;
      
      recognition.continuous = false;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.maxAlternatives = 1;
      
      recognition.onstart = () => {
        console.log('🎤 Voice recognition started');
        setIsListening(true);
      };
      
      recognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          } else {
            interimTranscript += transcript;
          }
        }
        
        const currentTranscript = finalTranscript || interimTranscript;
        setTranscript(currentTranscript);
        
        // If we have a final result, process it
        if (finalTranscript) {
          console.log('🗣️ Final transcript:', finalTranscript);
          onVoiceCommand(finalTranscript.trim());
          setTranscript('');
          setIsListening(false);
        }
      };
      
      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        setTranscript('');
        
        if (event.error === 'no-speech') {
          // Automatically restart if no speech detected
          setTimeout(() => {
            if (isActive && audioEnabled) {
              startListening();
            }
          }, 1000);
        }
      };
      
      recognition.onend = () => {
        console.log('🎤 Voice recognition ended');
        setIsListening(false);
        
        // Auto-restart listening if still active
        if (isActive && audioEnabled && !transcript) {
          timeoutRef.current = setTimeout(() => {
            startListening();
          }, 500);
        }
      };
    } else {
      console.warn('Speech recognition not supported in this browser');
      setIsSupported(false);
    }
    
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, [isActive, audioEnabled, transcript, onVoiceCommand]);

  const startListening = () => {
    if (!isSupported || !audioEnabled || !recognitionRef.current) return;
    
    try {
      if (isListening) {
        recognitionRef.current.stop();
      } else {
        recognitionRef.current.start();
      }
    } catch (error) {
      console.error('Error starting speech recognition:', error);
    }
  };

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsListening(false);
    setTranscript('');
  };

  // Auto-start listening when component becomes active
  useEffect(() => {
    if (isActive && audioEnabled && isSupported) {
      const timer = setTimeout(() => {
        startListening();
      }, 1000);
      
      return () => clearTimeout(timer);
    } else {
      stopListening();
    }
  }, [isActive, audioEnabled, isSupported]);

  if (!isSupported) {
    return (
      <div className="voice-input voice-input-unsupported">
        <div className="voice-status">
          <span className="voice-icon">🚫</span>
          <span className="voice-text">Voice input not supported in this browser</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`voice-input ${isActive ? 'active' : ''}`}>
      <div className="voice-controls">
        <button
          className={`voice-button ${isListening ? 'listening' : ''}`}
          onClick={isListening ? stopListening : startListening}
          disabled={!audioEnabled}
          aria-label={isListening ? 'Stop listening' : 'Start voice input'}
        >
          <span className="voice-icon">
            {isListening ? '🎤' : '🎙️'}
          </span>
          <span className="voice-label">
            {isListening ? 'Listening...' : 'Voice Input'}
          </span>
        </button>
      </div>
      
      {transcript && (
        <div className="voice-transcript">
          <div className="transcript-content">
            <span className="transcript-icon">💬</span>
            <span className="transcript-text">{transcript}</span>
          </div>
        </div>
      )}
      
      <div className="voice-status">
        <div className={`status-indicator ${isListening ? 'active' : 'inactive'}`}>
          <span className="status-dot"></span>
          <span className="status-text">
            {isListening ? 'Listening for voice commands' : 'Voice input ready'}
          </span>
        </div>
      </div>
      
      <div className="voice-commands-help">
        <div className="help-title">Try saying:</div>
        <div className="help-commands">
          <span>"What do you see?"</span>
          <span>"Describe this scene"</span>
          <span>"How many people are here?"</span>
          <span>"What colors do you see?"</span>
          <span>"Read any text"</span>
        </div>
      </div>
    </div>
  );
};

export default VoiceInput;
