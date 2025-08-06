// Speech Module - Handles speech recognition and synthesis
class SpeechManager {
    constructor() {
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.isListening = false;
        this.isAssistantSpeaking = false;
        this.isSpeechEnabled = true;
        this.conversationMode = false;
        this.conversationActive = false;
        this.browserSupported = false;
        this.errorCount = 0;
        this.maxErrors = 3;
        this.consecutiveNetworkErrors = 0;
        this.maxConsecutiveNetworkErrors = 5;
        this.networkErrorBackoffTime = 1000; // Start with 1 second
        this.maxBackoffTime = 30000; // Max 30 seconds
        this.isInErrorBackoff = false;
        this.lastNetworkErrorTime = 0;
        
        this.initializeSpeechRecognition();
    }

    initializeSpeechRecognition() {
        // Check for speech recognition support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            console.warn('🚫 Speech recognition not supported in this browser');
            this.browserSupported = false;
            this.showBrowserCompatibilityMessage();
            return;
        }
        
        this.browserSupported = true;

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';
        this.recognition.maxAlternatives = 1;

        console.log('🎤 Speech recognition initialized');
    }

    // Show browser compatibility message for unsupported browsers
    showBrowserCompatibilityMessage() {
        const message = 'Speech features require Chrome, Edge, or Safari. Text input is available as an alternative.';
        console.warn('🚫 Browser compatibility:', message);
        
        // Announce to screen readers
        this.announceToScreenReader(message);
        
        // Show visual notification if there's a way to display it
        if (typeof window.showNotification === 'function') {
            window.showNotification(message, 'warning');
        }
    }

    // Announce message to screen readers
    announceToScreenReader(message) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'assertive');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.style.position = 'absolute';
        announcement.style.left = '-10000px';
        announcement.textContent = message;
        document.body.appendChild(announcement);
        
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }

    // Handle network errors with circuit breaker pattern
    handleNetworkError(onErrorCallback) {
        this.consecutiveNetworkErrors++;
        this.lastNetworkErrorTime = Date.now();
        
        console.warn(`🚫 Network error ${this.consecutiveNetworkErrors}/${this.maxConsecutiveNetworkErrors}`);
        
        // Circuit breaker: Stop trying after too many consecutive network errors
        if (this.consecutiveNetworkErrors >= this.maxConsecutiveNetworkErrors) {
            this.isInErrorBackoff = true;
            const backoffTime = Math.min(this.networkErrorBackoffTime * Math.pow(2, this.consecutiveNetworkErrors - this.maxConsecutiveNetworkErrors), this.maxBackoffTime);
            
            console.warn(`🚫 Speech recognition circuit breaker activated. Backing off for ${backoffTime/1000}s`);
            
            // Announce to user
            const message = `Speech recognition temporarily unavailable due to network issues. Please use text input or try again in ${Math.round(backoffTime/1000)} seconds.`;
            this.announceToScreenReader(message);
            
            // Reset after backoff period
            setTimeout(() => {
                console.log('🔄 Speech recognition circuit breaker reset');
                this.consecutiveNetworkErrors = 0;
                this.isInErrorBackoff = false;
                this.announceToScreenReader('Speech recognition is available again.');
            }, backoffTime);
            
            if (onErrorCallback) {
                onErrorCallback('network_circuit_breaker');
            }
        } else {
            // Still within error threshold, provide user feedback but don't restart immediately
            const message = `Speech recognition network issue. ${this.maxConsecutiveNetworkErrors - this.consecutiveNetworkErrors} attempts remaining.`;
            console.warn(`🚫 ${message}`);
            
            if (onErrorCallback) {
                onErrorCallback('network');
            }
        }
    }

    // Start listening for voice input
    startListening() {
        if (!this.recognition) {
            console.error('Speech recognition not available');
            return false;
        }
        
        // Check if in error backoff mode
        if (this.isInErrorBackoff) {
            console.warn('🚫 Speech recognition in backoff mode, skipping start');
            return false;
        }

        if (this.isListening) {
            console.log('🎤 Already listening...');
            return true;
        }

        // Check if speech recognition is in a valid state
        try {
            // Validate browser support
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                throw new Error('Speech recognition not supported in this browser');
            }

            // Check if microphone permissions might be denied
            if (navigator.permissions) {
                navigator.permissions.query({name: 'microphone'}).then(result => {
                    if (result.state === 'denied') {
                        console.error('🎤 Microphone permission denied');
                    }
                }).catch(permError => {
                    console.log('🎤 Could not check microphone permissions:', permError);
                });
            }

            this.recognition.start();
            this.isListening = true;
            console.log('🎤 Started listening...');
            return true;
        } catch (error) {
            console.error('🎤 Error starting speech recognition:', error);
            
            // Provide specific error handling
            if (error.name === 'InvalidStateError') {
                console.error('🎤 Speech recognition is already active or in invalid state');
                // Try to reset the recognition
                this.resetRecognition();
            } else if (error.name === 'NotAllowedError') {
                console.error('🎤 Microphone access denied by user');
            } else if (error.name === 'NotSupportedError') {
                console.error('🎤 Speech recognition not supported');
            }
            
            this.isListening = false;
            return false;
        }
    }

    // Reset speech recognition to clean state
    resetRecognition() {
        console.log('🎤 Resetting speech recognition...');
        
        if (this.recognition) {
            try {
                this.recognition.stop();
            } catch (error) {
                console.log('🎤 Error stopping recognition during reset:', error);
            }
            
            this.isListening = false;
            
            // Reinitialize recognition
            setTimeout(() => {
                this.initializeRecognition();
                console.log('🎤 Speech recognition reset complete');
            }, 100);
        }
    }

    // Stop listening
    stopListening() {
        if (this.recognition && this.isListening) {
            try {
                this.recognition.stop();
                this.isListening = false;
                console.log('🎤 Stopped listening');
            } catch (error) {
                console.error('🎤 Error stopping speech recognition:', error);
                this.isListening = false;
                // Force reset if stop fails
                this.resetRecognition();
            }
        }
    }

    // Speak text using speech synthesis
    speakText(text, onEndCallback = null) {
        if (!this.isSpeechEnabled || !text) {
            if (onEndCallback) onEndCallback();
            return;
        }

        this.isAssistantSpeaking = true;
        
        // Create utterance
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Configure speech
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        // Use preferred voice if available
        const voices = this.synthesis.getVoices();
        const preferredVoice = voices.find(voice => 
            voice.lang.startsWith('en') && voice.name.includes('Female')
        ) || voices.find(voice => voice.lang.startsWith('en'));

        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }

        // Event handlers
        utterance.onstart = () => {
            console.log('🔊 Assistant speaking...');
            document.body.classList.add('speaking');
        };

        utterance.onend = () => {
            console.log('🔊 Assistant finished speaking');
            this.isAssistantSpeaking = false;
            document.body.classList.remove('speaking');
            
            if (onEndCallback) {
                onEndCallback();
            }
        };

        utterance.onerror = (event) => {
            console.error('🔊 Speech synthesis error:', event.error);
            this.isAssistantSpeaking = false;
            document.body.classList.remove('speaking');
            
            if (onEndCallback) {
                onEndCallback();
            }
        };

        // Speak the text
        this.synthesis.speak(utterance);
    }

    // Speak text and then restart listening (for conversation mode)
    speakTextAndThenListen(text, restartListeningCallback = null) {
        this.speakText(text, () => {
            // Restart listening if conversation mode is still active
            if (this.conversationMode && this.conversationActive) {
                setTimeout(() => {
                    if (restartListeningCallback) {
                        restartListeningCallback();
                    }
                }, 500); // Brief pause before listening again
            }
        });
    }

    // Set up recognition event handlers for conversation mode
    setupConversationRecognition(onResultCallback, onErrorCallback) {
        if (!this.recognition) {
            console.error('Speech recognition not available for conversation mode');
            return false;
        }

        let finalTranscript = '';
        let isProcessing = false;

        this.recognition.onresult = (event) => {
            let interimTranscript = '';
            finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            // If we have a final result, process it
            if (finalTranscript.trim() !== '' && !isProcessing) {
                isProcessing = true;
                console.log('🎤 Final transcript:', finalTranscript);

                if (onResultCallback) {
                    onResultCallback(finalTranscript.trim());
                }

                // Reset for next input
                finalTranscript = '';
                isProcessing = false;
            }
        };

        this.recognition.onerror = (event) => {
            console.error('🎤 Speech recognition error:', event.error);
            
            // Handle network errors with circuit breaker pattern
            if (event.error === 'network') {
                this.handleNetworkError(onErrorCallback);
            } else {
                // Handle other errors normally
                if (onErrorCallback) {
                    onErrorCallback(event.error);
                }
            }
        };

        this.recognition.onend = () => {
            console.log('🎤 Recognition ended');
            this.isListening = false;
        };

        return true;
    }

    // Toggle speech enabled/disabled
    toggleSpeech() {
        this.isSpeechEnabled = !this.isSpeechEnabled;
        console.log(`🔊 Speech ${this.isSpeechEnabled ? 'enabled' : 'disabled'}`);
        return this.isSpeechEnabled;
    }

    // Start conversation mode
    startConversationMode() {
        this.conversationMode = true;
        this.conversationActive = true;
        console.log('💬 Conversation mode started');
    }

    // Stop conversation mode
    stopConversationMode() {
        this.conversationMode = false;
        this.conversationActive = false;
        this.stopListening();
        
        // Stop any ongoing speech
        if (this.synthesis.speaking) {
            this.synthesis.cancel();
            this.isAssistantSpeaking = false;
            document.body.classList.remove('speaking');
        }
        
        console.log('💬 Conversation mode stopped');
    }

    // Check if conversation mode is active
    isConversationModeActive() {
        return this.conversationMode && this.conversationActive;
    }

    // Get available voices
    getAvailableVoices() {
        return this.synthesis.getVoices();
    }

    // Check if speech recognition is supported
    isSpeechRecognitionSupported() {
        return this.recognition !== null;
    }

    // Check if speech synthesis is supported
    isSpeechSynthesisSupported() {
        return 'speechSynthesis' in window;
    }
}

// Export for use in other modules
window.SpeechManager = SpeechManager;
