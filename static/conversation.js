// Conversation Module - Handles conversation logic and backend communication
class ConversationManager {
    constructor(visionManager, speechManager) {
        this.visionManager = visionManager;
        this.speechManager = speechManager;
        this.serverUrl = 'http://localhost:5000';
        this.sessionId = null;
        this.conversationHistory = [];
        this.conversationContainer = null;
        
        this.initialize();
    }

    initialize() {
        this.conversationContainer = document.getElementById('conversation');
        this.createSession();
    }

    // Create a new conversation session
    async createSession() {
        try {
            const response = await fetch(`${this.serverUrl}/api/v1/conversation/sessions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.sessionId = data.session_id;
                console.log(`💬 Session created: ${this.sessionId}`);
            } else {
                console.error('Failed to create session');
                this.sessionId = `mock_session_${Date.now()}`;
            }
        } catch (error) {
            console.error('Error creating session:', error);
            this.sessionId = `demo_session_${Date.now()}`;
            // Don't auto-start conversation mode or add welcome message
        }
    }

    // Add message to conversation display
    addMessage(message, sender) {
        if (!this.conversationContainer) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const timestamp = new Date().toLocaleTimeString();
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${message}</div>
                <div class="message-time">${timestamp}</div>
            </div>
        `;
        
        this.conversationContainer.appendChild(messageDiv);
        this.conversationContainer.scrollTop = this.conversationContainer.scrollHeight;
        
        // Add to history
        this.conversationHistory.push({ 
            role: sender === 'user' ? 'user' : 'assistant', 
            content: message,
            timestamp: new Date().toISOString()
        });

        // Announce to screen readers
        if (sender === 'assistant') {
            this.announceToScreenReader(message);
        }
    }

    // Process conversation message with vision awareness
    async processMessage(message, retryCount = 0) {
        const maxRetries = 2;
        
        try {
            console.log('💬 Processing conversation message:', message, `(attempt ${retryCount + 1}/${maxRetries + 1})`);

            // Validate input
            if (!message || typeof message !== 'string' || message.trim().length === 0) {
                throw new Error('Invalid message: message must be a non-empty string');
            }

            if (message.length > 1000) {
                console.warn('💬 Message is very long:', message.length, 'characters');
                message = message.substring(0, 1000) + '...';
            }

            // Add user message to conversation
            this.addMessage(message, 'user');

            // 🎯 VISION-AWARE CONVERSATION FLOW
            console.log('🧠 Starting vision-aware conversation flow...');
            
            // Step 1: Capture current image from video feed with timeout
            let imageData = null;
            try {
                const capturePromise = this.visionManager.captureImageFromVideo();
                const timeoutPromise = new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Image capture timeout')), 5000)
                );
                imageData = await Promise.race([capturePromise, timeoutPromise]);
            } catch (captureError) {
                console.warn('🧠 Image capture failed:', captureError.message);
                // Continue without visual context
            }
            
            // Step 2: Get vision caption if image is available
            let caption = null;
            if (imageData) {
                try {
                    const captionPromise = this.visionManager.fetchCaption(imageData);
                    const timeoutPromise = new Promise((_, reject) => 
                        setTimeout(() => reject(new Error('Caption generation timeout')), 10000)
                    );
                    caption = await Promise.race([captionPromise, timeoutPromise]);
                } catch (captionError) {
                    console.warn('🧠 Caption generation failed:', captionError.message);
                    // Continue without visual context
                }
            }
            
            // Step 3: Compose vision-grounded prompt
            let finalPrompt = message;
            if (caption && caption.trim().length > 0) {
                finalPrompt = `Current visual context: ${caption}. User question: ${message}`;
                console.log('🧠 Vision-grounded prompt:', finalPrompt.substring(0, 100) + '...');
            } else {
                console.log('🧠 No visual context available, using text-only prompt');
            }

            // Step 4: Send to conversation backend with retry logic
            const response = await this.sendToBackendWithRetry(finalPrompt, caption, retryCount);

            if (response.success) {
                const assistantResponse = response.response;

                // Validate response
                if (!assistantResponse || typeof assistantResponse !== 'string') {
                    throw new Error('Invalid response from backend');
                }

                // Add assistant response to conversation
                this.addMessage(assistantResponse, 'assistant');

                // Speak the response and then continue listening
                this.speechManager.speakTextAndThenListen(assistantResponse, () => {
                    if (this.speechManager.isConversationModeActive()) {
                        setTimeout(() => {
                            this.speechManager.startListening();
                        }, 500);
                    }
                });

            } else {
                throw new Error(response.error || 'Unknown error occurred');
            }

        } catch (error) {
            console.error('💬 Error in conversation:', error);
            
            // Retry logic for network errors
            if (retryCount < maxRetries && this.isRetryableError(error)) {
                console.log(`💬 Retrying conversation (${retryCount + 1}/${maxRetries})...`);
                await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1))); // Exponential backoff
                return this.processMessage(message, retryCount + 1);
            }
            
            // Fallback response based on error type
            const fallbackResponse = this.getFallbackResponse(error, retryCount >= maxRetries);
            this.addMessage(fallbackResponse, 'assistant');
            
            this.speechManager.speakTextAndThenListen(fallbackResponse, () => {
                if (this.speechManager.isConversationModeActive()) {
                    setTimeout(() => {
                        this.speechManager.startListening();
                    }, 1000);
                }
            });
        }
    }

    // Send regular message (for text input)
    async sendMessage(message, imageData = null) {
        try {
            console.log('💬 Sending message:', message);

            // Add user message to conversation
            this.addMessage(message, 'user');

            // Prepare request data
            const requestData = {
                message: message,
                session_id: this.sessionId
            };

            // Add image if provided
            if (imageData) {
                requestData.image = imageData;
            }

            const response = await fetch(`${this.serverUrl}/api/v1/conversation/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                const assistantResponse = data.response;
                this.addMessage(assistantResponse, 'assistant');

                // Speak response if speech is enabled
                if (this.speechManager.isSpeechEnabled) {
                    this.speechManager.speakText(assistantResponse);
                }

                return assistantResponse;
            } else {
                throw new Error(data.error || 'Unknown error occurred');
            }

        } catch (error) {
            console.error('Error sending message:', error);
            
            // Fallback response
            const fallbackResponse = "I'm having trouble connecting right now. Please try again.";
            this.addMessage(fallbackResponse, 'assistant');
            
            if (this.speechManager.isSpeechEnabled) {
                this.speechManager.speakText(fallbackResponse);
            }
            
            return fallbackResponse;
        }
    }

    // Start vision-aware conversation mode
    startConversationMode() {
        console.log('💬 Starting vision-aware conversation mode...');
        
        // Check if speech recognition is available and not in error state
        const speechAvailable = this.speechManager && 
                               this.speechManager.browserSupported && 
                               !this.speechManager.isInErrorBackoff &&
                               this.speechManager.consecutiveNetworkErrors < 3;
        
        console.log('🔍 Speech availability check:', {
            speechManager: !!this.speechManager,
            browserSupported: this.speechManager?.browserSupported,
            isInErrorBackoff: this.speechManager?.isInErrorBackoff,
            consecutiveNetworkErrors: this.speechManager?.consecutiveNetworkErrors
        });
        
        if (speechAvailable) {
            console.log('🎤 Attempting speech-based conversation mode...');
            
            // Start speech manager conversation mode
            this.speechManager.startConversationMode();
            
            // Set up speech recognition for conversation
            const success = this.speechManager.setupConversationRecognition(
                (transcript) => this.processMessage(transcript),
                (error) => {
                    console.warn('💬 Speech recognition failed, switching to text mode...');
                    this.handleConversationError(error);
                    // Immediately switch to text mode on speech error
                    setTimeout(() => this.startTextConversationMode(), 1000);
                }
            );
            
            if (success) {
                // Try to start listening with timeout
                const listeningStarted = this.speechManager.startListening();
                
                if (listeningStarted) {
                    // Give speech recognition 3 seconds to work, then fallback to text
                    setTimeout(() => {
                        if (this.speechManager.consecutiveNetworkErrors >= 3) {
                            console.log('💬 Speech recognition failing, switching to text mode...');
                            this.speechManager.stopConversationMode();
                            this.startTextConversationMode();
                        }
                    }, 3000);
                    
                    // Welcome message for voice mode
                    const welcomeMessage = "I'm ready to chat! I can see through your camera and answer questions about what I observe. Speak or I'll switch to text input shortly.";
                    this.addMessage(welcomeMessage, 'assistant');
                    this.speechManager.speakTextAndThenListen(welcomeMessage, () => {
                        if (this.speechManager.isConversationModeActive()) {
                            this.speechManager.startListening();
                        }
                    });
                    return; // Successfully started voice mode
                }
            }
        }
        
        // Fallback to text input mode immediately
        console.log('💬 Speech unavailable or failing, starting text-based conversation mode...');
        this.startTextConversationMode();
    }

    // Start text-based conversation mode when speech is unavailable
    startTextConversationMode() {
        console.log('💬 Starting text-based conversation mode...');
        
        // Show text input fallback interface
        const textInputFallback = document.getElementById('textInputFallback');
        const conversationTextInput = document.getElementById('conversationTextInput');
        const sendTextBtn = document.getElementById('sendTextBtn');
        const conversationHistoryText = document.getElementById('conversationHistoryText');
        
        if (textInputFallback) {
            textInputFallback.style.display = 'flex';
            
            // Focus on text input
            if (conversationTextInput) {
                conversationTextInput.focus();
            }
            
            // Add welcome message to text conversation
            const welcomeMessage = "I'm ready to chat! I can see through your camera and answer questions about what I observe. Type your question below:";
            this.addTextMessage(welcomeMessage, 'assistant');
            
            // Set up text input event listeners
            this.setupTextInputListeners();
        } else {
            console.error('❌ Text input fallback interface not found!');
        }
    }
    
    // Set up text input event listeners
    setupTextInputListeners() {
        const conversationTextInput = document.getElementById('conversationTextInput');
        const sendTextBtn = document.getElementById('sendTextBtn');
        const closeTextModalBtn = document.getElementById('closeTextModalBtn');
        
        if (conversationTextInput && sendTextBtn) {
            // Send message on button click
            sendTextBtn.addEventListener('click', () => {
                this.sendTextMessage();
            });
            
            // Send message on Enter key
            conversationTextInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendTextMessage();
                }
            });
            
            // Close on Escape key
            conversationTextInput.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    this.stopConversationMode();
                }
            });
        }
        
        // Close button event listener
        if (closeTextModalBtn) {
            closeTextModalBtn.addEventListener('click', () => {
                console.log('💬 Closing text conversation mode...');
                this.stopConversationMode();
            });
        }
        
        // Click outside modal to close
        const textInputFallback = document.getElementById('textInputFallback');
        if (textInputFallback) {
            textInputFallback.addEventListener('click', (e) => {
                if (e.target === textInputFallback) {
                    this.stopConversationMode();
                }
            });
        }
    }
    
    // Send text message
    sendTextMessage() {
        const conversationTextInput = document.getElementById('conversationTextInput');
        const sendTextBtn = document.getElementById('sendTextBtn');
        
        if (conversationTextInput) {
            const message = conversationTextInput.value.trim();
            
            if (message.length > 0) {
                // Disable input while processing
                conversationTextInput.disabled = true;
                sendTextBtn.disabled = true;
                
                // Add user message to text conversation
                this.addTextMessage(message, 'user');
                
                // Clear input
                conversationTextInput.value = '';
                
                // Process the message (same vision-aware flow)
                this.processMessage(message).finally(() => {
                    // Re-enable input
                    conversationTextInput.disabled = false;
                    sendTextBtn.disabled = false;
                    conversationTextInput.focus();
                });
            }
        }
    }
    
    // Add message to text conversation history
    addTextMessage(message, sender) {
        const conversationHistoryText = document.getElementById('conversationHistoryText');
        
        if (conversationHistoryText) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `conversation-message ${sender}`;
            messageDiv.textContent = message;
            
            conversationHistoryText.appendChild(messageDiv);
            
            // Scroll to bottom
            conversationHistoryText.scrollTop = conversationHistoryText.scrollHeight;
        }
    }

    // Stop conversation mode
    stopConversationMode() {
        console.log('💬 Stopping conversation mode...');
        
        // Stop speech manager
        if (this.speechManager) {
            this.speechManager.stopConversationMode();
        }
        
        // Hide text input fallback interface
        const textInputFallback = document.getElementById('textInputFallback');
        if (textInputFallback) {
            textInputFallback.style.display = 'none';
        }
        
        // Clear conversation history
        this.clearConversation();
        
        // Clear text conversation history
        const conversationHistoryText = document.getElementById('conversationHistoryText');
        if (conversationHistoryText) {
            conversationHistoryText.innerHTML = '';
        }
        
        // Update UI state
        this.updateConversationModeUI(false);
    }

    // Handle conversation errors
    handleConversationError(errorType) {
        let shouldContinue = true;
        let errorMessage = '';

        switch(errorType) {
            case 'network':
                errorMessage = "I'm having network issues. Let me try to continue listening...";
                break;
            case 'not-allowed':
            case 'permission-denied':
                errorMessage = "I need microphone access to continue our conversation. Please allow microphone access and try again.";
                shouldContinue = false;
                break;
            case 'no-speech':
                // Don't announce no-speech errors, just continue
                shouldContinue = true;
                break;
            case 'audio-capture':
                errorMessage = "I can't access your microphone. Please check your microphone connection.";
                shouldContinue = false;
                break;
            default:
                errorMessage = "I encountered an issue with speech recognition. Let me try to continue...";
        }

        if (errorMessage) {
            console.log('💬 Conversation error:', errorMessage);
            if (errorType !== 'no-speech') {
                this.addMessage(errorMessage, 'assistant');
            }
        }

        if (!shouldContinue) {
            this.stopConversationMode();
        } else {
            // Try to restart listening after a delay
            setTimeout(() => {
                if (this.speechManager.isConversationModeActive()) {
                    this.speechManager.startListening();
                }
            }, 1000);
        }
    }

    // Announce message to screen readers
    announceToScreenReader(message) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.style.position = 'absolute';
        announcement.style.left = '-10000px';
        announcement.style.width = '1px';
        announcement.style.height = '1px';
        announcement.style.overflow = 'hidden';
        announcement.textContent = `Assistant: ${message}`;
        
        document.body.appendChild(announcement);
        
        // Remove after announcement
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }

    // Clear conversation history
    clearConversation() {
        this.conversationHistory = [];
        if (this.conversationContainer) {
            this.conversationContainer.innerHTML = '';
        }
        console.log('💬 Conversation history cleared');
    }

    // Get conversation history
    getConversationHistory() {
        return this.conversationHistory;
    }

    // Send request to backend with retry logic
    async sendToBackendWithRetry(message, caption, retryCount) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout
        
        try {
            const response = await fetch(`${this.serverUrl}/api/v1/conversation/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId,
                    has_visual_context: !!caption,
                    retry_count: retryCount
                }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Validate response structure
            if (typeof data !== 'object' || data === null) {
                throw new Error('Invalid response format from backend');
            }
            
            return data;
            
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new Error('Request timeout - backend took too long to respond');
            }
            
            throw error;
        }
    }
    
    // Check if error is retryable
    isRetryableError(error) {
        const retryableErrors = [
            'NetworkError',
            'TypeError', // Often network-related
            'Request timeout',
            'HTTP 500',
            'HTTP 502',
            'HTTP 503',
            'HTTP 504'
        ];
        
        return retryableErrors.some(retryableError => 
            error.message.includes(retryableError) || 
            error.name === retryableError
        );
    }
    
    // Get appropriate fallback response based on error
    getFallbackResponse(error, allRetriesExhausted) {
        if (error.message.includes('timeout')) {
            return allRetriesExhausted 
                ? "I'm taking longer than usual to respond. The backend might be overloaded. Please try again in a moment."
                : "That's taking a bit longer than expected. Let me try again...";
        }
        
        if (error.message.includes('NetworkError') || error.message.includes('fetch')) {
            return allRetriesExhausted
                ? "I'm having trouble connecting to my backend services. Please check your internet connection and try again."
                : "I'm having connection issues. Let me try that again...";
        }
        
        if (error.message.includes('HTTP 5')) {
            return allRetriesExhausted
                ? "My backend services are experiencing issues. Please try again in a few minutes."
                : "There seems to be a server issue. Let me retry that...";
        }
        
        if (error.message.includes('Invalid message')) {
            return "I didn't quite understand that. Could you please rephrase your question?";
        }
        
        if (error.message.includes('Invalid response')) {
            return allRetriesExhausted
                ? "I received an unexpected response. Please try asking your question again."
                : "Something went wrong with my response. Let me try again...";
        }
        
        // Generic fallback
        return allRetriesExhausted
            ? "I'm having trouble processing that right now. Please try again, or ask a different question."
            : "Something went wrong. Let me try that again...";
    }

    // Check if conversation mode is active
    isConversationModeActive() {
        return this.speechManager.isConversationModeActive();
    }
    
    // Health check for backend connectivity
    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.serverUrl}/api/v1/model/info`, {
                method: 'GET',
                signal: AbortSignal.timeout(5000)
            });
            
            return response.ok;
        } catch (error) {
            console.warn('💬 Backend health check failed:', error.message);
            return false;
        }
    }
    
    // Get system status for debugging
    getSystemStatus() {
        return {
            sessionId: this.sessionId,
            conversationHistory: this.conversationHistory.length,
            cameraActive: this.visionManager.isCameraActive(),
            speechEnabled: this.speechManager.isSpeechEnabled,
            conversationMode: this.isConversationModeActive(),
            isListening: this.speechManager.isListening,
            isAssistantSpeaking: this.speechManager.isAssistantSpeaking
        };
    }
}

// Export for use in other modules
window.ConversationManager = ConversationManager;
