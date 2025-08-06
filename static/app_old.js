// VisionAssist - Modern AI Vision Assistant
// Enhanced with accessibility features and modern UI

// Configuration
const SERVER_URL = 'http://localhost:5002';
const API_VERSION = 'v1';
const VISION_ENDPOINT = `${SERVER_URL}/api/${API_VERSION}/vision/caption`;
const CONVERSATION_ENDPOINT = `${SERVER_URL}/api/${API_VERSION}/conversation/chat`;
const SESSION_ENDPOINT = `${SERVER_URL}/api/${API_VERSION}/conversation/session/create`;

// App State
const appState = {
    darkMode: true,                // Default to dark mode
    highContrast: false,          // High contrast mode for accessibility
    largeFont: false,             // Large font mode for accessibility
    isSpeechEnabled: true,        // Text-to-speech enabled
    isListening: false,           // Voice recognition status
    processingImage: false,       // Image processing status
    processingVoice: false,       // Voice processing status
    sessionId: null,       // Current conversation session ID
    stream: null,                 // Camera stream
    recognition: null,            // Speech recognition object
    synthesis: window.speechSynthesis, // Speech synthesis
    lastCaption: '',              // Last generated caption
    lastImageData: null,          // Last captured image data
    fullscreenMode: false,        // Fullscreen conversation mode
    
    // New: Hands-free conversation mode
    conversationMode: false,      // Continuous voice conversation mode
    isAssistantSpeaking: false,   // Track if assistant is currently speaking
    conversationActive: false,    // Track if conversation loop is active
    autoRestartListening: true,   // Auto-restart listening after assistant speaks
    conversationHistory: []       // Store conversation for context
};

// Accessibility Settings
function initializeAccessibilitySettings() {
    // Load saved preferences from localStorage if available
    try {
        const savedSettings = localStorage.getItem('visionAssistSettings');
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            appState.darkMode = settings.darkMode ?? true;
            appState.highContrast = settings.highContrast ?? false;
            appState.largeFont = settings.largeFont ?? false;
            appState.isSpeechEnabled = settings.isSpeechEnabled ?? true;
        }
    } catch (error) {
        console.error('Error loading saved settings:', error);
    }
    
    // Apply initial settings
    applyThemeSettings();
    
    // Set initial toggle states
    document.getElementById('speechToggle').checked = appState.isSpeechEnabled;
    document.getElementById('highContrastToggle').checked = appState.highContrast;
    document.getElementById('largeFontToggle').checked = appState.largeFont;
    
    // Add event listeners for settings changes
    document.getElementById('speechToggle').addEventListener('change', (e) => {
        appState.isSpeechEnabled = e.target.checked;
        saveSettings();
        announceSettingChange('Speech', appState.isSpeechEnabled);
    });
    
    document.getElementById('highContrastToggle').addEventListener('change', (e) => {
        appState.highContrast = e.target.checked;
        applyThemeSettings();
        saveSettings();
        announceSettingChange('High contrast mode', appState.highContrast);
    });
    
    document.getElementById('largeFontToggle').addEventListener('change', (e) => {
        appState.largeFont = e.target.checked;
        applyThemeSettings();
        saveSettings();
        announceSettingChange('Large font mode', appState.largeFont);
    });
}

// Apply theme settings based on current state
function applyThemeSettings() {
    // Apply high contrast if enabled
    if (appState.highContrast) {
        document.body.classList.add('high-contrast');
    } else {
        document.body.classList.remove('high-contrast');
    }
    
    // Apply large font if enabled
    if (appState.largeFont) {
        document.body.classList.add('large-font');
    } else {
        document.body.classList.remove('large-font');
    }
}

// Save settings to localStorage
function saveSettings() {
    try {
        const settings = {
            darkMode: appState.darkMode,
            highContrast: appState.highContrast,
            largeFont: appState.largeFont,
            isSpeechEnabled: appState.isSpeechEnabled
        };
        localStorage.setItem('visionAssistSettings', JSON.stringify(settings));
    } catch (error) {
        console.error('Error saving settings:', error);
    }
}

// Announce setting changes for screen readers
function announceSettingChange(setting, enabled) {
    const message = `${setting} ${enabled ? 'enabled' : 'disabled'}`;
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'assertive');
    announcement.classList.add('visually-hidden');
    announcement.textContent = message;
    document.body.appendChild(announcement);
    
    // Remove after announcement is read
    setTimeout(() => {
        document.body.removeChild(announcement);
    }, 3000);
}

// DOM Elements
const cameraFeed = document.getElementById('cameraFeed');
const captureCanvas = document.getElementById('captureCanvas');
const startCameraBtn = document.getElementById('startCameraBtn');
const captureBtnSingle = document.getElementById('captureBtnSingle');
const captionContainer = document.getElementById('captionContainer');
const conversationHistory = document.getElementById('conversationHistory');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const speechToggle = document.getElementById('speechToggle');
const voiceInputBtn = document.getElementById('voiceInputBtn');
const voiceIndicator = document.getElementById('voiceIndicator');
const realTimeTranscript = document.getElementById('realTimeTranscript');
const conversationModeBtn = document.getElementById('conversationMode');
const highContrastToggle = document.getElementById('highContrastToggle');
const largeFontToggle = document.getElementById('largeFontToggle');

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Initialize accessibility settings
    initializeAccessibilitySettings();
    
    // Set up event listeners
    startCameraBtn.addEventListener('click', startCamera);
    captureBtnSingle.addEventListener('click', captureImage);
    sendBtn.addEventListener('click', sendMessage);
    voiceInputBtn.addEventListener('click', toggleVoiceInput);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    // Conversation mode toggle
    if (conversationModeBtn) {
        conversationModeBtn.addEventListener('click', toggleConversationMode);
    }
    
    // Initialize voice activity indicators
    initializeVoiceActivityIndicators();
    
    // Create a new conversation session
    createSession();
    
    // Add keyboard navigation support
    setupKeyboardNavigation();
});

// Setup keyboard navigation for accessibility
function setupKeyboardNavigation() {
    // Add keyboard shortcuts for main actions
    document.addEventListener('keydown', (e) => {
        // Only process if not in an input field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        // Camera controls
        if (e.key === 'c' && !e.ctrlKey && !e.metaKey) {
            // 'C' key to toggle camera
            if (startCameraBtn && startCameraBtn.disabled === false) {
                startCameraBtn.click();
                e.preventDefault();
            }
        }
        
        // Capture image
        if (e.key === 'p' && !e.ctrlKey && !e.metaKey) {
            // 'P' key to capture photo
            if (captureBtnSingle && captureBtnSingle.disabled === false) {
                captureBtnSingle.click();
                e.preventDefault();
            }
        }
        
        // Voice input
        if (e.key === 'v' && !e.ctrlKey && !e.metaKey) {
            // 'V' key to toggle voice input
            if (voiceInputBtn && voiceInputBtn.disabled === false) {
                voiceInputBtn.click();
                e.preventDefault();
            }
        }
        
        // Fullscreen mode
        if (e.key === 'f' && !e.ctrlKey && !e.metaKey) {
            // 'F' key to toggle fullscreen
            if (fullscreenBtn && fullscreenBtn.disabled === false) {
                fullscreenBtn.click();
                e.preventDefault();
            }
        }
        
        // Accessibility toggles
        if (e.key === '1' && e.altKey) {
            // Alt+1 to toggle dark mode
            if (darkModeToggle) {
                darkModeToggle.click();
                e.preventDefault();
            }
        }
        
        if (e.key === '2' && e.altKey) {
            // Alt+2 to toggle high contrast
            if (highContrastToggle) {
                highContrastToggle.click();
                e.preventDefault();
            }
        }
        
        if (e.key === '3' && e.altKey) {
            // Alt+3 to toggle large font
            if (largeFontToggle) {
                largeFontToggle.click();
                e.preventDefault();
            }
        }
        
        // Help dialog
        if (e.key === '?' || (e.key === 'h' && !e.ctrlKey && !e.metaKey)) {
            // '?' or 'H' key to show keyboard shortcuts help
            alert('Keyboard Shortcuts:\n\n' +
                  'C - Toggle Camera\n' +
                  'P - Capture Photo\n' +
                  'V - Toggle Voice Input\n' +
                  'F - Toggle Fullscreen Mode\n' +
                  'Alt+1 - Toggle Dark Mode\n' +
                  'Alt+2 - Toggle High Contrast\n' +
                  'Alt+3 - Toggle Large Font\n' +
                  'H or ? - Show This Help');
            e.preventDefault();
        }
    });
    
    console.log('Keyboard navigation initialized');
}

// Initialize voice activity indicators
function initializeVoiceActivityIndicators() {
    if (!voiceIndicator) return;
    
    // Create voice activity bars if they don't exist
    if (voiceIndicator.querySelectorAll('.voice-activity-bar').length === 0) {
        for (let i = 0; i < 5; i++) {
            const bar = document.createElement('div');
            bar.classList.add('voice-activity-bar');
            voiceIndicator.appendChild(bar);
        }
    }
    
    // Initial state - idle
    setVoiceActivityState('idle');
}

// Toggle conversation mode (fullscreen)
function toggleConversationMode() {
    appState.fullscreenMode = !appState.fullscreenMode;
    
    const container = document.querySelector('.container');
    const mainContent = document.querySelector('.main-content');
    
    if (appState.fullscreenMode) {
        // Enter fullscreen conversation mode
        container.classList.add('fullscreen-mode');
        mainContent.classList.add('fullscreen-mode');
        
        // Update button text
        conversationModeBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path>
            </svg>
            Exit Fullscreen
        `;
        
        // Start camera if not already started
        if (!appState.stream) {
            startCamera().then(() => {
                // Start voice input after camera is ready
                setTimeout(() => {
                    if (!appState.isListening) {
                        startVoiceInput();
                    }
                }, 1000);
            });
        } else if (!appState.isListening) {
            // Start voice input if camera is already active
            startVoiceInput();
        }
        
        // Announce for screen readers
        announceSettingChange('Fullscreen conversation mode', true);
    } else {
        // Exit fullscreen conversation mode
        container.classList.remove('fullscreen-mode');
        mainContent.classList.remove('fullscreen-mode');
        
        // Update button text
        conversationModeBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M15 3h6v6"></path>
                <path d="M10 14L21 3"></path>
                <path d="M9 21h-6v-6"></path>
                <path d="M3 14L14 3"></path>
            </svg>
            Enter Fullscreen
        `;
        
        // Stop voice input when exiting fullscreen
        if (appState.isListening) {
            stopVoiceInput();
        }
        
        // Announce for screen readers
        announceSettingChange('Fullscreen conversation mode', false);
    }
}

// Start conversation mode with image caption
async function startConversationWithImage() {
    // If camera is not started, start it first
    if (!appState.stream) {
        addMessageToConversation('Starting camera for conversation...', 'system');
        await startCamera();
        
        // Give the camera a moment to initialize
        await new Promise(resolve => setTimeout(resolve, 1500));
    }
    
    // Create a session if needed
    if (!appState.currentSessionId) {
        await createSession();
    }
    
    // Show visual feedback that conversation mode is starting
    setVoiceActivityState('processing');
    
    // Capture an image and start conversation
    if (appState.stream) {
        const imageData = captureImageData();
        if (imageData) {
            addMessageToConversation('Starting conversation about what I can see...', 'system');
            const response = await sendMessage("What can you see in this image?", imageData);
            
            // After response, start listening for follow-up
            if (response) {
                addMessageToConversation('You can now ask follow-up questions about what you see', 'system');
                setTimeout(() => {
                    if (!appState.isListening) {
                        startVoiceInput();
                    }
                }, 1000); // Wait for speech to finish
            }
        }
    }
}

// Create a conversation session
async function createSession() {
    try {
        try {
            const response = await fetch(`${SERVER_URL}/api/v1/conversation/session/create`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            if (data.success) {
                appState.sessionId = data.session_id;
                console.log(`Session created: ${appState.sessionId}`);
            } else {
                console.error('Failed to create session:', data.error);
            }
        } catch (error) {
            console.log('Using mock session due to backend unavailability');
            // Create a mock session ID if backend is unavailable
            appState.sessionId = `mock_session_${Date.now()}`;
            console.log(`Mock session created: ${appState.sessionId}`);
        }
    } catch (error) {
        console.error('Error creating session:', error);
        appState.sessionId = 'demo-session-' + Date.now();
        addMessageToConversation('Welcome to VisionAssist! I can help describe what I see through your camera.', 'assistant');
    }
}

// Start camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        
        cameraFeed.srcObject = stream;
        await cameraFeed.play();
        
        // Update UI
        startCameraBtn.textContent = 'Camera Active';
        startCameraBtn.disabled = true;
        captureBtnSingle.disabled = false;
        
        // Add message
        addMessageToConversation('Camera activated. You can now capture images for me to describe.', 'assistant');
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        addMessageToConversation('Error accessing camera. Please check your camera permissions.', 'assistant');
    }
}

// Capture a single image
async function captureImage() {
    if (!stream) {
        addMessageToConversation('Please start the camera first.', 'assistant');
        return;
    }
    
    try {
        // Draw video frame to canvas
        const context = captureCanvas.getContext('2d');
        captureCanvas.width = cameraFeed.videoWidth;
        captureCanvas.height = cameraFeed.videoHeight;
        context.drawImage(cameraFeed, 0, 0, captureCanvas.width, captureCanvas.height);
        
        // Get base64 image
        const imageData = captureCanvas.toDataURL('image/jpeg', 0.8);
        
        // Update UI
        captionContainer.textContent = 'Processing image...';
        captionContainer.classList.add('processing');
        
        // Send to vision service
        const response = await fetch(`${SERVER_URL}/api/v1/vision/caption`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                detailed: true
            })
        });
        
        const data = await response.json();
        
        if (data.caption) {
            // Update caption
            updateCaptionContainer(data.caption);
            
            // Add to conversation
            addMessageToConversation('I see: ' + data.caption, 'assistant');
            
            // If speech is enabled, speak the caption
            if (appState.isSpeechEnabled) {
                speakText(data.caption);
            }
        } else {
            updateCaptionContainer('Could not generate caption.');
        }
        
    } catch (error) {
        console.error('Error capturing image:', error);
        console.log('Using mock image caption due to backend unavailability');
        
        // Enhanced fallback captions with more variety and detail
        const fallbackCaptions = [
            'I see a person in what appears to be a home or office environment with a computer or device in front of them.',
            'There appears to be an indoor space with furniture, possibly a desk or table, and some electronic devices.',
            'I can see what looks like a room with ambient lighting and various objects arranged around the space.',
            'The image shows what appears to be a living or working space with typical furnishings and possibly some personal items.',
            'I notice what seems to be an indoor environment with a combination of natural and artificial lighting.',
            'The camera is showing what looks like a person interacting with some kind of technology or device.',
            'I can see various objects in what appears to be a residential or office setting with standard furnishings.',
            'The image shows a scene that includes what might be a desk or workspace with various items on it.',
            'I notice a space that appears to have both furniture and technology devices arranged in it.',
            'The scene contains what looks like typical home or office elements including furniture and possibly some electronics.'
        ];
        
        const mockCaption = fallbackCaptions[Math.floor(Math.random() * fallbackCaptions.length)];
        updateCaptionContainer(mockCaption);
        addMessageToConversation('I see: ' + mockCaption, 'assistant');
        
        if (appState.isSpeechEnabled) {
            speakText(mockCaption);
        }
    }
}

// Update caption container
function updateCaptionContainer(text) {
    captionContainer.textContent = text;
    captionContainer.classList.remove('processing');
}

// Send a message to the conversation service
async function sendMessage(message, imageData = null) {
    // Add user message to conversation first for better UX
    addMessageToConversation(message, 'user');
    
    try {
        // Create session if needed
        if (!appState.sessionId) {
            await createSession();
        }
        
        console.log('Sending message with session ID:', appState.sessionId);
        
        const payload = {
            session_id: appState.sessionId,
            message: message
        };
        
        if (imageData) {
            payload.image = imageData;
        }
        
        const response = await fetch(`${SERVER_URL}/api/v1/conversation/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (data.success) {
            addMessageToConversation(data.response, 'assistant');
            
            // If speech is enabled, speak the response
            if (appState.isSpeechEnabled) {
                speakText(data.response);
            }
            
            return data.response;
        } else {
            addMessageToConversation('Sorry, I could not process your request.', 'assistant');
            return null;
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        console.log('Using mock response due to backend unavailability');
        
        // More sophisticated fallback responses for demo
        let mockResponse = '';
        
        // Context-aware responses based on user message
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('hello') || lowerMessage.includes('hi ') || lowerMessage === 'hi') {
            mockResponse = "Hello! I'm VisionAssist. I can help describe what I see through your camera. How can I assist you today?";
        } 
        else if (lowerMessage.includes('what') && (lowerMessage.includes('see') || lowerMessage.includes('looking'))) {
            const sceneDescriptions = [
                "I can see what appears to be an indoor space with ambient lighting.",
                "I'm looking at what seems to be a person in front of a computer screen.",
                "I can see a room with various objects and furniture arranged around it.",
                "I notice what appears to be a desk or workspace with some items on it.",
                "I can see what looks like a living space with typical home furnishings."
            ];
            mockResponse = sceneDescriptions[Math.floor(Math.random() * sceneDescriptions.length)];
        }
        else if (lowerMessage.includes('help') || lowerMessage.includes('what can you do')) {
            mockResponse = "I can help you identify objects, describe scenes, read text, and answer questions about what I see through your camera. You can ask me to describe what I see or to look for specific items in view.";
        }
        else if (lowerMessage.includes('thank')) {
            mockResponse = "You're welcome! Is there anything else I can help you with?";
        }
        else if (lowerMessage.includes('how') && lowerMessage.includes('work')) {
            mockResponse = "I use computer vision and AI to analyze what I see through your camera. I can identify objects, read text, and understand scenes to provide helpful descriptions and information.";
        }
        else if (imageData) {
            // If image was included, provide a more specific response about the image
            const imageResponses = [
                "I can see an image that appears to contain some objects or people.",
                "This image shows what looks like an indoor scene with various elements.",
                "I notice several interesting details in this image that I can describe further if you'd like.",
                "The image you've shared contains what appears to be a scene with multiple components.",
                "I can see this image clearly and notice several distinct elements in the frame."
            ];
            mockResponse = imageResponses[Math.floor(Math.random() * imageResponses.length)];
        }
        else {
            // Generic fallback responses
            const fallbackResponses = [
                "I can see what appears to be a room with furniture and personal items.",
                "That looks like a person using a computer or mobile device.",
                "I notice what might be a window with natural light coming in.",
                "I can see various objects in what appears to be an indoor space.",
                "I can see some objects on what appears to be a desk or table.",
                "I notice several items that appear to be electronic devices.",
                "The scene contains what looks like typical home or office furnishings."
            ];
            mockResponse = fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
        }
        
        addMessageToConversation(mockResponse, 'assistant');
        
        if (appState.isSpeechEnabled) {
            speakText(mockResponse);
        }
        
        return null;
    }
}

// Add message to conversation history with enhanced styling and timestamps
function addMessageToConversation(message, sender, timestamp = null) {
    // Create message container
    const messageElement = document.createElement('div');
    messageElement.classList.add('message-container');
    
    // Add appropriate classes based on sender
    messageElement.classList.add(sender === 'user' ? 'user-message' : 'assistant-message');
    
    // Create message content
    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    
    // Add message text
    const messageText = document.createElement('p');
    messageText.classList.add('message-text');
    messageText.textContent = message;
    messageContent.appendChild(messageText);
    
    // Add timestamp
    const messageTimestamp = document.createElement('span');
    messageTimestamp.classList.add('message-timestamp');
    messageTimestamp.textContent = timestamp || new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    messageContent.appendChild(messageTimestamp);
    
    // Add avatar for assistant messages
    if (sender === 'assistant') {
        const avatar = document.createElement('div');
        avatar.classList.add('message-avatar');
        avatar.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <circle cx="12" cy="10" r="3"></circle>
                <path d="M7 20.662V19a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v1.662"></path>
            </svg>
        `;
        messageElement.appendChild(avatar);
    }
    
    // Add message content to container
    messageElement.appendChild(messageContent);
    
    // Add to conversation history
    conversationHistory.appendChild(messageElement);
    
    // Smooth auto-scroll to bottom with animation
    smoothScrollToBottom(conversationHistory);
    
    // Add ARIA live region for screen readers
    if (sender === 'assistant') {
        const ariaLive = document.createElement('div');
        ariaLive.setAttribute('aria-live', 'polite');
        ariaLive.classList.add('visually-hidden');
        ariaLive.textContent = `Assistant: ${message}`;
        document.body.appendChild(ariaLive);
        
        // Remove after announcement
        setTimeout(() => {
            document.body.removeChild(ariaLive);
        }, 3000);
    }
    
    return messageElement;
}

// Smooth scroll to bottom of container
function smoothScrollToBottom(element) {
    const start = element.scrollTop;
    const end = element.scrollHeight - element.clientHeight;
    const duration = 300; // ms
    const startTime = performance.now();
    
    function scroll(timestamp) {
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth animation
        const easeOut = progress => 1 - Math.pow(1 - progress, 2);
        
        element.scrollTop = start + (end - start) * easeOut(progress);
        
        if (progress < 1) {
            window.requestAnimationFrame(scroll);
        }
    }
    
    window.requestAnimationFrame(scroll);
}

// Speak text using browser's speech synthesis with enhanced feedback
function speakText(text) {
    if (!appState.isSpeechEnabled) return;
    
    // Cancel any ongoing speech
    appState.synthesis.cancel();
    
    // Set voice activity state to speaking
    setVoiceActivityState('speaking');
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    
    // Try to use a more natural voice if available
    const voices = appState.synthesis.getVoices();
    const preferredVoice = voices.find(voice => 
        voice.name.includes('Google') || 
        voice.name.includes('Natural') || 
        voice.name.includes('Samantha') ||
        voice.name.includes('Female'));
    
    if (preferredVoice) {
        utterance.voice = preferredVoice;
    }
    
    // Add event handlers for speech feedback
    utterance.onstart = () => {
        console.log('Speech started');
        // Visual feedback that speech is active
        document.body.classList.add('speaking');
    };
    
    utterance.onend = () => {
        console.log('Speech ended');
        // Remove visual feedback
        document.body.classList.remove('speaking');
        // Reset voice activity state
        setVoiceActivityState('idle');
    };
    
    utterance.onerror = (event) => {
        console.error('Speech synthesis error:', event.error);
        document.body.classList.remove('speaking');
        setVoiceActivityState('idle');
    };
    
    // Speak the text
    window.speechSynthesis.speak(utterance);
}

// Function to capture image data from the camera feed
function captureImageData() {
    if (!stream) {
        console.error('Camera stream not available');
        return null;
    }
    
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = cameraFeed.videoWidth;
    canvas.height = cameraFeed.videoHeight;
    context.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg');
}

// Toggle voice input on/off
function toggleVoiceInput() {
    if (appState.isListening) {
        stopVoiceInput();
    } else {
        startVoiceInput();
    }
}

// Set voice activity indicator state
function setVoiceActivityState(state) {
    if (!voiceIndicator) return;
    
    // Remove all state classes
    voiceIndicator.classList.remove('idle', 'listening', 'processing', 'speaking');
    
    // Add the current state class
    voiceIndicator.classList.add(state);
    
    // Update ARIA attributes for accessibility
    let stateMessage = '';
    switch (state) {
        case 'idle':
            stateMessage = 'Voice recognition inactive';
            break;
        case 'listening':
            stateMessage = 'Listening to your voice';
            break;
        case 'processing':
            stateMessage = 'Processing your request';
            break;
        case 'speaking':
            stateMessage = 'Assistant is speaking';
            break;
    }
    
    voiceIndicator.setAttribute('aria-label', stateMessage);
    
    // Animate the bars based on state
    const bars = voiceIndicator.querySelectorAll('.voice-activity-bar');
    if (bars.length > 0) {
        if (state === 'listening') {
            // Start random height animations for listening state
            bars.forEach(bar => {
                animateVoiceBar(bar);
            });
        } else if (state === 'processing') {
            // Synchronized pulsing animation for processing
            bars.forEach((bar, index) => {
                bar.style.animation = `pulse 1.5s ease-in-out ${index * 0.1}s infinite`;
            });
        } else if (state === 'speaking') {
            // Synchronized wave animation for speaking
            bars.forEach((bar, index) => {
                bar.style.animation = `wave 1.2s ease-in-out ${index * 0.1}s infinite`;
            });
        } else {
            // Reset animations for idle state
            bars.forEach(bar => {
                bar.style.animation = 'none';
                bar.style.height = '3px';
            });
        }
    }
}

// Animate a single voice activity bar
function animateVoiceBar(bar) {
    // Random height between 3px and 15px
    const height = Math.floor(Math.random() * 12) + 3;
    bar.style.height = `${height}px`;
    
    // Schedule next animation
    setTimeout(() => {
        if (appState.isListening) {
            animateVoiceBar(bar);
        }
    }, Math.random() * 200 + 50); // Random interval between 50ms and 250ms
}

// Function to handle voice input with continuous recognition
function startVoiceInput() {
    // Check if speech recognition is supported
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        addMessageToConversation('Speech recognition is not supported in your browser. Please try using Chrome or Edge.', 'assistant');
        return;
    }
    
    // Initialize recognition if not already done
    try {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        appState.recognition = new SpeechRecognition();
        appState.recognition.continuous = true; // Enable continuous recognition
        appState.recognition.interimResults = true; // Enable interim results for real-time display
        appState.recognition.lang = 'en-US';
        
        // Clear any previous transcript
        if (realTimeTranscript) {
            realTimeTranscript.textContent = '';
            realTimeTranscript.classList.add('active');
        }
        
        // Visual feedback that we're listening
        appState.isListening = true;
        voiceInputBtn.classList.add('listening');
        voiceInputBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="1" y1="1" x2="23" y2="23"></line><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"></path><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg> Stop Listening';
        
        // Set voice activity state to listening
        setVoiceActivityState('listening');
        
        // Announce for screen readers
        announceSettingChange('Voice recognition', true);
        
        let finalTranscript = '';
        
        appState.recognition.onresult = (event) => {
            let interimTranscript = '';
            
            // Process results
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                
                if (event.results[i].isFinal) {
                    finalTranscript += transcript + ' ';
                    
                    // If we detect a natural pause (period, question mark, etc.), send the message
                    if (/[.!?]\s*$/.test(transcript) || transcript.length > 60) {
                        sendVoiceMessage(finalTranscript.trim());
                        finalTranscript = '';
                    }
                } else {
                    interimTranscript += transcript;
                }
            }
            
            // Update real-time transcript display
            if (realTimeTranscript) {
                // Show final transcript in bold
                realTimeTranscript.innerHTML = `
                    <span class="final-transcript">${finalTranscript}</span>
                    <span class="interim-transcript">${interimTranscript}</span>
                `;
                
                // Auto-scroll to bottom
                realTimeTranscript.scrollTop = realTimeTranscript.scrollHeight;
            }
            
            // Also update the input field for visual feedback
            userInput.value = finalTranscript + interimTranscript;
        };
        
        appState.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            
            // Handle specific error types with retry logic
            let errorMessage = '';
            let shouldRetry = false;
            
            switch(event.error) {
                case 'network':
                    errorMessage = 'Network connection issue. Speech recognition may work better with a stable internet connection. You can still use text input.';
                    shouldRetry = true;
                    break;
                case 'not-allowed':
                case 'permission-denied':
                    errorMessage = 'Microphone access was denied. Please allow microphone access in your browser settings and refresh the page.';
                    break;
                case 'aborted':
                    // Don't show error for user-initiated stops
                    if (appState.isListening) {
                        errorMessage = 'Speech recognition was stopped. You can start it again or use text input.';
                    }
                    break;
                case 'audio-capture':
                    errorMessage = 'No microphone was found. Please ensure your microphone is connected and try again.';
                    break;
                case 'service-not-allowed':
                    errorMessage = 'Speech recognition service is not available. This may be due to browser restrictions. Please use text input.';
                    break;
                case 'no-speech':
                    // Automatically retry for no-speech errors
                    if (appState.isListening) {
                        console.log('No speech detected, continuing to listen...');
                        return; // Don't stop listening
                    }
                    break;
                default:
                    errorMessage = `Speech recognition encountered an issue (${event.error}). You can try again or use text input.`;
                    shouldRetry = true;
            }
            
            // Stop current recognition
            stopVoiceInput();
            
            // Show error message if there is one
            if (errorMessage) {
                addMessageToConversation(errorMessage, 'assistant');
            }
            
            // Auto-retry for network errors after a short delay
            if (shouldRetry && event.error === 'network') {
                setTimeout(() => {
                    if (!appState.isListening) {
                        console.log('Auto-retrying speech recognition...');
                        // Don't auto-retry to avoid loops, just suggest manual retry
                        addMessageToConversation('💡 Tip: You can try the voice input again, or continue with text input below.', 'assistant');
                    }
                }, 2000);
            }
            
            // Focus on text input as fallback
            userInput.focus();
        };
        
        appState.recognition.onend = () => {
            // If we still have text in the final transcript, send it
            if (finalTranscript.trim() !== '') {
                sendVoiceMessage(finalTranscript.trim());
            }
            
            stopVoiceInput();
        };
        
        appState.recognition.start();
        
    } catch (error) {
        console.error('Error starting speech recognition:', error);
        addMessageToConversation('Error starting speech recognition.', 'system');
        stopVoiceInput();
    }
}

// Stop voice input
function stopVoiceInput() {
    if (appState.recognition) {
        appState.recognition.stop();
    }
    
    appState.isListening = false;
    voiceInputBtn.classList.remove('listening');
    voiceInputBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg> Voice Input';
    
    // Hide real-time transcript
    if (realTimeTranscript) {
        realTimeTranscript.classList.remove('active');
        setTimeout(() => {
            realTimeTranscript.textContent = '';
        }, 300); // Wait for fade-out animation
    }
    
    // Reset voice activity state
    setVoiceActivityState('idle');
    
    // Announce for screen readers
}

// Send a message from voice input
function sendVoiceMessage(transcript) {
if (!transcript || transcript.trim() === '') return;

// Update UI
userInput.value = transcript;

// Send the message
sendMessage(transcript);

// Clear input
userInput.value = '';
}

// ========================================
// HANDS-FREE CONVERSATION MODE (ChatGPT Style)
// ========================================

// Start hands-free conversation mode
function startConversationMode() {
console.log(' Starting hands-free conversation mode...');

// Check if speech recognition is supported
if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
addMessageToConversation(' Speech recognition is not supported in your browser. Please try using Chrome or Edge.', 'assistant');
return;
}

appState.conversationMode = true;
appState.conversationActive = true;

// Update UI
const conversationBtn = document.getElementById('conversationMode');
if (conversationBtn) {
conversationBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="9" x2="15" y2="15"></line><line x1="15" y1="9" x2="9" y2="15"></line></svg><span>Stop Conversation</span>';
conversationBtn.classList.add('active');
}

// Visual feedback
document.body.classList.add('conversation-mode');

// Initialize speech recognition for continuous mode
initializeContinuousRecognition();

// Welcome message
const welcomeMessage = "Hello! I'm ready to help you with visual assistance. I can see through your camera and describe what's around you. What would you like to know?";
addMessageToConversation(welcomeMessage, 'assistant');

// Speak welcome and start listening
speakTextAndThenListen(welcomeMessage);

// Announce for accessibility
announceSettingChange('Hands-free conversation mode', true);
}

// Stop hands-free conversation mode
function stopConversationMode() {
console.log(' Stopping hands-free conversation mode...');

appState.conversationMode = false;
appState.conversationActive = false;
appState.isAssistantSpeaking = false;

// Stop any ongoing recognition
if (appState.recognition) {
appState.recognition.stop();
}

// Stop any ongoing speech
if (appState.synthesis) {
appState.synthesis.cancel();
}

// Update UI
const conversationBtn = document.getElementById('conversationMode');
if (conversationBtn) {
conversationBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg><span>Start Conversation Mode</span>';
conversationBtn.classList.remove('active');
}

// Remove visual feedback
document.body.classList.remove('conversation-mode');
setVoiceActivityState('idle');

// Goodbye message
addMessageToConversation('Conversation mode stopped. You can start it again anytime!', 'assistant');

// Announce for accessibility
announceSettingChange('Hands-free conversation mode', false);
}

// Toggle conversation mode
function toggleConversationMode() {
if (appState.conversationMode) {
stopConversationMode();
} else {
startConversationMode();
}
}

// Initialize continuous speech recognition
function initializeContinuousRecognition() {
try {
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
appState.recognition = new SpeechRecognition();

// Configuration for continuous conversation
appState.recognition.continuous = false; // We'll manually restart for better control
appState.recognition.interimResults = true;
appState.recognition.lang = 'en-US';
appState.recognition.maxAlternatives = 1;

let finalTranscript = '';
let isProcessing = false;

appState.recognition.onstart = () => {
console.log(' Listening started...');
appState.isListening = true;
setVoiceActivityState('listening');

// Update real-time transcript
if (realTimeTranscript) {
realTimeTranscript.textContent = 'Listening...';
realTimeTranscript.classList.add('active');
}
};

appState.recognition.onresult = (event) => {
if (isProcessing) return; // Prevent multiple processing

let interimTranscript = '';
finalTranscript = '';

// Process all results
for (let i = event.resultIndex; i < event.results.length; i++) {
const transcript = event.results[i][0].transcript;

if (event.results[i].isFinal) {
finalTranscript += transcript;
} else {
interimTranscript += transcript;
}
}

// Update real-time display
if (realTimeTranscript) {
realTimeTranscript.textContent = finalTranscript + interimTranscript;
}

// If we have a final result, process it
if (finalTranscript.trim() !== '' && !isProcessing) {
isProcessing = true;
console.log(' Final transcript:', finalTranscript);

// Clear real-time transcript
if (realTimeTranscript) {
realTimeTranscript.textContent = '';
realTimeTranscript.classList.remove('active');
}

// Process the message
processConversationMessage(finalTranscript.trim());

// Reset for next input
finalTranscript = '';
isProcessing = false;
}
};

appState.recognition.onerror = (event) => {
console.error(' Speech recognition error:', event.error);

// Handle errors gracefully in conversation mode
if (appState.conversationMode) {
handleConversationError(event.error);
}
};

appState.recognition.onend = () => {
console.log(' Recognition ended');
appState.isListening = false;

// Auto-restart if conversation mode is still active and assistant isn't speaking
if (appState.conversationMode && appState.conversationActive && !appState.isAssistantSpeaking) {
setTimeout(() => {
if (appState.conversationMode && !appState.isAssistantSpeaking) {
console.log(' Auto-restarting recognition...');
startListening();
}
}, 500); // Small delay to prevent rapid restarts
} else {
setVoiceActivityState('idle');
}
};

} catch (error) {
console.error(' Failed to initialize speech recognition:', error);
addMessageToConversation('Failed to initialize speech recognition. Please ensure you\'re using a supported browser like Chrome.', 'assistant');
}
}

// Start listening (used in conversation mode)
function startListening() {
if (!appState.recognition || appState.isAssistantSpeaking) {
return;
}

try {
appState.recognition.start();
} catch (error) {
console.error('Error starting recognition:', error);
// Try again after a short delay
setTimeout(() => {
if (appState.conversationMode && !appState.isAssistantSpeaking) {
startListening();
}
}, 1000);
}
}

// Capture image from video feed for conversation mode
async function captureImageFromVideo() {
if (!stream || !cameraFeed.videoWidth || !cameraFeed.videoHeight) {
console.log(' No camera feed available for vision context');
return null;
}
    
try {
// Create a temporary canvas for capture
const tempCanvas = document.createElement('canvas');
const context = tempCanvas.getContext('2d');
        
// Set canvas dimensions to match video
tempCanvas.width = cameraFeed.videoWidth;
tempCanvas.height = cameraFeed.videoHeight;
        
// Draw current video frame to canvas
context.drawImage(cameraFeed, 0, 0, tempCanvas.width, tempCanvas.height);
        
// Get base64 image data
const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
console.log(' Captured image from video feed for vision context');
        
return imageData;
} catch (error) {
console.error('Error capturing image from video:', error);
return null;
}
}

// Fetch caption from vision API
async function fetchCaption(imageData) {
if (!imageData) {
return null;
}
    
try {
console.log(' Getting vision caption for conversation context...');
        
const response = await fetch(`${SERVER_URL}/api/v1/vision/caption`, {
method: 'POST',
headers: {
'Content-Type': 'application/json'
},
body: JSON.stringify({
image: imageData,
detailed: true
})
});
        
if (!response.ok) {
throw new Error(`Vision API error: ${response.status}`);
}
        
const data = await response.json();
const caption = data.caption || data.description || 'Unable to describe the image';
        
console.log(' Vision caption received:', caption);
return caption;
        
} catch (error) {
console.error('Error fetching caption:', error);
return null;
}
}

// Process conversation message with vision awareness
async function processConversationMessage(message) {
try {
console.log(' Processing conversation message:', message);

    // Add user message to conversation
    addMessageToConversation(message, 'user');
    appState.conversationHistory.push({ role: 'user', content: message });

    // Set processing state
    setVoiceActivityState('processing');

    // 🎯 VISION-AWARE CONVERSATION FLOW
    console.log('🧠 Starting vision-aware conversation flow...');
    
    // Step 1: Capture current image from video feed
    const imageData = await captureImageFromVideo();
    
    // Step 2: Get vision caption if image is available
    let caption = null;
    if (imageData) {
        caption = await fetchCaption(imageData);
    }
    
    // Step 3: Compose vision-grounded prompt
    let finalPrompt = message;
    if (caption) {
        finalPrompt = `Current visual context: ${caption}. User question: ${message}`;
        console.log('🧠 Vision-grounded prompt:', finalPrompt);
    } else {
        console.log('🧠 No visual context available, using text-only prompt');
    }

    // Step 4: Send to conversation backend
    const response = await fetch(CONVERSATION_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message: finalPrompt,
            session_id: appState.sessionId,
            has_visual_context: !!caption
        })
    });

if (!response.ok) {
throw new Error(`HTTP error! status: ${response.status}`);
}

const data = await response.json();

if (data.success) {
const assistantResponse = data.response;

// Add assistant response to conversation
addMessageToConversation(assistantResponse, 'assistant');
appState.conversationHistory.push({ role: 'assistant', content: assistantResponse });

// Speak the response and then continue listening
speakTextAndThenListen(assistantResponse);

} else {
throw new Error(data.error || 'Unknown error occurred');
}

} catch (error) {
console.error(' Error processing conversation message:', error);

// Fallback response
const errorResponse = "I'm having trouble processing your request right now. Could you please try again?";
addMessageToConversation(errorResponse, 'assistant');
speakTextAndThenListen(errorResponse);
}
}

// Speak text and then restart listening
function speakTextAndThenListen(text) {
if (!appState.isSpeechEnabled || !text) {
// If speech is disabled, just restart listening
if (appState.conversationMode) {
setTimeout(() => startListening(), 500);
}
return;
}

appState.isAssistantSpeaking = true;
setVoiceActivityState('speaking');

// Create utterance
const utterance = new SpeechSynthesisUtterance(text);

// Configure speech
utterance.rate = 1.0;
utterance.pitch = 1.0;
utterance.volume = 1.0;

// Use preferred voice if available
const voices = appState.synthesis.getVoices();
const preferredVoice = voices.find(voice => 
voice.lang.startsWith('en') && voice.name.includes('Female')
) || voices.find(voice => voice.lang.startsWith('en'));

if (preferredVoice) {
utterance.voice = preferredVoice;
}

// Event handlers
utterance.onstart = () => {
console.log(' Assistant speaking...');
document.body.classList.add('speaking');
};

utterance.onend = () => {
console.log(' Assistant finished speaking');
appState.isAssistantSpeaking = false;
document.body.classList.remove('speaking');

// Restart listening if conversation mode is still active
if (appState.conversationMode && appState.conversationActive) {
setTimeout(() => {
startListening();
}, 500); // Brief pause before listening again
} else {
setVoiceActivityState('idle');
}
};

utterance.onerror = (event) => {
console.error(' Speech synthesis error:', event.error);
appState.isAssistantSpeaking = false;
document.body.classList.remove('speaking');

// Still restart listening even if speech failed
if (appState.conversationMode && appState.conversationActive) {
setTimeout(() => startListening(), 500);
}
};

// Speak the text
appState.synthesis.speak(utterance);
}

// Handle conversation mode errors
function handleConversationError(errorType) {
let shouldContinue = true;
let errorMessage = '';

switch(errorType) {
case 'network':
errorMessage = 'I\'m having network issues. Let me try to continue listening...';
break;
case 'not-allowed':
case 'permission-denied':
errorMessage = 'I need microphone access to continue our conversation. Please allow microphone access and try again.';
shouldContinue = false;
break;
case 'no-speech':
// Don't announce no-speech errors, just continue
shouldContinue = true;
break;
case 'audio-capture':
errorMessage = 'I can\'t access your microphone. Please check your microphone connection.';
shouldContinue = false;
break;
default:
errorMessage = 'I encountered an issue with speech recognition. Let me try to continue...';
}

if (errorMessage) {
console.log(' Conversation error:', errorMessage);
if (errorType !== 'no-speech') {
addMessageToConversation(errorMessage, 'assistant');
}
}

if (!shouldContinue) {
stopConversationMode();
} else {
// Try to restart listening after a delay
setTimeout(() => {
if (appState.conversationMode && !appState.isAssistantSpeaking) {
startListening();
}
}, 1000);
}
}
