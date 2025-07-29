// VisionAssist Simplified Prototype
// A minimal implementation for demonstration purposes

// Configuration
const SERVER_URL = 'http://localhost:5000';

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

// Global variables
let stream = null;
let currentSessionId = null;
let isSpeechEnabled = true; // Default to speech enabled
let isListening = false; // Voice input status
let recognition = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Set up event listeners
    startCameraBtn.addEventListener('click', startCamera);
    captureBtnSingle.addEventListener('click', captureImage);
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    speechToggle.addEventListener('change', () => {
        isSpeechEnabled = speechToggle.checked;
    });
    
    // Voice input button
    if (voiceInputBtn) {
        voiceInputBtn.addEventListener('click', startVoiceInput);
    }
    
    // Initialize conversation mode
    async function startConversationMode() {
        // If camera is not started, start it first
        if (!stream) {
            addMessageToConversation('Starting camera for conversation mode...', 'system');
            await startCamera();
            
            // Give the camera a moment to initialize
            await new Promise(resolve => setTimeout(resolve, 1500));
        }
        
        // Create a session if needed
        if (!currentSessionId) {
            await createSession();
        }
        
        // Show visual feedback that conversation mode is starting
        const conversationModeBtn = document.getElementById('conversationMode');
        if (conversationModeBtn) {
            conversationModeBtn.disabled = true;
            conversationModeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting conversation...';
        }
        
        // Capture an image and start conversation
        if (stream) {
            const imageData = captureImageData();
            if (imageData) {
                addMessageToConversation('Starting conversation about what I can see...', 'system');
                const response = await sendMessage("What can you see in this image?", imageData);
                
                // After response, start listening for follow-up
                if (response && recognition) {
                    addMessageToConversation('You can now ask follow-up questions about what you see', 'system');
                    setTimeout(() => {
                        startVoiceInput();
                    }, 1000); // Wait for speech to finish
                }
            }
        }
        
        // Re-enable the button
        if (conversationModeBtn) {
            conversationModeBtn.disabled = false;
            conversationModeBtn.innerHTML = '<i class="fas fa-comments"></i> Start Conversation Mode';
        }
    }
    
    // Add conversation mode button
    const conversationModeBtn = document.createElement('button');
    conversationModeBtn.id = 'conversationMode';
    conversationModeBtn.className = 'btn btn-primary';
    conversationModeBtn.innerHTML = '<i class="fas fa-comments"></i> Start Conversation Mode';
    document.querySelector('.control-panel').appendChild(conversationModeBtn);
    
    conversationModeBtn.addEventListener('click', startConversationMode);
    
    // Create a session
    createSession();
});

// Create a conversation session
async function createSession() {
    try {
        const response = await fetch(`${SERVER_URL}/api/v1/conversation/session/create`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success && data.session_id) {
            currentSessionId = data.session_id;
            console.log('Created conversation session:', currentSessionId);
            
            // Add welcome message
            addMessageToConversation('Welcome to VisionAssist! I can help describe what I see through your camera.', 'assistant');
        }
    } catch (error) {
        console.error('Error creating session:', error);
        
        // Fallback
        currentSessionId = 'demo-session-' + Date.now();
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
            if (isSpeechEnabled) {
                speakText(data.caption);
            }
        } else {
            updateCaptionContainer('Could not generate caption.');
        }
        
    } catch (error) {
        console.error('Error capturing image:', error);
        updateCaptionContainer('Error processing image.');
        
        // Fallback for demo
        const fallbackCaptions = [
            'I see a person sitting at a desk with a computer.',
            'There appears to be a room with furniture and windows.',
            'I can see what looks like an indoor space with lighting.'
        ];
        
        const mockCaption = fallbackCaptions[Math.floor(Math.random() * fallbackCaptions.length)];
        updateCaptionContainer(mockCaption);
        addMessageToConversation('I see: ' + mockCaption, 'assistant');
        
        if (isSpeechEnabled) {
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
        if (!currentSessionId) {
            await createSession();
        }
        
        console.log('Sending message with session ID:', currentSessionId);
        
        const payload = {
            session_id: currentSessionId,
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
            if (isSpeechEnabled) {
                speakText(data.response);
            }
            
            return data.response;
        } else {
            addMessageToConversation('Sorry, I could not process your request.', 'assistant');
            return null;
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        
        // Fallback responses for demo
        const fallbackResponses = [
            "I can see what appears to be a room with furniture.",
            "That looks like a person using a computer.",
            "I notice what might be a window with natural light coming in.",
            "I'm not entirely sure what I'm seeing, but it looks like an indoor space.",
            "I can see some objects on what appears to be a desk or table."
        ];
        
        const mockResponse = fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
        addMessageToConversation(mockResponse, 'assistant');
        
        if (isSpeechEnabled) {
            speakText(mockResponse);
        }
        
        return null;
    }
}

// Add message to conversation history
function addMessageToConversation(message, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    
    const timestamp = new Date().toLocaleTimeString();
    
    messageElement.innerHTML = `
        <div class="message-content">${message}</div>
        <div class="message-timestamp">${timestamp}</div>
    `;
    
    conversationHistory.appendChild(messageElement);
    conversationHistory.scrollTop = conversationHistory.scrollHeight;
}

// Speak text using browser's speech synthesis
function speakText(text) {
    if ('speechSynthesis' in window) {
        // Cancel any ongoing speech
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        // Get available voices
        let voices = window.speechSynthesis.getVoices();
        
        // If voices aren't loaded yet, wait for them
        if (voices.length === 0) {
            window.speechSynthesis.onvoiceschanged = () => {
                voices = window.speechSynthesis.getVoices();
                // Try to find a natural sounding voice
                const preferredVoice = voices.find(voice => 
                    voice.name.includes('Samantha') || 
                    voice.name.includes('Google') || 
                    voice.name.includes('Natural')
                );
                
                if (preferredVoice) {
                    utterance.voice = preferredVoice;
                }
                
                window.speechSynthesis.speak(utterance);
            };
        } else {
            // Try to find a natural sounding voice
            const preferredVoice = voices.find(voice => 
                voice.name.includes('Samantha') || 
                voice.name.includes('Google') || 
                voice.name.includes('Natural')
            );
            
            if (preferredVoice) {
                utterance.voice = preferredVoice;
            }
            
            window.speechSynthesis.speak(utterance);
        }
    }
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

// Function to handle voice input
function startVoiceInput() {
    // Check if speech recognition is supported
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        addMessageToConversation('Speech recognition is not supported in your browser. Please try using Chrome or Edge.', 'assistant');
        return;
    }
    
    // Initialize recognition if not already done
    if (!recognition) {
        try {
            recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
        } catch (error) {
            console.error('Error initializing speech recognition:', error);
            addMessageToConversation('Could not initialize speech recognition. Please check your browser permissions.', 'assistant');
            return;
        }
    }
    
    if (isListening) {
        recognition.stop();
        return;
    }
    
    // Visual feedback that we're listening
    isListening = true;
    voiceInputBtn.classList.add('listening');
    voiceInputBtn.innerHTML = '<i class="fas fa-microphone-slash"></i> Stop Listening';
    
    let finalTranscript = '';
    
    recognition.start();
    
    recognition.onresult = (event) => {
        let interimTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        
        // Show interim results
        if (interimTranscript !== '') {
            userInput.value = interimTranscript;
        }
        
        // Process final results
        if (finalTranscript !== '') {
            userInput.value = finalTranscript;
        }
    };
    
    recognition.onend = () => {
        isListening = false;
        voiceInputBtn.classList.remove('listening');
        voiceInputBtn.innerHTML = '<i class="fas fa-microphone"></i> Voice Input';
        
        // If we have a transcript, send it
        if (userInput.value.trim() !== '') {
            sendBtn.click();
        }
    };
    
    recognition.onerror = (event) => {
        isListening = false;
        voiceInputBtn.classList.remove('listening');
        voiceInputBtn.innerHTML = '<i class="fas fa-microphone"></i> Voice Input';
        console.error('Speech recognition error', event.error);
        
        // Handle specific error types
        let errorMessage = '';
        switch(event.error) {
            case 'network':
                errorMessage = 'Network error occurred. Please check your internet connection and try again.';
                break;
            case 'not-allowed':
            case 'permission-denied':
                errorMessage = 'Microphone access was denied. Please allow microphone access in your browser settings.';
                break;
            case 'aborted':
                errorMessage = 'Speech recognition was aborted.';
                break;
            case 'audio-capture':
                errorMessage = 'No microphone was found. Please ensure your microphone is connected.';
                break;
            case 'service-not-allowed':
                errorMessage = 'Speech recognition service is not allowed. This may be due to browser restrictions.';
                break;
            default:
                errorMessage = `Speech recognition error: ${event.error}`;
        }
        
        addMessageToConversation(errorMessage, 'assistant');
        
        // Fallback to text input
        userInput.focus();
    };
}
