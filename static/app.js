// VisionAssist App - Auto-Capture AI Response Logger
class VisionAssistApp {
    constructor() {
        // Initialize managers
        this.visionManager = new VisionManager();
        this.speechManager = new SpeechManager();
        this.conversationManager = null;
        
        // UI elements
        this.elements = {};
        
        // App state
        this.state = {
            cameraActive: false,
            voiceMode: false,
            isListening: false,
            autoCaptureEnabled: true,
            autoCaptureInterval: 3000, // 3 seconds
            captureCountdown: 3
        };
        
        // Auto-capture timer
        this.autoCaptureTimer = null;
        this.countdownTimer = null;
        
        // Response logger
        this.responseLogger = {
            entries: [],
            maxEntries: 100
        };
        
        this.initialize();
    }

    async initialize() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initializeApp());
        } else {
            this.initializeApp();
        }
    }

    async initializeApp() {
        console.log('🚀 Initializing VisionAssist App...');
        
        // Initialize UI elements
        this.initializeElements();
        
        // Initialize managers
        this.visionManager.initialize();
        this.conversationManager = new ConversationManager(this.visionManager, this.speechManager);
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Initialize response logger
        this.initializeResponseLogger();
        
        // Auto-start camera and begin capture cycle
        await this.autoStartCamera();
        
        console.log('✅ VisionAssist App initialized successfully');
    }

    initializeElements() {
        // Camera elements
        this.elements.cameraFeed = document.getElementById('cameraFeed');
        this.elements.captionContainer = document.getElementById('captionContainer');
        this.elements.captureIndicator = document.getElementById('captureIndicator');
        this.elements.autoCaptureStatus = document.getElementById('autoCaptureStatus');
        this.elements.captureCountdown = document.getElementById('captureCountdown');
        
        // Status elements
        this.elements.systemStatus = document.getElementById('systemStatus');
        this.elements.statusText = document.getElementById('statusText');
        
        // Logger elements
        this.elements.responseTimeline = document.getElementById('responseTimeline');
        this.elements.clearLogBtn = document.getElementById('clearLogBtn');
        this.elements.exportLogBtn = document.getElementById('exportLogBtn');
        this.elements.pauseAutoCapture = document.getElementById('pauseAutoCapture');
        
        // Voice elements
        this.elements.voiceIndicator = document.getElementById('voiceIndicator');
        this.elements.realTimeTranscript = document.getElementById('realTimeTranscript');
        this.elements.conversationModeBtn = document.getElementById('conversationModeBtn');
    }

    setupEventListeners() {
        // Auto-capture controls
        this.elements.pauseAutoCapture?.addEventListener('click', () => this.toggleAutoCapture());
        this.elements.conversationModeBtn?.addEventListener('click', () => this.handleVoiceMode());
        
        // Logger controls
        this.elements.clearLogBtn?.addEventListener('click', () => this.clearResponseLog());
        this.elements.exportLogBtn?.addEventListener('click', () => this.exportResponseLog());
        
        // Listen for AI responses from other components
        document.addEventListener('aiResponse', (event) => this.logAIResponse(event.detail));
        document.addEventListener('visionCaption', (event) => this.logVisionResponse(event.detail));
        document.addEventListener('conversationResponse', (event) => this.logConversationResponse(event.detail));
        
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => this.handleVisibilityChange());
    }

    initializeResponseLogger() {
        // Show initial placeholder
        this.updateTimelinePlaceholder();
    }

    async autoStartCamera() {
        try {
            this.updateSystemStatus('starting', 'Starting camera...');
            
            const success = await this.visionManager.startCamera();
            
            if (success) {
                this.state.cameraActive = true;
                this.updateSystemStatus('active', 'Camera active');
                this.updateCaptionDisplay('Camera started - beginning auto-capture in 3 seconds...');
                
                // Log camera start
                this.logSystemEvent('Camera started automatically');
                
                // Start auto-capture cycle
                this.startAutoCaptureLoop();
            } else {
                this.updateSystemStatus('error', 'Camera failed');
                this.updateCaptionDisplay('Failed to start camera. Please check permissions and refresh the page.');
                this.logSystemEvent('Camera failed to start automatically', 'error');
            }
        } catch (error) {
            console.error('Error auto-starting camera:', error);
            this.updateSystemStatus('error', 'Camera error');
            this.updateCaptionDisplay('Error starting camera automatically');
            this.logSystemEvent(`Camera auto-start error: ${error.message}`, 'error');
        }
    }

    startAutoCaptureLoop() {
        if (!this.state.autoCaptureEnabled || !this.state.cameraActive) return;
        
        // Start countdown
        this.startCountdown();
        
        // Set up the capture timer
        this.autoCaptureTimer = setInterval(() => {
            if (this.state.autoCaptureEnabled && this.state.cameraActive) {
                this.performAutoCapture();
                this.startCountdown(); // Restart countdown for next capture
            }
        }, this.state.autoCaptureInterval);
        
        this.logSystemEvent('Auto-capture started (3-second intervals)');
    }

    startCountdown() {
        // Clear any existing countdown
        if (this.countdownTimer) {
            clearInterval(this.countdownTimer);
        }
        
        this.state.captureCountdown = 3;
        this.updateCountdownDisplay();
        
        this.countdownTimer = setInterval(() => {
            this.state.captureCountdown--;
            this.updateCountdownDisplay();
            
            if (this.state.captureCountdown <= 0) {
                clearInterval(this.countdownTimer);
                this.showCaptureIndicator();
            }
        }, 1000);
    }

    async performAutoCapture() {
        if (!this.state.cameraActive) return;
        
        try {
            this.updateCaptionDisplay('Analyzing image...', 'processing');
            
            // Capture image first
            const imageData = await this.visionManager.captureImageFromVideo();
            
            if (!imageData) {
                this.updateCaptionDisplay('Failed to capture image');
                this.logSystemEvent('Auto-capture: Failed to capture image', 'error');
                return;
            }
            
            // Get caption using simpler endpoint
            const caption = await this.visionManager.fetchCaption(imageData, true);
            
            if (caption) {
                this.updateCaptionDisplay(caption);
                this.logVisionResponse({
                    caption: caption,
                    confidence: 'Auto-captured',
                    processing_time: 'Real-time',
                    auto_captured: true
                });
            } else {
                this.updateCaptionDisplay('Failed to analyze image');
                this.logSystemEvent('Auto-capture analysis failed', 'error');
            }
        } catch (error) {
            console.error('Error in auto-capture:', error);
            this.updateCaptionDisplay('Error analyzing image');
            this.logSystemEvent(`Auto-capture error: ${error.message}`, 'error');
        }
    }

    toggleAutoCapture() {
        if (this.state.autoCaptureEnabled) {
            // Pause auto-capture
            this.state.autoCaptureEnabled = false;
            this.stopAutoCaptureLoop();
            this.elements.pauseAutoCapture.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5,3 19,12 5,21"/>
                </svg>
            `;
            this.elements.pauseAutoCapture.title = 'Resume Auto-Capture';
            this.updateSystemStatus('paused', 'Auto-capture paused');
            this.logSystemEvent('Auto-capture paused');
        } else {
            // Resume auto-capture
            this.state.autoCaptureEnabled = true;
            this.startAutoCaptureLoop();
            this.elements.pauseAutoCapture.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="6" y="4" width="4" height="16"/>
                    <rect x="14" y="4" width="4" height="16"/>
                </svg>
            `;
            this.elements.pauseAutoCapture.title = 'Pause Auto-Capture';
            this.updateSystemStatus('active', 'Auto-capture resumed');
            this.logSystemEvent('Auto-capture resumed');
        }
    }

    stopAutoCaptureLoop() {
        if (this.autoCaptureTimer) {
            clearInterval(this.autoCaptureTimer);
            this.autoCaptureTimer = null;
        }
        if (this.countdownTimer) {
            clearInterval(this.countdownTimer);
            this.countdownTimer = null;
        }
        this.hideCountdownDisplay();
    }

    async handleVoiceMode() {
        if (!this.state.cameraActive) {
            this.logSystemEvent('Camera must be active for voice mode', 'warning');
            return;
        }
        
        try {
            if (!this.state.voiceMode) {
                // Start voice mode
                this.state.voiceMode = true;
                this.elements.conversationModeBtn.classList.add('active');
                
                // Start conversation mode
                await this.conversationManager.startConversationMode();
                this.showVoiceIndicator();
                this.logSystemEvent('Voice mode activated');
            } else {
                // Stop voice mode
                this.state.voiceMode = false;
                this.elements.conversationModeBtn.classList.remove('active');
                
                // Stop conversation mode
                this.conversationManager.stopConversationMode();
                this.hideVoiceIndicator();
                this.logSystemEvent('Voice mode deactivated');
            }
        } catch (error) {
            console.error('Error toggling voice mode:', error);
            this.logSystemEvent(`Voice mode error: ${error.message}`, 'error');
        }
    }

    handleVisibilityChange() {
        if (document.hidden) {
            // Page is hidden, pause auto-capture to save resources
            if (this.state.autoCaptureEnabled) {
                this.stopAutoCaptureLoop();
                this.logSystemEvent('Auto-capture paused (page hidden)');
            }
        } else {
            // Page is visible again, resume auto-capture
            if (this.state.autoCaptureEnabled && this.state.cameraActive) {
                this.startAutoCaptureLoop();
                this.logSystemEvent('Auto-capture resumed (page visible)');
            }
        }
    }

    // Response Logger Methods
    logAIResponse(data) {
        this.addLogEntry({
            type: 'ai_response',
            content: data.response || data.content,
            metadata: data,
            timestamp: new Date()
        });
    }

    logVisionResponse(data) {
        this.addLogEntry({
            type: 'vision_analysis',
            content: data.caption,
            metadata: {
                confidence: data.confidence,
                processing_time: data.processing_time,
                model_info: data.model_info,
                auto_captured: data.auto_captured || false
            },
            timestamp: new Date()
        });
    }

    logConversationResponse(data) {
        this.addLogEntry({
            type: 'conversation',
            content: data.response,
            metadata: {
                context_used: data.context_used,
                session_id: data.session_id
            },
            timestamp: new Date()
        });
    }

    logSystemEvent(message, level = 'info') {
        this.addLogEntry({
            type: 'system',
            content: message,
            level: level,
            timestamp: new Date()
        });
    }

    addLogEntry(entry) {
        // Add to entries array
        this.responseLogger.entries.unshift(entry);
        
        // Limit number of entries
        if (this.responseLogger.entries.length > this.responseLogger.maxEntries) {
            this.responseLogger.entries = this.responseLogger.entries.slice(0, this.responseLogger.maxEntries);
        }
        
        // Update UI
        this.renderResponseTimeline();
        
        // Dispatch custom event for other components
        document.dispatchEvent(new CustomEvent('logEntryAdded', { detail: entry }));
    }

    renderResponseTimeline() {
        const timeline = this.elements.responseTimeline;
        if (!timeline) return;
        
        // Clear existing content
        timeline.innerHTML = '';
        
        if (this.responseLogger.entries.length === 0) {
            this.updateTimelinePlaceholder();
            return;
        }
        
        // Render entries
        this.responseLogger.entries.forEach(entry => {
            const entryElement = this.createLogEntryElement(entry);
            timeline.appendChild(entryElement);
        });
        
        // Scroll to top for newest entry
        timeline.scrollTop = 0;
    }

    createLogEntryElement(entry) {
        const entryDiv = document.createElement('div');
        entryDiv.className = 'response-entry';
        entryDiv.setAttribute('data-type', entry.type);
        
        // Add auto-capture indicator if applicable
        if (entry.metadata?.auto_captured) {
            entryDiv.classList.add('auto-captured');
        }
        
        // Create header
        const header = document.createElement('div');
        header.className = 'response-header';
        
        const typeSpan = document.createElement('div');
        typeSpan.className = 'response-type';
        typeSpan.innerHTML = `
            ${this.getTypeIcon(entry.type)}
            <span>${this.getTypeLabel(entry.type)}</span>
            ${entry.metadata?.auto_captured ? '<span class="auto-badge">AUTO</span>' : ''}
        `;
        
        const timestamp = document.createElement('div');
        timestamp.className = 'response-timestamp';
        timestamp.textContent = this.formatTimestamp(entry.timestamp);
        
        header.appendChild(typeSpan);
        header.appendChild(timestamp);
        
        // Create content
        const content = document.createElement('div');
        content.className = 'response-content';
        content.textContent = entry.content;
        
        entryDiv.appendChild(header);
        entryDiv.appendChild(content);
        
        return entryDiv;
    }

    getTypeIcon(type) {
        const icons = {
            'vision_analysis': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
            'conversation': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
            'ai_response': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 12l2 2 4-4"/><path d="M21 12c0 4.97-4.03 9-9 9s-9-4.03-9-9 4.03-9 9-9c2.12 0 4.07.74 5.61 1.98"/></svg>',
            'system': '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
        };
        return icons[type] || icons['system'];
    }

    getTypeLabel(type) {
        const labels = {
            'vision_analysis': 'Vision Analysis',
            'conversation': 'Conversation',
            'ai_response': 'AI Response',
            'system': 'System'
        };
        return labels[type] || 'Unknown';
    }

    formatTimestamp(timestamp) {
        return timestamp.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }

    updateTimelinePlaceholder() {
        const timeline = this.elements.responseTimeline;
        if (!timeline) return;
        
        timeline.innerHTML = `
            <div class="timeline-placeholder">
                <div class="placeholder-icon">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="12" cy="12" r="10"/>
                        <polyline points="12,6 12,12 16,14"/>
                    </svg>
                </div>
                <p>Auto-capture will begin shortly. AI responses will appear here every 3 seconds.</p>
            </div>
        `;
    }

    clearResponseLog() {
        this.responseLogger.entries = [];
        this.updateTimelinePlaceholder();
        this.logSystemEvent('Response log cleared');
    }

    exportResponseLog() {
        try {
            const exportData = {
                timestamp: new Date().toISOString(),
                entries: this.responseLogger.entries,
                metadata: {
                    total_entries: this.responseLogger.entries.length,
                    auto_capture_enabled: this.state.autoCaptureEnabled,
                    capture_interval: this.state.autoCaptureInterval,
                    export_version: '1.0'
                }
            };
            
            const dataStr = JSON.stringify(exportData, null, 2);
            const dataBlob = new Blob([dataStr], { type: 'application/json' });
            
            const link = document.createElement('a');
            link.href = URL.createObjectURL(dataBlob);
            link.download = `visionassist-auto-log-${new Date().toISOString().split('T')[0]}.json`;
            link.click();
            
            this.logSystemEvent('Response log exported successfully');
        } catch (error) {
            console.error('Error exporting log:', error);
            this.logSystemEvent(`Export failed: ${error.message}`, 'error');
        }
    }

    // UI Helper Methods
    updateSystemStatus(status, text) {
        const statusDot = this.elements.systemStatus;
        const statusText = this.elements.statusText;
        
        if (statusDot) {
            statusDot.className = 'status-dot';
            statusDot.classList.add(status);
        }
        
        if (statusText) {
            statusText.textContent = text;
        }
    }

    updateCaptionDisplay(text, state = 'normal') {
        const container = this.elements.captionContainer;
        if (!container) return;
        
        // Remove all state classes
        container.classList.remove('processing');
        
        if (state === 'processing') {
            container.classList.add('processing');
            container.innerHTML = `
                <div class="caption-content processing">
                    <div class="processing-spinner"></div>
                    <p>${text}</p>
                </div>
            `;
        } else {
            container.innerHTML = `
                <div class="caption-content">
                    <p>${text}</p>
                </div>
            `;
        }
    }

    updateCountdownDisplay() {
        const countdown = this.elements.captureCountdown;
        const status = this.elements.autoCaptureStatus;
        
        if (countdown) {
            countdown.textContent = this.state.captureCountdown;
        }
        
        if (status) {
            status.style.opacity = this.state.captureCountdown > 0 ? '1' : '0.5';
        }
    }

    hideCountdownDisplay() {
        const status = this.elements.autoCaptureStatus;
        if (status) {
            status.style.opacity = '0.3';
        }
    }

    showCaptureIndicator() {
        const indicator = this.elements.captureIndicator;
        if (indicator) {
            indicator.classList.add('capturing');
            setTimeout(() => {
                indicator.classList.remove('capturing');
            }, 500);
        }
    }

    showVoiceIndicator() {
        this.elements.voiceIndicator?.classList.add('active');
    }

    hideVoiceIndicator() {
        this.elements.voiceIndicator?.classList.remove('active');
    }

    // Voice activity updates
    updateVoiceActivity(isListening) {
        this.state.isListening = isListening;
        const indicator = this.elements.voiceIndicator;
        if (!indicator) return;
        
        if (isListening) {
            indicator.classList.add('listening');
            indicator.querySelector('.voice-status').textContent = 'Listening...';
        } else {
            indicator.classList.remove('listening');
            indicator.querySelector('.voice-status').textContent = 'Voice Mode Active';
        }
    }

    updateTranscript(text, isFinal = false) {
        const transcript = this.elements.realTimeTranscript;
        if (!transcript) return;
        
        if (text) {
            transcript.textContent = text;
            transcript.classList.add('active');
        } else {
            transcript.classList.remove('active');
        }
    }

    // Cleanup method
    destroy() {
        this.stopAutoCaptureLoop();
        if (this.conversationManager) {
            this.conversationManager.stopConversationMode();
        }
    }
}

// Initialize the app when the script loads
const visionAssistApp = new VisionAssistApp();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    visionAssistApp.destroy();
});

// Export for debugging
window.visionAssistApp = visionAssistApp;
