// Vision Module - Handles camera, image capture, and vision API calls
class VisionManager {
    constructor() {
        this.stream = null;
        this.cameraFeed = null;
        this.captureCanvas = null;
        this.serverUrl = 'http://localhost:5000';
    }

    initialize() {
        this.cameraFeed = document.getElementById('cameraFeed');
        // Use existing canvas if present; otherwise leave null and create on demand
        this.captureCanvas = document.getElementById('captureCanvas') || null;
    }

    // Start camera
    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });
            
            this.cameraFeed.srcObject = this.stream;
            await this.cameraFeed.play();
            
            console.log('📹 Camera started successfully');
            return true;
        } catch (error) {
            console.error('Error accessing camera:', error);
            return false;
        }
    }

    // Capture image from video feed for conversation mode
    async captureImageFromVideo() {
        // Comprehensive validation
        if (!this.stream) {
            console.log('🎥 No camera stream available for vision context');
            return null;
        }
        
        if (!this.cameraFeed) {
            console.error('🎥 Camera feed element not found');
            return null;
        }
        
        if (this.cameraFeed.readyState !== 4) {
            console.log('🎥 Camera feed not ready (readyState:', this.cameraFeed.readyState, ')');
            return null;
        }
        
        if (!this.cameraFeed.videoWidth || !this.cameraFeed.videoHeight) {
            console.log('🎥 Invalid video dimensions:', this.cameraFeed.videoWidth, 'x', this.cameraFeed.videoHeight);
            return null;
        }
        
        // Validate dimensions are reasonable
        if (this.cameraFeed.videoWidth < 10 || this.cameraFeed.videoHeight < 10) {
            console.error('🎥 Video dimensions too small:', this.cameraFeed.videoWidth, 'x', this.cameraFeed.videoHeight);
            return null;
        }
        
        if (this.cameraFeed.videoWidth > 4096 || this.cameraFeed.videoHeight > 4096) {
            console.warn('🎥 Video dimensions very large:', this.cameraFeed.videoWidth, 'x', this.cameraFeed.videoHeight);
        }
        
        try {
            // Create a temporary canvas for capture
            const tempCanvas = document.createElement('canvas');
            const context = tempCanvas.getContext('2d');
            
            if (!context) {
                throw new Error('Failed to get 2D canvas context');
            }
            
            // Set canvas dimensions to match video
            tempCanvas.width = this.cameraFeed.videoWidth;
            tempCanvas.height = this.cameraFeed.videoHeight;
            
            // Validate canvas dimensions were set correctly
            if (tempCanvas.width !== this.cameraFeed.videoWidth || tempCanvas.height !== this.cameraFeed.videoHeight) {
                throw new Error(`Canvas dimensions mismatch: expected ${this.cameraFeed.videoWidth}x${this.cameraFeed.videoHeight}, got ${tempCanvas.width}x${tempCanvas.height}`);
            }
            
            // Draw current video frame to canvas
            context.drawImage(this.cameraFeed, 0, 0, tempCanvas.width, tempCanvas.height);
            
            // Get base64 image data with error handling
            let imageData;
            try {
                imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
            } catch (canvasError) {
                console.error('🎥 Canvas toDataURL failed:', canvasError);
                // Try with lower quality
                try {
                    imageData = tempCanvas.toDataURL('image/jpeg', 0.5);
                    console.log('🎥 Captured with reduced quality due to canvas error');
                } catch (fallbackError) {
                    throw new Error(`Canvas export failed: ${fallbackError.message}`);
                }
            }
            
            // Validate image data
            if (!imageData || !imageData.startsWith('data:image/')) {
                throw new Error('Invalid image data generated');
            }
            
            // Check image data size (should be reasonable)
            const imageSizeKB = Math.round(imageData.length * 0.75 / 1024); // Rough base64 to bytes conversion
            if (imageSizeKB > 5000) { // 5MB limit
                console.warn('🎥 Large image captured:', imageSizeKB, 'KB');
            }
            
            console.log('🎥 Captured image from video feed for vision context (', imageSizeKB, 'KB)');
            
            return imageData;
        } catch (error) {
            console.error('🎥 Error capturing image from video:', error);
            return null;
        }
    }

    // Capture image using canvas
    async captureImage() {
        if (!this.stream) {
            throw new Error('Camera not started');
        }
        
        try {
            // Ensure we have a canvas to draw to
            const canvas = this.captureCanvas || document.createElement('canvas');
            const context = canvas.getContext('2d');
            if (!context) throw new Error('Failed to get 2D context');

            canvas.width = this.cameraFeed.videoWidth;
            canvas.height = this.cameraFeed.videoHeight;
            context.drawImage(this.cameraFeed, 0, 0, canvas.width, canvas.height);
            
            // Get base64 image
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            console.log('📸 Image captured successfully');
            
            return imageData;
        } catch (error) {
            console.error('Error capturing image:', error);
            throw error;
        }
    }

    // Fetch caption from vision API
    async fetchCaption(imageData, detailed = true) {
        if (!imageData) {
            return null;
        }
        
        try {
            console.log('🔍 Getting vision caption for conversation context...');
            
            const response = await fetch(`${this.serverUrl}/api/v1/vision/caption`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    detailed: detailed
                })
            });
            
            if (!response.ok) {
                throw new Error(`Vision API error: ${response.status}`);
            }
            
            const data = await response.json();
            const caption = data.caption || data.description || 'Unable to describe the image';
            
            console.log('🔍 Vision caption received:', caption);
            return caption;
            
        } catch (error) {
            console.error('Error fetching caption:', error);
            return null;
        }
    }

    // Get detailed analysis with attention and Grad-CAM
    async getDetailedAnalysis(imageData) {
        try {
            console.log('🔍 Sending image for detailed analysis...');
            console.log('🔍 Server URL:', this.serverUrl);
            console.log('🔍 Image data length:', imageData ? imageData.length : 'null');
            
            const response = await fetch(`${this.serverUrl}/api/v1/vision/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    include_attention: true,
                    include_gradcam: true
                })
            });
            
            console.log('🔍 Response status:', response.status);
            console.log('🔍 Response ok:', response.ok);
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('🔍 API Error Response:', errorText);
                throw new Error(`Analysis API error: ${response.status} - ${errorText}`);
            }
            
            const result = await response.json();
            console.log('🔍 Analysis result:', result);
            console.log('🔍 Analysis object keys:', Object.keys(result));
            if (result.analysis) {
                console.log('🔍 Analysis data:', result.analysis);
                console.log('🔍 Analysis keys:', Object.keys(result.analysis));
            }
            return result;
        } catch (error) {
            console.error('❌ Error getting detailed analysis:', error);
            console.error('❌ Error details:', {
                message: error.message,
                stack: error.stack,
                serverUrl: this.serverUrl
            });
            throw error;
        }
    }

    // Stop camera
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            console.log('📹 Camera stopped');
        }
    }

    // Check if camera is active
    isCameraActive() {
        return this.stream !== null;
    }
}

// Export for use in other modules
window.VisionManager = VisionManager;
