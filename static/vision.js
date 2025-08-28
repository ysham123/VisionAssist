// Vision Module - Streamlined camera and vision API handling
class VisionManager {
    constructor() {
        this.stream = null;
        this.cameraFeed = null;
        this.serverUrl = window.APP_CONFIG?.apiBaseUrl || '';
        this.retryCount = 0;
        this.maxRetries = 3;
    }

    initialize() {
        this.cameraFeed = document.getElementById('cameraFeed');
        return this.cameraFeed !== null;
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });
            
            if (this.cameraFeed) {
                this.cameraFeed.srcObject = this.stream;
                await this.cameraFeed.play();
                console.log('📹 Camera started');
                return true;
            }
            return false;
        } catch (error) {
            console.error('Camera error:', error);
            return false;
        }
    }

    captureImage() {
        if (!this.stream || !this.cameraFeed || this.cameraFeed.readyState !== 4) {
            return null;
        }
        
        try {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            canvas.width = this.cameraFeed.videoWidth;
            canvas.height = this.cameraFeed.videoHeight;
            context.drawImage(this.cameraFeed, 0, 0);
            
            return canvas.toDataURL('image/jpeg', 0.8);
        } catch (error) {
            console.error('Capture error:', error);
            return null;
        }
    }

    // Alias for backward compatibility
    captureImageFromVideo() {
        return this.captureImage();
    }

    async getCaption(imageData) {
        if (!imageData) return 'No image available';
        
        try {
            const response = await fetch(`${this.serverUrl}/api/v1/vision/caption`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            if (response.ok) {
                const data = await response.json();
                this.retryCount = 0;
                return data.caption || 'Unable to describe image';
            }
            
            throw new Error(`API error: ${response.status}`);
        } catch (error) {
            console.error('Caption error:', error);
            
            // Simple fallback without complex retry logic
            if (this.retryCount < this.maxRetries) {
                this.retryCount++;
                return 'Processing image...';
            }
            
            return 'Vision service temporarily unavailable';
        }
    }

    // Alias for backward compatibility
    async fetchCaption(imageData, detailed = false) {
        return this.getCaption(imageData);
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            console.log('📹 Camera stopped');
        }
    }

    isActive() {
        return this.stream !== null;
    }
}

window.VisionManager = VisionManager;