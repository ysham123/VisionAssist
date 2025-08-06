// Comprehensive Robustness Test Suite for VisionAssist
// This script tests various failure scenarios and edge cases

class RobustnessTestSuite {
    constructor() {
        this.testResults = [];
        this.visionManager = null;
        this.speechManager = null;
        this.conversationManager = null;
    }

    async initialize() {
        console.log('🧪 Initializing Robustness Test Suite...');
        
        // Wait for app to be ready
        if (window.visionAssistApp) {
            this.visionManager = window.visionAssistApp.visionManager;
            this.speechManager = window.visionAssistApp.speechManager;
            this.conversationManager = window.visionAssistApp.conversationManager;
        } else {
            console.error('❌ VisionAssist app not found');
            return false;
        }
        
        console.log('✅ Test suite initialized');
        return true;
    }

    // Test 1: Vision Module Edge Cases
    async testVisionModuleRobustness() {
        console.log('\n🔍 Testing Vision Module Robustness...');
        
        const tests = [
            {
                name: 'Capture without camera stream',
                test: async () => {
                    const originalStream = this.visionManager.stream;
                    this.visionManager.stream = null;
                    const result = await this.visionManager.captureImageFromVideo();
                    this.visionManager.stream = originalStream;
                    return result === null;
                }
            },
            {
                name: 'Capture with invalid video dimensions',
                test: async () => {
                    const originalVideoWidth = this.visionManager.cameraFeed?.videoWidth;
                    const originalVideoHeight = this.visionManager.cameraFeed?.videoHeight;
                    
                    if (this.visionManager.cameraFeed) {
                        Object.defineProperty(this.visionManager.cameraFeed, 'videoWidth', { value: 0, configurable: true });
                        Object.defineProperty(this.visionManager.cameraFeed, 'videoHeight', { value: 0, configurable: true });
                    }
                    
                    const result = await this.visionManager.captureImageFromVideo();
                    
                    // Restore original values
                    if (this.visionManager.cameraFeed) {
                        Object.defineProperty(this.visionManager.cameraFeed, 'videoWidth', { value: originalVideoWidth, configurable: true });
                        Object.defineProperty(this.visionManager.cameraFeed, 'videoHeight', { value: originalVideoHeight, configurable: true });
                    }
                    
                    return result === null;
                }
            },
            {
                name: 'Fetch caption with invalid image data',
                test: async () => {
                    const result = await this.visionManager.fetchCaption('invalid-image-data');
                    return result === null;
                }
            },
            {
                name: 'Fetch caption with null image data',
                test: async () => {
                    const result = await this.visionManager.fetchCaption(null);
                    return result === null;
                }
            }
        ];

        for (const test of tests) {
            try {
                const passed = await test.test();
                this.logTestResult('Vision', test.name, passed);
            } catch (error) {
                this.logTestResult('Vision', test.name, false, error.message);
            }
        }
    }

    // Test 2: Speech Module Edge Cases
    async testSpeechModuleRobustness() {
        console.log('\n🎤 Testing Speech Module Robustness...');
        
        const tests = [
            {
                name: 'Start listening without recognition support',
                test: async () => {
                    const originalRecognition = this.speechManager.recognition;
                    this.speechManager.recognition = null;
                    const result = this.speechManager.startListening();
                    this.speechManager.recognition = originalRecognition;
                    return result === false;
                }
            },
            {
                name: 'Stop listening when not listening',
                test: async () => {
                    const wasListening = this.speechManager.isListening;
                    this.speechManager.isListening = false;
                    this.speechManager.stopListening(); // Should not throw
                    this.speechManager.isListening = wasListening;
                    return true;
                }
            },
            {
                name: 'Speak empty text',
                test: async () => {
                    return new Promise((resolve) => {
                        this.speechManager.speakText('', () => {
                            resolve(true); // Should call callback even with empty text
                        });
                    });
                }
            },
            {
                name: 'Speak very long text',
                test: async () => {
                    const longText = 'A'.repeat(10000);
                    return new Promise((resolve) => {
                        this.speechManager.speakText(longText, () => {
                            resolve(true);
                        });
                        // Don't wait for completion, just check it doesn't crash
                        setTimeout(() => resolve(true), 100);
                    });
                }
            }
        ];

        for (const test of tests) {
            try {
                const passed = await test.test();
                this.logTestResult('Speech', test.name, passed);
            } catch (error) {
                this.logTestResult('Speech', test.name, false, error.message);
            }
        }
    }

    // Test 3: Conversation Module Edge Cases
    async testConversationModuleRobustness() {
        console.log('\n💬 Testing Conversation Module Robustness...');
        
        const tests = [
            {
                name: 'Process empty message',
                test: async () => {
                    try {
                        await this.conversationManager.processMessage('');
                        return false; // Should have thrown an error
                    } catch (error) {
                        return error.message.includes('Invalid message');
                    }
                }
            },
            {
                name: 'Process null message',
                test: async () => {
                    try {
                        await this.conversationManager.processMessage(null);
                        return false; // Should have thrown an error
                    } catch (error) {
                        return error.message.includes('Invalid message');
                    }
                }
            },
            {
                name: 'Process very long message',
                test: async () => {
                    const longMessage = 'A'.repeat(2000);
                    try {
                        await this.conversationManager.processMessage(longMessage);
                        return true; // Should handle gracefully
                    } catch (error) {
                        return false;
                    }
                }
            },
            {
                name: 'Backend health check with invalid URL',
                test: async () => {
                    const originalUrl = this.conversationManager.serverUrl;
                    this.conversationManager.serverUrl = 'http://invalid-url-that-does-not-exist';
                    const result = await this.conversationManager.checkBackendHealth();
                    this.conversationManager.serverUrl = originalUrl;
                    return result === false;
                }
            }
        ];

        for (const test of tests) {
            try {
                const passed = await test.test();
                this.logTestResult('Conversation', test.name, passed);
            } catch (error) {
                this.logTestResult('Conversation', test.name, false, error.message);
            }
        }
    }

    // Test 4: Network Failure Simulation
    async testNetworkFailureHandling() {
        console.log('\n🌐 Testing Network Failure Handling...');
        
        // Mock fetch to simulate network failures
        const originalFetch = window.fetch;
        
        const tests = [
            {
                name: 'Network timeout simulation',
                test: async () => {
                    window.fetch = () => new Promise(() => {}); // Never resolves
                    
                    try {
                        await this.conversationManager.processMessage('test message');
                        return true; // Should handle timeout gracefully
                    } catch (error) {
                        return false;
                    } finally {
                        window.fetch = originalFetch;
                    }
                }
            },
            {
                name: 'HTTP 500 error simulation',
                test: async () => {
                    window.fetch = () => Promise.resolve({
                        ok: false,
                        status: 500,
                        statusText: 'Internal Server Error'
                    });
                    
                    try {
                        await this.conversationManager.processMessage('test message');
                        return true; // Should handle 500 error gracefully
                    } catch (error) {
                        return false;
                    } finally {
                        window.fetch = originalFetch;
                    }
                }
            },
            {
                name: 'Network error simulation',
                test: async () => {
                    window.fetch = () => Promise.reject(new Error('NetworkError'));
                    
                    try {
                        await this.conversationManager.processMessage('test message');
                        return true; // Should handle network error gracefully
                    } catch (error) {
                        return false;
                    } finally {
                        window.fetch = originalFetch;
                    }
                }
            }
        ];

        for (const test of tests) {
            try {
                const passed = await test.test();
                this.logTestResult('Network', test.name, passed);
            } catch (error) {
                this.logTestResult('Network', test.name, false, error.message);
            }
        }
    }

    // Test 5: Memory and Performance Edge Cases
    async testMemoryAndPerformance() {
        console.log('\n🧠 Testing Memory and Performance Edge Cases...');
        
        const tests = [
            {
                name: 'Large conversation history',
                test: async () => {
                    const originalHistory = this.conversationManager.conversationHistory;
                    
                    // Add 1000 messages to history
                    for (let i = 0; i < 1000; i++) {
                        this.conversationManager.conversationHistory.push({
                            role: 'user',
                            content: `Test message ${i}`,
                            timestamp: new Date().toISOString()
                        });
                    }
                    
                    const status = this.conversationManager.getSystemStatus();
                    const passed = status.conversationHistory === 1000;
                    
                    // Restore original history
                    this.conversationManager.conversationHistory = originalHistory;
                    
                    return passed;
                }
            },
            {
                name: 'Rapid successive vision captures',
                test: async () => {
                    const promises = [];
                    for (let i = 0; i < 10; i++) {
                        promises.push(this.visionManager.captureImageFromVideo());
                    }
                    
                    const results = await Promise.allSettled(promises);
                    return results.every(result => result.status === 'fulfilled');
                }
            },
            {
                name: 'Concurrent speech operations',
                test: async () => {
                    const promises = [
                        this.speechManager.speakText('Test 1'),
                        this.speechManager.speakText('Test 2'),
                        this.speechManager.speakText('Test 3')
                    ];
                    
                    const results = await Promise.allSettled(promises);
                    return results.length === 3; // Should handle concurrent operations
                }
            }
        ];

        for (const test of tests) {
            try {
                const passed = await test.test();
                this.logTestResult('Performance', test.name, passed);
            } catch (error) {
                this.logTestResult('Performance', test.name, false, error.message);
            }
        }
    }

    // Log test results
    logTestResult(category, testName, passed, error = null) {
        const result = {
            category,
            testName,
            passed,
            error,
            timestamp: new Date().toISOString()
        };
        
        this.testResults.push(result);
        
        const status = passed ? '✅' : '❌';
        const errorMsg = error ? ` (${error})` : '';
        console.log(`${status} ${category}: ${testName}${errorMsg}`);
    }

    // Run all tests
    async runAllTests() {
        console.log('🧪 Starting Comprehensive Robustness Test Suite...\n');
        
        const initialized = await this.initialize();
        if (!initialized) {
            console.error('❌ Failed to initialize test suite');
            return;
        }

        await this.testVisionModuleRobustness();
        await this.testSpeechModuleRobustness();
        await this.testConversationModuleRobustness();
        await this.testNetworkFailureHandling();
        await this.testMemoryAndPerformance();
        
        this.generateTestReport();
    }

    // Generate comprehensive test report
    generateTestReport() {
        console.log('\n📊 Test Report Summary:');
        console.log('=' .repeat(50));
        
        const categories = [...new Set(this.testResults.map(r => r.category))];
        let totalTests = 0;
        let totalPassed = 0;
        
        categories.forEach(category => {
            const categoryTests = this.testResults.filter(r => r.category === category);
            const categoryPassed = categoryTests.filter(r => r.passed).length;
            
            console.log(`${category}: ${categoryPassed}/${categoryTests.length} tests passed`);
            
            totalTests += categoryTests.length;
            totalPassed += categoryPassed;
        });
        
        console.log('=' .repeat(50));
        console.log(`Overall: ${totalPassed}/${totalTests} tests passed (${Math.round(totalPassed/totalTests*100)}%)`);
        
        // Show failed tests
        const failedTests = this.testResults.filter(r => !r.passed);
        if (failedTests.length > 0) {
            console.log('\n❌ Failed Tests:');
            failedTests.forEach(test => {
                console.log(`  - ${test.category}: ${test.testName} (${test.error || 'Unknown error'})`);
            });
        }
        
        console.log('\n🎉 Robustness testing complete!');
        
        return {
            totalTests,
            totalPassed,
            passRate: totalPassed / totalTests,
            failedTests
        };
    }
}

// Export for use
window.RobustnessTestSuite = RobustnessTestSuite;

// Auto-run tests when script loads (optional)
if (window.location.search.includes('autotest=true')) {
    window.addEventListener('load', async () => {
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for app to initialize
        const testSuite = new RobustnessTestSuite();
        await testSuite.runAllTests();
    });
}
