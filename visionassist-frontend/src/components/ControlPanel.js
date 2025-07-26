import React from 'react';
import './ControlPanel.css';

const ControlPanel = ({ 
  isActive, 
  isLoading, 
  audioEnabled, 
  onStart, 
  onStop, 
  onToggleAudio, 
  onClearHistory, 
  onSaveCaption 
}) => {
  return (
    <div className="control-panel">
      <div className="control-container">
        <div className="control-header">
          <h2 className="control-title">
            <span className="control-icon">🎛️</span>
            VisionAssist Controls
          </h2>
        </div>

        <div className="control-grid">
          {/* Primary Controls */}
          <div className="control-section primary-controls">
            <h3 className="section-title">Primary Controls</h3>
            <div className="button-group">
              {!isActive ? (
                <button
                  className="btn btn-success btn-large"
                  onClick={onStart}
                  disabled={isLoading}
                  aria-label="Start VisionAssist real-time captioning"
                >
                  {isLoading ? (
                    <>
                      <span className="spinner"></span>
                      Starting...
                    </>
                  ) : (
                    <>
                      <span className="btn-icon">▶️</span>
                      Start VisionAssist
                    </>
                  )}
                </button>
              ) : (
                <button
                  className="btn btn-danger btn-large"
                  onClick={onStop}
                  aria-label="Stop VisionAssist real-time captioning"
                >
                  <span className="btn-icon">⏹️</span>
                  Stop VisionAssist
                </button>
              )}
            </div>
          </div>

          {/* Audio Controls */}
          <div className="control-section audio-controls">
            <h3 className="section-title">Audio Feedback</h3>
            <div className="button-group">
              <button
                className={`btn ${audioEnabled ? 'btn-primary' : 'btn-secondary'}`}
                onClick={onToggleAudio}
                aria-label={`${audioEnabled ? 'Disable' : 'Enable'} audio feedback`}
                aria-pressed={audioEnabled}
              >
                <span className="btn-icon">
                  {audioEnabled ? '🔊' : '🔇'}
                </span>
                {audioEnabled ? 'Audio On' : 'Audio Off'}
              </button>
            </div>
            <p className="control-description">
              {audioEnabled 
                ? 'Captions will be spoken aloud for accessibility'
                : 'Audio feedback is disabled'
              }
            </p>
          </div>

          {/* Caption Actions */}
          <div className="control-section caption-actions">
            <h3 className="section-title">Caption Actions</h3>
            <div className="button-group">
              <button
                className="btn btn-primary"
                onClick={onSaveCaption}
                disabled={!isActive}
                aria-label="Save current caption"
              >
                <span className="btn-icon">💾</span>
                Save Caption
              </button>
              
              <button
                className="btn btn-secondary"
                onClick={onClearHistory}
                aria-label="Clear caption history"
              >
                <span className="btn-icon">🗑️</span>
                Clear History
              </button>
            </div>
          </div>

          {/* Settings */}
          <div className="control-section settings">
            <h3 className="section-title">Settings</h3>
            <div className="settings-grid">
              <div className="setting-item">
                <label className="setting-label" htmlFor="caption-frequency">
                  Caption Frequency
                </label>
                <select 
                  id="caption-frequency" 
                  className="setting-select"
                  defaultValue="3"
                  aria-label="Select caption generation frequency"
                >
                  <option value="1">Every 1 second</option>
                  <option value="2">Every 2 seconds</option>
                  <option value="3">Every 3 seconds</option>
                  <option value="5">Every 5 seconds</option>
                </select>
              </div>
              
              <div className="setting-item">
                <label className="setting-label" htmlFor="speech-rate">
                  Speech Rate
                </label>
                <select 
                  id="speech-rate" 
                  className="setting-select"
                  defaultValue="normal"
                  disabled={!audioEnabled}
                  aria-label="Select speech rate for audio feedback"
                >
                  <option value="slow">Slow</option>
                  <option value="normal">Normal</option>
                  <option value="fast">Fast</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="quick-actions">
          <h3 className="section-title">Quick Actions</h3>
          <div className="quick-buttons">
            <button
              className="btn btn-outline"
              onClick={() => window.open('#help', '_blank')}
              aria-label="Open help documentation"
            >
              <span className="btn-icon">❓</span>
              Help
            </button>
            
            <button
              className="btn btn-outline"
              onClick={() => window.open('#accessibility', '_blank')}
              aria-label="View accessibility features"
            >
              <span className="btn-icon">♿</span>
              Accessibility
            </button>
            
            <button
              className="btn btn-outline"
              onClick={() => window.open('#feedback', '_blank')}
              aria-label="Provide feedback"
            >
              <span className="btn-icon">📝</span>
              Feedback
            </button>
          </div>
        </div>

        {/* Status Information */}
        <div className="status-info">
          <div className="info-grid">
            <div className="info-item">
              <span className="info-icon">🤖</span>
              <div className="info-content">
                <span className="info-title">AI Model</span>
                <span className="info-value">BLIP Production</span>
              </div>
            </div>
            
            <div className="info-item">
              <span className="info-icon">⚡</span>
              <div className="info-content">
                <span className="info-title">Performance</span>
                <span className="info-value">Real-time</span>
              </div>
            </div>
            
            <div className="info-item">
              <span className="info-icon">🎯</span>
              <div className="info-content">
                <span className="info-title">Accuracy</span>
                <span className="info-value">BLEU &gt; 0.7</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
