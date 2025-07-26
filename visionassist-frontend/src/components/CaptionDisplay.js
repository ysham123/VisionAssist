import React from 'react';
import './CaptionDisplay.css';

const CaptionDisplay = ({ currentCaption, confidence, captionHistory, isActive }) => {
  // Format confidence as percentage
  const confidencePercent = Math.round(confidence * 100);
  
  // Get confidence color
  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return '#28a745'; // Green
    if (conf >= 0.6) return '#ffc107'; // Yellow
    return '#dc3545'; // Red
  };

  return (
    <div className="caption-display">
      <div className="caption-container">
        <div className="caption-header">
          <h2 className="caption-title">
            <span className="caption-icon">💬</span>
            Live Caption
          </h2>
          {isActive && (
            <div className="confidence-badge" style={{ color: getConfidenceColor(confidence) }}>
              <span className="confidence-icon">🎯</span>
              {confidencePercent}%
            </div>
          )}
        </div>

        {/* Current Caption */}
        <div className="current-caption-section">
          <div className="current-caption" role="region" aria-live="polite" aria-label="Current caption">
            <div className="caption-text">
              {currentCaption}
            </div>
            
            {isActive && (
              <div className="caption-meta">
                <span className="caption-time">
                  {new Date().toLocaleTimeString()}
                </span>
                <span className="caption-confidence">
                  Confidence: {confidencePercent}%
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Caption History */}
        <div className="caption-history-section">
          <h3 className="history-title">
            <span className="history-icon">📝</span>
            Recent Captions ({captionHistory.length})
          </h3>
          
          <div className="caption-history" role="region" aria-label="Caption history">
            {captionHistory.length > 0 ? (
              <ul className="history-list">
                {captionHistory.map((entry) => (
                  <li key={entry.id} className="history-item">
                    <div className="history-content">
                      <div className="history-text">{entry.caption}</div>
                      <div className="history-meta">
                        <span className="history-time">{entry.timestamp}</span>
                        <span 
                          className="history-confidence"
                          style={{ color: getConfidenceColor(entry.confidence) }}
                        >
                          {Math.round(entry.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="history-empty">
                <span className="empty-icon">📋</span>
                <p>No captions yet. Start VisionAssist to begin generating captions.</p>
              </div>
            )}
          </div>
        </div>

        {/* Caption Stats */}
        {captionHistory.length > 0 && (
          <div className="caption-stats">
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-value">{captionHistory.length}</span>
                <span className="stat-label">Total Captions</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">
                  {Math.round(
                    captionHistory.reduce((sum, entry) => sum + entry.confidence, 0) / 
                    captionHistory.length * 100
                  )}%
                </span>
                <span className="stat-label">Avg Confidence</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">
                  {Math.round(
                    captionHistory.reduce((sum, entry) => sum + entry.caption.length, 0) / 
                    captionHistory.length
                  )}
                </span>
                <span className="stat-label">Avg Length</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CaptionDisplay;
