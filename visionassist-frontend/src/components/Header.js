import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header" role="banner">
      <div className="container">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo">
              <span className="logo-icon">👁️</span>
              <h1 className="logo-text">VisionAssist</h1>
            </div>
            <p className="tagline">AI-Powered Real-Time Vision Assistant</p>
          </div>
          
          <nav className="nav" role="navigation" aria-label="Main navigation">
            <ul className="nav-list">
              <li className="nav-item">
                <a href="#about" className="nav-link">About</a>
              </li>
              <li className="nav-item">
                <a href="#features" className="nav-link">Features</a>
              </li>
              <li className="nav-item">
                <a href="#accessibility" className="nav-link">Accessibility</a>
              </li>
              <li className="nav-item">
                <a href="#help" className="nav-link">Help</a>
              </li>
            </ul>
          </nav>
        </div>
        
        <div className="header-info">
          <div className="info-badge">
            <span className="badge-icon">🌟</span>
            <span className="badge-text">Production Ready</span>
          </div>
          <div className="info-badge">
            <span className="badge-icon">♿</span>
            <span className="badge-text">WCAG 2.1 AA</span>
          </div>
          <div className="info-badge">
            <span className="badge-icon">🚀</span>
            <span className="badge-text">Real-Time AI</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
