import React from 'react';
import './Footer.css';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer" role="contentinfo">
      <div className="container">
        <div className="footer-content">
          <div className="footer-section">
            <div className="footer-logo">
              <span className="footer-logo-icon">👁️</span>
              <span className="footer-logo-text">VisionAssist</span>
            </div>
            <p className="footer-description">
              Empowering visual accessibility through AI-powered real-time image captioning.
              Helping 2.2 billion visually impaired users worldwide.
            </p>
          </div>

          <div className="footer-section">
            <h3 className="footer-title">Features</h3>
            <ul className="footer-links">
              <li><a href="#real-time" className="footer-link">Real-time Captioning</a></li>
              <li><a href="#ai-powered" className="footer-link">AI-Powered Recognition</a></li>
              <li><a href="#audio-feedback" className="footer-link">Audio Feedback</a></li>
              <li><a href="#accessibility" className="footer-link">Full Accessibility</a></li>
            </ul>
          </div>

          <div className="footer-section">
            <h3 className="footer-title">Support</h3>
            <ul className="footer-links">
              <li><a href="#help" className="footer-link">Help Center</a></li>
              <li><a href="#tutorials" className="footer-link">Tutorials</a></li>
              <li><a href="#contact" className="footer-link">Contact Us</a></li>
              <li><a href="#feedback" className="footer-link">Feedback</a></li>
            </ul>
          </div>

          <div className="footer-section">
            <h3 className="footer-title">Accessibility</h3>
            <ul className="footer-links">
              <li><a href="#wcag" className="footer-link">WCAG 2.1 AA Compliant</a></li>
              <li><a href="#screen-readers" className="footer-link">Screen Reader Support</a></li>
              <li><a href="#keyboard" className="footer-link">Keyboard Navigation</a></li>
              <li><a href="#high-contrast" className="footer-link">High Contrast Mode</a></li>
            </ul>
          </div>
        </div>

        <div className="footer-bottom">
          <div className="footer-bottom-content">
            <div className="footer-copyright">
              <p>&copy; {currentYear} VisionAssist. Built with ❤️ for accessibility.</p>
            </div>
            
            <div className="footer-badges">
              <div className="badge">
                <span className="badge-icon">🌟</span>
                <span className="badge-text">Production Ready</span>
              </div>
              <div className="badge">
                <span className="badge-icon">♿</span>
                <span className="badge-text">Accessible</span>
              </div>
              <div className="badge">
                <span className="badge-icon">🚀</span>
                <span className="badge-text">Real-time AI</span>
              </div>
            </div>

            <div className="footer-tech">
              <span className="tech-label">Powered by:</span>
              <div className="tech-stack">
                <span className="tech-item">React</span>
                <span className="tech-item">BLIP AI</span>
                <span className="tech-item">WebRTC</span>
                <span className="tech-item">Speech API</span>
              </div>
            </div>
          </div>
        </div>

        {/* Accessibility Statement */}
        <div className="accessibility-statement">
          <p>
            <strong>Accessibility Commitment:</strong> VisionAssist is designed with accessibility as a core principle. 
            We follow WCAG 2.1 AA guidelines and continuously work to improve the experience for all users. 
            If you encounter any accessibility barriers, please <a href="#contact" className="footer-link">contact us</a>.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
