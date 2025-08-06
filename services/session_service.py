"""
Session Service
Handles session management and cleanup
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

class SessionService:
    """Service for handling session management"""
    
    def __init__(self):
        self.sessions = {}
        self._session_timeout = 24 * 3600  # 24 hours
        
    def create_session(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        
        session = {
            'id': session_id,
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'metadata': {}
        }
        
        self.sessions[session_id] = session
        logger.info(f"Created session: {session_id}")
        
        return {
            'success': True,
            'session_id': session_id,
            'created_at': session['created_at'].isoformat()
        }
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def update_activity(self, session_id: str):
        """Update session last activity"""
        if session_id in self.sessions:
            self.sessions[session_id]['last_activity'] = datetime.now()
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if (current_time - session['last_activity']).total_seconds() > self._session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
        
        return len(expired_sessions)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            'service': 'session',
            'status': 'healthy',
            'active_sessions': len(self.sessions),
            'timestamp': datetime.now().isoformat()
        }
