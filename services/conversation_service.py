"""
Conversation Service
Handles conversational AI and context management
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid

from config import config

logger = logging.getLogger(__name__)

class ConversationService:
    """Service for handling conversational AI operations"""
    
    def __init__(self):
        self.sessions = {}  # In-memory for now, will move to database
        self._cleanup_interval = 3600  # 1 hour
        self._session_timeout = 24 * 3600  # 24 hours
        
    def create_session(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        
        session = {
            'id': session_id,
            'user_id': user_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'messages': [],
            'context': {},
            'metadata': {
                'total_messages': 0,
                'vision_requests': 0,
                'conversation_requests': 0
            }
        }
        
        self.sessions[session_id] = session
        
        logger.info(f"Created conversation session: {session_id}")
        
        return {
            'success': True,
            'session_id': session_id,
            'created_at': session['created_at'].isoformat()
        }
    
    def process_message(self, session_id: str, message: str, 
                            image_caption: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a conversation message with optional visual context
        
        Args:
            session_id: Session identifier
            message: User message
            image_caption: Optional image caption for visual context
            
        Returns:
            AI response with metadata
        """
        try:
            # Get or create session
            session = self._get_or_create_session(session_id)
            
            # Update session activity
            session['last_activity'] = datetime.now()
            
            # Build context-aware prompt
            prompt = self._build_contextual_prompt(message, image_caption, session)
            
            # Generate response (mock for now, will integrate with LLM)
            response = self._generate_response(prompt, session)
            
            # Store conversation in session
            self._store_conversation(session, message, response, image_caption)
            
            return {
                'success': True,
                'response': response,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'context_used': image_caption is not None
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'success': False,
                'error': 'Failed to process conversation message',
                'error_code': 'CONVERSATION_ERROR'
            }
    
    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.create_session()
            # Use the most recently created session
            session_id = max(self.sessions.keys(), key=lambda k: self.sessions[k]['created_at'])
        
        return self.sessions[session_id]
    
    def _build_contextual_prompt(self, message: str, image_caption: Optional[str], 
                               session: Dict[str, Any]) -> str:
        """Build context-aware prompt for AI"""
        prompt_parts = []
        
        # Add system context
        prompt_parts.append(
            "You are VisionAssist, an AI assistant designed to help visually impaired users. "
            "Provide helpful, clear, and concise responses focused on accessibility and navigation."
        )
        
        # Add visual context if available
        if image_caption:
            prompt_parts.append(f"Current visual context: {image_caption}")
        
        # Add recent conversation history (last 5 messages)
        recent_messages = session['messages'][-5:] if session['messages'] else []
        if recent_messages:
            prompt_parts.append("Recent conversation:")
            for msg in recent_messages:
                prompt_parts.append(f"User: {msg['user_message']}")
                prompt_parts.append(f"Assistant: {msg['ai_response']}")
        
        # Add current user message
        prompt_parts.append(f"User question: {message}")
        
        return "\n\n".join(prompt_parts)
    
    def _generate_response(self, prompt: str, session: Dict[str, Any]) -> str:
        """Generate AI response (mock implementation)"""
        # This is a mock implementation - in production, integrate with:
        # - OpenAI API
        # - Ollama
        # - Hugging Face Transformers
        # - Other LLM services
        
        # Simple keyword-based responses for demonstration
        message_lower = prompt.lower()
        
        if any(word in message_lower for word in ['navigate', 'direction', 'where', 'location']):
            return "Based on what I can see, I'll help you navigate safely. Please describe what you're trying to reach or avoid."
        
        elif any(word in message_lower for word in ['read', 'text', 'sign', 'label']):
            return "I can help you read text in images. Please point your camera at the text you'd like me to read."
        
        elif any(word in message_lower for word in ['color', 'what color', 'colors']):
            return "I can identify colors in images. The visual context shows various colors that I can describe in detail."
        
        elif any(word in message_lower for word in ['object', 'what is', 'identify']):
            return "I can identify objects in your environment. Based on the current image, I can describe what I see to help you understand your surroundings."
        
        else:
            return "I'm here to help you with visual assistance. I can describe images, read text, identify objects, and help with navigation. What would you like me to help you with?"
    
    def _store_conversation(self, session: Dict[str, Any], user_message: str, 
                          ai_response: str, image_caption: Optional[str]):
        """Store conversation in session"""
        conversation_entry = {
            'timestamp': datetime.now(),
            'user_message': user_message,
            'ai_response': ai_response,
            'image_caption': image_caption,
            'message_id': str(uuid.uuid4())
        }
        
        session['messages'].append(conversation_entry)
        session['metadata']['total_messages'] += 1
        
        if image_caption:
            session['metadata']['vision_requests'] += 1
        else:
            session['metadata']['conversation_requests'] += 1
    
    def get_session_history(self, session_id: str, limit: int = 50) -> Dict[str, Any]:
        """Get conversation history for a session"""
        if session_id not in self.sessions:
            return {
                'success': False,
                'error': 'Session not found',
                'error_code': 'SESSION_NOT_FOUND'
            }
        
        session = self.sessions[session_id]
        messages = session['messages'][-limit:] if limit else session['messages']
        
        return {
            'success': True,
            'session_id': session_id,
            'messages': [
                {
                    'timestamp': msg['timestamp'].isoformat(),
                    'user_message': msg['user_message'],
                    'ai_response': msg['ai_response'],
                    'had_visual_context': msg['image_caption'] is not None,
                    'message_id': msg['message_id']
                }
                for msg in messages
            ],
            'metadata': session['metadata']
        }
    
    def cleanup_expired_sessions(self):
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
            'service': 'conversation',
            'status': 'healthy',
            'active_sessions': len(self.sessions),
            'total_messages': sum(s['metadata']['total_messages'] for s in self.sessions.values()),
            'timestamp': datetime.now().isoformat()
        }
