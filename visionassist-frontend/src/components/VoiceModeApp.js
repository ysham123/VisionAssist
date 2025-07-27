/**
 * VoiceModeApp - Enhanced VisionAssist App with Voice Mode Interface
 * Integrates the new voice-first UI with existing backend infrastructure
 */
import React, { useState, useEffect, useContext } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import VoiceModeInterface from './VoiceModeInterface';
import AccessibilityProvider, { useAccessibility } from './AccessibilityProvider';
import ErrorBoundary from './ErrorBoundary';
import { 
  API_CONFIG, 
  FEATURES, 
  validateBase64Image, 
  sanitizeTextInput, 
  getSecureHeaders,
  logger 
} from '../config';

// Authentication context
const AuthContext = React.createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Authentication provider component
const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [tokens, setTokens] = useState({
    accessToken: localStorage.getItem('accessToken'),
    refreshToken: localStorage.getItem('refreshToken')
  });

  // Check authentication status on mount
  useEffect(() => {
    const checkAuth = async () => {
      const accessToken = localStorage.getItem('accessToken');
      
      if (accessToken) {
        try {
          const response = await fetch('/api/v1/auth/verify', {
            headers: {
              'Authorization': `Bearer ${accessToken}`,
              ...getSecureHeaders()
            }
          });

          if (response.ok) {
            const data = await response.json();
            setUser(data.user);
            setIsAuthenticated(true);
          } else {
            // Try to refresh token
            await refreshTokens();
          }
        } catch (error) {
          logger.error('Auth check failed:', error);
          logout();
        }
      }
      
      setIsLoading(false);
    };

    checkAuth();
  }, []);

  // Refresh tokens
  const refreshTokens = async () => {
    const refreshToken = localStorage.getItem('refreshToken');
    
    if (!refreshToken) {
      logout();
      return false;
    }

    try {
      const response = await fetch('/api/v1/auth/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getSecureHeaders()
        },
        body: JSON.stringify({ refresh_token: refreshToken })
      });

      if (response.ok) {
        const data = await response.json();
        const newTokens = data.tokens;
        
        localStorage.setItem('accessToken', newTokens.access_token);
        localStorage.setItem('refreshToken', newTokens.refresh_token);
        
        setTokens({
          accessToken: newTokens.access_token,
          refreshToken: newTokens.refresh_token
        });
        
        return true;
      } else {
        logout();
        return false;
      }
    } catch (error) {
      logger.error('Token refresh failed:', error);
      logout();
      return false;
    }
  };

  // Login function
  const login = async (email, password) => {
    try {
      const response = await fetch('/api/v1/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getSecureHeaders()
        },
        body: JSON.stringify({ 
          email: sanitizeTextInput(email), 
          password 
        })
      });

      const data = await response.json();

      if (response.ok && data.success) {
        const { user: userData, tokens: userTokens } = data;
        
        localStorage.setItem('accessToken', userTokens.access_token);
        localStorage.setItem('refreshToken', userTokens.refresh_token);
        
        setUser(userData);
        setIsAuthenticated(true);
        setTokens({
          accessToken: userTokens.access_token,
          refreshToken: userTokens.refresh_token
        });
        
        logger.info('User logged in successfully');
        return { success: true };
      } else {
        return { 
          success: false, 
          error: data.error || 'Login failed' 
        };
      }
    } catch (error) {
      logger.error('Login error:', error);
      return { 
        success: false, 
        error: 'Network error occurred' 
      };
    }
  };

  // Register function
  const register = async (email, password, firstName, lastName) => {
    try {
      const response = await fetch('/api/v1/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...getSecureHeaders()
        },
        body: JSON.stringify({
          email: sanitizeTextInput(email),
          password,
          first_name: sanitizeTextInput(firstName),
          last_name: sanitizeTextInput(lastName)
        })
      });

      const data = await response.json();

      if (response.ok && data.success) {
        logger.info('User registered successfully');
        return { success: true };
      } else {
        return { 
          success: false, 
          error: data.error || 'Registration failed' 
        };
      }
    } catch (error) {
      logger.error('Registration error:', error);
      return { 
        success: false, 
        error: 'Network error occurred' 
      };
    }
  };

  // Logout function
  const logout = async () => {
    try {
      const accessToken = localStorage.getItem('accessToken');
      const refreshToken = localStorage.getItem('refreshToken');
      
      if (accessToken) {
        await fetch('/api/v1/auth/logout', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Type': 'application/json',
            ...getSecureHeaders()
          },
          body: JSON.stringify({ refresh_token: refreshToken })
        });
      }
    } catch (error) {
      logger.error('Logout error:', error);
    } finally {
      localStorage.removeItem('accessToken');
      localStorage.removeItem('refreshToken');
      setUser(null);
      setIsAuthenticated(false);
      setTokens({ accessToken: null, refreshToken: null });
      logger.info('User logged out');
    }
  };

  const value = {
    user,
    isAuthenticated,
    isLoading,
    tokens,
    login,
    register,
    logout,
    refreshTokens
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Login/Register modal component
const AuthModal = ({ isOpen, onClose, mode, onSwitchMode }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const { login, register } = useAuth();
  const { announce } = useAccessibility();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      let result;
      if (mode === 'login') {
        result = await login(email, password);
      } else {
        result = await register(email, password, firstName, lastName);
      }

      if (result.success) {
        if (mode === 'register') {
          announce('Registration successful. Please log in.');
          onSwitchMode('login');
        } else {
          announce('Login successful');
          onClose();
        }
      } else {
        setError(result.error);
        announce(`Error: ${result.error}`);
      }
    } catch (error) {
      setError('An unexpected error occurred');
      announce('An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 backdrop-blur-sm bg-black/50"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <motion.div
        className="w-full max-w-md p-6 bg-white dark:bg-gray-900 rounded-2xl shadow-xl"
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
      >
        <h2 className="text-2xl font-bold text-center mb-6 text-gray-900 dark:text-white">
          {mode === 'login' ? 'Sign In' : 'Create Account'}
        </h2>

        {error && (
          <div className="mb-4 p-3 bg-red-100 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg">
            <p className="text-red-700 dark:text-red-300 text-sm">{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {mode === 'register' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="firstName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  First Name
                </label>
                <input
                  id="firstName"
                  type="text"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required={mode === 'register'}
                />
              </div>
              <div>
                <label htmlFor="lastName" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Last Name
                </label>
                <input
                  id="lastName"
                  type="text"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required={mode === 'register'}
                />
              </div>
            </div>
          )}

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full p-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
              minLength={8}
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 px-4 bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white font-medium rounded-lg transition-colors focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            {isLoading ? 'Processing...' : (mode === 'login' ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        <div className="mt-6 text-center">
          <button
            onClick={() => onSwitchMode(mode === 'login' ? 'register' : 'login')}
            className="text-blue-500 hover:text-blue-600 text-sm font-medium"
          >
            {mode === 'login' 
              ? "Don't have an account? Sign up" 
              : "Already have an account? Sign in"
            }
          </button>
        </div>

        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          aria-label="Close modal"
        >
          ×
        </button>
      </motion.div>
    </motion.div>
  );
};

// Main app component
const VoiceModeApp = () => {
  return (
    <div className="app">
      {/* Skip link for accessibility */}
      <a 
        href="#main-content" 
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 bg-blue-500 text-white px-4 py-2 rounded-lg z-50"
      >
        Skip to main content
      </a>

      {/* Main content - Direct access to VoiceModeInterface */}
      <main id="main-content" role="main">
        <VoiceModeInterface />
      </main>
    </div>
  );
};

// Root app with providers (no authentication required)
const App = () => {
  return (
    <ErrorBoundary>
      <AccessibilityProvider>
        <VoiceModeApp />
      </AccessibilityProvider>
    </ErrorBoundary>
  );
};

export default App;
