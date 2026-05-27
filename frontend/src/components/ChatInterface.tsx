'use client';

import { useState, useRef, useEffect, useMemo } from 'react';
import { Send, Bot, User, Loader2, Heart, AlertTriangle } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  type?: 'text' | 'suggestion' | 'crisis' | 'assessment';
}

type SupportedLang = 'en' | 'hi' | 'mr' | 'kn';

interface ChatInterfaceProps {
  onTriggerAssessment?: () => void;
  onTriggerCrisis?: () => void;
}

export default function ChatInterface({ onTriggerAssessment, onTriggerCrisis }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: uuidv4(),
      content: "Hello! I'm your AI mental health companion. I'm here to listen, support, and help you navigate through whatever you're experiencing. How are you feeling today?",
      sender: 'bot',
      timestamp: new Date(),
      type: 'text'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [userId] = useState(() => uuidv4()); // Anonymous user ID
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [language, setLanguage] = useState<SupportedLang>('en');

  // Cache of translations by message id and language
  const [translations, setTranslations] = useState<Record<string, Partial<Record<SupportedLang, string>>>>({});
  const [isTranslating, setIsTranslating] = useState(false);

  const languageOptions: { label: string; value: SupportedLang }[] = useMemo(
    () => [
      { label: 'English', value: 'en' },
      { label: 'हिन्दी', value: 'hi' },
      { label: 'मराठी', value: 'mr' },
      { label: 'ಕನ್ನಡ', value: 'kn' }
    ],
    []
  );

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Helper: translate array of texts to selected language
  const translateTexts = async (texts: string[], target: SupportedLang, source?: SupportedLang): Promise<string[]> => {
    try {
      const res = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
        cache: 'no-store',
        body: JSON.stringify({ texts, targetLang: target, sourceLang: source || 'en' })
      });
      if (!res.ok) throw new Error('Translate failed');
      const data = await res.json();
      return (data.translations as string[]) || texts;
    } catch (e) {
      console.error('Translation error:', e);
      return texts;
    }
  };

  // Whenever messages or language change, translate any missing items
  useEffect(() => {
    const run = async () => {
      if (!messages.length) return;
      if (language === 'en') return; // originals are English
      const pending = messages.filter((m) => !translations[m.id]?.[language]);
      if (!pending.length) return;
      setIsTranslating(true);
      // Originals are authored in English
      const translated = await translateTexts(pending.map((m) => m.content), language, 'en');
      setTranslations((prev) => {
        const next = { ...prev };
        pending.forEach((m, idx) => {
          next[m.id] = { ...(next[m.id] || {}), [language]: translated[idx] };
        });
        return next;
      });
      setIsTranslating(false);
    };
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, language]);

  // Crisis keywords that trigger immediate professional referral
  const crisisKeywords = [
    'suicide', 'kill myself', 'end it all', 'not worth living', 'want to die',
    'self harm', 'hurt myself', 'crisis', 'emergency', 'help me'
  ];

  // Assessment triggers
  const assessmentTriggers = [
    'anxiety', 'depression', 'stress', 'panic', 'overwhelmed', 'sad', 'worried',
    'fear', 'nervous', 'down', 'hopeless', 'empty'
  ];

  const checkForCrisis = (message: string): boolean => {
    return crisisKeywords.some(keyword => 
      message.toLowerCase().includes(keyword.toLowerCase())
    );
  };

  const checkForAssessment = (message: string): boolean => {
    return assessmentTriggers.some(keyword => 
      message.toLowerCase().includes(keyword.toLowerCase())
    );
  };

  const generateBotResponse = async (userMessage: string): Promise<{content: string, type: string}> => {
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          userId: userId
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from AI');
      }

      const data = await response.json();
      
      let messageType: 'text' | 'suggestion' | 'crisis' | 'assessment' = 'text';
      
      if (data.isCrisis) {
        messageType = 'crisis';
      } else if (data.needsAssessment) {
        messageType = 'assessment';
      }

      return {
        content: data.response,
        type: messageType
      };
    } catch (error) {
      console.error('Error calling AI API:', error);
      return {
        content: "I'm sorry, I'm having trouble processing your message right now. Please try again, and remember that if you're in crisis, please contact emergency services immediately.",
        type: 'text'
      };
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: uuidv4(),
      content: inputValue.trim(),
      sender: 'user',
      timestamp: new Date(),
      type: 'text'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Translate outgoing user message to English for backend intent detection if needed
      let backendMessage = userMessage.content;
      if (language !== 'en') {
        const [translatedToEn] = await translateTexts([userMessage.content], 'en', language);
        backendMessage = translatedToEn || userMessage.content;
      }

      const botResponse = await generateBotResponse(backendMessage);
      
      const botMessage: Message = {
        id: uuidv4(),
        content: botResponse.content,
        sender: 'bot',
        timestamp: new Date(),
        type: botResponse.type as 'text' | 'suggestion' | 'crisis' | 'assessment'
      };

      setMessages(prev => [...prev, botMessage]);

      // Immediately translate the user and bot messages for display if not English
      if (language !== 'en') {
        const toTranslate: { id: string; text: string }[] = [userMessage, botMessage]
          .map((m) => ({ id: m.id, text: m.content }));
        if (toTranslate.length) {
          const translated = await translateTexts(toTranslate.map(t => t.text), language, 'en');
          setTranslations(prev => {
            const next = { ...prev };
            toTranslate.forEach((t, idx) => {
              next[t.id] = { ...(next[t.id] || {}), [language]: translated[idx] };
            });
            return next;
          });
        }
      }

      // Trigger appropriate actions based on message type
      if (botMessage.type === 'crisis' && onTriggerCrisis) {
        setTimeout(() => onTriggerCrisis(), 2000);
      } else if (botMessage.type === 'assessment' && onTriggerAssessment) {
        setTimeout(() => onTriggerAssessment(), 2000);
      }
    } catch (error) {
      console.error('Error generating response:', error);
      const errorMessage: Message = {
        id: uuidv4(),
        content: "I'm sorry, I'm having trouble processing your message right now. Please try again, and remember that if you're in crisis, please contact emergency services immediately.",
        sender: 'bot',
        timestamp: new Date(),
        type: 'text'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Chat Header */}
      <div className="bg-white dark:bg-slate-800 rounded-t-xl border border-slate-200 dark:border-slate-700 p-4 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary/20 rounded-lg">
              <Bot className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h2 className="font-semibold text-primary">
                AI Mental Health Companion
              </h2>
              <p className="text-sm text-slate-600 dark:text-slate-400">
                Anonymous ID: {userId.slice(0, 8)}...
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <label htmlFor="language" className="text-sm text-slate-600 dark:text-slate-300">Language</label>
            <select
              id="language"
              value={language}
              onChange={(e) => setLanguage(e.target.value as SupportedLang)}
              className="border border-slate-300 dark:border-slate-600 rounded-md px-2 py-1 text-sm bg-white dark:bg-slate-700 dark:text-slate-100"
            >
              {languageOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            {isTranslating && (
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
            )}
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="bg-white dark:bg-slate-800 border-x border-slate-200 dark:border-slate-700 h-96 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.sender === 'user'
                  ? 'bg-primary text-primary-foreground shadow-lg'
                  : message.type === 'crisis'
                  ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-800 dark:text-red-200'
                  : message.type === 'assessment'
                  ? 'bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 text-amber-800 dark:text-amber-200'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-slate-100'
              }`}
            >
              <div className="flex items-start space-x-2">
                {message.sender === 'bot' && (
                  <Bot className="h-4 w-4 mt-0.5 flex-shrink-0" />
                )}
                {message.sender === 'user' && (
                  <User className="h-4 w-4 mt-0.5 flex-shrink-0" />
                )}
                <div className="flex-1">
                  <p className="text-sm leading-relaxed">
                    {language === 'en' ? message.content : (translations[message.id]?.[language] || message.content)}
                  </p>
                  {message.type === 'crisis' && (
                    <div className="mt-2 flex items-center space-x-1 text-xs">
                      <AlertTriangle className="h-3 w-3" />
                      <span>Crisis Support Available</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start animate-fade-in">
            <div className="bg-slate-100 dark:bg-slate-700 px-4 py-2 rounded-lg shadow-lg">
              <div className="flex items-center space-x-2">
                <Bot className="h-4 w-4 text-primary" />
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                <span className="text-sm text-slate-600 dark:text-slate-400">Thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white dark:bg-slate-800 rounded-b-xl border border-slate-200 dark:border-slate-700 p-4 shadow-lg">
        <div className="flex space-x-2">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Share what's on your mind... (Press Enter to send, Shift+Enter for new line)"
            className="flex-1 resize-none border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring bg-input text-foreground"
            rows={2}
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="bg-primary hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground text-primary-foreground p-2 rounded-lg transition-all duration-300 transform hover:scale-105 disabled:cursor-not-allowed shadow-lg"
          >
            <Send className="h-5 w-5" />
          </button>
        </div>
        
        <div className="mt-2 text-xs text-slate-600 dark:text-slate-400">
          <Heart className="h-3 w-3 inline mr-1" />
          Your conversations are anonymous and secure. If you&apos;re in crisis, please contact emergency services.
        </div>
      </div>
    </div>
  );
}
