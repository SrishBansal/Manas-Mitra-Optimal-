'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Heart, MessageSquare, BookOpen, Phone, Menu, X, Sparkles, User, Bot, Moon, Sun, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
};

const SUGGESTED_PROMPTS = [
  "I'm feeling anxious today",
  "Help me relax",
  "I need someone to talk to",
  "Tips for better sleep",
];

const MENTAL_HEALTH_RESOURCES = [
  { name: "KIRAN Mental Health Rehab", number: "1800-599-0019", description: "24/7 mental health support" },
  { name: "Vandrevala Foundation", number: "1860-266-2345", description: "Crisis intervention and emotional support" },
  { name: "Snehi", number: "+91-22-2772-6771", description: "Suicide prevention helpline" },
  { name: "iCall", number: "+91-22-2556-3291", description: "Psychosocial helpline (Mon-Sat, 8am-10pm)" },
];

// Strip markdown formatting from text
const stripMarkdown = (text: string): string => {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1')  // Bold **text**
    .replace(/\*(.*?)\*/g, '$1')      // Italic *text*
    .replace(/__(.*?)__/g, '$1')      // Bold __text__
    .replace(/_(.*?)_/g, '$1')        // Italic _text_
    .trim();
};

export default function ManasMitraChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [resourcesOpen, setResourcesOpen] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Load theme from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null;
    if (savedTheme) {
      setTheme(savedTheme);
      document.documentElement.setAttribute('data-theme', savedTheme);
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg.content }),
      });

      const data = await response.json();

      const rawReply = data.reply || data.error || "I'm having trouble connecting right now. Please try again.";
      const cleanedReply = stripMarkdown(rawReply);

      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: cleanedReply,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestedPrompt = (prompt: string) => {
    setInput(prompt);
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 fixed lg:relative z-30 w-64 h-full glass-card border-r border-calm-border flex flex-col`}>
        {/* Sidebar Header */}
        <div className="p-6 border-b border-calm-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center shadow-lg">
                <Heart size={22} className="text-white" fill="white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-calm-text">Manas Mitra</h1>
                <p className="text-xs text-calm-muted">AI Companion</p>
              </div>
            </div>
            <button onClick={() => setSidebarOpen(false)} className="lg:hidden text-calm-muted hover:text-calm-text transition-colors">
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Sidebar Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 font-medium transition-all hover:shadow-md">
            <MessageSquare size={18} />
            <span>New Chat</span>
          </button>
          <button
            onClick={() => setResourcesOpen(true)}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-700/50 text-calm-muted hover:text-calm-text transition-all"
          >
            <BookOpen size={18} />
            <span>Resources</span>
          </button>
        </nav>

        {/* Emergency Call */}
        <div className="p-4 border-t border-calm-border">
          <a
            href="tel:1800-599-0019"
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/50 font-medium transition-all btn-hover"
          >
            <Phone size={16} />
            <span className="text-sm">Crisis: 1800-599-0019</span>
          </a>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-20 lg:hidden backdrop-blur-sm"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Resources Modal */}
      <AnimatePresence>
        {resourcesOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
              onClick={() => setResourcesOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="fixed inset-0 z-50 flex items-center justify-center p-4"
            >
              <div className="glass-card max-w-lg w-full p-6 rounded-2xl max-h-[80vh] overflow-y-auto">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Info size={24} className="text-primary-500" />
                    <h2 className="text-xl font-bold text-calm-text">Mental Health Resources</h2>
                  </div>
                  <button
                    onClick={() => setResourcesOpen(false)}
                    className="text-calm-muted hover:text-calm-text transition-colors"
                  >
                    <X size={24} />
                  </button>
                </div>
                <p className="text-sm text-calm-muted mb-6">
                  Professional help is available 24/7. Don't hesitate to reach out.
                </p>
                <div className="space-y-4">
                  {MENTAL_HEALTH_RESOURCES.map((resource, idx) => (
                    <div key={idx} className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-calm-border">
                      <h3 className="font-semibold text-calm-text mb-1">{resource.name}</h3>
                      <a
                        href={`tel:${resource.number}`}
                        className="text-primary-600 dark:text-primary-400 font-mono text-sm hover:underline"
                      >
                        {resource.number}
                      </a>
                      <p className="text-xs text-calm-muted mt-1">{resource.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen">
        {/* Header */}
        <header className="glass-card border-b border-calm-border p-4 flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(true)}
              className="lg:hidden text-calm-muted hover:text-calm-text transition-colors"
            >
              <Menu size={24} />
            </button>
            <div>
              <h2 className="text-lg font-semibold text-calm-text">Chat with Manas Mitra</h2>
              <p className="text-xs text-calm-muted">Your empathetic AI companion</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-xs text-calm-muted hidden sm:inline">Online</span>
            </div>
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700/50 text-calm-muted hover:text-calm-text transition-all btn-hover"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
            </button>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto scrollbar-thin p-4 md:p-6">
          <div className="max-w-4xl mx-auto">
            {messages.length === 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="h-full flex flex-col items-center justify-center text-center py-12 space-y-6"
              >
                <div className="relative">
                  <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-primary-100 to-primary-200 dark:from-primary-900/50 dark:to-primary-800/50 flex items-center justify-center shadow-2xl">
                    <Heart size={44} className="text-primary-600 dark:text-primary-400" />
                  </div>
                  <div className="absolute -top-2 -right-2 w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-600 rounded-full flex items-center justify-center shadow-lg">
                    <Sparkles size={16} className="text-white" />
                  </div>
                </div>

                <div className="space-y-2">
                  <h3 className="text-3xl font-bold text-calm-text bg-gradient-to-r from-primary-600 to-primary-500 bg-clip-text text-transparent">
                    Welcome to Manas Mitra
                  </h3>
                  <p className="text-calm-muted max-w-md text-lg">
                    "You don't have to carry it all alone. I'm here to listen without judgment."
                  </p>
                </div>

                <div className="w-full max-w-2xl">
                  <p className="text-sm font-medium text-calm-muted mb-4">Try one of these:</p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {SUGGESTED_PROMPTS.map((prompt, idx) => (
                      <motion.button
                        key={idx}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        onClick={() => handleSuggestedPrompt(prompt)}
                        className="glass-card px-5 py-4 rounded-xl text-sm text-left hover:bg-primary-50 dark:hover:bg-primary-900/30 hover:border-primary-200 dark:hover:border-primary-700 transition-all group btn-hover shadow-md"
                      >
                        <span className="text-calm-text group-hover:text-primary-700 dark:group-hover:text-primary-400 font-medium">{prompt}</span>
                      </motion.button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            <AnimatePresence mode="popLayout">
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ duration: 0.2 }}
                  className={`flex gap-3 mb-6 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {msg.role === 'assistant' && (
                    <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center shadow-md">
                      <Bot size={18} className="text-white" />
                    </div>
                  )}

                  <div
                    className={`max-w-[75%] md:max-w-[65%] ${msg.role === 'user'
                        ? 'glass-card bg-gradient-to-br from-primary-500 to-primary-600 text-white border-primary-400 shadow-lg'
                        : msg.content.startsWith('System:') || msg.content.includes('trouble connecting')
                          ? 'glass-card bg-red-50 dark:bg-red-900/30 text-red-800 dark:text-red-300 border-red-200 dark:border-red-700'
                          : 'glass-card text-calm-text shadow-md'
                      } px-5 py-4 rounded-2xl ${msg.role === 'user' ? 'rounded-br-md' : 'rounded-bl-md'}`}
                  >
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                    <p className={`text-xs mt-2 ${msg.role === 'user' ? 'text-primary-100' : 'text-calm-muted'}`}>
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>

                  {msg.role === 'user' && (
                    <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-slate-200 to-slate-300 dark:from-slate-600 dark:to-slate-700 flex items-center justify-center shadow-md">
                      <User size={18} className="text-slate-600 dark:text-slate-300" />
                    </div>
                  )}
                </motion.div>
              ))}

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex gap-3 mb-6"
                >
                  <div className="flex-shrink-0 w-9 h-9 rounded-full bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center shadow-md">
                    <Bot size={18} className="text-white" />
                  </div>
                  <div className="glass-card px-6 py-4 rounded-2xl rounded-bl-md flex items-center gap-2 shadow-md">
                    <span className="w-2 h-2 bg-primary-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                    <span className="w-2 h-2 bg-primary-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                    <span className="w-2 h-2 bg-primary-400 rounded-full animate-bounce"></span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={messagesEndRef} className="h-4" />
          </div>
        </div>

        {/* Input Area */}
        <footer className="glass-card border-t border-calm-border p-4 shadow-lg">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSubmit} className="flex items-end gap-3">
              <div className="flex-1 glass-card rounded-2xl px-5 py-4 border border-calm-border focus-within:border-primary-400 dark:focus-within:border-primary-500 focus-within:shadow-lg transition-all">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                  placeholder="Type your message... (Shift+Enter for new line)"
                  className="w-full bg-transparent focus:outline-none text-calm-text text-sm placeholder:text-calm-muted resize-none"
                  rows={1}
                  disabled={isLoading}
                  style={{ minHeight: '24px', maxHeight: '120px' }}
                />
              </div>
              <button
                type="submit"
                disabled={!input.trim() || isLoading}
                className="p-4 bg-gradient-to-br from-primary-500 to-primary-600 text-white rounded-xl hover:from-primary-600 hover:to-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg hover:shadow-2xl disabled:hover:shadow-lg flex-shrink-0 btn-hover"
              >
                <Send size={20} />
              </button>
            </form>
            <p className="text-[10px] text-center text-calm-muted mt-3 font-medium">
              Manas Mitra is an AI companion, not a replacement for professional help.
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}
