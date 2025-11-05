'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2 } from 'lucide-react';

type Message = {
  role: 'user' | 'assistant';
  content: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from server');
      }

      const data = await response.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [
        ...prev,
        { 
          role: 'assistant', 
          content: 'Sorry, I encountered an error. Please try again.' 
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-indigo-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto p-4">
          <h1 className="text-2xl font-bold text-indigo-600">Manas Mitra</h1>
          <p className="text-gray-600">Your mental health companion</p>
        </div>
      </header>

      <main className="max-w-4xl mx-auto p-4 pb-24">
        <div className="bg-white rounded-xl shadow-md p-4 h-[70vh] overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center text-gray-500">
              <Bot className="w-12 h-12 mb-4 text-indigo-400" />
              <h2 className="text-xl font-medium mb-2">Welcome to Manas Mitra</h2>
              <p className="max-w-md">I'm here to support your mental wellbeing. How can I help you today?</p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`flex items-start max-w-3xl ${
                      message.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                    }`}
                  >
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                        message.role === 'user' ? 'bg-indigo-100 text-indigo-600 ml-3' : 'bg-indigo-600 text-white mr-3'
                      }`}
                    >
                      {message.role === 'user' ? (
                        <User className="w-4 h-4" />
                      ) : (
                        <Bot className="w-4 h-4" />
                      )}
                    </div>
                    <div
                      className={`p-4 rounded-2xl ${
                        message.role === 'user'
                          ? 'bg-indigo-600 text-white rounded-tr-none'
                          : 'bg-gray-100 text-gray-800 rounded-tl-none'
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex items-center justify-start">
                  <div className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center mr-3">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl rounded-tl-none">
                    <Loader2 className="w-5 h-5 animate-spin text-indigo-600" />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="mt-4 flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 p-4 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-indigo-600 text-white p-4 rounded-full hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </main>
    </div>
  );
}
