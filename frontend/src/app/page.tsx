'use client';

import { useState, useEffect } from 'react';
import { useTheme } from "next-themes";
import { Heart, MessageCircle, Shield, Users, Sparkles, Moon, Sun } from 'lucide-react';
import ChatInterface from '@/components/ChatInterface';
import WelcomeScreen from '@/components/WelcomeScreen';
import RelaxationTechniques from '@/components/RelaxationTechniques';
import PsychologicalTests from '@/components/PsychologicalTests';
import ProfessionalReferral from '@/components/ProfessionalReferral';

export default function Home() {
  const [currentView, setCurrentView] = useState<'welcome' | 'chat' | 'relaxation' | 'tests' | 'referral'>('welcome');
  const { setTheme, theme } = useTheme();
  // Avoid hydration mismatch by waiting for mount (optional but good practice)
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
      {/* Header */}
      <header className="bg-background/95 backdrop-blur-sm border-b border-border sticky top-0 z-50 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-br from-teal-100 to-cyan-100 dark:from-teal-800 dark:to-cyan-800 rounded-lg">
                <Heart className="h-6 w-6 text-teal-600 dark:text-cyan-400" />
              </div>
              <div>
                <h1 className="text-xl font-semibold bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text text-transparent">
                  Manas Mitra
                </h1>
                <p className="text-sm text-muted-foreground">
                  Your AI Mental Health Companion
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <button
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                className="p-2 rounded-lg bg-secondary text-foreground hover:bg-muted transition-all duration-300"
                aria-label="Toggle theme"
              >
                {theme === "dark" ? (
                  <Sun className="h-5 w-5 text-slate-600 dark:text-slate-400" />
                ) : (
                  <Moon className="h-5 w-5 text-slate-600 dark:text-slate-400" />
                )}
              </button>

              <div className="flex items-center space-x-1 bg-slate-100 dark:bg-slate-800 rounded-lg p-1 shadow-md">
                <button
                  onClick={() => setCurrentView('welcome')}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-300 ${currentView === 'welcome'
                    ? 'bg-primary text-primary-foreground shadow-lg'
                    : 'text-muted-foreground hover:text-primary hover:bg-muted/50'
                    }`}
                >
                  <Sparkles className="h-4 w-4 inline mr-1" />
                  Home
                </button>
                <button
                  onClick={() => setCurrentView('chat')}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-300 ${currentView === 'chat'
                    ? 'bg-teal-600 text-white shadow-lg'
                    : 'text-slate-600 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 hover:bg-white/50 dark:hover:bg-slate-700/50'
                    }`}
                >
                  <MessageCircle className="h-4 w-4 inline mr-1" />
                  Chat
                </button>
                <button
                  onClick={() => setCurrentView('relaxation')}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-300 ${currentView === 'relaxation'
                    ? 'bg-teal-600 text-white shadow-lg'
                    : 'text-slate-600 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 hover:bg-white/50 dark:hover:bg-slate-700/50'
                    }`}
                >
                  <Moon className="h-4 w-4 inline mr-1" />
                  Relax
                </button>
                <button
                  onClick={() => setCurrentView('tests')}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-300 ${currentView === 'tests'
                    ? 'bg-teal-600 text-white shadow-lg'
                    : 'text-slate-600 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 hover:bg-white/50 dark:hover:bg-slate-700/50'
                    }`}
                >
                  <Users className="h-4 w-4 inline mr-1" />
                  Tests
                </button>
                <button
                  onClick={() => setCurrentView('referral')}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-300 ${currentView === 'referral'
                    ? 'bg-teal-600 text-white shadow-lg'
                    : 'text-slate-600 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-400 hover:bg-white/50 dark:hover:bg-slate-700/50'
                    }`}
                >
                  <Shield className="h-4 w-4 inline mr-1" />
                  Support
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentView === 'welcome' && <WelcomeScreen onStartChat={() => setCurrentView('chat')} />}
        {currentView === 'chat' && <ChatInterface />}
        {currentView === 'relaxation' && <RelaxationTechniques />}
        {currentView === 'tests' && <PsychologicalTests />}
        {currentView === 'referral' && <ProfessionalReferral />}
      </main>

      {/* Footer */}
      <footer className="bg-background/95 backdrop-blur-sm border-t border-border mt-16 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-sm text-muted-foreground">
            <p className="mb-2">
              <strong>Important:</strong> This is not a replacement for professional medical advice.
            </p>
            <p>
              If you&apos;re experiencing a mental health crisis, please contact your local emergency services or a mental health professional immediately.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
