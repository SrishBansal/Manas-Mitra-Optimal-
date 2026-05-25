'use client';

import { useState } from 'react';
import { Heart, MessageCircle, Shield, Users, Sparkles, ArrowRight, CheckCircle } from 'lucide-react';

interface WelcomeScreenProps {
  onStartChat: () => void;
}

export default function WelcomeScreen({ onStartChat }: WelcomeScreenProps) {
  const [showFeatures, setShowFeatures] = useState(false);

  const features = [
    {
      icon: MessageCircle,
      title: 'AI-Powered Conversations',
      description: 'Engage in meaningful conversations with our empathetic AI companion designed to understand and support your mental health journey.',
      color: 'text-primary'
    },
    {
      icon: Shield,
      title: 'Anonymous & Secure',
      description: 'Your privacy is our priority. All conversations are anonymous and secure, with no personal data stored.',
      color: 'text-primary'
    },
    {
      icon: Users,
      title: 'Professional Referrals',
      description: 'When needed, we can connect you with qualified mental health professionals in your area.',
      color: 'text-primary'
    },
    {
      icon: Sparkles,
      title: 'Relaxation Techniques',
      description: 'Access guided breathing exercises, meditation, and other proven relaxation techniques.',
      color: 'text-primary'
    }
  ];

  const benefits = [
    '24/7 availability for support',
    'Evidence-based psychological assessments',
    'Personalized coping strategies',
    'Crisis intervention guidance',
    'Progress tracking and insights'
  ];

  return (
    <div className="max-w-4xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-12 animate-fade-in">
        <div className="inline-flex items-center justify-center w-20 h-20 bg-primary rounded-full mb-6 shadow-2xl">
          <Heart className="h-10 w-10 text-primary-foreground" />
        </div>

        <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-6">
          Welcome to{' '}
          <span className="text-primary">
            Manas Mitra
          </span>
        </h1>

        <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
          Your compassionate AI companion for mental health support.
          Get immediate help, guidance, and professional referrals when you need them most.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
          <button
            onClick={onStartChat}
            className="group bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 shadow-lg shadow-primary/25 flex items-center space-x-2"
          >
            <span>Start Your Journey</span>
            <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
          </button>

          <button
            onClick={() => setShowFeatures(!showFeatures)}
            className="px-6 py-3 border-2 border-primary/20 text-primary rounded-xl font-medium hover:bg-secondary transition-all duration-300"
          >
            {showFeatures ? 'Hide Features' : 'Learn More'}
          </button>
        </div>
      </div>

      {/* Features Section */}
      {showFeatures && (
        <div className="grid md:grid-cols-2 gap-6 mb-12 animate-fade-in">
          {features.map((feature, index) => (
            <div
              key={index}
              className="bg-card p-6 rounded-xl border border-border hover:shadow-lg hover:shadow-primary/10 transition-all duration-300"
            >
              <div className="flex items-start space-x-4">
                <div className={`p-3 rounded-lg bg-secondary`}>
                  <feature.icon className={`h-6 w-6 ${feature.color}`} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-muted-foreground">
                    {feature.description}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Benefits Section */}
      <div className="bg-card rounded-2xl p-8 border border-border shadow-lg">
        <h2 className="text-2xl font-bold text-foreground mb-6 text-center">
          How Manas Mitra Can Help You
        </h2>

        <div className="grid md:grid-cols-2 gap-4 mb-8">
          {benefits.map((benefit, index) => (
            <div key={index} className="flex items-center space-x-3">
              <CheckCircle className="h-5 w-5 text-success flex-shrink-0" />
              <span className="text-foreground">{benefit}</span>
            </div>
          ))}
        </div>

        <div className="flex flex-col items-center gap-6">
          <div className="inline-flex items-center space-x-2 text-sm text-muted-foreground">
            <Shield className="h-4 w-4" />
            <span>Your conversations are completely anonymous and secure</span>
          </div>

          <button
            onClick={onStartChat}
            className="bg-primary hover:bg-primary/90 text-primary-foreground px-6 py-3 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 shadow-lg shadow-primary/25"
          >
            Begin Your Mental Health Journey
          </button>
        </div>
      </div>

      {/* Safety Notice */}
      <div className="mt-8 p-6 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl shadow-lg">
        <div className="flex items-start space-x-3">
          <Shield className="h-6 w-6 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-amber-800 dark:text-amber-200 mb-2">
              Important Safety Information
            </h3>
            <p className="text-amber-700 dark:text-amber-300 text-sm">
              Manas Mitra is designed to provide support and guidance, but it&apos;s not a replacement for professional medical care.
              If you&apos;re experiencing a mental health crisis or having thoughts of self-harm, please contact emergency services
              or a mental health professional immediately.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
