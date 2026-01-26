'use client';

import { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Heart, Wind, Waves, Sun, Moon, Timer } from 'lucide-react';

interface BreathingExercise {
  id: string;
  name: string;
  description: string;
  duration: number; // in seconds
  pattern: {
    inhale: number;
    hold: number;
    exhale: number;
    pause: number;
  };
  icon: React.ComponentType<any>;
  color: string;
}

interface Meditation {
  id: string;
  title: string;
  description: string;
  duration: number; // in minutes
  type: 'mindfulness' | 'body-scan' | 'loving-kindness' | 'progressive-relaxation';
  icon: React.ComponentType<any>;
  color: string;
  steps?: { title: string; instruction: string; duration: number }[]; // duration in seconds
}

export default function RelaxationTechniques() {
  const [activeExercise, setActiveExercise] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentPhase, setCurrentPhase] = useState<'inhale' | 'hold' | 'exhale' | 'pause'>('inhale');
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [selectedMeditation, setSelectedMeditation] = useState<string | null>(null);
  const [phaseRemaining, setPhaseRemaining] = useState(0);
  const [isMeditating, setIsMeditating] = useState(false);
  const [meditationRemaining, setMeditationRemaining] = useState(0);
  const [meditationStepIndex, setMeditationStepIndex] = useState(0);
  const [meditationStepRemaining, setMeditationStepRemaining] = useState(0);

  const breathingExercises: BreathingExercise[] = [
    {
      id: '4-7-8',
      name: '4-7-8 Breathing',
      description: 'A calming technique that helps reduce anxiety and promote sleep',
      duration: 300, // 5 minutes
      pattern: { inhale: 4, hold: 7, exhale: 8, pause: 0 },
      icon: Wind,
      color: 'text-blue-600 dark:text-blue-400'
    },
    {
      id: 'box-breathing',
      name: 'Box Breathing',
      description: 'Military technique for focus and stress reduction',
      duration: 240, // 4 minutes
      pattern: { inhale: 4, hold: 4, exhale: 4, pause: 4 },
      icon: Waves,
      color: 'text-green-600 dark:text-green-400'
    },
    {
      id: 'calm-breathing',
      name: 'Calm Breathing',
      description: 'Simple technique for immediate relaxation',
      duration: 180, // 3 minutes
      pattern: { inhale: 4, hold: 2, exhale: 6, pause: 2 },
      icon: Heart,
      color: 'text-purple-600 dark:text-purple-400'
    }
  ];

  const meditations: Meditation[] = [
    {
      id: 'mindfulness',
      title: 'Mindfulness Meditation',
      description: 'Focus on the present moment and observe your thoughts without judgment',
      duration: 10,
      type: 'mindfulness',
      icon: Sun,
      color: 'text-orange-600 dark:text-orange-400',
      steps: [
        { title: 'Settle', instruction: 'Sit comfortably. Soften your gaze or close your eyes. Take a few natural breaths.', duration: 60 },
        { title: 'Anchor', instruction: 'Place gentle attention on the breath at your nostrils or belly.', duration: 180 },
        { title: 'Notice', instruction: 'When the mind wanders, notice it kindly and gently return to the breath.', duration: 120 },
        { title: 'Widen Awareness', instruction: 'Include sounds and bodily sensations in awareness, letting them come and go.', duration: 120 },
        { title: 'Closing', instruction: 'Take a deeper breath, wiggle fingers and toes, and open your eyes.', duration: 60 }
      ]
    },
    {
      id: 'body-scan',
      title: 'Body Scan',
      description: 'Progressive relaxation technique to release tension throughout your body',
      duration: 15,
      type: 'body-scan',
      icon: Waves,
      color: 'text-blue-600 dark:text-blue-400',
      steps: [
        { title: 'Settle', instruction: 'Lie down or sit comfortably. Allow your body to be supported.', duration: 60 },
        { title: 'Head & Face', instruction: 'Bring attention to your forehead, eyes, jaw. Soften and release any tension.', duration: 120 },
        { title: 'Neck & Shoulders', instruction: 'Scan the neck and shoulders. Let them drop and relax with each exhale.', duration: 120 },
        { title: 'Arms & Hands', instruction: 'Move awareness down the arms into the hands and fingers. Soften the palms.', duration: 120 },
        { title: 'Chest & Back', instruction: 'Notice the breath moving your chest and the contact of your back with support.', duration: 120 },
        { title: 'Abdomen & Hips', instruction: 'Feel the belly and hips. Invite ease and warmth to spread.', duration: 120 },
        { title: 'Legs & Feet', instruction: 'Scan thighs, knees, calves, ankles, and toes. Release any tightness.', duration: 180 },
        { title: 'Closing', instruction: 'Take a deeper breath and gently reawaken movement in fingers and toes.', duration: 60 }
      ]
    },
    {
      id: 'loving-kindness',
      title: 'Loving-Kindness',
      description: 'Cultivate compassion and positive feelings toward yourself and others',
      duration: 12,
      type: 'loving-kindness',
      icon: Heart,
      color: 'text-pink-600 dark:text-pink-400',
      steps: [
        { title: 'Settle', instruction: 'Sit comfortably and soften the breath. Bring a gentle smile if you like.', duration: 60 },
        { title: 'Self Kindness', instruction: 'Silently repeat: “May I be safe. May I be healthy. May I be peaceful.”', duration: 150 },
        { title: 'Loved One', instruction: 'Bring to mind someone you care for. Offer: “May you be safe, healthy, and peaceful.”', duration: 150 },
        { title: 'Neutral Person', instruction: 'Bring to mind someone neutral. Offer them the same kind wishes.', duration: 120 },
        { title: 'All Beings', instruction: 'Expand outward: “May all beings be safe, healthy, and peaceful.”', duration: 120 },
        { title: 'Closing', instruction: 'Return to your breath. Notice any shift in mood or body.', duration: 60 }
      ]
    },
    {
      id: 'progressive-relaxation',
      title: 'Progressive Relaxation',
      description: 'Systematically tense and relax different muscle groups',
      duration: 20,
      type: 'progressive-relaxation',
      icon: Moon,
      color: 'text-indigo-600 dark:text-indigo-400',
      steps: [
        { title: 'Settle', instruction: 'Find a comfortable posture. Take a calming breath.', duration: 60 },
        { title: 'Hands & Forearms', instruction: 'Gently tense hands/forearms for 5 seconds, then release fully.', duration: 120 },
        { title: 'Upper Arms & Shoulders', instruction: 'Tense upper arms/shoulders 5 seconds, then release completely.', duration: 120 },
        { title: 'Face & Jaw', instruction: 'Scrunch facial muscles 5 seconds, then soften the entire face.', duration: 120 },
        { title: 'Chest & Back', instruction: 'Gently draw shoulder blades together 5 seconds, then relax.', duration: 120 },
        { title: 'Abdomen', instruction: 'Engage core for 5 seconds, then let the belly soften on the exhale.', duration: 120 },
        { title: 'Legs & Feet', instruction: 'Point toes and tense legs for 5 seconds, then release fully.', duration: 150 },
        { title: 'Closing', instruction: 'Breathe easily. Notice a sense of heaviness and calm.', duration: 60 }
      ]
    }
  ];

  const currentExercise = breathingExercises.find(ex => ex.id === activeExercise);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isPlaying && currentExercise) {
      interval = setInterval(() => {
        setTimeRemaining(prev => {
          if (prev <= 0) {
            setIsPlaying(false);
            setActiveExercise(null);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [isPlaying, currentExercise]);

  // Guided meditation countdown
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isMeditating && selectedMeditation) {
      interval = setInterval(() => {
        setMeditationRemaining(prev => {
          if (prev <= 0) {
            setIsMeditating(false);
            return 0;
          }
          return prev - 1;
        });
        setMeditationStepRemaining(prev => {
          const m = meditations.find(x => x.id === selectedMeditation);
          if (!m || !m.steps || m.steps.length === 0) return 0;
          if (prev > 1) return prev - 1;
          // Advance to next step in an atomic tick
          const nextIdx = meditationStepIndex + 1;
          if (nextIdx >= m.steps.length) {
            setIsMeditating(false);
            return 0;
          }
          setMeditationStepIndex(nextIdx);
          return m.steps[nextIdx].duration;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isMeditating, selectedMeditation, meditationStepIndex]);

  // Handle per-breath phase countdown and cycling
  useEffect(() => {
    if (!isPlaying || !currentExercise || timeRemaining <= 0) return;

    const pattern = currentExercise.pattern;
    const getNextPhase = (phase: typeof currentPhase): typeof currentPhase => {
      if (phase === 'inhale') return 'hold';
      if (phase === 'hold') return 'exhale';
      if (phase === 'exhale') return pattern.pause > 0 ? 'pause' : 'inhale';
      return 'inhale';
    };

    const getPhaseDuration = (phase: typeof currentPhase): number => {
      switch (phase) {
        case 'inhale': return pattern.inhale;
        case 'hold': return pattern.hold;
        case 'exhale': return pattern.exhale;
        case 'pause': return pattern.pause;
        default: return 0;
      }
    };

    // Initialize phaseRemaining if zero
    if (phaseRemaining <= 0) {
      const initial = getPhaseDuration(currentPhase);
      if (initial > 0) setPhaseRemaining(initial);
    }

    const phaseInterval = setInterval(() => {
      setPhaseRemaining(prev => {
        if (prev <= 1) {
          const nextPhase = getNextPhase(currentPhase);
          setCurrentPhase(nextPhase);
          const nextDur = getPhaseDuration(nextPhase);
          return nextDur > 0 ? nextDur : 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(phaseInterval);
  }, [isPlaying, currentExercise, currentPhase, timeRemaining, phaseRemaining]);

  const startBreathingExercise = (exerciseId: string) => {
    // Stop meditation if running
    if (isMeditating) {
      setIsMeditating(false);
      setSelectedMeditation(null);
      setMeditationRemaining(0);
    }
    setActiveExercise(exerciseId);
    setIsPlaying(true);
    setTimeRemaining(breathingExercises.find(ex => ex.id === exerciseId)?.duration || 0);
    setCurrentPhase('inhale');
    const ex = breathingExercises.find(ex => ex.id === exerciseId);
    setPhaseRemaining(ex ? ex.pattern.inhale : 0);
  };

  const stopExercise = () => {
    setIsPlaying(false);
    setActiveExercise(null);
    setTimeRemaining(0);
    setCurrentPhase('inhale');
    setPhaseRemaining(0);
  };

  const startMeditation = (meditationId: string) => {
    // Stop breathing exercise if running
    if (isPlaying) {
      setIsPlaying(false);
      setActiveExercise(null);
      setTimeRemaining(0);
      setCurrentPhase('inhale');
      setPhaseRemaining(0);
    }
    setSelectedMeditation(meditationId);
    setIsMeditating(true);
    const m = meditations.find(m => m.id === meditationId);
    if (m) {
      const total = (m.steps && m.steps.length > 0)
        ? m.steps.reduce((acc, s) => acc + s.duration, 0)
        : (m.duration * 60);
      setMeditationRemaining(total);
      setMeditationStepIndex(0);
      setMeditationStepRemaining(m.steps && m.steps[0] ? m.steps[0].duration : total);
    } else {
      setMeditationRemaining(0);
      setMeditationStepIndex(0);
      setMeditationStepRemaining(0);
    }
  };

  const stopMeditation = () => {
    setIsMeditating(false);
    setMeditationRemaining(0);
    setSelectedMeditation(null);
    setMeditationStepIndex(0);
    setMeditationStepRemaining(0);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getPhaseInstructions = () => {
    if (!currentExercise) return '';
    
    switch (currentPhase) {
      case 'inhale':
        return `Breathe in slowly for ${currentExercise.pattern.inhale} seconds`;
      case 'hold':
        return `Hold your breath for ${currentExercise.pattern.hold} seconds`;
      case 'exhale':
        return `Breathe out slowly for ${currentExercise.pattern.exhale} seconds`;
      case 'pause':
        return `Pause for ${currentExercise.pattern.pause} seconds`;
      default:
        return '';
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
          Relaxation Techniques
        </h1>
        <p className="text-lg text-slate-600 dark:text-slate-400">
          Take a moment to breathe, relax, and find your center
        </p>
      </div>

      {/* Active Exercise Display */}
      {activeExercise && currentExercise && (
        <div className="mb-8 bg-white dark:bg-slate-800 rounded-2xl p-8 border border-slate-200 dark:border-slate-700">
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <div className={`p-4 rounded-full bg-slate-100 dark:bg-slate-700 ${currentExercise.color}`}>
                {currentExercise && <currentExercise.icon className="h-8 w-8" />}
              </div>
            </div>
            
            <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
              {currentExercise.name}
            </h2>
            
            <div className="mb-6">
              <div className="text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                {formatTime(timeRemaining)}
              </div>
              <div className="text-lg text-slate-600 dark:text-slate-400">
                {getPhaseInstructions()}
              </div>
              <div className="mt-2 inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 text-sm">
                <span className="font-medium capitalize">{currentPhase}</span>
                <span>•</span>
                <span>{phaseRemaining}s</span>
              </div>
            </div>

            <div className="flex justify-center space-x-4">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2"
              >
                {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                <span>{isPlaying ? 'Pause' : 'Resume'}</span>
              </button>
              
              <button
                onClick={stopExercise}
                className="bg-slate-600 hover:bg-slate-700 text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2"
              >
                <RotateCcw className="h-5 w-5" />
                <span>Stop</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Breathing Exercises */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6">
          Breathing Exercises
        </h2>
        
        <div className="grid md:grid-cols-3 gap-6">
          {breathingExercises.map((exercise) => (
            <div
              key={exercise.id}
              className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 hover:shadow-lg transition-shadow"
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className={`p-2 rounded-lg bg-slate-100 dark:bg-slate-700 ${exercise.color}`}>
                  <exercise.icon className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                  {exercise.name}
                </h3>
              </div>
              
              <p className="text-slate-600 dark:text-slate-400 mb-4">
                {exercise.description}
              </p>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-1 text-sm text-slate-500 dark:text-slate-400">
                  <Timer className="h-4 w-4" />
                  <span>{Math.floor(exercise.duration / 60)} min</span>
                </div>
                
                <button
                  onClick={() => startBreathingExercise(exercise.id)}
                  disabled={activeExercise === exercise.id}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-400 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:cursor-not-allowed"
                >
                  {activeExercise === exercise.id ? 'Active' : 'Start'}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Guided Meditations */}
      <div>
        <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6">
          Guided Meditations
        </h2>

        {/* Active Meditation Display */}
        {selectedMeditation && (
          <div className="mb-6 bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700">
            {(() => {
              const m = meditations.find(md => md.id === selectedMeditation)!;
              return (
                <div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className={`p-3 rounded-lg bg-slate-100 dark:bg-slate-700 ${m.color}`}>
                        <m.icon className="h-6 w-6" />
                      </div>
                      <div>
                        <div className="font-semibold text-slate-900 dark:text-slate-100">{m.title}</div>
                        <div className="text-slate-600 dark:text-slate-400 text-sm">Total: {formatTime(meditationRemaining)}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => setIsMeditating(!isMeditating)}
                        className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2"
                      >
                        {isMeditating ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                        <span>{isMeditating ? 'Pause' : 'Resume'}</span>
                      </button>
                      <button
                        onClick={stopMeditation}
                        className="bg-slate-600 hover:bg-slate-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center space-x-2"
                      >
                        <RotateCcw className="h-4 w-4" />
                        <span>Stop</span>
                      </button>
                    </div>
                  </div>

                  {m.steps && m.steps.length > 0 && (
                    <div className="mt-6 p-4 rounded-xl bg-slate-50 dark:bg-slate-900/30 border border-slate-200 dark:border-slate-700">
                      <div className="flex items-center justify-between mb-2">
                        <div className="font-medium text-slate-900 dark:text-slate-100">Step {meditationStepIndex + 1}: {m.steps[meditationStepIndex]?.title}</div>
                        <div className="text-sm text-slate-600 dark:text-slate-400">{meditationStepRemaining}s</div>
                      </div>
                      <div className="text-slate-700 dark:text-slate-300">
                        {m.steps[meditationStepIndex]?.instruction}
                      </div>
                    </div>
                  )}

                  {m.steps && m.steps.length > 0 && (
                    <div className="mt-4">
                      <div className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Upcoming steps</div>
                      <ol className="space-y-1 text-sm text-slate-600 dark:text-slate-400">
                        {m.steps.slice(meditationStepIndex + 1, meditationStepIndex + 5).map((s, i) => (
                          <li key={i} className="flex items-center justify-between">
                            <span className="truncate mr-2">{s.title}</span>
                            <span className="tabular-nums">{s.duration}s</span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}
        
        <div className="grid md:grid-cols-2 gap-6">
          {meditations.map((meditation) => (
            <div
              key={meditation.id}
              className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 hover:shadow-lg transition-shadow"
            >
              <div className="flex items-start space-x-4">
                <div className={`p-3 rounded-lg bg-slate-100 dark:bg-slate-700 ${meditation.color}`}>
                  <meditation.icon className="h-6 w-6" />
                </div>
                
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
                    {meditation.title}
                  </h3>
                  
                  <p className="text-slate-600 dark:text-slate-400 mb-4">
                    {meditation.description}
                  </p>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-1 text-sm text-slate-500 dark:text-slate-400">
                      <Timer className="h-4 w-4" />
                      <span>{meditation.duration} min</span>
                    </div>
                    
                  <button
                    onClick={() => startMeditation(meditation.id)}
                      className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                    >
                      Start Meditation
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Relaxation Tips */}
      <div className="mt-12 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl p-8 border border-blue-200 dark:border-blue-800">
        <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-6 text-center">
          Quick Relaxation Tips
        </h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Immediate Relief
            </h3>
            <ul className="space-y-2 text-slate-600 dark:text-slate-400">
              <li>• Take 5 deep breaths slowly</li>
              <li>• Tense and release your shoulders</li>
              <li>• Look at something beautiful for 30 seconds</li>
              <li>• Listen to calming music</li>
              <li>• Stretch your arms and neck gently</li>
            </ul>
          </div>
          
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
              Daily Practices
            </h3>
            <ul className="space-y-2 text-slate-600 dark:text-slate-400">
              <li>• Practice gratitude each morning</li>
              <li>• Take short walks in nature</li>
              <li>• Limit screen time before bed</li>
              <li>• Keep a relaxation journal</li>
              <li>• Connect with loved ones regularly</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
