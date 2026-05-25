'use client';

import { useState } from 'react';
import { CheckCircle, Circle, ArrowRight, ArrowLeft, Brain, Heart, AlertTriangle, TrendingUp } from 'lucide-react';

interface Question {
  id: number;
  text: string;
  category: 'anxiety' | 'depression' | 'stress' | 'general';
}

interface TestResult {
  score: number;
  level: 'low' | 'mild' | 'moderate' | 'severe';
  description: string;
  recommendations: string[];
}

interface Test {
  id: string;
  name: string;
  description: string;
  questions: Question[];
  maxScore: number;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
}

export default function PsychologicalTests() {
  const [selectedTest, setSelectedTest] = useState<string | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<{ [key: number]: number }>({});
  const [showResults, setShowResults] = useState(false);
  const [testResults, setTestResults] = useState<TestResult | null>(null);

  const tests: Test[] = [
    {
      id: 'anxiety',
      name: 'Anxiety Assessment',
      description: 'Evaluate your anxiety levels and identify areas of concern',
      maxScore: 21,
      icon: AlertTriangle,
      color: 'text-primary',
      questions: [
        { id: 1, text: 'I feel nervous, anxious, or on edge', category: 'anxiety' },
        { id: 2, text: 'I am not able to stop or control worrying', category: 'anxiety' },
        { id: 3, text: 'I worry too much about different things', category: 'anxiety' },
        { id: 4, text: 'I have trouble relaxing', category: 'anxiety' },
        { id: 5, text: 'I am so restless that it is hard to sit still', category: 'anxiety' },
        { id: 6, text: 'I become easily annoyed or irritable', category: 'anxiety' },
        { id: 7, text: 'I feel afraid as if something awful might happen', category: 'anxiety' }
      ]
    },
    {
      id: 'gad7',
      name: 'GAD-7 (Anxiety)',
      description: 'Generalized Anxiety Disorder 7-item scale. Timeframe: last 2 weeks.',
      maxScore: 21,
      icon: AlertTriangle,
      color: 'text-primary',
      questions: [
        { id: 1, text: 'Feeling nervous, anxious, or on edge', category: 'anxiety' },
        { id: 2, text: 'Not being able to stop or control worrying', category: 'anxiety' },
        { id: 3, text: 'Worrying too much about different things', category: 'anxiety' },
        { id: 4, text: 'Trouble relaxing', category: 'anxiety' },
        { id: 5, text: 'Being so restless that it is hard to sit still', category: 'anxiety' },
        { id: 6, text: 'Becoming easily annoyed or irritable', category: 'anxiety' },
        { id: 7, text: 'Feeling afraid as if something awful might happen', category: 'anxiety' }
      ]
    },
    {
      id: 'phq7',
      name: 'PHQ-7 (Depression)',
      description: 'Patient Health Questionnaire 7-item depression screener. Timeframe: last 2 weeks.',
      maxScore: 21,
      icon: Heart,
      color: 'text-primary',
      questions: [
        { id: 1, text: 'Little interest or pleasure in doing things', category: 'depression' },
        { id: 2, text: 'Feeling down, depressed, or hopeless', category: 'depression' },
        { id: 3, text: 'Feeling tired or having little energy', category: 'depression' },
        { id: 4, text: 'Feeling bad about yourself — or that you are a failure', category: 'depression' },
        { id: 5, text: 'Trouble concentrating on things', category: 'depression' },
        { id: 6, text: 'Moving or speaking so slowly that other people could have noticed', category: 'depression' },
        { id: 7, text: 'Trouble falling or staying asleep, or sleeping too much', category: 'depression' }
      ]
    },
    {
      id: 'ghq12',
      name: 'GHQ-12 (General Health Questionnaire)',
      description: '12-item GHQ screening for general psychological distress. Timeframe: recent weeks.',
      maxScore: 36,
      icon: TrendingUp,
      color: 'text-primary',
      questions: [
        { id: 1, text: 'Been able to concentrate on what you’re doing', category: 'general' },
        { id: 2, text: 'Lost much sleep over worry', category: 'general' },
        { id: 3, text: 'Felt that you are playing a useful part in things', category: 'general' },
        { id: 4, text: 'Felt capable of making decisions about things', category: 'general' },
        { id: 5, text: 'Felt constantly under strain', category: 'general' },
        { id: 6, text: 'Felt you couldn’t overcome your difficulties', category: 'general' },
        { id: 7, text: 'Been able to enjoy your normal day-to-day activities', category: 'general' },
        { id: 8, text: 'Been able to face up to your problems', category: 'general' },
        { id: 9, text: 'Been feeling unhappy and depressed', category: 'general' },
        { id: 10, text: 'Been losing confidence in yourself', category: 'general' },
        { id: 11, text: 'Been thinking of yourself as a worthless person', category: 'general' },
        { id: 12, text: 'Been feeling reasonably happy, all things considered', category: 'general' }
      ]
    },
    {
      id: 'depression',
      name: 'Depression Screening',
      description: 'Assess your mood and identify signs of depression',
      maxScore: 27,
      icon: Heart,
      color: 'text-primary',
      questions: [
        { id: 1, text: 'Little interest or pleasure in doing things', category: 'depression' },
        { id: 2, text: 'Feeling down, depressed, or hopeless', category: 'depression' },
        { id: 3, text: 'Trouble falling or staying asleep, or sleeping too much', category: 'depression' },
        { id: 4, text: 'Feeling tired or having little energy', category: 'depression' },
        { id: 5, text: 'Poor appetite or overeating', category: 'depression' },
        { id: 6, text: 'Feeling bad about yourself or that you are a failure', category: 'depression' },
        { id: 7, text: 'Trouble concentrating on things', category: 'depression' },
        { id: 8, text: 'Moving or speaking so slowly that other people could have noticed', category: 'depression' },
        { id: 9, text: 'Thoughts that you would be better off dead', category: 'depression' }
      ]
    },
    {
      id: 'stress',
      name: 'Stress Level Assessment',
      description: 'Measure your current stress levels and coping strategies',
      maxScore: 40,
      icon: TrendingUp,
      color: 'text-primary',
      questions: [
        { id: 1, text: 'I feel overwhelmed by my responsibilities', category: 'stress' },
        { id: 2, text: 'I have difficulty concentrating due to stress', category: 'stress' },
        { id: 3, text: 'I feel irritable or short-tempered', category: 'stress' },
        { id: 4, text: 'I have trouble sleeping due to worry or stress', category: 'stress' },
        { id: 5, text: 'I feel physically tense or have muscle tension', category: 'stress' },
        { id: 6, text: 'I avoid situations that cause me stress', category: 'stress' },
        { id: 7, text: 'I feel like I have too much to do and not enough time', category: 'stress' },
        { id: 8, text: 'I feel anxious about the future', category: 'stress' },
        { id: 9, text: 'I have physical symptoms like headaches or stomachaches', category: 'stress' },
        { id: 10, text: 'I feel like I cannot control the important things in my life', category: 'stress' }
      ]
    }
  ];

  const currentTest = tests.find(test => test.id === selectedTest);
  const currentQuestionData = currentTest?.questions[currentQuestion];

  const handleAnswerSelect = (value: number) => {
    setAnswers(prev => ({
      ...prev,
      [currentQuestionData?.id || 0]: value
    }));
  };

  const nextQuestion = () => {
    if (currentTest && currentQuestion < currentTest.questions.length - 1) {
      setCurrentQuestion(prev => prev + 1);
    }
  };

  const previousQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(prev => prev - 1);
    }
  };

  const calculateResults = (): TestResult => {
    if (!currentTest) return { score: 0, level: 'low', description: '', recommendations: [] };

    const totalScore = currentTest.questions.reduce((sum, question) => {
      return sum + (answers[question.id] || 0);
    }, 0);

    let level: 'low' | 'mild' | 'moderate' | 'severe';
    let description: string;
    let recommendations: string[];

    // Specific scales with validated cutoffs
    if (currentTest.id === 'gad7') {
      // 0-4 minimal, 5-9 mild, 10-14 moderate, 15-21 severe
      if (totalScore <= 4) {
        level = 'low';
        description = 'Minimal anxiety symptoms (GAD-7).';
        recommendations = ['Continue healthy routines', 'Use relaxation as needed'];
      } else if (totalScore <= 9) {
        level = 'mild';
        description = 'Mild anxiety symptoms (GAD-7).';
        recommendations = ['Try breathing/mindfulness exercises', 'Monitor symptoms over time'];
      } else if (totalScore <= 14) {
        level = 'moderate';
        description = 'Moderate anxiety symptoms (GAD-7). Consider clinical evaluation.';
        recommendations = ['Consider professional support', 'Practice CBT-based strategies'];
      } else {
        level = 'severe';
        description = 'Severe anxiety symptoms (GAD-7). Clinical assessment recommended.';
        recommendations = ['Seek professional evaluation', 'Create a coping plan with support'];
      }
    } else if (currentTest.id === 'phq7') {
      // Scaled from PHQ thresholds: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-21 severe
      if (totalScore <= 4) {
        level = 'low';
        description = 'Minimal depressive symptoms (PHQ-7).';
        recommendations = ['Maintain healthy habits', 'Stay socially engaged'];
      } else if (totalScore <= 9) {
        level = 'mild';
        description = 'Mild depressive symptoms (PHQ-7).';
        recommendations = ['Increase pleasant activities', 'Use sleep hygiene strategies'];
      } else if (totalScore <= 14) {
        level = 'moderate';
        description = 'Moderate depressive symptoms (PHQ-7). Consider clinical evaluation.';
        recommendations = ['Consider professional support', 'Try behavioral activation'];
      } else {
        level = 'severe';
        description = 'Severe depressive symptoms (PHQ-7). Clinical assessment recommended.';
        recommendations = ['Seek professional evaluation', 'Prioritize safety and support'];
      }
    } else if (currentTest.id === 'ghq12') {
      // GHQ-12 Likert (0-3) total: 0-36. Heuristic cutoffs
      if (totalScore <= 12) {
        level = 'low';
        description = 'Low general psychological distress (GHQ-12).';
        recommendations = ['Maintain routines', 'Preventive self-care'];
      } else if (totalScore <= 20) {
        level = 'mild';
        description = 'Mild psychological distress (GHQ-12).';
        recommendations = ['Stress management', 'Enhance social support'];
      } else if (totalScore <= 27) {
        level = 'moderate';
        description = 'Moderate psychological distress (GHQ-12). Consider clinical screening.';
        recommendations = ['Consider professional advice', 'Structured coping strategies'];
      } else {
        level = 'severe';
        description = 'Severe psychological distress (GHQ-12). Clinical assessment recommended.';
        recommendations = ['Seek professional help', 'Develop a support and safety plan'];
      }
    } else {
      // Generic percentage-based fallback
      const percentage = (totalScore / currentTest.maxScore) * 100;
      if (percentage < 25) {
        level = 'low';
        description = 'Your scores indicate minimal symptoms in this area.';
        recommendations = [
          'Continue your current self-care practices',
          'Maintain healthy lifestyle habits',
          'Consider preventive mental health strategies'
        ];
      } else if (percentage < 50) {
        level = 'mild';
        description = 'You may be experiencing mild symptoms that could benefit from attention.';
        recommendations = [
          'Practice stress management techniques',
          'Consider talking to a trusted friend or family member',
          'Engage in regular physical activity',
          'Maintain a consistent sleep schedule'
        ];
      } else if (percentage < 75) {
        level = 'moderate';
        description = 'Your scores suggest moderate symptoms that may benefit from professional support.';
        recommendations = [
          'Consider speaking with a mental health professional',
          'Practice relaxation and mindfulness techniques',
          'Maintain social connections and support systems',
          'Consider lifestyle changes to reduce stress'
        ];
      } else {
        level = 'severe';
        description = 'Your scores indicate significant symptoms that would benefit from professional support.';
        recommendations = [
          'Seek professional mental health support as soon as possible',
          'Consider reaching out to a crisis helpline if needed',
          'Engage with your support network',
          'Prioritize self-care and safety'
        ];
      }
    }

    return { score: totalScore, level, description, recommendations };
  };

  const finishTest = () => {
    const results = calculateResults();
    setTestResults(results);
    setShowResults(true);
  };

  const resetTest = () => {
    setSelectedTest(null);
    setCurrentQuestion(0);
    setAnswers({});
    setShowResults(false);
    setTestResults(null);
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-600 dark:text-green-400';
      case 'mild': return 'text-yellow-600 dark:text-yellow-400';
      case 'moderate': return 'text-orange-600 dark:text-orange-400';
      case 'severe': return 'text-red-600 dark:text-red-400';
      default: return 'text-slate-600 dark:text-slate-400';
    }
  };

  const getLevelBgColor = (level: string) => {
    switch (level) {
      case 'low': return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
      case 'mild': return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'moderate': return 'bg-orange-50 dark:bg-orange-900/20 border-orange-200 dark:border-orange-800';
      case 'severe': return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      default: return 'bg-slate-50 dark:bg-slate-900/20 border-slate-200 dark:border-slate-800';
    }
  };

  if (showResults && testResults) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            Assessment Results
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400">
            {currentTest?.name} - Your Results
          </p>
        </div>

        <div className={`rounded-2xl p-8 border ${getLevelBgColor(testResults.level)}`}>
          <div className="text-center mb-8">
            <div className="flex justify-center mb-4">
              <div className={`p-4 rounded-full bg-slate-100 dark:bg-slate-700 ${currentTest?.color}`}>
                {currentTest && <currentTest.icon className="h-8 w-8" />}
              </div>
            </div>
            
            <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-2">
              {testResults.level.charAt(0).toUpperCase() + testResults.level.slice(1)} Level
            </h2>
            
            <div className="text-4xl font-bold mb-2">
              <span className={getLevelColor(testResults.level)}>
                {testResults.score}/{currentTest?.maxScore}
              </span>
            </div>
            
            <p className="text-lg text-slate-600 dark:text-slate-400">
              {testResults.description}
            </p>
          </div>

          <div className="mb-8">
            <h3 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-4">
              Recommendations
            </h3>
            <ul className="space-y-3">
              {testResults.recommendations.map((recommendation, index) => (
                <li key={index} className="flex items-start space-x-3">
                  <CheckCircle className="h-5 w-5 text-success mt-0.5 flex-shrink-0" />
                  <span className="text-slate-700 dark:text-slate-300">{recommendation}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="text-center">
            <button
              onClick={resetTest}
              className="bg-primary hover:bg-primary/90 text-primary-foreground px-6 py-3 rounded-lg font-medium transition-colors"
            >
              Take Another Assessment
            </button>
          </div>
        </div>

        <div className="mt-8 p-6 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="h-6 w-6 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold text-amber-800 dark:text-amber-200 mb-2">
                Important Note
              </h3>
              <p className="text-amber-700 dark:text-amber-300 text-sm">
                These assessments are for informational purposes only and are not a substitute for professional diagnosis. 
                If you&apos;re concerned about your mental health, please consult with a qualified mental health professional.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (selectedTest && currentTest) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            {currentTest.name}
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400">
            Question {currentQuestion + 1} of {currentTest.questions.length}
          </p>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-2xl p-8 border border-slate-200 dark:border-slate-700">
          <div className="mb-8">
            <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2 mb-4">
              <div
                className="bg-primary h-2 rounded-full transition-all duration-300"
                style={{ width: `${((currentQuestion + 1) / currentTest.questions.length) * 100}%` }}
              />
            </div>
          </div>

          <div className="mb-8">
            <h2 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-6">
              {currentQuestionData?.text}
            </h2>

            <div className="space-y-3">
              {[0, 1, 2, 3].map((value) => (
                <button
                  key={value}
                  onClick={() => handleAnswerSelect(value)}
                  className={`w-full p-4 rounded-lg border-2 transition-colors text-left ${
                    answers[currentQuestionData?.id || 0] === value
                      ? 'border-primary bg-primary/10'
                      : 'border-slate-200 dark:border-slate-600 hover:border-slate-300 dark:hover:border-slate-500'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    {answers[currentQuestionData?.id || 0] === value ? (
                      <CheckCircle className="h-5 w-5 text-primary" />
                    ) : (
                      <Circle className="h-5 w-5 text-slate-400" />
                    )}
                    <span className="text-slate-900 dark:text-slate-100">
                      {value === 0 && 'Not at all'}
                      {value === 1 && 'Several days'}
                      {value === 2 && 'More than half the days'}
                      {value === 3 && 'Nearly every day'}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="flex justify-between">
            <button
              onClick={previousQuestion}
              disabled={currentQuestion === 0}
              className="flex items-center space-x-2 px-4 py-2 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ArrowLeft className="h-4 w-4" />
              <span>Previous</span>
            </button>

            {currentQuestion === currentTest.questions.length - 1 ? (
              <button
                onClick={finishTest}
                disabled={answers[currentQuestionData?.id || 0] === undefined}
                className="bg-success hover:bg-success/90 disabled:bg-muted text-primary-foreground px-6 py-2 rounded-lg font-medium transition-colors disabled:cursor-not-allowed"
              >
                Finish Assessment
              </button>
            ) : (
              <button
                onClick={nextQuestion}
                disabled={answers[currentQuestionData?.id || 0] === undefined}
                className="flex items-center space-x-2 bg-primary hover:bg-primary/90 disabled:bg-muted text-primary-foreground px-4 py-2 rounded-lg font-medium transition-colors disabled:cursor-not-allowed"
              >
                <span>Next</span>
                <ArrowRight className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100 mb-4">
          Psychological Assessments
        </h1>
        <p className="text-lg text-slate-600 dark:text-slate-400">
          Take evidence-based assessments to better understand your mental health
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {tests.map((test) => (
          <div
            key={test.id}
            className="bg-white dark:bg-slate-800 p-6 rounded-xl border border-slate-200 dark:border-slate-700 hover:shadow-lg transition-shadow"
          >
            <div className="flex items-center space-x-3 mb-4">
              <div className={`p-3 rounded-lg bg-slate-100 dark:bg-slate-700 ${test.color}`}>
                <test.icon className="h-6 w-6" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
                {test.name}
              </h3>
            </div>
            
            <p className="text-slate-600 dark:text-slate-400 mb-4">
              {test.description}
            </p>
            
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm text-slate-500 dark:text-slate-400">
                {test.questions.length} questions
              </div>
              <div className="text-sm text-slate-500 dark:text-slate-400">
                ~{Math.ceil(test.questions.length * 0.5)} min
              </div>
            </div>
            
            <button
              onClick={() => setSelectedTest(test.id)}
              className="w-full bg-primary hover:bg-primary/90 text-primary-foreground py-2 rounded-lg font-medium transition-colors"
            >
              Start Assessment
            </button>
          </div>
        ))}
      </div>

      <div className="mt-12 bg-primary/10 rounded-2xl p-8 border border-primary/20">
        <div className="text-center">
          <Brain className="h-12 w-12 text-primary mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mb-4">
            About These Assessments
          </h2>
          <p className="text-slate-600 dark:text-slate-400 max-w-3xl mx-auto">
            These assessments are based on validated psychological screening tools used by mental health professionals. 
            They can help identify areas where you might benefit from additional support or professional guidance. 
            Remember, these are screening tools, not diagnostic instruments.
          </p>
        </div>
      </div>
    </div>
  );
}
