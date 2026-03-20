<<<<<<< HEAD
import { NextRequest, NextResponse } from 'next/server';

// Mental health response templates
const responseTemplates = {
  greeting: [
    "Hello! I'm here to listen and support you. How are you feeling today?",
    "Hi there! I'm your AI mental health companion. What's on your mind?",
    "Welcome! I'm here to provide a safe space for you to share. How can I help you today?"
  ],
  anxiety: [
    "I understand that anxiety can feel overwhelming. You're not alone in this. What specific situations are causing you the most anxiety right now?",
    "Anxiety is a common experience, and it's brave of you to reach out. Let's work together to find some strategies that might help. What techniques have worked for you in the past?",
    "I hear you, and I want you to know that your feelings are valid. Anxiety can be managed with the right tools and support. What would be most helpful for you right now?"
  ],
  depression: [
    "I'm sorry you're experiencing this. Depression can make everything feel harder, but you're showing great strength by reaching out. What small things have brought you comfort recently?",
    "You're not alone in this struggle. Depression is treatable, and there is hope. What support do you need right now?",
    "I want you to know that your feelings matter and that there are people who care about you. What would help you feel a little better today?"
  ],
  stress: [
    "Stress can feel overwhelming, but there are strategies we can explore together. What specific situations are causing you the most stress?",
    "I can help you develop some coping strategies for managing stress. What works best for you when you're feeling overwhelmed?",
    "Let's work together to find ways to reduce your stress. What relaxation techniques have you tried before?"
  ],
  crisis: [
    "I'm concerned about what you're sharing. Your safety is the most important thing right now. Please reach out to a mental health professional immediately. You can call the National Suicide Prevention Lifeline at 988, or contact your local emergency services. You're not alone, and there are people who want to help you.",
    "Your safety is my top priority. If you're having thoughts of self-harm, please contact emergency services immediately at 911 or the National Suicide Prevention Lifeline at 988. There are people who care about you and want to help.",
    "I'm worried about you. Please reach out to a crisis helpline right now. You can call 988 for the Suicide & Crisis Lifeline, or text HOME to 741741 for the Crisis Text Line. You don't have to face this alone."
  ],
  general: [
    "I'm here to listen and support you. Can you tell me more about what you're experiencing?",
    "Thank you for sharing that with me. How can I best support you right now?",
    "I understand this is important to you. What would be most helpful for you in this moment?",
    "You're not alone in this. I'm here to help you work through whatever you're facing.",
    "It sounds like you're going through a lot. What support do you need right now?"
  ]
};

// Crisis keywords that require immediate professional referral
const crisisKeywords = [
  'suicide', 'kill myself', 'end it all', 'not worth living', 'want to die',
  'self harm', 'hurt myself', 'crisis', 'emergency', 'help me', 'end my life',
  'better off dead', 'no point', 'give up', 'hopeless', 'worthless'
];

// Assessment triggers
const assessmentTriggers = [
  'anxiety', 'depression', 'stress', 'panic', 'overwhelmed', 'sad', 'worried',
  'fear', 'nervous', 'down', 'hopeless', 'empty', 'lonely', 'isolated'
];

function detectIntent(message: string): string {
  const lowerMessage = message.toLowerCase();

  // Check for crisis indicators first
  if (crisisKeywords.some(keyword => lowerMessage.includes(keyword))) {
    return 'crisis';
  }

  // Check for specific mental health concerns
  if (lowerMessage.includes('anxiety') || lowerMessage.includes('anxious') || lowerMessage.includes('worry')) {
    return 'anxiety';
  }

  if (lowerMessage.includes('depression') || lowerMessage.includes('depressed') || lowerMessage.includes('sad')) {
    return 'depression';
  }

  if (lowerMessage.includes('stress') || lowerMessage.includes('stressed') || lowerMessage.includes('overwhelmed')) {
    return 'stress';
  }

  if (lowerMessage.includes('hello') || lowerMessage.includes('hi') || lowerMessage.includes('hey')) {
    return 'greeting';
  }

  // Check if assessment might be helpful
  if (assessmentTriggers.some(keyword => lowerMessage.includes(keyword))) {
    return 'assessment';
  }

  return 'general';
}

function generateResponse(intent: string, message: string): string {
  const templates = responseTemplates[intent as keyof typeof responseTemplates] || responseTemplates.general;
  const randomTemplate = templates[Math.floor(Math.random() * templates.length)];

  // Add personalized touches based on the message
  if (intent === 'assessment') {
    return randomTemplate + " Would you like me to guide you through some questions that can help identify the best support strategies for you?";
  }

  return randomTemplate;
}

export async function POST(request: NextRequest) {
  try {
    const { message, userId } = await request.json();

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    // Detect the intent locally for immediate frontend flags (crisis/assessment)
    // This preserves the safe/fast local checks while offloading the heavy NLP generation
    const intent = detectIntent(message);
    const isCrisis = intent === 'crisis';
    const needsAssessment = intent === 'assessment';

    let responseText = "";
    let emotion = "neutral";

    try {
      // Connect to Python Backend
      const backendRes = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
          user_id: userId,
          session_id: userId, // Use userId as session_id for conversation continuity
        }),
      });

      if (backendRes.ok) {
        const data = await backendRes.json();
        responseText = data.reply;
        emotion = data.emotion;
      } else {
        console.error("Backend responded with error:", await backendRes.text());
        // Fallback to local generation if backend fails
        responseText = generateResponse(intent, message);
      }
    } catch (backendError: any) {
      console.error("Failed to connect to backend:", backendError);
      return NextResponse.json(
        {
          response: `[System Error] Connection to backend failed: ${backendError?.message || backendError}. URL: http://127.0.0.1:8000/chat`,
          intent: 'error',
          isCrisis: false,
          needsAssessment: false,
          timestamp: new Date().toISOString()
        },
        { status: 200 }
      );
    }

    return NextResponse.json({
      response: responseText,
      intent,
      isCrisis,
      needsAssessment,
      emotion, // Pass back the detected emotion if available
      timestamp: new Date().toISOString(),
      userId: userId || 'anonymous'
    });

  } catch (error) {
    console.error('Error processing chat message:', error);
    return NextResponse.json(
      {
        response: "I'm sorry, I'm having trouble processing your message right now. Please try again, and remember that if you're in crisis, please contact emergency services immediately.",
        intent: 'error',
        isCrisis: false,
        needsAssessment: false,
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    message: 'MindCare AI Chat API is running',
    version: '1.0.0',
    timestamp: new Date().toISOString()
  });
=======
import { checkSafety, CRISIS_RESPONSE_TEMPLATE } from "@/lib/safety";
import { MANAS_MITRA_SYSTEM_PROMPT } from "@/lib/prompt";
import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function POST(req: Request) {
    try {
        const body = await req.json();
        const message = body.message || "";

        if (!message || typeof message !== "string") {
            return NextResponse.json({ error: "Invalid message" }, { status: 400 });
        }

        // 1. Safety Check (Keyword Based)
        if (checkSafety(message)) {
            return NextResponse.json({
                reply: CRISIS_RESPONSE_TEMPLATE,
                needs_help: true
            });
        }

        // 2. Check API Key
        const apiKey = process.env.GEMINI_API_KEY;
        if (!apiKey) {
            return NextResponse.json({ error: "Server Configuration Error: Missing API Key" }, { status: 500 });
        }

        // 3. Call Gemini (Using verified working model found via scan)
        // gemini-2.5-flash was confirmed to work with this key
        const MODEL_ID = "gemini-2.5-flash";

        const callGemini = async (retryCount = 0): Promise<string> => {
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=${apiKey}`;

            const payload = {
                contents: [{ role: "user", parts: [{ text: message }] }],
                systemInstruction: { parts: [{ text: MANAS_MITRA_SYSTEM_PROMPT }] },
                generationConfig: {
                    temperature: 0.7,
                    maxOutputTokens: 1000,
                }
            };

            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`Gemini API Error (Attempt ${retryCount + 1}):`, response.status, errorText);

                // Retry once on network/server errors (5xx)
                if (retryCount < 1 && response.status >= 500) {
                    console.log("Retrying API call...");
                    return callGemini(retryCount + 1);
                }

                // If 429 or 404, throw immediately
                throw new Error(`Gemini API Failed (${response.status}): ${errorText}`);
            }

            const data = await response.json();
            const reply = data.candidates?.[0]?.content?.parts?.[0]?.text;

            if (!reply) {
                if (data.promptFeedback?.blockReason) {
                    return "I cannot respond to that due to safety guidelines.";
                }
                throw new Error("Empty response from Gemini");
            }

            return reply;
        };

        const text = await callGemini();

        // 4. Log to Supabase
        if (supabase) {
            supabase.from('chats').insert([
                { role: 'user', content: message },
                { role: 'assistant', content: text },
                { role: 'model', content: MODEL_ID }
            ]).then(({ error }) => {
                if (error) console.error('Supabase Log Error:', error);
            });
        }

        return NextResponse.json({ reply: text });

    } catch (error: any) {
        console.error("Critical Chat API Error:", error);

        const errorMessage = error.message || "Unknown error";

        if (errorMessage.includes("429")) {
            return NextResponse.json({
                error: `System: Gemini API Quota Exceeded for ${errorMessage.includes('gemini') ? 'selected model' : 'API key'}.`
            }, { status: 429 });
        }

        return NextResponse.json({
            error: `System: Unable to generate response. (${errorMessage})`
        }, { status: 500 });
    }
>>>>>>> daf68bcc64963c83bb108ae13c37eeb71ca39222
}
