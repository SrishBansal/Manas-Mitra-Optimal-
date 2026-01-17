import { GoogleGenerativeAI } from "@google/generative-ai";
import { checkSafety, CRISIS_RESPONSE_TEMPLATE } from "@/lib/safety";
import { MANAS_MITRA_SYSTEM_PROMPT } from "@/lib/prompt";
import { NextResponse } from "next/server";
import { supabase } from "@/lib/supabase";

export async function POST(req: Request) {
    try {
        const { message } = await req.json();

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
            return NextResponse.json({
                reply: "I'm having a little trouble connecting to my thoughts right now (Mission Control: GEMINI_API_KEY is missing). I'm mostly here to listen, though."
            });
        }

        // 3. Call Gemini
        const genAI = new GoogleGenerativeAI(apiKey);
        const model = genAI.getGenerativeModel({
            model: "gemini-pro",
            // systemInstruction removed for compatibility
        });

        // Prepend system prompt to message for robust instruction following
        const finalPrompt = `${MANAS_MITRA_SYSTEM_PROMPT}\n\nUser: ${message}\nAssistant:`;

        const result = await model.generateContent(finalPrompt);
        const response = await result.response;
        const text = response.text();

        // 4. Log to Supabase (Fire and Forget)
        if (supabase) {
            supabase.from('chats').insert([
                { role: 'user', content: message },
                { role: 'assistant', content: text }
            ]).then(({ error }) => {
                if (error) console.error('Supabase Log Error:', error);
            });
        }

        return NextResponse.json({ reply: text });

    } catch (error) {
        console.error("Chat API Error:", error);

        // Fallback to Mock Persona if API fails (e.g. Invalid Key/Permissions)
        const lowerMsg = (await req.json().catch(() => ({}))).message?.toLowerCase() || "";
        let fallbackReply = "I'm having a little trouble connecting to my thoughts right now, but I'm here. (System: API Connection Failed, using offline mode)";

        if (lowerMsg.includes("happy") || lowerMsg.includes("good")) {
            fallbackReply = "That sounds really bright! It's so good to feel that lightness, isn't it? What's making you smile today?";
        } else if (lowerMsg.includes("sad") || lowerMsg.includes("lonely")) {
            fallbackReply = "I hear you. It can feel really heavy carrying that all by yourself. Do you want to talk about what's bringing this on?";
        } else if (lowerMsg.includes("breakup") || lowerMsg.includes("broke up")) {
            fallbackReply = "Breakups leave such a quiet ache. It's really hard when a connection changes like that. How are you holding up right now?";
        } else if (lowerMsg.includes("stress") || lowerMsg.includes("exam")) {
            fallbackReply = "Exams can make everything feel tight and rushed. Remember to take a breath—you're more than just these scores. What's the subject worrying you most?";
        } else if (lowerMsg.includes("kill") || lowerMsg.includes("die")) {
            // Should have been caught by safety check, but just in case
            fallbackReply = "I am really concerned about you. Please call 1800-599-0019 immediately.";
        }

        return NextResponse.json({ reply: fallbackReply });
    }
}
