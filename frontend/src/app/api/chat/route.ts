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
}
