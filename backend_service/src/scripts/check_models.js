const { GoogleGenerativeAI } = require("@google/generative-ai");
require('dotenv').config({ path: '../.env.local' });

async function listModels() {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
        console.error("API Key not found!");
        return;
    }

    console.log("Using API Key:", apiKey.substring(0, 5) + "...");

    const genAI = new GoogleGenerativeAI(apiKey);
    try {
        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
        // There isn't a direct listModels on the client instance in node SDK easily w/o admin?
        // Actually looking at docs, genAI.getGenerativeModel is just a factory.
        // We can try to generate content to see if it works.

        // But to list models, we might need a different call. 
        // The error suggests "Call ListModels to see the list".
        // It seems newer SDK might not expose listModels easily OR I need to use the model manager.

        // Let's just try a simple generation with a known safe model
        console.log("Attempting generation with gemini-1.5-flash...");
        const result = await model.generateContent("Hello");
        console.log("Success! Response:", result.response.text());

    } catch (error) {
        console.error("Error:", error.message);
    }
}

listModels();
