const https = require('https');
const fs = require('fs');
const path = require('path');

let API_KEY = process.env.GEMINI_API_KEY;

if (!API_KEY) {
    try {
        const envPath = path.resolve(__dirname, '../../.env.local');
        const envContent = fs.readFileSync(envPath, 'utf8');
        const match = envContent.match(/GEMINI_API_KEY=(.+)/);
        if (match && match[1]) {
            API_KEY = match[1].trim();
        }
    } catch (e) {
        console.error("Could not read .env.local", e.message);
    }
}

if (!API_KEY) {
    console.error("No API Key found. Set GEMINI_API_KEY env var or check .env.local");
    process.exit(1);
}

function httpsRequest(url, method, data = null) {
    return new Promise((resolve, reject) => {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const req = https.request(url, options, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => resolve({ status: res.statusCode, body: body }));
        });

        req.on('error', (e) => reject(e));

        if (data) {
            req.write(JSON.stringify(data));
        }
        req.end();
    });
}

async function findWorkingModel() {
    console.log("Fetching available models...");
    const listUrl = `https://generativelanguage.googleapis.com/v1beta/models?key=${API_KEY}`;

    try {
        const listRes = await httpsRequest(listUrl, 'GET');
        if (listRes.status !== 200) {
            console.error(`Failed to list models: ${listRes.status} ${listRes.body}`);
            return;
        }

        const models = JSON.parse(listRes.body).models;
        console.log(`Found ${models.length} models. Testing generation capability...`);

        const generationModels = models.filter(m =>
            m.supportedGenerationMethods.includes("generateContent")
        );

        console.log(`Found ${generationModels.length} generation models.`);

        for (const model of generationModels) {
            const name = model.name.replace('models/', ''); // simplify logs
            process.stdout.write(`Testing ${name}... `);

            const genUrl = `https://generativelanguage.googleapis.com/v1beta/${model.name}:generateContent?key=${API_KEY}`;
            const payload = {
                contents: [{ role: "user", parts: [{ text: "Hi" }] }],
                generationConfig: { maxOutputTokens: 10 }
            };

            const genRes = await httpsRequest(genUrl, 'POST', payload);

            if (genRes.status === 200) {
                console.log("SUCCESS! ✅");
                console.log(`\n>>> FOUND WORKING MODEL: ${model.name}`);
                console.log("Response:", genRes.body);
                return; // Stop after finding the first working one
            } else {
                console.log(`FAILED (${genRes.status}) ❌`);
                // console.log(genRes.body); // debug
            }
        }

        console.log("\nAll models failed.");

    } catch (e) {
        console.error("Script error:", e);
    }
}

findWorkingModel();
