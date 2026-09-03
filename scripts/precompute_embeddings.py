import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv(".env")
load_dotenv("backend/.env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DISTORTIONS_DATA = [
    {
        "name": "Catastrophizing",
        "definition": "Expecting the absolute worst outcome in a situation, even with minimal evidence.",
        "framework": "Gently explore the actual probability of the feared outcome. Encourage the user to consider the best-case, worst-case, and most likely scenarios.",
        "keywords": ["fail", "ruined", "disaster", "heart attack", "die", "worst", "hate me", "fired", "end of the world", "hopeless"],
        "examples": [
            "I'm going to fail this exam and my life will be completely ruined.",
            "If I make one mistake in my presentation, I'll be fired and end up homeless.",
            "Everything is going wrong, this is a total disaster and there's no way out.",
            "They haven't texted back, they must hate me and our friendship is over.",
            "My chest feels tight, I'm probably having a heart attack and going to die."
        ]
    },
    {
        "name": "All-or-Nothing Thinking",
        "definition": "Viewing things in black-and-white, absolute terms (e.g., 'always a failure', 'perfect or ruined').",
        "framework": "Highlight the middle ground and shades of gray. Shift focus from binary outcomes to progressive learning and self-compassion.",
        "keywords": ["complete failure", "perfect", "ruined", "total waste", "either", "unproductive", "never right"],
        "examples": [
            "If I don't get an A+, I'm a complete failure.",
            "I ate one cookie, so my entire diet is ruined now.",
            "Either they completely agree with me or they hate my guts.",
            "I messed up one slide, the whole presentation was a disaster.",
            "I couldn't finish all my tasks today, so I was totally unproductive."
        ]
    },
    {
        "name": "Emotional Reasoning",
        "definition": "Assuming that your subjective feelings reflect objective reality (e.g., 'I feel guilty, so I must be bad').",
        "framework": "Help the user distinguish between temporary emotional states and objective, observable facts. Validate the feeling, but challenge the factual conclusion.",
        "keywords": ["feel lonely", "feel stupid", "feel guilty", "feel like an idiot", "feel anxious", "nobody cares", "unsolvable"],
        "examples": [
            "I feel so lonely, so nobody must care about me.",
            "I feel like an idiot, which means I am stupid.",
            "I'm feeling so anxious, so something terrible is definitely about to happen.",
            "I feel guilty, so I must have done something awful to them.",
            "I feel overwhelmed, so this problem must be completely unsolvable."
        ]
    },
    {
        "name": "Overgeneralization",
        "definition": "Drawing broad, negative conclusions based on a single event (often using words like 'always', 'never', 'everyone').",
        "framework": "Gently prompt the user to look for exceptions to their perceived universal patterns, pointing out specific, positive counter-instances.",
        "keywords": ["always", "never", "everyone", "nobody", "every single time", "all the time"],
        "examples": [
            "I always mess up my relationships, I'll be alone forever.",
            "Nothing ever goes right for me in this city.",
            "I failed this job interview, so I will never get hired anywhere.",
            "Everyone is always happier and more successful than me.",
            "I can never do anything right."
        ]
    },
    {
        "name": "Should Statements",
        "definition": "Using rigid rules ('should', 'must', 'ought to') to motivate yourself or judge others, leading to guilt or anger.",
        "framework": "Help the user reframe self-imposed demands into flexible preferences or choices (e.g., changing 'I should study' to 'It is helpful for my goals if I study').",
        "keywords": ["should", "must", "ought to", "supposed to", "have to be strong", "shouldn't feel"],
        "examples": [
            "I should be studying right now instead of resting, I'm so lazy.",
            "I must always be strong and never show any weakness.",
            "I shouldn't feel sad about this minor thing, I should be happier.",
            "They should know what's wrong without me telling them.",
            "I ought to have figured my life out by now."
        ]
    }
]

def main():
    print(f"Initializing Gemini Client with key prefix: {GEMINI_API_KEY[:6]}...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    model_name = "gemini-embedding-001"

    precomputed = []

    for dist in DISTORTIONS_DATA:
        print(f"\nProcessing distortion category: {dist['name']}")
        for example in dist["examples"]:
            try:
                res = client.models.embed_content(model=model_name, contents=example)
                vec = []
                if hasattr(res, "embedding") and res.embedding and hasattr(res.embedding, "values"):
                    vec = res.embedding.values
                elif hasattr(res, "embeddings") and res.embeddings and len(res.embeddings) > 0:
                    vec = res.embeddings[0].values

                if vec:
                    precomputed.append({
                        "example": example,
                        "distortion_name": dist["name"],
                        "definition": dist["definition"],
                        "framework": dist["framework"],
                        "embedding": list(vec)
                    })
                    print(f"  [OK] Embedded '{example[:40]}...' (length: {len(vec)})")
                else:
                    print(f"  [FAIL] Vector empty for '{example}'")
            except Exception as e:
                print(f"  [ERROR] Failed to embed '{example}': {e}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, "distortion_embeddings.json"))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(precomputed, f)

    file_size_kb = os.path.getsize(out_path) / 1024
    print(f"\nSuccessfully generated {len(precomputed)} precomputed embeddings!")
    print(f"Saved to: {out_path} ({file_size_kb:.2f} KB)")

if __name__ == "__main__":
    main()
