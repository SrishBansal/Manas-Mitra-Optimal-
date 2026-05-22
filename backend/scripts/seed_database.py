import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

def seed_database():
    # Define path for persistent database
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.abspath(os.path.join(current_dir, "..", "chroma_db"))
    
    print(f"Initializing ChromaDB client at: {DB_PATH}")
    os.makedirs(DB_PATH, exist_ok=True)
    
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Initialize all-MiniLM-L6-v2 embedding function
    MODEL_PATH = os.path.abspath(os.path.join(current_dir, "..", "..", "all-MiniLM-L6-v2")).replace("\\", "/")
    print(f"Loading embedding function from local path: {MODEL_PATH} on CPU...")
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_PATH,
        device="cpu"
    )
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="cognitive_distortions",
        embedding_function=embedding_func
    )
    
    # Define the 5 cognitive distortions data
    distortions = [
        {
            "name": "Catastrophizing",
            "definition": "Expecting the absolute worst outcome in a situation, even with minimal evidence.",
            "framework": "Gently explore the actual probability of the feared outcome. Encourage the user to consider the best-case, worst-case, and most likely scenarios.",
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
            "examples": [
                "I should be studying right now instead of resting, I'm so lazy.",
                "I must always be strong and never show any weakness.",
                "I shouldn't feel sad about this minor thing, I should be happier.",
                "They should know what's wrong without me telling them.",
                "I ought to have figured my life out by now."
            ]
        }
    ]
    
    # Flatten database records
    documents = []
    metadatas = []
    ids = []
    
    for dist in distortions:
        name = dist["name"]
        definition = dist["definition"]
        framework = dist["framework"]
        for idx, ex in enumerate(dist["examples"]):
            documents.append(ex)
            metadatas.append({
                "distortion": name,
                "definition": definition,
                "framework": framework
            })
            ids.append(f"{name.lower().replace(' ', '_')}_{idx}")
            
    print(f"Seeding database with {len(documents)} typical thoughts...")
    
    # Clear collection if it exists to ensure clean seed
    try:
        client.delete_collection("cognitive_distortions")
        print("Cleared existing collection.")
    except Exception:
        pass
        
    collection = client.get_or_create_collection(
        name="cognitive_distortions",
        embedding_function=embedding_func
    )
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print("Database seeding completed successfully!")

if __name__ == "__main__":
    import sys
    sys.stderr = sys.stdout
    import traceback
    try:
        seed_database()
    except Exception as e:
        print("An error occurred during database seeding:")
        traceback.print_exc()
        sys.exit(1)

