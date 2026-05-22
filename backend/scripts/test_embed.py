import sys
import os
import traceback

print("Starting test direct offline...")
try:
    print("Importing SentenceTransformer...")
    from sentence_transformers import SentenceTransformer
    print("Imported successfully.")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "..", "..", "all-MiniLM-L6-v2"))
    print(f"Loading local model from: {model_path}")
    
    model = SentenceTransformer(model_path, device='cpu')
    print("Model initialized successfully!")
    
    print("Encoding text...")
    embeddings = model.encode(["hello world"])
    print(f"Success! Shape: {embeddings.shape}")
except Exception as e:
    print("Exception caught:")
    traceback.print_exc()
except BaseException as e:
    print("BaseException caught:")
    traceback.print_exc()
