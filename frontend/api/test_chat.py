import urllib.request
import json

url = 'http://localhost:8000/chat'
data = {
    "message": "Mera kuch bhi karne ka mann nahi karta hai, mujhe lagta hai sab kuch bekar hai."
}
req = urllib.request.Request(url, json.dumps(data).encode('utf-8'), {'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error: {e}")
