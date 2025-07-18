from pymongo import MongoClient
from pprint import pprint

def inspect_memories():
    client = MongoClient('mongodb://localhost:27017')
    db = client['ai_memory']
    memories = db['memories']
    
    print("\nMemory Statistics:")
    print(f"Total memories: {memories.count_documents({})}")
    print(f"User inputs: {memories.count_documents({'type': 'user_input'})}")
    print(f"AI responses: {memories.count_documents({'type': 'ai_response'})}")
    
    print("\nSample Memories:")
    for mem in memories.find().sort("timestamp", -1).limit(3):
        print(f"\n[{mem['timestamp']}] {mem['type'].upper()} (importance: {mem['importance']})")
        print(f"Content: {mem['content']}")
        if 'metadata' in mem:
            print(f"Metadata: {mem['metadata']}")

if __name__ == "__main__":
    inspect_memories()