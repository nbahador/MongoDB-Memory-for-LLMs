from flask import Flask, render_template, request, jsonify
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
import hashlib
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = Flask(__name__)

# Initialize with an open-access model (Microsoft Phi-2)
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

class LongTermMemory:
    def __init__(self, connection_string: str, db_name: str = "ai_memory", agent_id: str = "default_agent"):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.agent_id = agent_id
        self.memories = self.db["memories"]
        self.ensure_indexes()
        
    def ensure_indexes(self) -> None:
        self.memories.create_index([("agent_id", 1)])
        self.memories.create_index([("timestamp", -1)])
        self.memories.create_index([("tags", 1)])
        self.memories.create_index([("importance", -1)])
        self.memories.create_index([("memory_id", 1)], unique=True)
        
        existing_indexes = self.memories.index_information()
        if "content_text" not in existing_indexes:
            self.memories.create_index([("content", "text")], name="content_text")
        
    def _generate_memory_id(self, content: str) -> str:
        return hashlib.sha256(f"{self.agent_id}_{content}".encode()).hexdigest()
    
    def _ensure_timezone(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    
    def store_memory(
        self,
        content: str,
        memory_type: str = "conversation",
        importance: float = 0.5,
        tags: list = None,
        metadata: dict = None,
        timestamp: datetime = None
    ) -> str:
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}
            
        cleaned_content = self._clean_duplicate_sentences(content)
            
        memory_id = self._generate_memory_id(cleaned_content)
        timestamp = self._ensure_timezone(timestamp or datetime.now(timezone.utc))
        
        memory_doc = {
            "memory_id": memory_id,
            "agent_id": self.agent_id,
            "content": cleaned_content,
            "type": memory_type,
            "importance": max(0.0, min(1.0, importance)),
            "tags": tags,
            "metadata": metadata,
            "timestamp": timestamp,
            "last_accessed": timestamp,
            "access_count": 1,
            "created_at": timestamp,
            "updated_at": timestamp
        }
        
        try:
            self.memories.insert_one(memory_doc)
        except:
            update_data = {
                "$set": {
                    "content": cleaned_content,
                    "type": memory_type,
                    "importance": max(0.0, min(1.0, importance)),
                    "tags": tags,
                    "metadata": metadata,
                    "updated_at": timestamp,
                    "last_accessed": timestamp
                },
                "$inc": {"access_count": 1}
            }
            self.memories.update_one(
                {"memory_id": memory_id, "agent_id": self.agent_id},
                update_data
            )
        
        return memory_id
    
    def _clean_duplicate_sentences(self, text: str) -> str:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            if sentence not in seen_sentences:
                seen_sentences.add(sentence)
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences) + ('.' if text.endswith('.') else '')
    
    def retrieve_memories(
        self,
        query: str = None,
        tags: list = None,
        memory_types: list = None,
        min_importance: float = 0.0,
        max_count: int = 10,
        recency_weight: float = 0.5,
        importance_weight: float = 0.5
    ) -> list:
        filters = {"agent_id": self.agent_id, "importance": {"$gte": min_importance}}
        
        if tags:
            filters["tags"] = {"$in": tags}
            
        if memory_types:
            filters["type"] = {"$in": memory_types}
            
        if query:
            try:
                text_memories = list(self.memories.find(
                    {"$text": {"$search": query}, "agent_id": self.agent_id},
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]))
                memory_ids = [mem["memory_id"] for mem in text_memories]
                filters["memory_id"] = {"$in": memory_ids}
                memories = list(self.memories.find(filters))
                
                if not memories:
                    regex = re.compile(query, re.IGNORECASE)
                    filters["content"] = regex
                    memories = list(self.memories.find(filters))
            except:
                regex = re.compile(query, re.IGNORECASE)
                filters["content"] = regex
                memories = list(self.memories.find(filters))
        else:
            memories = list(self.memories.find(filters))
        
        now = datetime.now(timezone.utc)
        for memory in memories:
            memory_timestamp = self._ensure_timezone(memory["timestamp"])
            recency = (now - memory_timestamp).total_seconds()
            max_recency = (now - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
            normalized_recency = 1 - (recency / max_recency)
            
            memory["relevance_score"] = (
                (recency_weight * normalized_recency) + 
                (importance_weight * memory["importance"])
            )
            
            self.memories.update_one(
                {"_id": memory["_id"]},
                {"$set": {"last_accessed": now}, "$inc": {"access_count": 1}}
            )
        
        memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return memories[:max_count]

# Initialize memory system
memory = LongTermMemory("mongodb://localhost:27017", agent_id="chat_agent")

def generate_response(prompt, context_memories=None, max_length=150):
    """Generate a response using the LLM with optional context from memory"""
    # Prepare the prompt with context
    if context_memories:
        context = "\n".join([f"Context: {mem['content']}" for mem in context_memories])
        full_prompt = f"{context}\n\nQuestion: {prompt}\nAnswer:"
    else:
        full_prompt = f"Question: {prompt}\nAnswer:"
    
    # Generate response with better parameters
    response = text_generator(
        full_prompt,
        max_new_tokens=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2,
        num_return_sequences=1
    )
    
    # Extract and clean the response
    full_response = response[0]['generated_text'].replace(full_prompt, "").strip()
    
    # Post-processing to ensure quality
    clean_response = full_response.split("\n")[0].split(".")[0] + "."
    clean_response = clean_response.strip()
    
    return clean_response

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    # Store user message
    memory.store_memory(
        user_message,
        memory_type="user_input",
        importance=0.7,
        tags=["chat", "user_message"]
    )
    
    # Retrieve relevant memories for context
    relevant_memories = memory.retrieve_memories(
        query=user_message,
        max_count=3,
        importance_weight=0.7
    )
    
    # Generate response using LLM
    try:
        response = generate_response(user_message, relevant_memories)
        
        # Basic sanity check for response quality
        if not response or len(response) < 5 or response.count(".") > 3:
            response = "Let me think differently about that. " + generate_response(
                "Please rephrase your answer to: " + user_message,
                relevant_memories,
                max_length=100
            )
    except Exception as e:
        print(f"Error generating response: {e}")
        response = "I'm having trouble generating a good response right now. Could you ask that differently?"
    
    # Store AI response (after cleaning duplicates)
    memory.store_memory(
        response,
        memory_type="ai_response",
        importance=0.8,
        tags=["chat", "ai_response"],
        metadata={"generation_model": model_name}
    )
    
    return jsonify({
        'response': response,
        'context_memories': [mem['content'] for mem in relevant_memories]
    })

if __name__ == '__main__':
    app.run(debug=True)