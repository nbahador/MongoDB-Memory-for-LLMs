# LLM Chat with Memory

## Overview

This is a chatbot web application with long-term memory capabilities using MongoDB. Uses MongoDB as Storage to persistently store conversations. Optimizes retrieval with indexes on agent_id, timestamp, tags. Removes duplicate sentences before storage. Combines recency and importance for retrieval. Tracks last_accessed, access_count, and updated_at.

---

## Workflow

| Step | Action | Description |
|------|--------|-------------|
| 1 | User Input | User sends a message → Stored in MongoDB |
| 2 | Memory Retrieval | System retrieves relevant memories (if any) |
| 3 | Response Generation | LLM generates a response using the prompt + context |
| 4 | Memory Storage | AI response is stored in memory |
| 5 | Display | Response is displayed in the chat UI |

---

## Screenshots

### 🖥️ Web Application Interface
**LLM Chat with Memory** application running on localhost:5000
- Shows conversation about William Shakespeare's writing style and Charles Dickens' novel themes
- Displays memory context indicators for both responses
- Image: [Web Interface Screenshot](https://github.com/nbahador/MongoDB-Memory-for-LLMs/blob/main/img/img3.png)

### 📊 MongoDB Compass - Memories Collection
**MongoDB Compass interface** showing "memories" collection
- Contains conversational data with metadata
- Includes timestamps, agent IDs, content about literary topics
- Shows importance scores and access tracking information
- Image: [MongoDB Collection Screenshot](https://github.com/nbahador/MongoDB-Memory-for-LLMs/blob/main/img/img1.png)

### 📄 MongoDB Document Details
**Individual memory document** about Charles Dickens' novels
- Content: Main themes of social injustice, poverty, class divisions, human compassion
- Tagged as "ai_response" with importance score of 0.8
- Image: [MongoDB Document Screenshot](https://github.com/nbahador/MongoDB-Memory-for-LLMs/blob/main/img/img2.png)

---

## How to Run

### Requirements
| Component | Details |
|-----------|---------|
| **Python Framework** | Flask |
| **Database** | MongoDB (`mongodb://localhost:27017`) |
| **AI Libraries** | Hugging Face transformers + PyTorch |

### Setup Commands

```bash
# Start Flask app
python app.py

# Access chat UI
http://localhost:5000

# (Optional) Inspect memories
python inspect_memories.py
```

---

## 1. Core Components

### A. Flask Web Application (`app.py`)
- **Framework**: Uses Flask to handle HTTP requests/responses
- **Routes**:
  - `/` → Renders a chat interface (chat.html)
  - `/chat` (POST) → Processes user messages, generates AI responses, and stores memories

### B. AI Model (Microsoft Phi-2)
Uses Hugging Face's transformers to load:
- **Tokenizer**: `AutoTokenizer.from_pretrained("microsoft/phi-2")`
- **Model**: `AutoModelForCausalLM.from_pretrained(...)`
- **Text Generation Pipeline**: `pipeline("text-generation", ...)`

**Response Generation**:
- Takes a user prompt + context from memory
- Uses parameters like `temperature=0.7`, `top_k=50`, and `repetition_penalty=1.2` for balanced responses

### C. Long-Term Memory System (LongTermMemory Class)
- **Storage**: Uses MongoDB to persistently store conversations
- **Features**:
  - Memory Indexing: Optimizes retrieval with indexes on agent_id, timestamp, tags, etc.
  - Deduplication: Removes duplicate sentences before storage
  - Relevance Scoring: Combines recency and importance for retrieval
  - Memory Updates: Tracks last_accessed, access_count, and updated_at

### D. Memory Inspection Script (`inspect_memories.py`)
A utility to check stored memories:
- Counts total memories, user inputs, and AI responses
- Displays the latest 3 memories with timestamps and importance

---

## 2. Key Features

### A. Memory-Augmented Chat
**Stores**:
- User messages (`memory_type="user_input"`)
- AI responses (`memory_type="ai_response"`)

**Retrieves Relevant Memories**:
- Uses text search + regex fallback
- Weights recency and importance for ranking

### B. Response Generation
| Process | Description |
|---------|-------------|
| Context Retrieval | Retrieves context from memory |
| Prompt Building | Feeds prompt + context to the LLM |
| Response Processing | Post-processes the response to improve quality |

### C. Web Interface (`chat.html`)
**Frontend Features**:
- Displays chat bubbles (user vs AI)
- Shows context usage (how many memories influenced the response)
- Handles real-time messaging with JavaScript
