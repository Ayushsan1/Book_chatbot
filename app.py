import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
import certifi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGODB_URI", "").strip()

db_enabled = True
try:
    client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where(), server_api=ServerApi('1'))
    client.admin.command("ping")
    db = client["chatbot"]
    collection = db["users"]
    print("✅ MongoDB connected successfully!")
except Exception as e:
    print(f"⚠️  MongoDB connection failed: {e}")
    db_enabled = False

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# In-memory fallback history (used when MongoDB is unavailable)
memory_history = []

prompt=ChatPromptTemplate.from_messages(
    [
    ("system",
     "You are an expert book recommendation assistant. "
     "When a user asks about a subject or topic they want to study, you help them find the best books available. "
     "For each recommended book, provide: "
     "1. Book title and author. "
     "2. Approximate pricing (new, used, and eBook formats where available). "
     "3. An analyzed review summarizing strengths, weaknesses, and who it is best suited for. "
     "4. Where to buy it (e.g., Amazon, Google Books, Open Library). "
     "Always rank books from most recommended to least, and tailor suggestions to the user's level (beginner, intermediate, advanced) if they mention it."
    ),
    ("placeholder","{history}"),
    ("user","{question}")
    ]
)

llm=ChatGroq(groq_api_key=groq_api_key,model_name="openai/gpt-oss-20b")

chain = prompt | llm


def get_history(user_id):
    if not db_enabled:
        return memory_history
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []
    for chat in chats:
        history.append((chat["role"], chat["message"]))
    return history

@app.get("/")
def home():
    return {"message": "Welcome to the GENAI learning chatbot"}

@app.post("/chat")
def chat(request: ChatRequest):
    user_id = request.user_id
    question = request.question
    history = get_history(user_id)
    response = chain.invoke({"history": history, "question": question})
    if db_enabled:
        collection.insert_one({
            "user_id": user_id,
            "role": "user",
            "message": question,
            "timestamp": datetime.now()
        })
        collection.insert_one({
            "user_id": user_id,
            "role": "assistant",
            "message": response.content,
            "timestamp": datetime.now()
        })
    else:
        memory_history.append(("user", question))
        memory_history.append(("assistant", response.content))
    return {"response": response.content}

