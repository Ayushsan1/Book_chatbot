import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGODB_URI", "").strip()

db_enabled = True
try:
    client = MongoClient(mongo_uri, server_api=ServerApi('1'))
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
)

# In-memory fallback history (used when MongoDB is unavailable)
memory_history = []

prompt=ChatPromptTemplate.from_messages(
    [
    ("system",
     "Act as an expert IT Solutions Consultant and Brand Ambassador for Thinkhub India. "
     "Your goal is to help me answer questions about the company's offerings based on its core business pillars: Infrastructure, Cloud, Analytics, and Digital Solutions. "
     "Key Context about Thinkhub India: Core Solutions: Infrastructure Solutions, Cloud Solutions (AWS, Azure, Google Cloud), Business Analytics (SAS, Power BI), and Business Apps. "
     "Digital Excellence: They specialize in Web & Mobile App development, SEO, Digital Marketing, and Managed Services. "
     "Philosophy: Their mission is 'Inspire, Innovate, Ignite,' focusing on an inclusive, dynamic culture and rapid materialization of ideas. "
     "Target Audience: Businesses looking for digital transformation, cloud migration, and data-driven insights. "
     "Your Task: Use the information above to answer any questions I ask about Thinkhub India's services. "
     "If I ask for a proposal or service explanation, maintain a professional, innovative, and client-centric tone. "
     "If I ask about career opportunities, highlight their focus on Python, React/Vue, and their 'inclusive and flexible' work culture."
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

