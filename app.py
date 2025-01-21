import os
import json
import logging
import re
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Form, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import mysql.connector
from dotenv import load_dotenv
import uvicorn
import threading
from pymongo import MongoClient
from langchain_groq import ChatGroq

import search
import auth


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URI = os.getenv("DATABASE_URI")   

db_config = {
    "host": os.getenv("DB_HOST", ""),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "desi_restaurants"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "auth_plugin": 'mysql_native_password',
    "use_pure": True
}

client = MongoClient(DATABASE_URI)
db = client["Review"]  # For storing users, history, and API data
books = db['Users']

# Initialize FastAPI application
app = FastAPI()


class UserAdd(BaseModel):
    email: str
    password: str
    name: str 

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    query: str
    parsed_params: list
    results: list



def initialize_chat_model():
    try:
        chat_groq = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
        return chat_groq
    except Exception as e:
        raise RuntimeError(f"ChatGroq initialization failed: {e}")



llm = initialize_chat_model()

@app.post("/api/search", response_model=SearchResponse)
async def search_query(query: str = Form(...)):
    try:
        user_query = query.strip()
        
        # Get LLM messages and invoke
        messages = search.get_llm_messages(user_query)
        print("*********************************************************************")
        print(messages)
        print("*********************************************************************")
        ai_msg = llm.invoke(messages)
        print("*********************************************************************")
        print(ai_msg)
        print("*********************************************************************")
        # Parse parameters
        parsed_params = search.extract_json_from_response(ai_msg.content)
        print("*********************************************************************")
        print(parsed_params)
        print("*********************************************************************")
        # Execute search with parsed parameters
        results = search.execute_query(parsed_params)
        
        return SearchResponse(
            query=user_query,
            parsed_params=parsed_params,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/signup")
async def signup(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    result_holder = {}

    # Create a UserAdd object for validation
    user = {"name":name, 
            "email":email, 
            "password":password}
    print("Validated UserAdd object:", user)

    # Start the sign-up process in a background thread
    thread = threading.Thread(target=auth.handle_signup, args=(user, result_holder))
    thread.start()
    thread.join()  # Wait for the thread to finish

    # Handle the result or error from the background thread
    if "error" in result_holder:
        raise HTTPException(status_code=400, detail=result_holder["error"])

    return {"message": result_holder["message"]}

@app.post("/api/signin")
async def signin(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    ):
    result_holder = {}

    user = {"email":email, 
            "password":password}

    # Start the sign-in process in a background thread
    thread = threading.Thread(target=auth.handle_signin, args=(user, request, result_holder))
    thread.start()
    thread.join()  # Wait for the thread to finish

    # Handle the result or error from the background thread
    if "error" in result_holder:
        raise HTTPException(status_code=401, detail=result_holder["error"])

    return {
        "message": result_holder["message"],
        "ip": result_holder["ip"],
        "name": result_holder["name"]
    }
    

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        # workers=8
    )
