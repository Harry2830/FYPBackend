import os
import json
import logging
import re
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Form, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
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
GROQ_API_KEYY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")   

# Initialize FastAPI application
app = FastAPI()

origins = [
    "http://localhost:3000",  
    "https://ai.myedbox.com/",
    "https://logsaga.com/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Explicitly specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


class SearchResponse(BaseModel):
    query: str
    parsed_params: list
    results: list


def initialize_chat_model():
    try:
        chat_groq = ChatGroq(
            api_key=GROQ_API_KEYY,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
        )
        return chat_groq
    except Exception as e:
        raise RuntimeError(f"ChatGroq initialization failed: {e}")



llm = initialize_chat_model()

@app.post("/api/api/search", response_model=SearchResponse)
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



@app.post("/api/api/signup")
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

@app.post("/api/api/signin")
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
    
@app.post("/api/api/recent")
async def recent():
    result_holder = {}

    # Start the sign-in process in a background thread
    thread = threading.Thread(target=search.recent_reviews, args=(result_holder,))
    thread.start()
    thread.join()  # Wait for the thread to finish

    # Handle the result or error from the background thread
    if "error" in result_holder:
        raise HTTPException(status_code=401, detail=result_holder["error"])

    return {
        "reviews": result_holder["reviews"],
    }
    
@app.post("/api/api/top-rated-restaurants")
async def top_rated_restaurants():
    result_holder = {}

    # Start the query process in a background thread
    thread = threading.Thread(target=search.get_top_rated_restaurants, args=(result_holder,))
    thread.start()
    thread.join()  # Wait for the thread to finish

    # Handle the result or error from the background thread
    if "error" in result_holder:
        raise HTTPException(status_code=401, detail=result_holder["error"])

    return {
        "restaurants": result_holder["restaurants"],
    }  

@app.post("/api/api/newsletter")
async def signin(
    email: str = Form(...),
    ):
    result_holder = {}

    user = {"email":email}

    # Start the sign-in process in a background thread
    thread = threading.Thread(target=auth.handle_newsletter, args=(user, result_holder))
    thread.start()
    thread.join()  # Wait for the thread to finish

    # Handle the result or error from the background thread
    if "error" in result_holder:
        raise HTTPException(status_code=401, detail=result_holder["error"])

    return {
        "message": result_holder["message"],
    }
    
@app.get("/api/api/media/{folder}/{filename}")
async def get_media_file(folder: str, filename: str):
    try:
        file_path = os.path.join("media", folder, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        # workers=4
    )
