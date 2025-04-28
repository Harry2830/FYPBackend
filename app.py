import os
import json
import logging
import re
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Query, Form, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
import mysql.connector
from dotenv import load_dotenv
import uvicorn
import threading
from pymongo import MongoClient
from typing import List, Optional
from langchain_groq import ChatGroq
import search
import auth

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI application
app = FastAPI()

origins = [
    "http://localhost:3000",  
    "https://ai.myedbox.com",
    "https://logsaga.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Explicitly specify the frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
class Restaurant(BaseModel):
    restaurant_id: int
    name: str
    location_latitude: float
    location_longitude: float
    cuisine_type: List[str]
    average_rating: float
    location_name: str

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

# -------- GET /api/restaurants (Code 1) --------
@app.get("/api/restaurants", response_model=List[Restaurant])
def list_restaurants(
    cuisine: Optional[str] = Query(
        None,
        description="Filter by cuisine type (e.g. Pakistani). Matches if the restaurant has this cuisine in its JSON array."
    ),
    minRating: float = Query(
        0.0,
        ge=0.0,
        le=5.0,
        description="Minimum average rating (0.0–5.0)."
    )
):
    try:
        conn = auth.get_direct_db_connection()
        cursor = conn.cursor(dictionary=True)

        sql = """
        SELECT
            restaurant_id,
            name,
            CAST(location_latitude AS DECIMAL(9,6)) AS location_latitude,
            CAST(location_longitude AS DECIMAL(9,6)) AS location_longitude,
            cuisine_type,
            CAST(average_rating AS DECIMAL(2,1))         AS average_rating,
            location_name
        FROM recommendar_restaurant
        WHERE average_rating >= %s
        """
        params = [minRating]

        if cuisine:
            sql += " AND JSON_CONTAINS(cuisine_type, %s)"
            params.append(json.dumps(cuisine))

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        for row in rows:
            row["cuisine_type"] = json.loads(row["cuisine_type"])

        return rows

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()

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
    
@app.post("/api/recent")
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
    
@app.post("/api/top-rated-restaurants")
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

    
@app.post("/api/newsletter")
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
    
@app.get("/media/{folder}/{filename}")
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
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        # workers=8
    )
