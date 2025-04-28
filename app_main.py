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


@app.post("/api/search")
async def search_query(
    query: str = Form(...),
    user_latitude: float = Form(None),
    user_longitude: float = Form(None)
):
    try:
        user_query = query.strip()
        print(f"User query: {user_query}")

        # Extract parameters using LLM
        structured_output, parsed = search.extract_parameters_with_llm(user_query)
            
        aspect = parsed.get("aspect")
        target = parsed.get("target")
        location_pref = parsed.get("location_preference", "none").lower()

        # Query database based on extracted parameters
        final_results = {}
        with auth.get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            if aspect and aspect.lower() != "general":
                print(f"📥 Running DB queries for aspect='{aspect}', target='{target}', location_preference='{location_pref}'")
                
                if location_pref == "yes" and user_latitude is not None and user_longitude is not None:
                    location_results = search.get_location_based_recommendations(cursor, aspect, user_latitude, user_longitude)
                    final_results["LocationBased"] = location_results
                    print(f"📍 {len(final_results['LocationBased'])} closest restaurant(s) selected based on user location")
                else:
                    final_results = search.get_aspect_based_recommendations(cursor, aspect)
            else:
                final_results = search.get_overall_recommendations(cursor)

        return {
            "query": user_query,
            "structured_output": structured_output,
            "parsed": parsed,
            "results": final_results
        }

    except Exception as e:
        print(f"Error: {e}")
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
