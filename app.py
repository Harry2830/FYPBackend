import os
import json
import logging
import re
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Query, Form, Request,Body
from pydantic import BaseModel, ValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from dotenv import load_dotenv
import uvicorn
import threading
from pymongo import MongoClient
from typing import List, Optional, Dict ,Any
from langchain_groq import ChatGroq
import search
import auth
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import process

# Load environment
dotenv_loaded = load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI application
app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://ai.myedbox.com",
    "https://logsaga.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------- Shared Models ---------------------
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

class DishOut(BaseModel):
    dish_id:     int
    dish_name:   str
    restaurant:  str
    restaurant_location: str
    price:       float
    category:    Optional[str]
    description: Optional[str]
    cuisines:    List[str]
    avg_rating:  Optional[float]
    popularity:  Optional[float]

class RecommendationRequest(BaseModel):
    liked_restaurants: List[str]
    liked_dishes:      List[str]

class RestaurantRec(BaseModel):
    restaurant_id: int
    name:          str
    food_type:     str

class DishRec(BaseModel):
    dish_id:     int
    Restaurant:  str
    Item:        str
    PricePKR:    float
    Description: str


class InteractiveDishOut(BaseModel):
    dish_id: int
    Restaurant: str
    Item: str
    PricePKR: float
    Description: Optional[str]

class LikeBranchesRequest(BaseModel):
    email: str
    branch_ids: List[int]

class LikeDishesRequest(BaseModel):
    email: str
    dish_ids: List[int]

# --------------------- ChatGroq Init ---------------------
def initialize_chat_model():
    try:
        return ChatGroq(api_key=GROQ_API_KEY,
                        model="llama-3.3-70b-versatile",
                        temperature=0.2)
    except Exception as e:
        raise RuntimeError(f"ChatGroq init failed: {e}")
llm = initialize_chat_model()

# --------------------- Data & Model Load ---------------------
# Load full review datasets for encoding
df_reviews_rest = pd.read_csv("restaurant_final_clean_real_names.csv")
df_reviews_dish = pd.read_csv("dish_final_clean_real_names.csv")

# Metadata for endpoints
meta_rest = df_reviews_rest[["restaurant_id","name","food type"]].drop_duplicates()
name_rest = meta_rest[["restaurant_id","name"]]
meta_dish = df_reviews_dish[["dish_id","Restaurant","Item","PricePKR","Description"]].drop_duplicates()
name_dish = meta_dish[["dish_id","Item"]]

# Build label encoders based on review data
user_enc_rest = LabelEncoder().fit(df_reviews_rest["user_id"])
item_enc_rest = LabelEncoder().fit(df_reviews_rest["restaurant_id"])
user_enc_dish = LabelEncoder().fit(df_reviews_dish["user_id"])
item_enc_dish = LabelEncoder().fit(df_reviews_dish["dish_id"])

# NCF definition for both recommenders
class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=128):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(emb_dim*2,128), nn.ReLU(),
            nn.Linear(128,64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,16),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16,1),    nn.Sigmoid()
        )
    def forward(self, u, i):
        return self.fc_layers(torch.cat([self.user_embedding(u),
                                  self.item_embedding(i)],1)).squeeze()

# Instantiate and load weights using encoder class sizes
num_users_rest = len(user_enc_rest.classes_)
num_items_rest = len(item_enc_rest.classes_)
model_rest = NCF(num_users_rest, num_items_rest)
model_rest.load_state_dict(torch.load("best_model_hit60.pth"))
model_rest.eval()

num_users_dish = len(user_enc_dish.classes_)
num_items_dish = len(item_enc_dish.classes_)
model_dish = NCF(num_users_dish, num_items_dish)
model_dish.load_state_dict(torch.load("best_model_hit60_dish.pth"))
model_dish.eval()

# Helper: expand embed for new user
def expand_user_embedding(model):
    with torch.no_grad():
        w = model.user_embedding.weight
        new_w = torch.cat([w, torch.randn(1, w.size(1))], dim=0)
        model.user_embedding = nn.Embedding.from_pretrained(new_w, freeze=False)
    return new_w.size(0) - 1

# Fuzzy-match names to IDs
def fuzzy_match(names: List[str], choices: List[str], ids: List[int], thresh=80) -> List[int]:
    matched = []
    for name in names:
        match, score, _ = process.extractOne(name, choices)
        if score >= thresh:
            matched.append(ids[choices.index(match)])
    return matched

# --------------------- Recommendation Endpoints ---------------------
@app.post("/recommend/restaurants", response_model=List[RestaurantRec])
async def recommend_restaurants(
    email: str = Form(...),
    liked_restaurants: List[str] = Form(...),
    liked_dishes: List[str]      = Form(default=[]),
):
    req = RecommendationRequest(
        liked_restaurants=liked_restaurants,
        liked_dishes=liked_dishes
    )
    choices = name_rest['name'].tolist()
    ids_list = name_rest['restaurant_id'].tolist()
    liked_ids = fuzzy_match(req.liked_restaurants, choices, ids_list)
    if not liked_ids:
        raise HTTPException(status_code=400, detail="No valid restaurant matches.")
    new_uid = expand_user_embedding(model_rest)
    all_items = torch.arange(len(item_enc_rest.classes_))
    users = torch.full_like(all_items, new_uid)
    with torch.no_grad():
        scores = model_rest(users, all_items).numpy()
    mask = item_enc_rest.transform(liked_ids)
    scores[mask] = -1
    top_idx = scores.argsort()[-5:][::-1]
    top_ids = item_enc_rest.inverse_transform(top_idx)
    recs = meta_rest.set_index('restaurant_id').loc[top_ids].reset_index()
    recs.rename(columns={'food type':'food_type'}, inplace=True)

    # Persist all 5 picks
    try:
        conn = auth.get_direct_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_restaurant_recommendations WHERE user_email=%s", (email,))
        for rank, rid in enumerate(top_ids, start=1):
            cursor.execute(
                "INSERT INTO user_restaurant_recommendations"
                " (user_email, restaurant_id, rec_rank) VALUES (%s,%s,%s)",
                (email, int(rid), rank)
            )
        conn.commit()
    except Exception as e:
        logging.error(f"Failed to save restaurant recs: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

    return recs.to_dict('records')

@app.post("/recommend/dishes", response_model=List[DishRec])
async def recommend_dishes(
    email: str = Form(...),
    liked_restaurants: List[str] = Form(default=[]),
    liked_dishes:      List[str] = Form(...),
):
    req = RecommendationRequest(
        liked_restaurants=liked_restaurants,
        liked_dishes=liked_dishes
    )
    choices = name_dish['Item'].tolist()
    ids_list = name_dish['dish_id'].tolist()
    liked_ids = fuzzy_match(req.liked_dishes, choices, ids_list)
    if not liked_ids:
        raise HTTPException(status_code=400, detail="No valid dish matches.")
    new_uid = expand_user_embedding(model_dish)
    all_items = torch.arange(len(item_enc_dish.classes_))
    users = torch.full_like(all_items, new_uid)
    with torch.no_grad():
        scores = model_dish(users, all_items).numpy()
    mask = item_enc_dish.transform(liked_ids)
    scores[mask] = -1
    top_idx = scores.argsort()[-5:][::-1]
    top_ids = item_enc_dish.inverse_transform(top_idx)
    recs = meta_dish.set_index('dish_id').loc[top_ids].reset_index()

    # Persist all 5 picks
    try:
        conn = auth.get_direct_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_dish_recommendations WHERE user_email=%s", (email,))
        for rank, did in enumerate(top_ids, start=1):
            cursor.execute(
                "INSERT INTO user_dish_recommendations"
                " (user_email, dish_id, rec_rank) VALUES (%s,%s,%s)",
                (email, int(did), rank)
            )
        conn.commit()
    except Exception as e:
        logging.error(f"Failed to save dish recs: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

    return recs.to_dict('records')


# --------------------- Interactive Flow Endpoints ---------------------

# Step 1: List all restaurant branches (use these IDs in the next call)
@app.get("/interactive/restaurants", response_model=List[Restaurant])
def list_all_restaurants():
    """
    Returns every restaurant branch with its ID and details. Clients should pick three `restaurant_id` values.
    """
    conn = auth.get_direct_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT restaurant_id, name, location_latitude, location_longitude, cuisine_type, average_rating, location_name FROM recommendar_restaurant"
    )
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    for r in rows:
        r["cuisine_type"] = json.loads(r["cuisine_type"])
    return rows



class InteractiveDishOut(BaseModel):
    dish_id:     int
    Item:        str
    PricePKR:    float
    Description: str
    Restaurant:  str

class LikeBranchesRequest(BaseModel):
    branch_ids: List[int]

@app.post(
    "/interactive/restaurants/like",
    response_model=List[InteractiveDishOut],
)
def interactive_get_dishes(req: LikeBranchesRequest = Body(...)):
    """
    Step 2: Given 3 branch_ids, return all dishes from those branches.
    """
    if not req.branch_ids:
        raise HTTPException(400, "No branch_ids provided")
    placeholders = ",".join("%s" for _ in req.branch_ids)
    sql = f"""
      SELECT DISTINCT
        d.dish_id,
        d.dish_name       AS Item,
        COALESCE(o.price_override, d.price) AS PricePKR,
        d.description     AS Description,
        r.name            AS Restaurant
      FROM recommendar_dishoffering o
      JOIN recommendar_dish d       ON o.dish_id       = d.dish_id
      JOIN recommendar_restaurant r ON o.restaurant_id = r.restaurant_id
      WHERE o.restaurant_id IN ({placeholders})
    """
    conn   = auth.get_direct_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(sql, tuple(req.branch_ids))
    rows = cursor.fetchall()
    cursor.close(); conn.close()
    return rows


from fastapi import Body
from typing import Any, Dict, List
from pydantic import BaseModel

class LikeDishesRequest(BaseModel):
    email:    str
    dish_ids: List[int]

@app.post(
    "/interactive/dishes/like",
    response_model=Dict[str, Any],
)
async def interactive_like_and_recommend(
    req: LikeDishesRequest = Body(...)
):
    """
    Persist the 3 liked dish_ids, then call your /recommend REST & DISH endpoints,
    await them, save the top-5 picks, and return both lists.
    """
    email    = req.email
    dish_ids = req.dish_ids

    if not email or len(dish_ids) != 3:
        raise HTTPException(400, "Must supply email and exactly 3 dish_ids")

    # --- Persist the user's 3 likes ---
    conn = auth.get_direct_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM user_dish_recommendations WHERE user_email=%s",
            (email,),
        )
        for rank, did in enumerate(dish_ids, start=1):
            cursor.execute(
                "INSERT INTO user_dish_recommendations "
                "(user_email, dish_id, rec_rank) VALUES (%s,%s,%s)",
                (email, did, rank),
            )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    # --- Build the parameters for restaurant recs ---
    # fetch the restaurant *names* behind those dish_ids
    conn = auth.get_direct_db_connection()
    try:
        cursor = conn.cursor()
        placeholders = ",".join("%s" for _ in dish_ids)
        cursor.execute(
            f"SELECT DISTINCT r.name FROM recommendar_dishoffering o "
            f" JOIN recommendar_restaurant r ON o.restaurant_id=r.restaurant_id"
            f" WHERE o.dish_id IN ({placeholders})",
            tuple(dish_ids),
        )
        liked_restaurant_names = [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

    # --- Await the two recommendation endpoints ---
    rest_recs = await recommend_restaurants(
        email=email,
        liked_restaurants=liked_restaurant_names,
        liked_dishes=[],
    )
    # now fetch the dish names for the liked IDs
    conn = auth.get_direct_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT dish_name FROM recommendar_dish WHERE dish_id IN ({placeholders})",
            tuple(dish_ids),
        )
        liked_dish_names = [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

    dish_recs = await recommend_dishes(
        email=email,
        liked_restaurants=[],
        liked_dishes=liked_dish_names,
    )

    return {
        "restaurant_recs": rest_recs,
        "dish_recs":       dish_recs
    }


from pydantic import BaseModel
from typing import List

class BrandOut(BaseModel):
    name: str

@app.get("/interactive/restaurant-brands", response_model=List[BrandOut])
def list_brands():
    """
    Return each restaurant brand _once_ (no branches).
    Clients pick 3 brand names from this list.
    """
    conn    = auth.get_direct_db_connection()
    cursor  = conn.cursor()
    cursor.execute("SELECT DISTINCT name FROM recommendar_restaurant")
    names = [row[0] for row in cursor.fetchall()]
    cursor.close(); conn.close()
    return [{"name": n} for n in names]

class BrandListRequest(BaseModel):
    brand_names: List[str]

@app.post(
    "/interactive/restaurant-branches",
    response_model=List[Restaurant],
)
def list_branches_by_brand(req: BrandListRequest):
    """
    Given a list of brand names, return every branch (with its ID & details).
    """
    conn    = auth.get_direct_db_connection()
    cursor  = conn.cursor(dictionary=True)
    placeholders = ",".join("%s" for _ in req.brand_names)
    sql = f"""
      SELECT
        restaurant_id, name, location_latitude, location_longitude,
        cuisine_type, average_rating, location_name
      FROM recommendar_restaurant
      WHERE name IN ({placeholders})
    """
    cursor.execute(sql, tuple(req.brand_names))
    rows = cursor.fetchall()
    cursor.close(); conn.close()

    for r in rows:
        r["cuisine_type"] = json.loads(r["cuisine_type"])
    return rows





@app.post("/api/dishes/search", response_model=List[DishOut])
def search_dishes_form(
    search:       Optional[str]      = Form(None),
    cuisines:     Optional[List[str]] = Form(None),
    price_ranges: Optional[List[int]] = Form(None),
    ratings:      Optional[List[int]] = Form(None),
    sort_by:      str                 = Form("popularity"),
):
    with auth.get_direct_db_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        try:
            sql = """
            SELECT
              d.dish_id,
              d.dish_name,
              r.name AS restaurant,
              r.location_name AS restaurant_location,
              COALESCE(o.price_override, d.price) AS price,
              d.category,
              d.description,
              GROUP_CONCAT(DISTINCT r.cuisine_type) AS cuisines_csv,
              AVG(dr.rating)                     AS avg_rating,
              MAX(o.popularity_score)            AS popularity
            FROM recommendar_dish d
            JOIN recommendar_dishoffering o 
              ON d.dish_id = o.dish_id
            JOIN recommendar_restaurant r   
              ON o.restaurant_id = r.restaurant_id
            LEFT JOIN recommendar_dishofferingreview dr
              ON o.dish_offering_id = dr.dish_offering_id
            """
            where_clauses = []
            params = []

            if search:
                where_clauses.append("d.dish_name LIKE %s")
                params.append(f"%{search}%")

            if cuisines:
                or_c = []
                for c in cuisines:
                    or_c.append("JSON_CONTAINS(r.cuisine_type, %s)")
                    params.append(json.dumps(c))
                where_clauses.append("(" + " OR ".join(or_c) + ")")

            if price_ranges:
                max_price = max(price_ranges)
                where_clauses.append("COALESCE(o.price_override, d.price) <= %s")
                params.append(max_price)

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " GROUP BY d.dish_id"

            if ratings:
                min_rating = min(ratings)
                sql += " HAVING avg_rating >= %s"
                params.append(min_rating)

            sort_map = {
                "popularity": "popularity DESC",
                "price":      "price ASC",
                "rating":     "avg_rating DESC",
            }
            sql += f" ORDER BY {sort_map.get(sort_by, 'popularity DESC')}"
            sql += " LIMIT 100"

            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

            # Convert the CSV into a Python list
            result = []
            for row in rows:
                csv = row.pop("cuisines_csv") or ""
                row["cuisines"] = csv.split(",") if csv else []
                result.append(row)

            return result

        except mysql.connector.Error as e:
            raise HTTPException(status_code=500, detail=f"DB error: {e}")
        finally:
            cursor.close()

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
        port=8080,
        reload=True,
        # workers=8
    )
