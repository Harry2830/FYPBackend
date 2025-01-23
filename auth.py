from passlib.context import CryptContext
from pymongo import MongoClient
import os
from datetime import datetime
import pytz
from dotenv import load_dotenv
from fastapi import Request

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URI = os.getenv("DATABASE_URI")   

client = MongoClient(DATABASE_URI)
db = client["Review"]  # For storing users, history, and API data
books = db['Users']

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_local_time():
    local_tz = pytz.timezone("Asia/Karachi")
    local_time = datetime.now(local_tz)
    return local_time.strftime("%Y-%m-%d %H:%M:%S")

# Utility function to get the client's IP address
def get_client_ip(request: Request):
    return request.client.host

# Function to hash the password
def hash_password(password: str):
    return pwd_context.hash(password)

# Function to verify the password
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def handle_signup(user: dict, result_holder: dict):
    try:
        
        # Check if user already exists
        existing_user = db.Users.find_one({"email": user['email']})
        if existing_user:
            result_holder["error"] = "User already exists"
            return

        # Hash password before saving
        hashed_password = hash_password(user['password'])

        # Save the new user to the "Users" collection
        db.Users.insert_one({
            "name": user['name'],
            "email": user['email'],
            "password": hashed_password,
            "created_at": get_local_time()
        })

        # Populate result holder with success message
        result_holder["message"] = "User created successfully"
    except Exception as e:
        result_holder["error"] = str(e)
        
def handle_signin(user: dict, request: Request, result_holder: dict):
    try:
        # Find the user by email
        db_user = db.Users.find_one({"email": user['email']})
        if not db_user:
            result_holder["error"] = "Invalid email or password"
            return

        # Verify the password (hash the input and compare with stored hash)
        if not verify_password(user['password'], db_user["password"]):
            result_holder["error"] = "Invalid email or password"
            return

        # Get the client's IP address
        client_ip = get_client_ip(request)

        # Store the login history with the IP address in the "History" collection
        login_history = {
            "email": user['email'],
            "time": get_local_time(),
            "ip": client_ip,
        }
        db.History.insert_one(login_history)

        # Populate the result holder with successful login info
        result_holder["message"] = "Login successful"
        result_holder["ip"] = client_ip
        result_holder["name"] = db_user.get("name")  # Fetch the name field from the database user
    except Exception as e:
        result_holder["error"] = str(e)
        

        
        
def handle_newsletter(user: dict, result_holder: dict):
    try:
        # Find the user by email
        db_user = db.Newsletter.find_one({"email": user['email']})
        if db_user:
            result_holder["message"] = "Already subscribed!"
            return

        # Store the login history with the IP address in the "History" collection
        newsletter_subscribe = {
            "email": user['email'],
            "time": get_local_time(),
        }
        db.Newsletter.insert_one(newsletter_subscribe)

        # Populate the result holder with successful login info
        result_holder["message"] = "Thank you for subscribing!"
    except Exception as e:
        result_holder["error"] = str(e)
        
        