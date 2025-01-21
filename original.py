import os
import json
import logging
import re
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Form, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import mysql.connector
from dotenv import load_dotenv
import uvicorn
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("restaurant_search.log")],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MySQL Configuration
db_config = {
    "host": os.getenv("DB_HOST", ""),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "desi_restaurants"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "auth_plugin": 'mysql_native_password',
    "use_pure": True
}

# Initialize FastAPI application
app = FastAPI()

# ChatGroq Initialization
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY is missing from environment variables")
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Define Request and Response Models
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
        logger.critical(f"Failed to initialize ChatGroq: {e}")
        raise RuntimeError(f"ChatGroq initialization failed: {e}")

llm = initialize_chat_model()

def get_llm_messages(query):
    system_prompt = """You are a parameter extraction system for a restaurant search application that handles both English and Hindi/Hinglish queries. For each query, extract key parameters and format them exactly like this example:

Parameter: limit
Value: 10
Description: Number of results requested

Parameter: dish_type 
Value: biryani
Description: Type of dish or cuisine requested

Parameter: action
Value: search
Description: Type of action requested

Rules:
1. For number limits (like "top 10"), always use parameter name "limit" with numeric value.
2. For food items (biryani, dosa etc), always use parameter name "dish_type".
3. For actions (find, search, batao, dikha etc), always use parameter name "action" with value "search".
4. For locations (area, jagah etc), use parameter name "location".
5. Always include descriptions for each parameter.
6. Format must match the example exactly.

Example inputs and outputs:

Input: "best biryani places in delhi"
Output:
Parameter: dish_type
Value: biryani
Description: Type of dish requested

Parameter: location
Value: delhi
Description: Search location requested

Parameter: action
Value: search
Description: Search action requested

Input: "top 5 dosa ki jagah batao"
Output:
Parameter: limit
Value: 5
Description: Number of results requested

Parameter: dish_type
Value: dosa
Description: Type of dish requested

Parameter: action
Value: search
Description: Search action requested"""

    # Create the messages list with system prompt and user query
    messages = [
        ("system", system_prompt),
        ("human", f"Extract parameters from this query: {query}")
    ]
    
    return messages


def execute_query(params):
    """
    Execute search query based on structured parameters.
    Returns a list of restaurant results.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            
            # Initialize query building components
            conditions = []
            query_params = []
            limit = 10  # default limit
            
            # Build query conditions based on parameters
            for param in params:
                if param["parameter"] == "dish_type":
                    conditions.append("(d.dish_name LIKE %s)")
                    query_params.append(f"%{param['value']}%")
                elif param["parameter"] == "limit":
                    limit = int(param['value'])
                # elif param["parameter"] == "location":
                #     conditions.append("r.location LIKE %s")
                #     query_params.append(f"%{param['value']}%")
            
            # Construct the base query
            sql_query = """
            SELECT r.name AS restaurant_name, d.dish_name, d.price, d.popularity_score
            FROM Dishes d
            JOIN Restaurants r ON d.restaurant_id = r.restaurant_id
            """
            
            # Add WHERE clause if we have conditions
            if conditions:
                sql_query += " WHERE " + " AND ".join(conditions)
            
            # Add LIMIT clause
            sql_query += f" LIMIT {limit}"
            
            logger.debug(f"Executing SQL: {sql_query}")
            logger.debug(f"With parameters: {query_params}")
            
            # Execute the query
            cursor.execute(sql_query, tuple(query_params))
            results = cursor.fetchall()
            
            # If no results found, log it
            if not results:
                logger.info("No restaurants found matching the criteria")
                return []
                
            return results
            
    except mysql.connector.Error as e:
        logger.error(f"Database query error: {e}")
        return [{"error": str(e)}]
    except Exception as e:
        logger.error(f"Unexpected error in execute_query: {e}")
        return [{"error": str(e)}]


@app.post("/search", response_model=SearchResponse)
async def search(query: str = Form(...)):
    try:
        user_query = query.strip()
        logger.debug(f"Received query: {user_query}")
        
        # Get LLM messages and invoke
        messages = get_llm_messages(user_query)
        print("*********************************************************************")
        print(messages)
        print("*********************************************************************")
        ai_msg = llm.invoke(messages)
        print("*********************************************************************")
        print(ai_msg)
        print("*********************************************************************")
        # Parse parameters
        parsed_params = extract_json_from_response(ai_msg.content)
        logger.debug(f"Parsed parameters: {parsed_params}")
        print("*********************************************************************")
        print(parsed_params)
        print("*********************************************************************")
        # Execute search with parsed parameters
        results = execute_query(parsed_params)
        
        return SearchResponse(
            query=user_query,
            parsed_params=parsed_params,
            results=results
        )
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Database connection manager
@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        yield conn
    except mysql.connector.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()

# Database query executor



# Initialize the ChatGroq model

# llm = ChatGroq(
#     model="mixtral-8x7b-32768",  # Choose the correct model
#     temperature=0,  # For deterministic responses
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other parameters...
# )

def extract_json_from_response(response_text):
    """
    Extract parameters from LLM response text with proper format handling.
        
        Example input format:
        Parameter: dish_type
        Value: pasta
        Description: Type of dish requested
        """
    logger.debug(f"Extracting parameters from response: {response_text}")
        
    parameters = []
    current_param = {}
        
        # Split the response into lines and process each line
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    
    for line in lines:
        # Skip empty lines
        if not line:
            continue
                
            # Parse Parameter line
        if line.startswith('Parameter:'):
                # If we have a complete parameter, add it to our list
            if all(k in current_param for k in ['parameter', 'value', 'description']):
                parameters.append(current_param)
                current_param = {}
                
                # Start new parameter
            param_value = line.split('Parameter:')[1].strip()
            current_param = {'parameter': param_value}
            
            # Parse Value line
        elif line.startswith('Value:'):
            value = line.split('Value:')[1].strip()
                # Convert numeric values if appropriate
            try:
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
            except ValueError:
                pass
            current_param['value'] = value
            
            # Parse Description line
        elif line.startswith('Description:'):
            current_param['description'] = line.split('Description:')[1].strip()
        
        # Don't forget to add the last parameter if complete
    if all(k in current_param for k in ['parameter', 'value', 'description']):
        parameters.append(current_param)
        
    logger.debug(f"Extracted parameters: {parameters}")
    return parameters

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
    )
