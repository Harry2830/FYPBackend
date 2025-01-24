import os
import mysql.connector
import math


db_config = {
    "host": os.getenv("DB_HOST", ""),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "desi_restaurants"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "auth_plugin": 'mysql_native_password',
    "use_pure": True
}

def extract_json_from_response(response_text):
    """
    Extract parameters from LLM response text with proper format handling.
        
        Example input format:
        Parameter: dish_type
        Value: pasta
        Description: Type of dish requested
        """
        
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
        
    return parameters

def get_db_connection():
    try:
        return mysql.connector.connect(**db_config)
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        raise


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

    Parameter: location_latitude
    Value: 24.8607
    Description: Latitude of the user's location (if provided)

    Parameter: location_longitude
    Value: 67.0011
    Description: Longitude of the user's location (if provided)

    Rules:
    1. For number limits (like "top 10"), always use parameter name "limit" with numeric value.
    2. For food items (biryani, dosa etc), always use parameter name "dish_type".
    3. For actions (find, search, batao, dikha etc), always use parameter name "action" with value "search".
    4. For locations (area, jagah etc), use parameter name "location".
    5. For latitude and longitude, use parameters "location_latitude" and "location_longitude".
    6. Always include descriptions for each parameter.
    7. Format must match the example exactly.

    Example inputs and outputs:

    Input: "best biryani places near latitude 24.8607 and longitude 67.0011"
    Output:
    Parameter: dish_type
    Value: biryani
    Description: Type of dish requested

    Parameter: location_latitude
    Value: 24.8607
    Description: Latitude of the user's location

    Parameter: location_longitude
    Value: 67.0011
    Description: Longitude of the user's location

    Parameter: action
    Value: search
    Description: Search action requested"""

    # Create the messages list with system prompt and user query
    messages = [
        ("system", system_prompt),
        ("human", f"Extract parameters from this query: {query}")
    ]
    
    return messages

# Haversine formula function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c  # in kilometers
    return distance

def execute_query(params):
    """
    Execute search query based on structured parameters.
    Returns a list of restaurant results.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Initialize query building components
        conditions = []
        query_params = []
        limit = 10  # default limit
        user_lat = None
        user_lon = None

        # Build query conditions based on parameters
        for param in params:
            if param["parameter"] == "dish_type":
                conditions.append("d.dish_name LIKE %s")
                query_params.append(f"%{param['value']}%")
            elif param["parameter"] == "limit":
                limit = int(param["value"])
            elif param["parameter"] == "location_latitude":
                user_lat = float(param["value"])
            elif param["parameter"] == "location_longitude":
                user_lon = float(param["value"])

        # Construct the base query
        sql_query = """
        SELECT r.name AS restaurant_name, d.dish_name, d.price, d.popularity_score, r.location_latitude AS restaurant_latitude, r.location_longitude AS restaurant_longitude, d.image,r.location_name
        FROM recommendar_dish d
        JOIN recommendar_restaurant r ON d.restaurant_id = r.restaurant_id
        """

        # Add WHERE clause if we have conditions
        if conditions:
            sql_query += " WHERE " + " AND ".join(conditions)

        # Calculate the distance between user and restaurant if latitude and longitude are provided
        if user_lat is not None and user_lon is not None:
            # Ensure latitude and longitude are part of the SELECT statement to use in HAVING clause
            sql_query += """
            HAVING (6371 * acos(cos(radians(%s)) * cos(radians(r.location_latitude)) * cos(radians(r.location_longitude) - radians(%s)) + sin(radians(%s)) * sin(radians(r.location_latitude)))) < 10
            """
            query_params.extend([user_lat, user_lon, user_lat])

        # Add LIMIT clause
        sql_query += f" LIMIT {limit}"

        # Execute the query
        cursor.execute(sql_query, tuple(query_params))
        results = cursor.fetchall()

        # If no results found, log it
        if not results:
            return []

        return results

    except mysql.connector.Error as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        if conn:
            conn.close()

def recent_reviews(result_holder: dict):
    """
    Execute search query based on structured parameters.
    Returns a list of restaurant results.
    """
    connection = None
    try:
        # Initialize reviews in case of an error or no results
        result_holder["reviews"] = []

        # Establish connection to the database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)  # Use dictionary=True to get results as dicts

        # SQL query to fetch top 10 reviews sorted by timestamp
        query = """
        SELECT 
            review_id, 
            review_text, 
            overall_sentiment, 
            timestamp, 
            restaurant_id, 
            ambiance_score, 
            dish_id, 
            food_quality_score, 
            sentiment_score, 
            service_experience_score
        FROM 
            recommendar_review
        ORDER BY 
            timestamp DESC
        LIMIT 10;
        """

        # Execute the query
        cursor.execute(query)

        # Fetch the top 10 results
        results = cursor.fetchall()

        # If no results, update the error key
        if not results:
            result_holder["error"] = "No reviews found."
            return

        # Add results to the result holder
        result_holder["reviews"] = results

    except mysql.connector.Error as err:
        # Handle database errors
        result_holder["error"] = f"Database error: {err}"

    finally:
        # Ensure resources are properly cleaned up
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            
def get_top_restaurants(result_holder: dict):
    """
    Fetch the top 3 restaurants with the highest Brilliant_count.
    """
    connection = None
    try:
        # Initialize restaurants in case of an error or no results
        result_holder["restaurants"] = []

        # Establish connection to the database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)  # Use dictionary=True to get results as dicts

        # SQL query to fetch top 3 restaurants with the highest Brilliant_count
        query = """
        SELECT 
            ra.restaurant_id, 
            rr.name, 
            rr.location_latitude, 
            rr.location_longitude, 
            rr.cuisine_type, 
            rr.average_rating, 
            rr.location_name, 
            ra.Brilliant_count
        FROM 
            recommendar_aspectanalytics ra
        INNER JOIN 
            recommendar_restaurant rr
        ON 
            ra.restaurant_id = rr.restaurant_id
        ORDER BY 
            ra.Brilliant_count DESC
        LIMIT 3;
        """

        # Execute the query
        cursor.execute(query)

        # Fetch the top 3 results
        results = cursor.fetchall()

        # If no results, update the error key
        if not results:
            result_holder["error"] = "No restaurants found."
            return

        # Add results to the result holder
        result_holder["restaurants"] = results

    except mysql.connector.Error as err:
        # Handle database errors
        result_holder["error"] = f"Database error: {err}"

    finally:
        # Ensure resources are properly cleaned up
        if connection and connection.is_connected():
            cursor.close()
            connection.close()