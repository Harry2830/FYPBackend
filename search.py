import os
import mysql.connector
import math
import prompt
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
    Returns up to `limit` unique dish items (deduped by dish_name).
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)

            # --- Parse params ---
            unique_limit = 10
            dish_filter   = None
            user_lat      = None
            user_lon      = None

            for p in params:
                if p["parameter"] == "dish_type":
                    dish_filter = f"%{p['value']}%"
                elif p["parameter"] == "limit":
                    unique_limit = int(p["value"])
                elif p["parameter"] == "location_latitude":
                    user_lat = float(p["value"])
                elif p["parameter"] == "location_longitude":
                    user_lon = float(p["value"])

            # --- Build SELECT fields ---
            select_fields = [
                "r.name            AS restaurant_name",
                "d.dish_name",
                "COALESCE(o.price_override, d.price) AS price",
                "r.location_latitude  AS restaurant_latitude",
                "r.location_longitude AS restaurant_longitude",
                "r.location_name",
            ]
            query_args = []

            # if we have user location, include distance for ordering/filtering
            if user_lat is not None and user_lon is not None:
                select_fields.insert(
                    0,
                    "(6371 * acos("
                      "cos(radians(%s)) * cos(radians(r.location_latitude)) * "
                      "cos(radians(r.location_longitude) - radians(%s)) + "
                      "sin(radians(%s)) * sin(radians(r.location_latitude))"
                    ")) AS distance"
                )
                query_args.extend([user_lat, user_lon, user_lat])

            sql = f"""
                SELECT {', '.join(select_fields)}
                  FROM recommendar_dishoffering o
                  JOIN recommendar_dish       d ON o.dish_id       = d.dish_id
                  JOIN recommendar_restaurant r ON o.restaurant_id = r.restaurant_id
            """

            # --- Build WHERE clauses ---
            where = []
            if dish_filter:
                where.append("d.dish_name LIKE %s")
                query_args.append(dish_filter)

            if user_lat is not None and user_lon is not None:
                where.append(
                    "(6371 * acos("
                      "cos(radians(%s)) * cos(radians(r.location_latitude)) * "
                      "cos(radians(r.location_longitude) - radians(%s)) + "
                      "sin(radians(%s)) * sin(radians(r.location_latitude))"
                    ")) < 10"
                )
                query_args.extend([user_lat, user_lon, user_lat])

            if where:
                sql += " WHERE " + " AND ".join(where)

            # order by distance if available
            if user_lat is not None and user_lon is not None:
                sql += " ORDER BY distance"

            # no SQL LIMIT here – we'll enforce a **unique** limit in Python
            cursor.execute(sql, tuple(query_args))
            rows = cursor.fetchall()

            # --- Dedupe by dish_name & slice to unique_limit ---
            seen = set()
            unique_results = []
            for row in rows:
                dish = row["dish_name"]
                if dish not in seen:
                    seen.add(dish)
                    unique_results.append(row)
                    if len(unique_results) >= unique_limit:
                        break

            return unique_results

    except mysql.connector.Error as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": str(e)}]


import mysql.connector

def recent_reviews(result_holder: dict):
    """
    Fetch the 10 most recent reviews, including restaurant name.
    """
    connection = None
    cursor = None
    try:
        # initialize in case of early return
        result_holder["reviews"] = []

        # connect to DB
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        # pull the latest 10 reviews
        query = """
        SELECT
            rr.review_id,
            rr.review_text,
            rr.timestamp,
            rr.restaurant_id,
            rr.dish_id,
            rr.ambiance_score,
            rr.food_quality_score,
            rr.sentiment_score,
            rr.service_experience_score,
            r.name AS restaurant_name
        FROM recommendar_review AS rr
        JOIN recommendar_restaurant AS r
          ON rr.restaurant_id = r.restaurant_id
        ORDER BY rr.timestamp DESC
        LIMIT 10;
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            result_holder["error"] = "No reviews found."
        else:
            result_holder["reviews"] = rows

    except mysql.connector.Error as err:
        result_holder["error"] = f"Database error: {err}"

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def get_top_rated_restaurants(result_holder: dict):
    """
    Fetch the top 3 restaurants ranked by the average of 
    Food, Service, and Ambiance scores.
    """
    connection = None
    cursor = None

    try:
        # initialize in case of early return
        result_holder["restaurants"] = []

        # open db connection
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT
            ra.restaurant_id,
            rr.name                AS restaurant_name,
            rr.location_latitude,
            rr.location_longitude,
            rr.cuisine_type,
            rr.average_rating,
            rr.location_name,
            AVG(ra.average_sentiment_score) AS overall_average_score
        FROM recommendar_aspectanalytics AS ra
        JOIN recommendar_restaurant   AS rr
          ON ra.restaurant_id = rr.restaurant_id
        WHERE ra.aspect_type IN ('Food', 'Service', 'Ambiance')
        GROUP BY
            ra.restaurant_id,
            rr.name,
            rr.location_latitude,
            rr.location_longitude,
            rr.cuisine_type,
            rr.average_rating,
            rr.location_name
        ORDER BY overall_average_score DESC
        LIMIT 3;
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            result_holder["error"] = "No restaurants found."
        else:
            result_holder["restaurants"] = rows

    except mysql.connector.Error as err:
        result_holder["error"] = f"Database error: {err}"

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


chat_groq = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama3-70b-8192",
    temperature=0.2,
)

def extract_parameters_with_llm(query):
    """Extract aspect, target and location_preference using LLM"""
    aspect_prompt = prompt.get_aspect_extraction_prompt()
    messages = [
        ("system", aspect_prompt),
        ("human", f"Input: {query}\nOutput:")
    ]
    response = chat_groq.invoke(messages)
    structured_output = response.content.strip()
    print(f"LLaMA Response:\n{structured_output}")
    
    return structured_output, parse_llm_response(structured_output)


def parse_llm_response(structured_output):
    """Parse the LLM response to extract parameters"""
    parsed = {}
    current_param = None
    lines = structured_output.splitlines()
    for line in lines:
        line = line.strip()
        if line.lower().startswith("parameter:"):
            current_param = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("value:") and current_param:
            value = line.split(":", 1)[-1].strip()
            parsed[current_param] = value
            current_param = None
    return parsed

def get_location_based_recommendations(cursor, aspect, user_latitude, user_longitude):
    """Get recommendations based on sentiment score and location proximity"""
    query_sql = """
        SELECT a.analytics_id, a.average_sentiment_score, r.name AS restaurant_name,
               r.location_latitude, r.location_longitude, r.location_name
        FROM recommendar_aspectanalytics a
        JOIN recommendar_restaurant r ON a.restaurant_id = r.restaurant_id
        WHERE a.aspect_type = %s
        ORDER BY a.average_sentiment_score DESC
        LIMIT 10
    """
    cursor.execute(query_sql, (aspect.capitalize(),))
    candidates = cursor.fetchall()
    print(f"📤 {len(candidates)} candidate row(s) returned by sentiment score")

    # Define 10 kilometer boundary (approximate in lat/long)
    # 1 degree of latitude/longitude is roughly 111km at the equator
    # So 10km is approximately 0.09 degrees
    max_distance_degrees = 0.09
    
    filtered_candidates = []
    for candidate in candidates:
        try:
            r_lat = float(candidate["location_latitude"])
            r_lon = float(candidate["location_longitude"])
            
            # Calculate distance in degrees (approximate)
            distance_degrees = ((r_lat - user_latitude) ** 2 + (r_lon - user_longitude) ** 2) ** 0.5
            
            # Calculate distance in kilometers (approximate)
            distance_km = distance_degrees * 111
            
            # Only include restaurants within 10km
            if distance_km <= 10:
                candidate["distance"] = distance_km
                filtered_candidates.append(candidate)
        except Exception:
            continue
    
    # Sort by distance
    filtered_candidates.sort(key=lambda x: x["distance"])
    
    print(f"Found {len(filtered_candidates)} restaurants within 10km radius")
    return filtered_candidates[:3]

def get_aspect_based_recommendations(cursor, aspect):
    """Get recommendations based on aspect categories"""
    final_results = {}
    categories = {
        "Brilliant": "Brilliant_count",
        "Average": "Average_count",
        "BelowAverage": "BelowAverage_count",
        "Good": "Good_count",
        "NotRecommended": "NotRecommended_count"
    }
    selected_fields = {
        "Brilliant": ["analytics_id", "Brilliant_count", "restaurant_name", "location_latitude", "location_longitude", "location_name"],
        "Average": ["analytics_id", "Average_count", "restaurant_name", "location_latitude", "location_longitude", "location_name"],
        "BelowAverage": ["analytics_id", "BelowAverage_count", "restaurant_name", "location_latitude", "location_longitude", "location_name"],
        "Good": ["analytics_id", "Good_count", "restaurant_name", "location_latitude", "location_longitude", "location_name"],
        "NotRecommended": ["analytics_id", "NotRecommended_count", "restaurant_name", "location_latitude", "location_longitude", "location_name"]
    }
    
    for cat, col in categories.items():
        query_sql = f"""
            SELECT a.analytics_id, a.{col}, r.name AS restaurant_name, 
                   r.location_latitude, r.location_longitude, r.location_name 
            FROM recommendar_aspectanalytics a
            JOIN recommendar_restaurant r ON a.restaurant_id = r.restaurant_id
            WHERE a.aspect_type = %s
            ORDER BY a.{col} DESC
            LIMIT 3
        """
        cursor.execute(query_sql, (aspect.capitalize(),))
        rows = cursor.fetchall()
        filtered_rows = [{k: row[k] for k in selected_fields[cat]} for row in rows]
        final_results[cat] = {
            "count": len(filtered_rows),
            "data": filtered_rows
        }
        print(f"📤 {len(filtered_rows)} row(s) returned for category {cat}")
    
    return final_results