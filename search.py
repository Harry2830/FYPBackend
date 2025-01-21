import os
import mysql.connector

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
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Initialize query building components
        conditions = []
        query_params = []
        limit = 10  # default limit

        # Build query conditions based on parameters
        for param in params:
            if param["parameter"] == "dish_type":
                conditions.append("d.dish_name LIKE %s")
                query_params.append(f"%{param['value']}%")
            elif param["parameter"] == "limit":
                limit = int(param["value"])

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
