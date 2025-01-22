import os
import mysql.connector
from mysql.connector import Error

# Load database configuration from environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306))  # Default to 3306 if not set
}

def test_db_connection():
    try:
        print("Attempting to connect to the database...")
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"Connected to MySQL server version: {db_info}")
            print(f"Database: {DB_CONFIG['database']} is accessible!")
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            print("Database connection closed.")

if __name__ == "__main__":
    print("Database Configuration:")
    for key, value in DB_CONFIG.items():
        print(f"{key}: {value if key != 'password' else '********'}")  # Mask password
    print("\nTesting connection...\n")
    test_db_connection()
