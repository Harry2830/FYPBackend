import mysql.connector
from typing import Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration - for testing, print these values (without password)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306))
}

def print_connection_info():
    """Print connection information (excluding password)"""
    print("\nTrying to connect with:")
    for key, value in DB_CONFIG.items():
        if key != 'password':
            print(f"{key}: {value}")

def execute_sql_file(file_path: str):
    """Execute SQL commands from a .sql file against the database."""
    try:
        # Print connection info first
        print_connection_info()

        # Attempt to connect to the database
        print("\nAttempting database connection...")
        conn = mysql.connector.connect(**DB_CONFIG)
        print("Connection successful!")

        cursor = conn.cursor()

        # Open the SQL file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as sql_file:
            sql_commands = sql_file.read()

            # Split the SQL commands into individual statements (in case there are multiple)
            for command in sql_commands.split(';'):
                command = command.strip()
                if command:
                    try:
                        cursor.execute(command)
                        print(f"Executed command: {command[:50]}...")  # Display part of the command for reference
                    except mysql.connector.Error as e:
                        print(f"Error executing command: {e}")
                        continue
        
        # Commit the transaction (in case there were any changes)
        conn.commit()

    except mysql.connector.Error as e:
        print(f"\nDatabase error: {e}")
        print("\nPlease verify your database configuration and ensure: ")
        print("1. The database server is running")
        print("2. Your IP has access to the database")
        print("3. The database and user exist with proper permissions")
        print("4. The connection details (host, port, etc.) are correct")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    print("\nDatabase Schema Information")
    print("=" * 50)

    # If environment variables are missing, print helpful message
    required_vars = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("\nMissing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease create a .env file with the following variables:")
        print("""
        DB_HOST=your_database_host
        DB_USER=your_database_user
        DB_PASSWORD=your_database_password
        DB_NAME=your_database_name
        DB_PORT=your_database_port  # Optional, defaults to 3306
        """)
    else:
        # Provide the path to the SQL file to be executed
        sql_file_path = 'rec_db.sql'  # Update with your actual .sql file path
        execute_sql_file(sql_file_path)
