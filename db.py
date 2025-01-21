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

def print_table_columns():
    """Print columns for each specified table"""
    tables = [
        'AspectAnalytics',
        'AspectSentiments',
        'Dishes',
        'NLPRecommendations',
        'Restaurants',
        'Reviews',
        'Users'
    ]
    
    try:
        # Print connection info first
        print_connection_info()
        
        # Try to connect
        print("\nAttempting database connection...")
        conn = mysql.connector.connect(**DB_CONFIG)
        print("Connection successful!")
        
        cursor = conn.cursor()
        
        for table in tables:
            print(f"\n{'='*50}")
            print(f"Table: {table}")
            print('='*50)
            
            try:
                # First, verify the table exists
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = DATABASE()
                    AND table_name = %s
                """, (table,))
                
                if cursor.fetchone()[0] == 0:
                    print(f"Table {table} does not exist in the database!")
                    continue
                
                # Get column information
                cursor.execute("""
                    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                    AND TABLE_NAME = %s
                    ORDER BY ORDINAL_POSITION
                """, (table,))
                
                columns = cursor.fetchall()
                
                if not columns:
                    print(f"No columns found for table {table}")
                    continue
                
                # Print column details
                print("\nColumns:")
                print("-" * 80)
                print(f"{'Column Name':<30} {'Data Type':<15} {'Nullable':<10} {'Key':<10}")
                print("-" * 80)
                
                for column in columns:
                    name, data_type, nullable, key = column
                    print(f"{str(name):<30} {str(data_type):<15} "
                          f"{str(nullable):<10} {str(key or '-'):<10}")
                
                # Try to get sample data
                try:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 1")
                    sample_data = cursor.fetchone()
                    
                    if sample_data:
                        print("\nSample Data (First Row):")
                        print("-" * 80)
                        column_names = [desc[0] for desc in cursor.description]
                        for i, value in enumerate(sample_data):
                            print(f"{column_names[i]}: {value}")
                    else:
                        print("\nNo sample data available")
                        
                except mysql.connector.Error as e:
                    print(f"\nError getting sample data: {e}")
                
                print("\n")
                
            except mysql.connector.Error as e:
                print(f"Error processing table {table}: {e}")
                continue
            
    except mysql.connector.Error as e:
        print(f"\nDatabase error: {e}")
        print("\nPlease verify your database configuration and ensure:")
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
        print_table_columns()