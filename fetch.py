import os
import mysql.connector
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306))
}

def fetch_tables_and_save_to_csv():
    try:
        # Connect to the database
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # Fetch all tables
        # cursor.execute("SHOW TABLES;")
        tables = ["recommendar_restaurant"]

        # Loop through each table and fetch its data
        for table in tables:
            table_name = table
            print(f"Fetching data from table: {table_name}")
            
            # Fetch all rows from the table
            cursor.execute(f"SELECT * FROM {table_name};")
            print(cursor)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]  # Get column names

            # Convert data to a Pandas DataFrame
            df = pd.DataFrame(rows, columns=columns)

            # Save the DataFrame to a CSV file
            csv_filename = f"{table_name}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == "__main__":
    fetch_tables_and_save_to_csv()
