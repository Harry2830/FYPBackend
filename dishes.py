import mysql.connector
from mysql.connector import Error
import os

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306))  # Default to 3306 if not set
}

def connect_to_database():
    """Function to connect to the database."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            print("Connected to the database")
            return connection
    except Error as e:
        print(f"Error: {e}")
        return None

def insert_dishes(cursor):
    """Insert sample dish data."""
    dishes = [
        ('Margherita Pizza', 8.99, 4.5, 1),
        ('Spaghetti Bolognese', 10.50, 4.3, 1),
        ('Garlic Bread', 4.25, 4.0, 1),
        ('Cheeseburger', 7.99, 4.2, 2),
        ('Chicken Wings', 9.50, 4.4, 2),
        ('Loaded Fries', 6.75, 4.1, 2),
        ('Grilled Salmon', 14.99, 4.8, 3),
        ('Caesar Salad', 7.25, 4.6, 3),
        ('Tiramisu', 6.50, 4.7, 3),
        ('Pepperoni Pizza', 9.99, 4.0, 4),
        ('Chicken Parmesan', 12.00, 4.2, 4),
        ('Cheese Stuffed Crust Pizza', 11.50, 4.6, 5),
        ('Veggie Delight Pizza', 10.00, 4.4, 5)
    ]
    
    query = """
    INSERT INTO recommendar_dish (dish_name, price, popularity_score, restaurant_id)
    VALUES (%s, %s, %s, %s)
    """
    
    cursor.executemany(query, dishes)

def main():
    """Main function to execute database operations."""
    connection = connect_to_database()
    if connection:
        try:
            cursor = connection.cursor()
            
            # Insert dishes data
            insert_dishes(cursor)
            print("Dishes data inserted successfully.")
            
            # Commit the changes
            connection.commit()

        except Error as e:
            print(f"Error while inserting data: {e}")
            connection.rollback()

        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
                print("Database connection closed.")
    else:
        print("Failed to connect to the database.")

if __name__ == "__main__":
    main()
