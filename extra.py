# import pandas as pd

# # Read the existing CSV file into a DataFrame
# df = pd.read_csv("corrected_restaurant_locations.csv")
# df = df.drop(["location_latitude","location_longitude"], axis=1)

# df.rename(columns={
#     'corrected_latitude': 'location_latitude',
#     'corrected_longitude': 'location_longitude'
# }, inplace=True)

# # Rename the columns
# print(df)

# df.to_csv("updated_restaurant_locations.csv", index=False)



import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': int(os.getenv('DB_PORT', 3306))
}

# Read the updated CSV into a DataFrame
df = pd.read_csv("updated_restaurant_locations.csv")

# Connect to the database
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# Loop through each row in the DataFrame and update the corresponding restaurant
for index, row in df.iterrows():
    restaurant_id = row['restaurant_id']
    location_latitude = row['location_latitude']
    location_longitude = row['location_longitude']
    
    # Update the latitude and longitude in the database
    update_query = """
    UPDATE recommendar_restaurant
    SET location_latitude = %s, location_longitude = %s
    WHERE restaurant_id = %s
    """
    cursor.execute(update_query, (location_latitude, location_longitude, restaurant_id))

# Commit the changes to the database
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()

print("Database updated successfully with new latitude and longitude values.")
