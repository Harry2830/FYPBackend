import requests
import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function to fetch latitude and longitude using Google Maps Geocoding API
def get_lat_lng(restaurant_name, api_key):
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {'address': restaurant_name, 'key': api_key}

    try:
        response = requests.get(geocode_url, params=params)
        data = response.json()

        if data['status'] == 'OK':
            lat = data['results'][0]['geometry']['location']['lat']
            lng = data['results'][0]['geometry']['location']['lng']
            return lat, lng
        else:
            print(f"Error for {restaurant_name}: {data['status']}")
            return None, None
    except Exception as e:
        print(f"Error fetching data for {restaurant_name}: {e}")
        return None, None

# Read the existing CSV file into a DataFrame
df = pd.read_csv("recommendar_restaurant.csv")

# Combine name and location_name to create a full address or unique identifier
df['full_address'] = df['name'] + " " + df['location_name'] + " Karachi"

# Google API Key (set this in your .env file or replace with the actual key)
api_key = os.getenv('GOOGLE_API_KEY', 'your_api_key_here')

# Fetch corrected latitude and longitude
def fetch_corrected_lat_lng(row):
    lat, lng = get_lat_lng(row['full_address'], api_key)
    return pd.Series({'corrected_latitude': lat, 'corrected_longitude': lng})

# Apply the function to fetch corrected latitudes and longitudes
df[['corrected_latitude', 'corrected_longitude']] = df.apply(fetch_corrected_lat_lng, axis=1)

# Save or display the updated DataFrame
print(df)

# Optionally, export the DataFrame to a new CSV file
df.to_csv("corrected_restaurant_locations.csv", index=False)
