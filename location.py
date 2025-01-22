import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_lat_lng(restaurant_name, api_key):
    # URL for the Geocoding API
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json"
    
    # Parameters for the API request
    params = {
        'address': restaurant_name,
        'key': api_key
    }

    # Send the request to the API
    response = requests.get(geocode_url, params=params)
    
    # Parse the response
    data = response.json()

    if data['status'] == 'OK':
        # Extract latitude and longitude
        lat = data['results'][0]['geometry']['location']['lat']
        lng = data['results'][0]['geometry']['location']['lng']
        return lat, lng
    else:
        return None, None

# Example usage
api_key = os.getenv('GOOGLE_API_KEY', 'sss')
restaurant_name = 'Broadway Pizza Phase 8 Karachi'  # Replace with the restaurant's name or address
latitude, longitude = get_lat_lng(restaurant_name, api_key)

if latitude and longitude:
    print()
    print(f"Latitude: {latitude}, Longitude: {longitude}")
else:
    print("Couldn't find the location.")
