# app.py
from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# OpenWeatherMap API key
API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
BASE_URL = "http://api.openweathermap.org/data/2.5/"

# Default city
DEFAULT_CITY = "New York"

# -----------------------------
# Helper Functions
# -----------------------------
def kelvin_to_celsius(kelvin):
    return round(kelvin - 273.15, 2)

def kelvin_to_fahrenheit(kelvin):
    return round((kelvin - 273.15) * 9/5 + 32, 2)

def get_weather_data(city):
    try:
        url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return None, data.get("message", "City not found")
        
        weather = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temp": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "description": data["weather"][0]["description"],
            "icon": data["weather"][0]["icon"]
        }
        return weather, None
    except Exception as e:
        return None, str(e)

def get_forecast_data(city):
    try:
        url = f"{BASE_URL}forecast?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != "200":
            return None, data.get("message", "Forecast not available")

        forecast = []
        for entry in data["list"]:
            forecast.append({
                "date": entry["dt_txt"],
                "temp": entry["main"]["temp"],
                "description": entry["weather"][0]["description"],
                "icon": entry["weather"][0]["icon"]
            })
        return forecast, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    city = DEFAULT_CITY
    if request.method == "POST":
        city = request.form.get("city", DEFAULT_CITY).strip()

    weather, error = get_weather_data(city)
    forecast, forecast_error = get_forecast_data(city)

    if error or forecast_error:
        return render_template("index.html", error=error or forecast_error)

    return render_template(
        "index.html",
        weather=weather,
        forecast=forecast,
        city=city
    )

@app.route("/map", methods=["GET"])
def map_view():
    return render_template("map.html")

if __name__ == "__main__":
    app.run(debug=True)