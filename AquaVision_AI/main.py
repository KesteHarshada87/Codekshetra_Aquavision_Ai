from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from groq import Groq
import os
from pydantic import BaseModel

app = FastAPI()



# ===============================
# LOCATION MODEL
# ===============================

class Location(BaseModel):
    latitude: float
    longitude: float


# ===============================
# STATIC FILES
# ===============================

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv("clean_dwlr.csv")

df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

df = df.sort_values(["station_name", "date"])

stations = df[["station_name", "latitude", "longitude"]].drop_duplicates().copy()

model = joblib.load("forecast_model.pkl")

encoder = joblib.load("station_encoder.pkl")

crop_model = joblib.load("crop_model.pkl")

area_encoder = joblib.load("area_encoder.pkl")

crop_encoder = joblib.load("crop_encoder.pkl")


# ===============================
# WATER STATUS
# ===============================

def get_water_status(depth):

    if depth < 5:
        return "critical"

    elif depth < 10:
        return "warning"

    else:
        return "safe"


# ===============================
# SAFE CROPS
# ===============================

def get_safe_crops(depth):

    status = get_water_status(depth)

    if status == "safe":
        return ["Rice", "Sugarcane", "Maize"]

    if status == "warning":
        return ["Wheat", "Maize", "Cotton"]

    return ["Millets", "Sorghum", "Pulses"]


# ===============================
# ML CROP PREDICTION
# ===============================

def predict_crop(area_name, avg_water):

    try:

        area_encoded = area_encoder.transform([area_name])[0]

        year = datetime.now().year

        production = avg_water * 100
        yield_val = avg_water * 10
        area_harvested = 50

        X = np.array([[area_encoded, year, production, yield_val, area_harvested]])

        pred = crop_model.predict(X)[0]

        crop = crop_encoder.inverse_transform([pred])[0]

        return crop

    except:

        return None


# ===============================
# STATION SEARCH (FIXED)
# ===============================

def find_station(name):

    name = name.lower()

    matches = df[df["station_name"].str.lower().str.contains(name)]

    if matches.empty:
        return None

    return matches.iloc[0]["station_name"]


# ===============================
# FORECAST FUNCTION (FIXED)
# ===============================

def generate_6_month_forecast(station_name):

    station_name = find_station(station_name)

    if station_name is None:
        return None

    station_data = df[df["station_name"] == station_name]

    if len(station_data) < 4:
        return None

    last4 = list(station_data["currentlevel"].tail(4).values)

    now = datetime.now()

    month = now.month
    year = now.year

    station_encoded = encoder.transform([station_name])[0]

    forecasts = []

    for i in range(6):

        X = np.array(
            [[last4[-1], last4[-2], last4[-3], last4[-4], month, year, station_encoded]]
        )

        pred = float(model.predict(X)[0])

        forecasts.append(round(pred, 2))

        last4.append(pred)
        last4.pop(0)

        month += 1

        if month > 12:
            month = 1
            year += 1

    avg_water = sum(forecasts) / len(forecasts)

    status = get_water_status(avg_water)

    if status == "critical":
        risk = "Critical"

    elif status == "warning":
        risk = "Warning"

    else:
        risk = "Safe"

    ml_crop = predict_crop(station_name, avg_water)

    safe_crops = get_safe_crops(avg_water)

    if ml_crop and ml_crop in safe_crops:

        crops = [ml_crop] + [c for c in safe_crops if c != ml_crop]

    else:

        crops = safe_crops

    return forecasts, round(avg_water, 2), risk, crops, station_name


# ===============================
# STATION FROM CHAT
# ===============================

def extract_station(text):

    text = text.lower()

    for station in df["station_name"].unique():

        if station.lower() in text:

            return station

    return None


# ===============================
# LOCATION FORECAST
# ===============================

@app.post("/predict_by_location")
def predict_by_location(loc: Location):

    lat = loc.latitude
    lon = loc.longitude

    stations["distance"] = np.sqrt(
        (stations["latitude"] - lat) ** 2 + (stations["longitude"] - lon) ** 2
    )

    nearest = stations.sort_values("distance").iloc[0]

    station_name = nearest["station_name"]

    result = generate_6_month_forecast(station_name)

    if not result:
        return {"error": "Not enough historical data"}

    forecasts, avg, risk, crops, station_name = result

    return {
        "station": station_name,
        "forecast": forecasts,
        "average_water_level": avg,
        "risk_level": risk,
        "recommended_crops": crops,
    }


# ===============================
# MANUAL FORECAST API (NEW)
# ===============================

@app.post("/predict6/{station_name}")
def predict6(station_name: str):

    result = generate_6_month_forecast(station_name)

    if not result:
        return {"error": "Station not found or insufficient data"}

    forecasts, avg, risk, crops, station_name = result

    return {
        "station": station_name,
        "forecast": forecasts,
        "average_water_level": avg,
        "risk_level": risk,
        "recommended_crops": crops,
    }


# ===============================
# CHATBOT
# ===============================

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.post("/chat")
async def chat(message: str):

    station = extract_station(message)

    if station:

        result = generate_6_month_forecast(station)

        if not result:
            return {"response": "Groundwater data not available for that station."}

        forecasts, avg, risk, crops, station_name = result

        response = f"""
Groundwater Forecast for {station_name}

6 Month Forecast: {forecasts}

Average Water Level: {avg} meters

Risk Level: {risk}

Recommended Crops:
{', '.join(crops)}
"""

        return {"response": response}

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": message}],
        temperature=0.3,
    )

    return {"response": completion.choices[0].message.content}


# ===============================
# DASHBOARD DATA
# ===============================

@app.get("/data")
def get_data():
    return df.head(2000).to_dict(orient="records")