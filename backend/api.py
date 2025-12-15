from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


try:
    model_path = Path('../models/xgboost_pipeline.joblib')
    model = joblib.load(model_path)
    print("Model loaded")
except FileNotFoundError:
    model = None
    print("Model not found")
    
    
class InputFeatures(BaseModel):
    accommodates: int       
    bedrooms: int
    bathrooms: int
    beds: int
    latitude: float
    longitude: float
    review_scores_rating: float
    host_listings_count: int
    number_of_reviews: int
    minimum_nights: int
    accommodate_per_bedroom: float
    beds_per_bedroom: float
    amenities_count: int
    host_response_rate: float
    review_scores_avg: float
    distance_to_center: float
    availability_ratio: float
    geo_cluster: int
    host_is_superhost: int 
    instant_bookable: int   
    host_identity_verified: int
    neighbourhood_cleansed: str
    room_type: str 
    
    
app = FastAPI()


@app.get("/")
def home():
    return {"message": "Predict AirBnB Rj it's on!"}


@app.post("/predict")
def predict_price(features: InputFeatures):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Criar DataFrame com os nomes das colunas que o modelo espera
    data_input = pd.DataFrame([{
        'accommodates': features.accommodates,
        'bedrooms': features.bedrooms,
        'bathrooms': features.bathrooms,
        'beds': features.beds,
        'latitude': features.latitude,
        'longitude': features.longitude,
        'review_scores_rating': features.review_scores_rating,
        'host_listings_count': features.host_listings_count,
        'number_of_reviews': features.number_of_reviews,
        'minimum_nights': features.minimum_nights,
        'accommodate_per_bedroom': features.accommodate_per_bedroom,
        'beds_per_bedroom': features.beds_per_bedroom,
        'amenities_count': features.amenities_count,
        'host_response_rate': features.host_response_rate,
        'review_scores_avg': features.review_scores_avg,
        'distance_to_center': features.distance_to_center,
        'availability_ratio': features.availability_ratio,
        'geo_cluster': features.geo_cluster,
        'host_is_superhost': features.host_is_superhost,
        'instant_bookable': features.instant_bookable,
        'host_identity_verified': features.host_identity_verified,
        'neighbourhood_cleansed': features.neighbourhood_cleansed,
        'room_type': features.room_type,
    }])

    try:
        prediction_log = model.predict(data_input)
        
        valor_predito = float(np.expm1(prediction_log[0]))
        
        return {
            "estimated_price": round(valor_predito, 2),
            "currency": "BRL"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict error: {str(e)}")
