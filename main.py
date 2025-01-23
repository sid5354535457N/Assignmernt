from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
from models import train_model, predict_model
from schemas import PredictionInput, PredictionResponse

app = FastAPI()

# Global variable to store the trained model
model=None


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df.to_csv("uploaded_data.csv", index=False)
    return {"message": "File uploaded successfully."}


@app.post("/train")
def train():
    global model
    df = pd.read_csv("uploaded_data.csv")
    model, metrics = train_model(df)
    return {"message": "Model trained successfully", "metrics": metrics}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    global model
    if model is None:
        return JSONResponse(content={"error": "Model is not trained yet"}, status_code=400)

    prediction, confidence = predict_model(model, input_data)
    return {"Downtime": prediction, "Confidence": confidence}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
