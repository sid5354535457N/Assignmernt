# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
from models import train_model, predict_model
from schemas import PredictionInput, PredictionResponse

app = FastAPI()

model = None

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df.to_csv("uploaded_data.csv", index=False)
        return {"message": "File uploaded successfully."}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/train")
def train():
    global model
    try:

        df = pd.read_csv("uploaded_data.csv")
        model, metrics = train_model(df)
        return {"message": "Model trained successfully", "metrics": metrics}
    except FileNotFoundError:
        return JSONResponse(content={"error": "No file uploaded yet. Please upload a dataset first."}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    global model
    if model is None:
        return JSONResponse(content={"error": "Model is not trained yet. Please train the model first."}, status_code=400)

    try:
        prediction, confidence = predict_model(model, input_data)
        return {"Downtime": prediction, "Confidence": confidence}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)