from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

def train_model(df):
    X = df[['product_id', 'defect_type', 'defect_date', 'defect_location', 'repair_cost', 'inspection_method']]
    X['defect_date'] = pd.to_datetime(X['defect_date']).astype(int)
    y = df['severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("F1 score:", f1)

    metrics = {"accuracy": accuracy, "f1_score": f1}
    return model, metrics

def predict_model(model, input_data):
    df = pd.DataFrame([input_data.dict()])
    df['defect_date'] = pd.to_datetime(df['defect_date']).astype(int)
    prediction = model.predict(df)
    confidence = max(model.predict_proba(df)[0])
    return ("Critical" if prediction[0] == 1 else "Non-Critical", confidence)