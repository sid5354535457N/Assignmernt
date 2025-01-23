from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def train_model(df):
    X = df[['Temperature', 'Run_Time']]
    y = df['Downtime_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    metrics = {"accuracy": accuracy, "f1_score": f1}
    return model, metrics


def predict_model(model, input_data):
    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)
    confidence = max(model.predict_proba(df)[0])
    return ("Yes" if prediction[0] == 1 else "No", confidence)
