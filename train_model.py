
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

        ###Load dataset
df =pd.read_csv(r"C:\Users\shubi\Desktop\lr_ra\diabetes_Preduction\diabetes.csv")
        ###Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

        ##Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
        ###Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

        ###Save Model
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl: ")


