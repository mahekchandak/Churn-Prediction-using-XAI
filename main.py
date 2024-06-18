import pandas as pd
from preprocessing import preprocess_data
from models import train_models
from evaluation import evaluate_models
from explainer import generate_shap_values

# Load data
train = pd.read_csv("data/train/cell2celltrain.csv")
test = pd.read_csv("data/train/cell2cellholdout.csv")

# Preprocess data
X_train, X_test, Y_train, Y_test = preprocess_data(train, test)

# Train models
models = train_models(X_train, Y_train)

# Evaluate models
evaluate_models(models, X_test, Y_test)

# Generate SHAP values
generate_shap_values(models, X_train, X_test, Y_test)

# Predict on new data
input_data_path = input("Enter the path to the input CSV file: ")
inp = pd.read_csv(input_data_path)
# Preprocess input data
inp = preprocess_data(inp)
for model_name, model in models.items():
    output = model.predict(inp)
    print(f"{model_name} predictions: {output}")