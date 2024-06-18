# Explainable AI for Financial Advisory: Churn Prediction
This repository hosts a prototype of an Explainable AI platform for Financial Advisory. This machine learning application predicts customer churn for a telecommunications company. The project uses various machine learning models, including Random Forest, Naive Bayes, and Support Vector Machine (SVM), and incorporates Explainable AI (XAI) techniques to interpret the model predictions using SHAP.

## Repository Contents
- `main.py`: The main entry point of the application.
- `preprocessing.py`: Contains functions for data preprocessing.
- `models.py`: Contains functions for training and evaluating machine learning models.
- `evaluation.py`: Contains functions for model evaluation, including confusion matrix and ROC curve plotting.
- `explainer.py`: Contains functions for generating SHAP values and creating SHAP summary plots.
- `requirements.txt`: Lists the required Python packages for the project.
- `data/`: Directory containing the input data files.
- `stats/`: Directory where the output figures and results are saved.

Please note that data, stats and explainer files and are not included due to GitHub file size limits.

## Requirements
To execute the churn_prediction_models.ipynb notebook or the prepare_clients_dataset.py script, you will need a Python 3.8.10 virtual environment. Follow the instructions below to set up the virtual environment using a Windows Command shell:

1. Create a virtual environment named .venv: `python -m venv .venv`
2. Activate the virtual environment: `.venv\scripts\activate.bat`
3. Install the required Python packages: `pip install -r requirements.txt`
4. Using the .venv virtual environment, run the script from the Windows Command shell or execute the notebook using your preferred notebook interface.

## Contributions 
Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

## License
This project is licensed under the [MIT License](LICENSE).
