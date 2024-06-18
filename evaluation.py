import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import pandas as pd

def plot_confusion_matrix(model, X_test, Y_test, model_name):
    Y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    f, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, linewidth=0.7, linecolor='black', fmt='g', ax=ax, cmap="BuPu")
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    # Save the figure
    figure_path = os.path.join('stats', f'{model_name}_confusion_matrix.png')
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close(f)

def plot_roc_curve(model, X_test, Y_test, model_name):
    if model_name == 'SVM':
        Y_proba = model._predict_proba_lr(X_test)[:, 1]
    else:
        Y_proba = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(Y_test, Y_proba)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend()

    # Save the figure
    figure_path = os.path.join('stats', f'{model_name}_roc_curve.png')
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()
def evaluate_models(models, X_test, Y_test):
    # Create the 'stats' directory if it doesn't exist
    os.makedirs('stats', exist_ok=True)

    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        plot_confusion_matrix(model, X_test, Y_test, model_name)
        plot_roc_curve(model, X_test, Y_test, model_name)
        score = model.score(X_test, Y_test)
        print(f"Accuracy of {model_name}: {score}")