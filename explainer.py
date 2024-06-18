import os
import pickle
import shap
import matplotlib.pyplot as plt
import pandas as pd

def generate_shap_values(models, X_train, X_test, Y_test):
    explainer_path = 'explainer.pkl'

    try:
        with open(explainer_path, 'rb') as f:
            explainer = pickle.load(f)
            print("Loaded existing explainer.")
    except FileNotFoundError:
        better_model, _ = max(models.items(), key=lambda x: x[1].score(X_test, Y_test))
        explainer = shap.Explainer(better_model, X_train)
        with open(explainer_path, 'wb') as f:
            pickle.dump(explainer, f)
            print("Created and saved a new explainer.")

    shap_values = explainer(X_test, check_additivity=False)

    # # Create the 'stats' directory if it doesn't exist
    # os.makedirs('stats', exist_ok=True)

    # Save the SHAP summary plot
    shap_values = pd.DataFrame(shap_values)
    shap_values.columns = [f"feature_{i}" for i in range(shap_values.shape[1])]
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    figure_path = os.path.join('stats', 'shap_summary_plot.png')
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()