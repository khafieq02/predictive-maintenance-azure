[README.md](https://github.com/user-attachments/files/31197283/README.md)
# Predictive Maintenance Pipeline

A machine-learning pipeline for predicting industrial machine failures and presenting maintenance risk insights in Power BI. The project uses a Random Forest classifier trained on the AI4I 2020 Predictive Maintenance Dataset.

## Project Overview

Unexpected machine failures can cause production downtime and maintenance costs. This project uses machine-operating measurements to estimate whether a machine-failure event is likely and to generate a failure-risk probability for dashboard analysis.

The workflow includes:

1. Loading and preparing industrial machine data.
2. Training a Random Forest classification model.
3. Handling class imbalance with `class_weight=balanced`.
4. Evaluating the model on a stratified held-out test set.
5. Exporting predictions and feature importance for Power BI.
6. Registering the dataset asset and trained model in Azure Machine Learning.
7. Presenting risk summaries in a Power BI dashboard.

## Dataset

This project uses the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from the UCI Machine Learning Repository.

AI4I 2020 is **synthetic industrial data** designed to reflect realistic predictive-maintenance conditions. It should not be interpreted as direct production data.

The target variable is `Machine failure`. Input variables include machine type, air temperature, process temperature, rotational speed, torque, and tool wear.

## Model and Evaluation

The model is a `RandomForestClassifier`. The data uses an 80/20 train-test split with `random_state=42` and stratification enabled.

Class imbalance was handled with:

```python
class_weight="balanced"
```

This gives greater training importance to the minority machine-failure class.

### Test-set results

The metrics below are from `outputs/metrics.json` and were calculated on 2,000 held-out test records containing 68 failure cases.

| Metric | Score |
|---|---:|
| Accuracy | 0.9795 |
| Precision | 0.7143 |
| Recall | 0.6618 |
| F1 score | 0.6870 |
| ROC-AUC | 0.9721 |

Accuracy alone can be misleading for an imbalanced failure-prediction problem, so precision, recall, F1 score, and ROC-AUC are also reported.

## Outputs

The pipeline produces the following outputs:

- `outputs/metrics.json` — model metrics and training configuration.
- `outputs/predictions_for_powerbi.csv` — test records, predictions, and failure-risk probabilities for Power BI.
- `outputs/feature_importance.csv` — Random Forest feature-importance values.
- `outputs/confusion_matrix.png` — confusion-matrix visualization.
- `outputs/feature_importance.png` — feature-importance visualization.

## Power BI Dashboard

The Power BI dashboard includes:

- Machine Type slicer for H, L, and M.
- Total test-records KPI card.
- Actual machine-failures KPI card.
- Predicted high-risk-cases KPI card.
- Model F1-score KPI card.
- Actual failure cases by machine type.
- Tool wear versus failure-risk-probability scatter plot.
- Random Forest feature-importance chart.
- Detailed risk-summary table containing machine measurements, actual failure, predicted failure, and failure-risk probability.

The dashboard screenshot is available at:

```text
evidence/powerbi_dashboard.png
```

### Reproduce the dashboard

1. Open Power BI Desktop.
2. Load `outputs/predictions_for_powerbi.csv`.
3. Create KPI cards for total records, actual failures, predicted high-risk cases, and F1 score.
4. Add a Machine Type slicer.
5. Create a column chart for actual failures by machine type.
6. Create a scatter plot with tool wear on the x-axis and failure-risk probability on the y-axis.
7. Load `outputs/feature_importance.csv` and create a horizontal bar chart.
8. Add a table containing machine measurements, actual failure, predicted failure, and failure-risk probability.

## Azure Machine Learning

The project used Azure Machine Learning to:

- Create the workspace `pm-predictive-maintenance-2026`.
- Use the `Malaysia West` region.
- Create or upload a dataset asset.
- Register the trained Random Forest model.
- Confirm that the workspace provisioning state was `Succeeded`.

The project does not claim online endpoint deployment because no endpoint deployment evidence is included.

## Repository Structure

```text
predictive-maintenance-pipeline/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
├── evidence/
│   ├── azure_workspace.png
│   ├── azure_dataset_asset.png
│   ├── azure_registered_model.png
│   └── powerbi_dashboard.png
├── notebooks/
├── outputs/
│   ├── predictions_for_powerbi.csv
│   ├── feature_importance.csv
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── powerbi/
└── src/
    └── train_model.py
```

Update the structure above if your actual filenames differ.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

From the project root, run:

```bash
python src/train_model.py
```

The script should train the model and generate the evaluation and dashboard files inside `outputs/`.

If the project is implemented primarily as a notebook, open the notebook inside `notebooks/` and run the cells in order.

## Limitations

- The AI4I 2020 dataset is synthetic rather than production sensor data.
- The model is evaluated on a fixed held-out test split.
- The predictions are intended for analysis and dashboard demonstration.
- An online Azure endpoint and real-time monitoring are not included.
- Feature importance indicates model reliance, not causal impact.

## Skills Demonstrated

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest classification
- Binary classification
- Class-imbalance handling
- Model evaluation
- Feature-importance analysis
- Power BI dashboard development
- Azure Machine Learning workspace and model registry
- Git and GitHub

## License and Data Attribution

This repository is for educational and portfolio purposes. Refer to the [UCI dataset page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) for dataset information and attribution details.
