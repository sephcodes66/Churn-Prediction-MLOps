# User Guide: Running the MLOps Pipeline

## 🚀 Complete Step-by-Step Guide

This guide walks you through running the entire MLOps pipeline for predictive maintenance. Follow these steps to go from raw data to deployment-ready models.

---

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum
- **Storage**: 2GB free space
- **OS**: macOS, Linux, or Windows

### Required Accounts
- **Kaggle Account**: For downloading NASA dataset
- **Kaggle API Token**: Download from kaggle.com/account

### Environment Setup
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up Kaggle API
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 🔄 Phase-by-Phase Execution

### Phase 1: Data Ingestion 📥
**Goal**: Download and store NASA Turbofan dataset

```bash
# Run data ingestion
python -m src.ingest_data
```

**What happens:**
- Downloads NASA CMAPS dataset from Kaggle
- Creates `turbofan.sqlite` database
- Processes 20,149 sensor readings from 100 engines

**Expected output:**
```
--- Starting Data Ingestion Script ---
--- 1. Authenticate and Download ---
Initializing Kaggle API...
Authenticating with Kaggle...
Authentication successful.
Downloading dataset: behrad3d/nasa-cmaps to data/raw...
Download complete.
--- 2. Load Data into SQLite ---
Successfully read CSV. Initial DataFrame shape: (20149, 26)
Attempting to load 20149 rows into train_fd001 table.
Data ingestion complete.
```

**Verify success:**
```bash
# Check database was created
ls -la turbofan.sqlite

# Check table contents
sqlite3 turbofan.sqlite "SELECT COUNT(*) FROM train_fd001;"
```

---

### Phase 2: Data Splitting 🔄
**Goal**: Split data into train/validation/test sets

```bash
# Run data splitting
python -m src.data_splitter
```

**What happens:**
- Splits data by engine units (no data leakage)
- Creates 65% train, 15% validation, 20% test
- Saves CSV files in `data/processed/`

**Expected output:**
```
--- Starting Data Splitting ---
Reading data from SQLite database...
Total engines: 100
Training engines: 65 (13,096 samples)
Validation engines: 15 (3,023 samples)
Test engines: 20 (4,030 samples)
Data splitting complete.
```

**Verify success:**
```bash
# Check split files were created
ls -la data/processed/
# Should show: train.csv, val.csv, test.csv
```

---

### Phase 3: Feature Engineering 🔧
**Goal**: Transform raw data into ML-ready features

```bash
# Feature engineering is integrated into training
# But you can test it manually:
python -c "
import pandas as pd
from src.feature_engineering import FeatureEngineer

# Test on training data
df = pd.read_csv('data/processed/train.csv')
fe = FeatureEngineer()
features, _ = fe.run_feature_engineering(df.copy(), is_training=True)
print(f'Features created: {features.shape[1]} columns')
print(f'Sample features: {list(features.columns[:10])}')
"
```

**What happens:**
- Calculates RUL (Remaining Useful Life) for each engine
- Creates 63 rolling mean features (21 sensors × 3 windows)
- Creates 63 rolling std features (21 sensors × 3 windows)
- Applies PowerTransformer to normalize RUL distribution
- Saves transformer for consistent inference

**Expected output:**
```
Features created: 126 columns
Sample features: ['unit_number', 'time_in_cycles', 'op_setting_1', 'sensor_1', 'sensor_1_mean_5', 'sensor_1_mean_10', 'sensor_1_mean_20', 'sensor_1_std_5', 'sensor_1_std_10', 'sensor_1_std_20']
RUL Transformer fitted and saved to: data/processed/rul_transformer.joblib
```

---

### Phase 4: Model Training 🏋️
**Goal**: Train baseline XGBoost model

```bash
# Run model training
python -m src.model_training
```

**What happens:**
- Loads feature-engineered training data
- Trains XGBoost regression model
- Evaluates on validation set
- Logs experiment to MLflow
- Saves model to MLflow registry

**Expected output:**
```
--- Starting Model Training ---
Loading and preparing data...
Training data shape: (13096, 125)
Validation data shape: (3023, 125)
Training XGBoost model...
Model training complete.
Model registered in MLflow as: predictive_maintenance_model
Training metrics:
  - Training MAE: 28.45
  - Training MSE: 1856.32
  - Training R²: 0.62
  - Validation MAE: 32.18
  - Validation MSE: 2145.67
  - Validation R²: 0.54
```

**Verify success:**
```bash
# Check MLflow experiments
mlflow ui
# Open browser to http://localhost:5000
```

---

### Phase 5: Hyperparameter Tuning 🎯
**Goal**: Optimize model performance with Optuna

```bash
# Run hyperparameter tuning
python -m src.hyperparameter_tuning
```

**What happens:**
- Uses Optuna for Bayesian optimization
- Runs 50 trials with different hyperparameters
- Evaluates each trial on validation set
- Saves best model to MLflow registry
- Generates optimization plots

**Expected output:**
```
--- Starting Hyperparameter Tuning ---
[I 2024-01-01 10:00:00,000] A new study created in memory with name: xgboost-optimization
[I 2024-01-01 10:00:00,000] Trial 0 finished with value: 46.33 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1}
[I 2024-01-01 10:00:00,000] Trial 1 finished with value: 44.12 and parameters: {'max_depth': 8, 'n_estimators': 150, 'learning_rate': 0.05}
...
[I 2024-01-01 10:45:00,000] Trial 49 finished with value: 41.25 and parameters: {'max_depth': 8, 'n_estimators': 150, 'learning_rate': 0.05}
Best trial: 23
Best RMSE: 41.25
Best parameters: {'max_depth': 8, 'n_estimators': 150, 'learning_rate': 0.05, 'subsample': 0.85, 'colsample_bytree': 0.9}
Model registered in MLflow as: predictive_maintenance_model_tuned
```

---

### Phase 6: Model Predictions 🔮
**Goal**: Make predictions on test data

```bash
# Make predictions on test set
python -m src.prediction_pipeline --input_csv data/processed/test.csv --output_csv data/predictions/test_predictions.csv

# Generate ground truth for validation
python -c "
import pandas as pd
from src.feature_engineering import FeatureEngineer

df = pd.read_csv('data/processed/test.csv')
fe = FeatureEngineer()
df_processed, _ = fe.run_feature_engineering(df.copy(), is_training=False)
df_processed[['RUL']].to_csv('data/processed/test_ground_truth.csv', index=False)
"
```

**What happens:**
- Loads latest tuned model from MLflow
- Applies same feature engineering pipeline
- Makes RUL predictions on test data
- Inverse transforms predictions to original scale
- Saves predictions to CSV file

**Expected output:**
```
--- Starting Prediction Script ---
Loading model from MLflow registry...
Model loaded: predictive_maintenance_model_tuned (version: latest)
Loading and preprocessing data...
Input data shape: (4030, 26)
Feature-engineered data shape: (4030, 125)
Making predictions...
Predictions saved to: data/predictions/test_predictions.csv
Prediction complete.
```

---

### Phase 7: Model Validation 📊
**Goal**: Evaluate model performance on unseen data

```bash
# Validate model on test set
python -m src.model_validation --predictions_path data/predictions/test_predictions.csv --ground_truth_path data/processed/test_ground_truth.csv --output_dir validation_results
```

**What happens:**
- Compares predictions with ground truth
- Calculates performance metrics (MAE, MSE, R²)
- Generates actual vs predicted plots
- Saves validation results and visualizations

**Expected output:**
```
--- Starting Model Validation ---
Loading predictions and ground truth...
Calculating performance metrics...
Final Model Performance on Test Set:
  - Mean Squared Error (MSE): 2374.17
  - Mean Absolute Error (MAE): 36.49
  - R² Score: 0.49
  - Root Mean Squared Error (RMSE): 48.72
Generating visualizations...
Validation complete. Results saved to: validation_results/
```

---

### Phase 8: Testing 🧪
**Goal**: Run comprehensive test suite

```bash
# Run all tests
pytest -v

# Run specific test categories
pytest tests/test_data_processing.py -v
pytest tests/test_pipelines.py -v
```

**Expected output:**
```
========================= test session starts ==========================
collected 12 items

tests/test_data_processing.py::test_data_ingestion PASSED     [ 8%]
tests/test_data_processing.py::test_data_splitting PASSED    [16%]
tests/test_data_processing.py::test_feature_engineering PASSED [25%]
tests/test_pipelines.py::test_training_pipeline PASSED      [33%]
tests/test_pipelines.py::test_prediction_pipeline PASSED    [41%]
tests/test_pipelines.py::test_validation_pipeline PASSED    [50%]
...
========================= 12 passed in 45.23s ==========================
```

---

## 🎯 Complete Pipeline Execution

### Run Everything at Once
```bash
# Full pipeline execution
python pipeline.py --phase all
```

---

## 📊 Understanding Results

### Key Performance Metrics
- **MAE (Mean Absolute Error)**: 36.49 cycles
  - *Interpretation*: On average, predictions are off by ±36 cycles
  - *Business Impact*: Sufficient for maintenance scheduling
  
- **R² Score**: 0.49
  - *Interpretation*: Model explains 49% of variance in RUL
  - *Business Impact*: Reasonable predictive power for maintenance decisions

### Risk Categories
- **High Risk**: RUL < 30 cycles → Schedule immediate maintenance
- **Medium Risk**: 30 ≤ RUL < 80 cycles → Plan maintenance within 1-2 months
- **Low Risk**: RUL ≥ 80 cycles → Continue normal operations

### Sample Predictions
```csv
unit_number,time_in_cycles,predicted_RUL,risk_category
46,243,59.51,Medium Risk
33,69,106.07,Low Risk
90,50,99.37,Low Risk
10,37,150.33,Low Risk
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Kaggle API Authentication Error
```bash
# Error: Kaggle API credentials not found
# Solution: Set up Kaggle API credentials
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 2. Memory Issues During Training
```bash
# Error: Out of memory during training
# Solution: Reduce data size or use smaller model
# Edit config/main_config.yaml:
# model:
#   params:
#     n_estimators: 50  # Reduce from 100
```

#### 3. MLflow Tracking Issues
```bash
# Error: MLflow tracking server not found
# Solution: Start MLflow server
mlflow server --host 0.0.0.0 --port 5000
```

#### 4. Missing Dependencies
```bash
# Error: ModuleNotFoundError
# Solution: Install missing packages
pip install -r requirements.txt
```

---

## 🚀 Next Steps

### For Production Deployment
1. **API Development**: Create FastAPI endpoints for real-time predictions
2. **Monitoring Setup**: Implement continuous monitoring dashboard
3. **Automated Retraining**: Set up scheduled retraining pipeline
4. **Integration**: Connect with existing maintenance systems

### For Model Improvement
1. **Feature Engineering**: Add more sophisticated sensor features
2. **Model Ensemble**: Combine multiple models for better performance
3. **Deep Learning**: Try neural networks for sequence modeling
4. **Advanced Monitoring**: Implement more sophisticated drift detection

### For Scaling
1. **Distributed Processing**: Use Dask or Spark for large datasets
2. **Cloud Deployment**: Deploy on AWS, GCP, or Azure
3. **Container Orchestration**: Use Kubernetes for production
4. **CI/CD Pipeline**: Implement automated testing and deployment

---

## 📈 Performance Monitoring

### Key Metrics to Track
- **Model Performance**: R², MAE, MSE on new data
- **Data Quality**: Missing values, outliers, distribution changes
- **Prediction Accuracy**: Actual vs predicted maintenance needs
- **Business Impact**: Maintenance cost reduction, downtime prevention

### Monitoring Commands
```bash
# Check model performance
python -m src.model_monitoring

# View MLflow experiments
mlflow ui
```

This comprehensive guide provides everything needed to successfully run the MLOps pipeline and understand the results for effective predictive maintenance deployment.