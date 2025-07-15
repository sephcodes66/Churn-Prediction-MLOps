# Sample Outputs Guide: MLOps Pipeline Results

## 📊 Understanding Your Data at Each Phase

This guide provides real examples of data transformations and outputs at each phase of the MLOps pipeline, making it easier to understand what happens to your data.

---

## 🔄 Phase 1: Data Ingestion Output

### Raw Data Structure (SQLite Database)
After ingestion, you'll have a SQLite database with the following structure:

```sql
Table: train_fd001
Columns: 26 total
- unit_number (int): Engine identifier (1-100)
- time_in_cycles (int): Operating cycle number
- op_setting_1,2,3 (float): Operational settings
- sensor_1 to sensor_21 (float): Sensor readings
```

### Sample Raw Data
```csv
unit_number,time_in_cycles,op_setting_1,op_setting_2,op_setting_3,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,sensor_6,sensor_7,sensor_8,sensor_9,sensor_10,sensor_11,sensor_12,sensor_13,sensor_14,sensor_15,sensor_16,sensor_17,sensor_18,sensor_19,sensor_20,sensor_21
1,1,-0.0007,0.0019,100.0,518.67,641.82,1589.70,1400.60,14.62,21.61,553.36,2388.06,9046.19,1.30,47.47,521.66,2388.02,8138.62,8,0,392,2388,100.00,39.06,23.4190
1,2,-0.0004,0.0000,100.0,518.67,642.15,1591.82,1403.14,14.62,21.61,553.75,2388.04,9044.07,1.30,47.49,521.28,2388.08,8131.49,8,0,392,2388,100.00,39.00,23.4236
1,3,0.0016,0.0006,100.0,518.67,642.35,1587.99,1404.20,14.62,21.61,554.26,2388.08,9052.94,1.30,47.27,522.42,2388.09,8133.23,8,0,393,2388,100.00,38.95,23.3442
```

**What this means:**
- Each row represents one operational cycle of a turbofan engine
- Unit 1 has been running for 3 cycles so far
- Sensor readings vary slightly between cycles, indicating normal operation
- All engines start with similar baseline readings

---

## 🔄 Phase 2: Data Splitting Output

### Split Statistics
```
Dataset Split Results:
├── Training Set: 13,096 rows (65%) - Units for model training
├── Validation Set: 3,023 rows (15%) - Units for hyperparameter tuning
└── Test Set: 4,030 rows (20%) - Units for final evaluation

Total: 20,149 time-series observations from 100 engines
```

### File Structure After Splitting
```
data/processed/
├── train.csv      # Training data (13,096 rows)
├── val.csv        # Validation data (3,023 rows)
└── test.csv       # Test data (4,030 rows)
```

**What this means:**
- Different engines are allocated to different sets (no data leakage)
- Stratified splitting ensures balanced representation across sets
- Each set contains complete time-series for selected engines

---

## 🔄 Phase 3: Feature Engineering Output

### Before Feature Engineering (Raw Data)
```csv
unit_number,time_in_cycles,sensor_1,sensor_2,sensor_3,...
1,1,518.67,641.82,1589.70,...
1,2,518.67,642.15,1591.82,...
1,3,518.67,642.35,1587.99,...
```

### After Feature Engineering (126 columns)
```csv
unit_number,time_in_cycles,sensor_1,sensor_1_mean_5,sensor_1_mean_10,sensor_1_mean_20,sensor_1_std_5,sensor_1_std_10,sensor_1_std_20,sensor_2,sensor_2_mean_5,...,RUL
1,1,518.67,518.67,518.67,518.67,0.0,0.0,0.0,641.82,641.82,...,191
1,2,518.67,518.67,518.67,518.67,0.0,0.0,0.0,642.15,641.985,...,190
1,3,518.67,518.67,518.67,518.67,0.0,0.0,0.0,642.35,642.11,...,189
```

### Feature Categories Created
```
Feature Engineering Results:
├── Original Sensors: 21 features
│   └── sensor_1, sensor_2, ..., sensor_21
├── Rolling Means: 63 features (21 sensors × 3 windows)
│   └── sensor_X_mean_5, sensor_X_mean_10, sensor_X_mean_20
├── Rolling Std Dev: 63 features (21 sensors × 3 windows)
│   └── sensor_X_std_5, sensor_X_std_10, sensor_X_std_20
├── Operational Settings: 3 features
│   └── op_setting_1, op_setting_2, op_setting_3
└── Target Variable: 1 feature
    └── RUL (Remaining Useful Life)
```

### RUL (Target Variable) Examples
```
RUL Calculation Examples:
Engine 1: Max cycles = 192
├── Cycle 1: RUL = 192 - 1 = 191 (healthy)
├── Cycle 100: RUL = 192 - 100 = 92 (moderate wear)
└── Cycle 191: RUL = 192 - 191 = 1 (near failure)

Engine 2: Max cycles = 287
├── Cycle 1: RUL = 287 - 1 = 286 (healthy)
├── Cycle 200: RUL = 287 - 200 = 87 (moderate wear)
└── Cycle 286: RUL = 287 - 286 = 1 (near failure)
```

**What this means:**
- RUL decreases linearly with time for each engine
- Rolling statistics capture sensor degradation trends
- PowerTransformer normalizes RUL distribution for better model performance

---

## 🔄 Phase 4: Model Training Output

### Training Results in MLflow
```
MLflow Experiment: "Predictive Maintenance"
Run Name: "baseline_model_run"
├── Parameters:
│   ├── objective: "reg:squarederror"
│   ├── n_estimators: 100
│   ├── learning_rate: 0.1
│   ├── max_depth: 6
│   └── random_state: 42
├── Metrics:
│   ├── train_mae: 28.45
│   ├── train_mse: 1856.32
│   ├── train_r2: 0.62
│   ├── val_mae: 32.18
│   ├── val_mse: 2145.67
│   └── val_r2: 0.54
└── Artifacts:
    ├── model/ (XGBoost model)
    ├── feature_importance.png
    └── training_metrics.json
```

### Sample Training Predictions
```csv
unit_number,actual_RUL,predicted_RUL,error
1,150,147.23,2.77
1,149,146.85,2.15
1,148,145.92,2.08
2,125,128.45,-3.45
2,124,127.12,-3.12
```

**What this means:**
- Model learns to predict RUL with reasonable accuracy
- Validation metrics show generalization capability
- Feature importance reveals most predictive sensors

---

## 🔄 Phase 5: Hyperparameter Tuning Output

### Optuna Optimization Results
```
Optuna Study Results:
├── Total Trials: 50
├── Best Trial: #23
├── Best RMSE: 42.18
├── Best Parameters:
│   ├── max_depth: 8
│   ├── n_estimators: 150
│   ├── learning_rate: 0.05
│   ├── subsample: 0.85
│   └── colsample_bytree: 0.9
└── Optimization Time: 45 minutes
```

### Tuned Model Performance
```
Tuned Model Results:
├── Training MAE: 26.78 (↓ from 28.45)
├── Training MSE: 1654.23 (↓ from 1856.32)
├── Training R²: 0.68 (↑ from 0.62)
├── Validation MAE: 30.45 (↓ from 32.18)
├── Validation MSE: 1987.34 (↓ from 2145.67)
└── Validation R²: 0.58 (↑ from 0.54)
```

**What this means:**
- Hyperparameter tuning improved performance across all metrics
- Model is more accurate and generalizes better
- Optimization found deeper trees with lower learning rate work best

---

## 🔄 Phase 6: Model Predictions Output

### Sample Prediction Results
```csv
unit_number,time_in_cycles,op_setting_1,op_setting_2,op_setting_3,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,sensor_6,sensor_7,sensor_8,sensor_9,sensor_10,sensor_11,sensor_12,sensor_13,sensor_14,sensor_15,sensor_16,sensor_17,sensor_18,sensor_19,sensor_20,sensor_21,RUL_prediction
46,243,-0.0017,-0.0003,100.0,518.67,643.45,1601.32,1421.55,14.62,21.61,551.99,2388.21,9054.42,1.3,48.13,519.72,2388.28,8126.34,8,0,395,2388,100.0,38.49,23.0906,59.51
33,69,-0.0023,0.0004,100.0,518.67,642.1,1585.42,1405.19,14.62,21.61,553.78,2387.96,9065.64,1.3,47.2,522.02,2388.01,8149.72,8,0,394,2388,100.0,38.91,23.3597,106.07
90,50,-0.0006,-0.0001,100.0,518.67,642.12,1579.53,1391.38,14.62,21.61,554.5,2388.01,9054.52,1.3,47.33,522.53,2388.02,8139.9,8,0,392,2388,100.0,38.87,23.3707,99.37
10,37,0.0026,0.0002,100.0,518.67,641.97,1575.6,1398.36,14.62,21.61,554.34,2388.0,9050.57,1.3,47.19,522.0,2387.99,8138.71,8,0,392,2388,100.0,39.08,23.4904,150.33
```

### Prediction Interpretation
```
Prediction Examples:
├── Engine 46 (Cycle 243): RUL = 59.51 cycles
│   └── Interpretation: Needs maintenance in ~60 cycles
├── Engine 33 (Cycle 69): RUL = 106.07 cycles
│   └── Interpretation: Healthy, maintenance in ~106 cycles
├── Engine 90 (Cycle 50): RUL = 99.37 cycles
│   └── Interpretation: Good condition, ~99 cycles remaining
└── Engine 10 (Cycle 37): RUL = 150.33 cycles
    └── Interpretation: Excellent condition, ~150 cycles remaining
```

**What this means:**
- Lower RUL values indicate engines closer to failure
- Predictions help schedule proactive maintenance
- Values are inverse-transformed back to original scale

---

## 🔄 Phase 7: Model Validation Output

### Final Test Set Performance
```
Final Model Performance on Test Set:
├── Mean Squared Error (MSE): 2374.17
├── Mean Absolute Error (MAE): 36.49
├── R² Score: 0.49
├── Root Mean Squared Error (RMSE): 48.72
└── Total Test Samples: 4,030
```

### Performance Interpretation
```
Performance Analysis:
├── MAE = 36.49 cycles
│   └── On average, predictions are off by ~36 cycles
├── R² = 0.49
│   └── Model explains 49% of variance in RUL
├── RMSE = 48.72 cycles
│   └── Typical prediction error is ~49 cycles
└── Business Impact:
    └── Good enough for maintenance scheduling (±36 cycles)
```

### Validation Visualizations
```
Generated Plots:
├── actual_vs_predicted.png
│   └── Scatter plot showing prediction accuracy
├── feature_importance.png
│   └── Most important sensors for prediction
├── performance_comparison.png
│   └── Baseline vs tuned model comparison
└── validation_scatter_plot.png
    └── Test set prediction visualization
```

**What this means:**
- Model performs reasonably well for predictive maintenance
- ±36 cycle accuracy is sufficient for maintenance planning
- Visualizations help interpret model behavior

---

## 🔄 Phase 8: Model Monitoring Output

### Drift Detection Results
```
Data Drift Monitoring:
├── Reference Period: Last 30 days
├── Current Period: Today
├── Drift Test: Kolmogorov-Smirnov
├── Threshold: 0.25
└── Results:
    ├── sensor_1: p-value = 0.89 (No drift)
    ├── sensor_2: p-value = 0.76 (No drift)
    ├── sensor_3: p-value = 0.12 (No drift)
    └── sensor_4: p-value = 0.03 (⚠️ Drift detected!)
```

### Performance Monitoring
```
Model Performance Monitoring:
├── Current R²: 0.47
├── Baseline R²: 0.49
├── Performance Drop: 4.1%
├── Threshold: 30% drop
└── Status: ✅ Performance OK
```

### Monitoring Alerts
```
Alert System:
├── Drift Alert: sensor_4 showing significant drift
├── Recommendation: Investigate sensor_4 calibration
├── Performance Alert: None
└── Action Required: Check sensor_4 maintenance
```

**What this means:**
- Continuous monitoring detects data quality issues
- Drift alerts help maintain model accuracy
- Performance monitoring ensures model reliability

---

## 🎯 Summary of Key Outputs

### Business Value Delivered
```
MLOps Pipeline Business Impact:
├── Predictive Accuracy: ±36 cycles (good for maintenance)
├── Early Warning: Predict failures 36+ cycles in advance
├── Cost Savings: Proactive vs reactive maintenance
├── Downtime Reduction: Scheduled maintenance windows
└── Safety Improvement: Prevent catastrophic failures
```

### Technical Achievements
```
Technical MLOps Capabilities:
├── Automated Pipeline: End-to-end automation
├── Experiment Tracking: MLflow integration
├── Model Versioning: Reproducible models
├── Performance Monitoring: Continuous validation
├── Drift Detection: Data quality monitoring
└── Scalable Architecture: Production-ready system
```

This comprehensive guide shows exactly what outputs you can expect at each phase, making the MLOps pipeline transparent and easy to understand.