> [!NOTE]
> This project was co-authored by a human and a coding agent. Its primary goal is to serve as a clear example of how different software components integrate and function collectively.

## For more detailed information on any of the topics, please refer to the documentation in the `docs/` directory. 


# MLOps Pipeline for Predictive Maintenance

## Overview

This repository contains a production-ready MLOps pipeline for predicting the Remaining Useful Life (RUL) of turbofan engines. It transforms raw sensor data into actionable maintenance insights using a modular, scalable, and automated system.

## Key Features

-   **End-to-End Automation:** A 9-phase pipeline handles everything from data ingestion to model deployment.
-   **Rigorous Validation:** Each phase incorporates automated statistical testing to ensure data and model quality.
-   **Advanced ML:** Compares multiple algorithms, uses Bayesian optimization for hyperparameter tuning, and provides model explanations with SHAP.
-   **Enterprise-Ready:** Built with containerization (Docker), real-time monitoring, and robust error handling.

## Quick Start

### Prerequisites

-   Python 3.8+
-   Kaggle API Credentials (`~/.kaggle/kaggle.json`)
-   Docker (Optional, for deployment)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ML-Ops-AD
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create required directories:**
    ```bash
    mkdir -p logs data/raw data/processed data/validation
    ```

### Running the Pipeline

-   **Run the full pipeline with validation:**
    ```bash
    python pipeline.py --phase all --validate
    ```

-   **Run a specific phase (e.g., model training):**
    ```bash
    python pipeline.py --phase 4
    ```

## MLOps Pipeline Phases

The pipeline automates the entire machine learning lifecycle across 9 distinct phases:

1.  **Data Ingestion:** Fetches and validates raw data.
2.  **Data Splitting:** Creates time-series aware train/test splits.
3.  **Feature Engineering:** Generates and selects relevant features.
4.  **Model Training:** Trains and evaluates multiple candidate models.
5.  **Hyperparameter Tuning:** Optimizes the best model for peak performance.
6.  **Prediction Pipeline:** Creates a production-ready inference pipeline.
7.  **Model Validation:** Performs final, in-depth model validation.
8.  **Model Monitoring:** Sets up real-time performance and data drift monitoring.
9.  **Model Deployment:** Packages and deploys the model as a service.

## Project Structure

```
ML-Ops-AD/
├── README.md              # This file
├── pipeline.py            # Main pipeline orchestrator
├── requirements.txt       # Python dependencies
├── config/                # All configuration files
├── src/                   # Source code for all pipeline phases
├── data/                  # Data storage (raw, processed, etc.)
├── tests/                 # Automated tests
└── docs/                  # Detailed documentation
```

## Configuration

-   **Main Configuration:** `config/main_config.yaml`
-   **Test Configuration:** `config/test_config.yaml`

Adjust these files to change dataset names, model parameters, or deployment settings.

## Testing and Validation

-   **Run Unit Tests:**
    ```bash
    pytest tests/
    ```
-   **Pipeline Validation:** Each pipeline run generates detailed validation reports and visualizations in the `data/validation/` directory, ensuring transparency and reliability.


