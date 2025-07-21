
import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import joblib
import yaml

# ml libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# For MLFlow integration
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/model_training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Data class to hold model training settings."""
    model_types: List[str]
    cv_folds: int
    cv_strategy: str
    optimization_metric: str
    optimization_direction: str
    n_trials: int
    timeout_minutes: int
    early_stopping_rounds: int
    random_state: int

class ModelSelector:
    """Selects the best model from a list of candidates."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.results = {}
        
    def get_model_candidates(self) -> Dict[str, Any]:
        """Returns a dictionary of candidate models for evaluation."""
        candidates = {}
        
        # Add linear models if specified in the config
        if 'linear' in self.config.model_types:
            candidates['linear'] = LinearRegression()
            candidates['ridge'] = Ridge(random_state=self.config.random_state)
            candidates['lasso'] = Lasso(random_state=self.config.random_state)
            candidates['elastic_net'] = ElasticNet(random_state=self.config.random_state)
            
        # Add tree-based models if specified in the config
        if 'tree' in self.config.model_types:
            candidates['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            candidates['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.config.random_state
            )
            candidates['xgboost'] = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
        # Add neural networks if specified in the config
        if 'neural' in self.config.model_types:
            candidates['mlp'] = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.config.random_state
            )
            
        # Add Support Vector Regression if specified in the config
        if 'svm' in self.config.model_types:
            candidates['svr'] = SVR(kernel='rbf')
            
        return candidates
        
    def evaluate_model_candidates(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """Evaluates candidate models using cross-validation."""
        logger.info("Evaluating model candidates...")
        
        candidates = self.get_model_candidates()
        results = {}
        
        # Set up cross-validation strategy
        if self.config.cv_strategy == 'time_series':
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            # Default to time-aware split
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            
        for name, model in candidates.items():
            try:
                logger.info(f"Evaluating {name}...")
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X, y, 
                    cv=cv, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                # Calculate performance metrics
                mse_scores = -cv_scores
                rmse_scores = np.sqrt(mse_scores)
                
                results[name] = {
                    'rmse_mean': rmse_scores.mean(),
                    'rmse_std': rmse_scores.std(),
                    'mse_mean': mse_scores.mean(),
                    'mse_std': mse_scores.std(),
                    'cv_scores': cv_scores.tolist(),
                    'model': model
                }
                
                logger.info(f"{name}: RMSE = {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                results[name] = {
                    'error': str(e),
                    'rmse_mean': float('inf'),
                    'rmse_std': float('inf')
                }
                
        # Select the best model based on RMSE
        best_model_name = min(results.keys(), key=lambda x: results[x].get('rmse_mean', float('inf')))
        best_model = results[best_model_name]['model']
        
        logger.info(f"Best model: {best_model_name} (RMSE: {results[best_model_name]['rmse_mean']:.4f})")
        
        self.results = results
        return results, best_model_name, best_model

class HyperparameterOptimizer:
    """Optimizes model hyperparameters using Optuna."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.study = None
        self.best_params = None
        
    def get_hyperparameter_space(self, model_name: str) -> Dict[str, Any]:
        """Returns the hyperparameter space for a given model."""
        spaces = {
            'xgboost': {
                'n_estimators': (50, 500),
                'max_depth': (3, 12),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10),
                'min_child_weight': (1, 10)
            },
            'random_forest': {
                'n_estimators': (50, 500),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'gradient_boosting': {
                'n_estimators': (50, 500),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 12),
                'subsample': (0.6, 1.0),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10)
            },
            'ridge': {
                'alpha': (0.1, 100.0)
            },
            'lasso': {
                'alpha': (0.1, 100.0)
            },
            'elastic_net': {
                'alpha': (0.1, 100.0),
                'l1_ratio': (0.1, 0.9)
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'alpha': (0.0001, 0.1),
                'learning_rate_init': (0.001, 0.1)
            }
        }
        
        return spaces.get(model_name, {})
        
    def optimize_hyperparameters(self, 
                                model_name: str, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                base_model: Any) -> Tuple[Dict, float]:
        """Optimizes model hyperparameters using Optuna."""
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        # Get hyperparameter space for the selected model
        param_space = self.get_hyperparameter_space(model_name)
        
        if not param_space:
            logger.warning(f"No hyperparameter space defined for {model_name}")
            return {}, float('inf')
            
        # Create Optuna objective function
        def objective(trial):
            params = {}
            
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                    
            # Create model with suggested hyperparameters
            if model_name == 'xgboost':
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    **params
                )
            elif model_name == 'random_forest':
                model = RandomForestRegressor(
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    **params
                )
            elif model_name == 'gradient_boosting':
                model = GradientBoostingRegressor(
                    random_state=self.config.random_state,
                    **params
                )
            elif model_name == 'ridge':
                model = Ridge(random_state=self.config.random_state, **params)
            elif model_name == 'lasso':
                model = Lasso(random_state=self.config.random_state, **params)
            elif model_name == 'elastic_net':
                model = ElasticNet(random_state=self.config.random_state, **params)
            elif model_name == 'mlp':
                model = MLPRegressor(
                    random_state=self.config.random_state,
                    max_iter=500,
                    **params
                )
            else:
                model = base_model
                
            # Perform cross-validation
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            
            return -scores.mean()  # Return positive MSE for minimization
            
        # Create an Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner()
        )
        
        # Start the optimization process
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_minutes * 60
        )
        
        self.study = study
        self.best_params = study.best_params
        
        logger.info(f"Optimization completed. Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        return study.best_params, study.best_value

class ModelEvaluator:
    """Evaluates the performance of a trained model."""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, 
                      model: Any, 
                      X_train: pd.DataFrame, 
                      y_train: pd.Series,
                      X_val: pd.DataFrame, 
                      y_val: pd.Series,
                      model_name: str) -> Dict:
        """Evaluates the model on training and validation sets."""
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate performance metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        val_metrics = self._calculate_metrics(y_val, y_val_pred)
        
        # Get feature importance
        feature_importance = self._get_feature_importance(model, X_train.columns)
        
        # Estimate model complexity
        complexity = self._estimate_model_complexity(model)
        
        # Analyze residuals
        residuals = self._analyze_residuals(y_val, y_val_pred)
        
        results = {
            'model_name': model_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': feature_importance,
            'model_complexity': complexity,
            'residual_analysis': residuals,
            'overfitting_score': self._calculate_overfitting_score(train_metrics, val_metrics)
        }
        
        self.evaluation_results[model_name] = results
        return results
        
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculates regression metrics."""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        }
        
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict:
        """Extracts feature importance from the model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                return {}
                
            # Create a dictionary of feature importances
            feature_importance = dict(zip(feature_names, importance))
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'feature_importance': sorted_importance,
                'top_10_features': dict(list(sorted_importance.items())[:10])
            }
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
            
    def _estimate_model_complexity(self, model: Any) -> Dict:
        """Estimates the complexity of the model."""
        complexity = {}
        
        if hasattr(model, 'n_estimators'):
            complexity['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            complexity['max_depth'] = model.max_depth
        if hasattr(model, 'n_features_in_'):
            complexity['n_features'] = model.n_features_in_
            
        # Estimate total parameters for tree-based models
        if hasattr(model, 'tree_') and hasattr(model.tree_, 'node_count'):
            complexity['tree_nodes'] = model.tree_.node_count
        elif hasattr(model, 'estimators_'):
            complexity['total_trees'] = len(model.estimators_)
            
        return complexity
        
    def _analyze_residuals(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Analyzes the prediction residuals."""
        residuals = y_true - y_pred
        
        return {
            'residual_mean': float(residuals.mean()),
            'residual_std': float(residuals.std()),
            'residual_skewness': float(residuals.skew()),
            'residual_kurtosis': float(residuals.kurtosis()),
            'residual_range': [float(residuals.min()), float(residuals.max())],
            'heteroscedasticity': self._test_heteroscedasticity(y_pred, residuals)
        }
        
    def _test_heteroscedasticity(self, y_pred: np.ndarray, residuals: pd.Series) -> Dict:
        """Tests for heteroscedasticity in residuals."""
        try:
            from scipy.stats import pearsonr
            
            # Correlation between predicted values and absolute residuals
            correlation, p_value = pearsonr(y_pred, np.abs(residuals))
            
            return {
                'correlation': float(correlation),
                'p_value': float(p_value),
                'heteroscedastic': bool(p_value < 0.05)
            }
        except Exception:
            return {'error': 'Could not test heteroscedasticity'}
            
    def _calculate_overfitting_score(self, train_metrics: Dict, val_metrics: Dict) -> float:
        """Calculates a score to measure overfitting."""
        try:
            train_r2 = train_metrics['r2']
            val_r2 = val_metrics['r2']
            
            # Higher score indicates more overfitting
            return max(0, train_r2 - val_r2)
        except Exception:
            return 0.0

class ModelTrainer:
    """Trains and evaluates machine learning models."""
    
    def __init__(self, config_path: str = "config/main_config.yaml"):
        self.config = self._load_config(config_path)
        self.model_config = self._create_model_config()
        self.selector = ModelSelector(self.model_config)
        self.optimizer = HyperparameterOptimizer(self.model_config)
        self.evaluator = ModelEvaluator()
        
    def _load_config(self, config_path: str) -> Dict:
        """Loads the main configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    def _create_model_config(self) -> ModelConfig:
        """Creates a ModelConfig object from the main config."""
        return ModelConfig(
            model_types=['linear', 'tree', 'neural'],
            cv_folds=5,
            cv_strategy='time_series',
            optimization_metric='rmse',
            optimization_direction='minimize',
            n_trials=50,
            timeout_minutes=30,
            early_stopping_rounds=10,
            random_state=42
        )
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Loads training and validation data."""
        logger.info("Loading training data...")
        
        # Load processed data from the specified directory
        processed_dir = Path(self.config["data"]["processed_data_dir"])
        
        train_df = pd.read_csv(processed_dir / "train_processed.csv")
        val_df = pd.read_csv(processed_dir / "val_processed.csv")
        
        # Separate features and target variable
        feature_cols = [col for col in train_df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
        
        X_train = train_df[feature_cols]
        y_train = train_df['RUL']
        X_val = val_df[feature_cols]
        y_val = val_df['RUL']
        
        logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
        
    def train_and_evaluate_models(self) -> Dict:
        """Trains, evaluates, and saves the best model."""
        logger.info("=== Starting Model Training and Evaluation ===")
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_data()
        
        # Select the best model candidate
        candidate_results, best_model_name, best_model = self.selector.evaluate_model_candidates(X_train, y_train)
        
        # Optimize hyperparameters for the best model
        best_params, best_score = self.optimizer.optimize_hyperparameters(
            best_model_name, X_train, y_train, best_model
        )
        
        # Train the final model with optimal parameters
        final_model = self._create_final_model(best_model_name, best_params)
        final_model.fit(X_train, y_train)
        
        # Perform a comprehensive evaluation
        evaluation_results = self.evaluator.evaluate_model(
            final_model, X_train, y_train, X_val, y_val, best_model_name
        )
        
        # Consolidate training results
        training_results = {
            'timestamp': datetime.now().isoformat(),
            'best_model': best_model_name,
            'best_params': best_params,
            'best_score': best_score,
            'candidate_results': candidate_results,
            'evaluation_results': evaluation_results,
            'model_config': {
                'model_types': self.model_config.model_types,
                'cv_folds': self.model_config.cv_folds,
                'n_trials': self.model_config.n_trials
            }
        }
        
        # Save the model and associated results
        self._save_model_and_results(final_model, training_results, X_val, y_val)
        
        return training_results
        
    def _create_final_model(self, model_name: str, params: Dict) -> Any:
        """Creates the final model with optimized hyperparameters."""
        if model_name == 'xgboost':
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.model_config.random_state,
                n_jobs=-1,
                **params
            )
        elif model_name == 'random_forest':
            return RandomForestRegressor(
                random_state=self.model_config.random_state,
                n_jobs=-1,
                **params
            )
        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(
                random_state=self.model_config.random_state,
                **params
            )
        elif model_name == 'ridge':
            return Ridge(random_state=self.model_config.random_state, **params)
        elif model_name == 'lasso':
            return Lasso(random_state=self.model_config.random_state, **params)
        elif model_name == 'elastic_net':
            return ElasticNet(random_state=self.model_config.random_state, **params)
        elif model_name == 'mlp':
            return MLPRegressor(
                random_state=self.model_config.random_state,
                max_iter=500,
                **params
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    def _save_model_and_results(self, 
                               model: Any, 
                               results: Dict, 
                               X_val: pd.DataFrame, 
                               y_val: pd.Series):
        """Saves the model, results, and logs to MLFlow."""
        logger.info("Saving model and results...")
        
        # Start an MLFlow run
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        
        with mlflow.start_run() as run:
            # Log model parameters
            mlflow.log_params(results['best_params'])
            mlflow.log_param('model_type', results['best_model'])
            
            # Log validation metrics
            val_metrics = results['evaluation_results']['val_metrics']
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)
                
            # Log training metrics
            train_metrics = results['evaluation_results']['train_metrics']
            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)
                
            # Log the model to MLFlow
            signature = infer_signature(X_val, y_val)
            
            if results['best_model'] == 'xgboost':
                mlflow.xgboost.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=X_val.head(5),
                    registered_model_name=self.config["model"]["name"]
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=X_val.head(5),
                    registered_model_name=self.config["model"]["name"]
                )
                
            # Save model and results locally
            os.makedirs("data/models", exist_ok=True)
            
            # Save the trained model
            joblib.dump(model, "data/models/final_model.joblib")
            
            # Save training results as JSON
            with open("data/models/training_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            # Create and save training visualizations
            self._create_training_visualizations(model, results, X_val, y_val)
            
            logger.info(f"Model saved to MLflow. Run ID: {run.info.run_id}")
            
    def _create_training_visualizations(self, 
                                      model: Any, 
                                      results: Dict, 
                                      X_val: pd.DataFrame, 
                                      y_val: pd.Series):
        """Creates and saves visualizations related to training."""
        os.makedirs("data/visualizations", exist_ok=True)
        
        # Plot 1: Model comparison
        self._plot_model_comparison(results['candidate_results'])
        
        # Plot 2: Feature importance
        self._plot_feature_importance(results['evaluation_results']['feature_importance'])
        
        # Plot 3: Predictions vs. actual values
        self._plot_predictions_vs_actual(model, X_val, y_val)
        
        # Plot 4: Residuals analysis
        self._plot_residuals(model, X_val, y_val)
        
        # Plot 5: Hyperparameter optimization history
        if self.optimizer.study is not None:
            self._plot_optimization_history(self.optimizer.study)
            
    def _plot_model_comparison(self, candidate_results: Dict):
        """Plots a comparison of candidate models."""
        if not candidate_results:
            return
            
        models = list(candidate_results.keys())
        rmse_means = [candidate_results[m].get('rmse_mean', float('inf')) for m in models]
        rmse_stds = [candidate_results[m].get('rmse_std', 0) for m in models]
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(models, rmse_means, yerr=rmse_stds, fmt='o', capsize=5)
        plt.title('Model Comparison - RMSE with Standard Deviation')
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/visualizations/model_comparison.png', dpi=300)
        plt.close()
        
    def _plot_feature_importance(self, feature_importance: Dict):
        """Plots the top 10 most important features."""
        if not feature_importance or 'top_10_features' not in feature_importance:
            return
            
        features = list(feature_importance['top_10_features'].keys())
        importance = list(feature_importance['top_10_features'].values())
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, importance)
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('data/visualizations/feature_importance.png', dpi=300)
        plt.close()
        
    def _plot_predictions_vs_actual(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series):
        """Plots predicted vs. actual values."""
        y_pred = model.predict(X_val)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_val, y_pred, alpha=0.6)
        
        # Add a line for perfect predictions
        min_val = min(y_val.min(), y_pred.min())
        max_val = max(y_val.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        plt.xlabel('Actual RUL')
        plt.ylabel('Predicted RUL')
        plt.title('Predictions vs Actual Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/visualizations/predictions_vs_actual.png', dpi=300)
        plt.close()
        
    def _plot_residuals(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series):
        """Plots various residual analyses."""
        y_pred = model.predict(X_val)
        residuals = y_val - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot residuals vs. predicted values
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted RUL')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predictions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot histogram of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Create a Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot residuals vs. actual values
        axes[1, 1].scatter(y_val, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Actual RUL')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Actual')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/visualizations/residuals_analysis.png', dpi=300)
        plt.close()
        
    def _plot_optimization_history(self, study):
        """Plots the hyperparameter optimization history."""
        try:
            import optuna.visualization as vis
            
            # Plot optimization history
            fig = vis.plot_optimization_history(study)
            fig.write_image('data/visualizations/optimization_history.png')
            
            # Plot parameter importances
            fig = vis.plot_param_importances(study)
            fig.write_image('data/visualizations/param_importances.png')
            
        except ImportError:
            logger.warning("Optuna visualization not available")
        except Exception as e:
            logger.warning(f"Could not create optimization plots: {e}")


def run_model_training():
    """Runs the model training and evaluation pipeline."""
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate_models()
    
    logger.info("=== Model Training Completed ===")
    logger.info(f"Best model: {results['best_model']}")
    logger.info(f"Validation RMSE: {results['evaluation_results']['val_metrics']['rmse']:.4f}")
    logger.info(f"Validation R²: {results['evaluation_results']['val_metrics']['r2']:.4f}")


if __name__ == "__main__":
    run_model_training()
