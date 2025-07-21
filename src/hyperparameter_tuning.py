
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
import pickle

# ml libraries
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TuningConfig:
    """Data class to hold hyperparameter tuning settings."""
    n_trials: int
    timeout_minutes: int
    optimization_strategy: str
    multi_objective: bool
    ensemble_tuning: bool
    cv_folds: int
    early_stopping: bool
    pruning_strategy: str
    sampler_type: str
    random_state: int

class EnsembleOptimizer:
    """Optimizes a set of base models for ensemble learning."""
    
    def __init__(self, base_models: List[str], config: TuningConfig):
        self.base_models = base_models
        self.config = config
        self.optimized_models = {}
        
    def optimize_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimizes each base model in the ensemble."""
        logger.info("Optimizing ensemble models...")
        
        # Optimize each model individually
        for model_name in self.base_models:
            logger.info(f"Optimizing {model_name}...")
            
            # Create an Optuna study
            study = optuna.create_study(
                direction='minimize',
                sampler=self._get_sampler(),
                pruner=self._get_pruner()
            )
            
            # Define the objective function for Optuna
            objective = self._create_model_objective(model_name, X, y)
            
            # Start the optimization process
            study.optimize(objective, n_trials=self.config.n_trials // len(self.base_models))
            
            # Store the optimization results
            self.optimized_models[model_name] = {
                'study': study,
                'best_params': study.best_params,
                'best_score': study.best_value,
                'model': self._create_model_with_params(model_name, study.best_params)
            }
            
        return self.optimized_models
        
    def _get_sampler(self):
        """Returns the Optuna sampler based on the configuration."""
        if self.config.sampler_type == 'tpe':
            return TPESampler(seed=self.config.random_state)
        elif self.config.sampler_type == 'cmaes':
            return CmaEsSampler(seed=self.config.random_state)
        else:
            return RandomSampler(seed=self.config.random_state)
            
    def _get_pruner(self):
        """Returns the Optuna pruner based on the configuration."""
        if self.config.pruning_strategy == 'median':
            return MedianPruner()
        elif self.config.pruning_strategy == 'successive_halving':
            return SuccessiveHalvingPruner()
        else:
            return None
            
    def _create_model_objective(self, model_name: str, X: pd.DataFrame, y: pd.Series):
        """Creates the objective function for Optuna optimization."""
        def objective(trial):
            # Get the hyperparameter space for the trial
            params = self._get_hyperparameter_space(model_name, trial)
            
            # Create the model with the suggested hyperparameters
            model = self._create_model_with_params(model_name, params)
            
            # Perform cross-validation
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            
            return -scores.mean()
            
        return objective
        
    def _get_hyperparameter_space(self, model_name: str, trial) -> Dict[str, Any]:
        """Returns the hyperparameter space for a given model."""
        if model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0)
            }
        elif model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        else:
            return {}
            
    def _create_model_with_params(self, model_name: str, params: Dict[str, Any]):
        """Creates a model instance with the given hyperparameters."""
        if model_name == 'xgboost':
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.config.random_state,
                n_jobs=-1,
                **params
            )
        elif model_name == 'random_forest':
            return RandomForestRegressor(
                random_state=self.config.random_state,
                n_jobs=-1,
                **params
            )
        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(
                random_state=self.config.random_state,
                **params
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

class MultiObjectiveOptimizer:
    """Optimizes a model for multiple objectives (e.g., performance and complexity)."""
    
    def __init__(self, config: TuningConfig):
        self.config = config
        
    def optimize_multi_objective(self, X: pd.DataFrame, y: pd.Series, model_name: str) -> Dict[str, Any]:
        """Performs multi-objective optimization."""
        logger.info(f"Multi-objective optimization for {model_name}...")
        
        # Create a multi-objective study in Optuna
        study = optuna.create_study(
            directions=['minimize', 'minimize'],  # [performance, complexity]
            sampler=TPESampler(seed=self.config.random_state)
        )
        
        # Define the multi-objective function
        def multi_objective(trial):
            # Get hyperparameters for the trial
            params = self._get_params_for_model(model_name, trial)
            
            # Create the model
            model = self._create_model(model_name, params)
            
            # Calculate performance using cross-validation
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            performance = -scores.mean()
            
            # Calculate model complexity
            complexity = self._calculate_complexity(model_name, params)
            
            return performance, complexity
            
        # Start the optimization
        study.optimize(multi_objective, n_trials=self.config.n_trials)
        
        # Find the best trade-off solution from the Pareto front
        best_solution = self._find_best_tradeoff(study)
        
        return {
            'study': study,
            'best_solution': best_solution,
            'pareto_front': study.best_trials
        }
        
    def _get_params_for_model(self, model_name: str, trial) -> Dict[str, Any]:
        """Returns the hyperparameter space for a given model."""
        if model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
            }
        else:
            return {}
            
    def _create_model(self, model_name: str, params: Dict[str, Any]):
        """Creates a model instance with the given hyperparameters."""
        if model_name == 'xgboost':
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.config.random_state,
                n_jobs=-1,
                **params
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    def _calculate_complexity(self, model_name: str, params: Dict[str, Any]) -> float:
        """Calculates a complexity score for the model."""
        if model_name == 'xgboost':
            # Complexity score based on number of estimators and max depth
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', 6)
            
            # Normalize and weight the complexity components
            complexity = (n_estimators / 500) * 0.6 + (max_depth / 12) * 0.4
            return complexity
        else:
            return 0.5
            
    def _find_best_tradeoff(self, study) -> Dict[str, Any]:
        """Finds the best trade-off solution from the Pareto front."""
        if not study.best_trials:
            return {}
            
        # Use a simple weighted approach to select the best trial
        best_trial = None
        best_score = float('inf')
        
        for trial in study.best_trials:
            # 80% weight on performance, 20% on complexity
            weighted_score = 0.8 * trial.values[0] + 0.2 * trial.values[1]
            
            if weighted_score < best_score:
                best_score = weighted_score
                best_trial = trial
                
        return {
            'trial': best_trial,
            'params': best_trial.params if best_trial else {},
            'performance': best_trial.values[0] if best_trial else 0,
            'complexity': best_trial.values[1] if best_trial else 0,
            'weighted_score': best_score
        }

class HyperparameterTuner:
    """Orchestrates the hyperparameter tuning process."""
    
    def __init__(self, config_path: str = "config/main_config.yaml"):
        self.config = self._load_config(config_path)
        self.tuning_config = self._create_tuning_config()
        self.ensemble_optimizer = EnsembleOptimizer(['xgboost', 'random_forest', 'gradient_boosting'], self.tuning_config)
        self.multi_objective_optimizer = MultiObjectiveOptimizer(self.tuning_config)
        
    def _load_config(self, config_path: str) -> Dict:
        """Loads the configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    def _create_tuning_config(self) -> TuningConfig:
        """Creates a TuningConfig object from the main config."""
        return TuningConfig(
            n_trials=self.config.get('tuning', {}).get('n_trials', 100),
            timeout_minutes=60,
            optimization_strategy='bayesian',
            multi_objective=True,
            ensemble_tuning=True,
            cv_folds=5,
            early_stopping=True,
            pruning_strategy='median',
            sampler_type='tpe',
            random_state=42
        )
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Loads processed training and validation data."""
        logger.info("Loading processed training data...")
        
        try:
            # Load data from CSV files
            train_df = pd.read_csv("data/processed/train_processed.csv")
            val_df = pd.read_csv("data/processed/val_processed.csv")
            
            # Separate features and target variable
            feature_cols = [col for col in train_df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
            
            X_train = train_df[feature_cols]
            y_train = train_df['RUL']
            X_val = val_df[feature_cols]
            y_val = val_df['RUL']
            
            logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
            
    def run_advanced_tuning(self) -> Dict[str, Any]:
        """Runs the advanced hyperparameter tuning pipeline."""
        logger.info("=== Starting Advanced Hyperparameter Tuning ===")
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_data()
        
        # Combine data for cross-validation
        X_full = pd.concat([X_train, X_val], axis=0)
        y_full = pd.concat([y_train, y_val], axis=0)
        
        results = {}
        
        # 1. Ensemble optimization
        if self.tuning_config.ensemble_tuning:
            logger.info("Running ensemble optimization...")
            ensemble_results = self.ensemble_optimizer.optimize_ensemble(X_full, y_full)
            results['ensemble_optimization'] = ensemble_results
            
        # 2. Multi-objective optimization
        if self.tuning_config.multi_objective:
            logger.info("Running multi-objective optimization...")
            multi_obj_results = self.multi_objective_optimizer.optimize_multi_objective(X_full, y_full, 'xgboost')
            results['multi_objective_optimization'] = multi_obj_results
            
        # 3. Select the best model from all candidates
        best_model_info = self._select_best_model(results, X_train, y_train, X_val, y_val)
        results['best_model'] = best_model_info
        
        # 4. Train the final model on the full training data
        final_model = self._train_final_model(best_model_info, X_train, y_train, X_val, y_val)
        results['final_model'] = final_model
        
        # 5. Save tuning results and the best model
        self._save_tuning_results(results)
        
        # 6. Create visualizations
        self._create_tuning_visualizations(results)
        
        # 7. Log results to MLFlow
        self._log_to_mlflow(results, X_val, y_val)
        
        logger.info("=== Advanced Hyperparameter Tuning Complete ===")
        return results
        
    def _select_best_model(self, results: Dict, X_train: pd.DataFrame, y_train: pd.Series, 
                          X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Selects the best model based on validation performance."""
        logger.info("Selecting best model...")
        
        candidates = []
        
        # Add candidates from ensemble optimization
        if 'ensemble_optimization' in results:
            for model_name, model_info in results['ensemble_optimization'].items():
                candidates.append({
                    'name': model_name,
                    'type': 'ensemble',
                    'params': model_info['best_params'],
                    'cross_val_score': model_info['best_score'],
                    'model': model_info['model']
                })
                
        # Add candidates from multi-objective optimization
        if 'multi_objective_optimization' in results:
            best_solution = results['multi_objective_optimization']['best_solution']
            if best_solution:
                candidates.append({
                    'name': 'xgboost_multi_objective',
                    'type': 'multi_objective',
                    'params': best_solution['params'],
                    'cross_val_score': best_solution['performance'],
                    'model': self._create_model_from_params('xgboost', best_solution['params'])
                })
                
        # Evaluate all candidates on the validation set
        best_candidate = None
        best_val_score = float('inf')
        
        for candidate in candidates:
            model = candidate['model']
            model.fit(X_train, y_train)
            
            val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            candidate['validation_rmse'] = val_rmse
            
            if val_rmse < best_val_score:
                best_val_score = val_rmse
                best_candidate = candidate
                
        logger.info(f"Best model selected: {best_candidate['name']} (RMSE: {best_val_score:.4f})")
        
        return best_candidate
        
    def _create_model_from_params(self, model_name: str, params: Dict[str, Any]):
        """Creates a model instance from a given set of parameters."""
        if model_name == 'xgboost':
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.tuning_config.random_state,
                n_jobs=-1,
                **params
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    def _train_final_model(self, best_model_info: Dict, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Trains the final selected model."""
        logger.info("Training final model...")
        
        # Create and train the model
        model = best_model_info['model']
        model.fit(X_train, y_train)
        
        # Evaluate on training and validation sets
        val_pred = model.predict(X_val)
        train_pred = model.predict(X_train)
        
        # Calculate performance metrics
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'mae': mean_absolute_error(y_train, train_pred),
            'r2': r2_score(y_train, train_pred)
        }
        
        val_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'mae': mean_absolute_error(y_val, val_pred),
            'r2': r2_score(y_val, val_pred)
        }
        
        # Get feature importance
        feature_importance = self._get_feature_importance(model, X_train.columns)
        
        return {
            'model': model,
            'model_name': best_model_info['name'],
            'best_params': best_model_info['params'],
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_importance': feature_importance
        }
        
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extracts feature importance from the model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                return {}
                
            # Create a sorted dictionary of feature importances
            feature_importance = dict(zip(feature_names, importance))
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
            
    def _save_tuning_results(self, results: Dict):
        """Saves the tuning results to files."""
        logger.info("Saving tuning results...")
        
        os.makedirs("data/tuning", exist_ok=True)
        
        # Prepare results for saving
        save_results = {
            'timestamp': datetime.now().isoformat(),
            'tuning_config': {
                'n_trials': self.tuning_config.n_trials,
                'optimization_strategy': self.tuning_config.optimization_strategy,
                'multi_objective': self.tuning_config.multi_objective,
                'ensemble_tuning': self.tuning_config.ensemble_tuning
            }
        }
        
        # Add ensemble results
        if 'ensemble_optimization' in results:
            save_results['ensemble_results'] = {}
            for model_name, model_info in results['ensemble_optimization'].items():
                save_results['ensemble_results'][model_name] = {
                    'best_params': model_info['best_params'],
                    'best_score': model_info['best_score'],
                    'n_trials': len(model_info['study'].trials)
                }
                
        # Add multi-objective results
        if 'multi_objective_optimization' in results:
            mo_results = results['multi_objective_optimization']
            save_results['multi_objective_results'] = {
                'best_solution': mo_results['best_solution'],
                'n_pareto_solutions': len(mo_results['pareto_front'])
            }
            
        # Add best model information
        if 'best_model' in results:
            save_results['best_model'] = {
                'name': results['best_model']['name'],
                'type': results['best_model']['type'],
                'params': results['best_model']['params'],
                'validation_rmse': results['best_model']['validation_rmse']
            }
            
        # Add final model information
        if 'final_model' in results:
            save_results['final_model'] = {
                'model_name': results['final_model']['model_name'],
                'best_params': results['final_model']['best_params'],
                'train_metrics': results['final_model']['train_metrics'],
                'val_metrics': results['final_model']['val_metrics'],
                'top_10_features': dict(list(results['final_model']['feature_importance'].items())[:10])
            }
            
        # Save results to a JSON file
        with open("data/tuning/tuning_results.json", 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
            
        # Save the best model
        if 'final_model' in results:
            joblib.dump(results['final_model']['model'], "data/tuning/best_tuned_model.joblib")
            
        logger.info("Tuning results saved to data/tuning/")
        
    def _create_tuning_visualizations(self, results: Dict):
        """Creates and saves visualizations related to tuning."""
        logger.info("Creating tuning visualizations...")
        
        os.makedirs("data/visualizations/tuning", exist_ok=True)
        
        # 1. Ensemble model comparison plot
        if 'ensemble_optimization' in results:
            self._plot_ensemble_comparison(results['ensemble_optimization'])
            
        # 2. Multi-objective optimization plot
        if 'multi_objective_optimization' in results:
            self._plot_multi_objective_results(results['multi_objective_optimization'])
            
        # 3. Hyperparameter importance plot
        if 'ensemble_optimization' in results:
            self._plot_hyperparameter_importance(results['ensemble_optimization'])
            
        # 4. Feature importance plot
        if 'final_model' in results:
            self._plot_feature_importance(results['final_model'])
            
        # 5. Training convergence plot
        if 'ensemble_optimization' in results:
            self._plot_training_convergence(results['ensemble_optimization'])
            
    def _plot_ensemble_comparison(self, ensemble_results: Dict):
        """Plots a comparison of the ensemble models."""
        models = list(ensemble_results.keys())
        scores = [ensemble_results[model]['best_score'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Ensemble Model Comparison - Cross-Validation RMSE')
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        # Add value labels to the bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{score:.3f}', ha='center', va='bottom')
                    
        plt.tight_layout()
        plt.savefig('data/visualizations/tuning/ensemble_comparison.png', dpi=300)
        plt.close()
        
    def _plot_multi_objective_results(self, mo_results: Dict):
        """Plots the results of multi-objective optimization."""
        if not mo_results.get('pareto_front'):
            return
            
        pareto_front = mo_results['pareto_front']
        
        performance_values = [trial.values[0] for trial in pareto_front]
        complexity_values = [trial.values[1] for trial in pareto_front]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(performance_values, complexity_values, alpha=0.7)
        plt.xlabel('Performance (RMSE)')
        plt.ylabel('Complexity')
        plt.title('Multi-Objective Optimization - Pareto Front')
        plt.grid(True, alpha=0.3)
        
        # Highlight the best solution
        best_solution = mo_results['best_solution']
        if best_solution:
            plt.scatter(best_solution['performance'], best_solution['complexity'], 
                       color='red', s=100, marker='*', label='Best Solution')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig('data/visualizations/tuning/multi_objective_results.png', dpi=300)
        plt.close()
        
    def _plot_hyperparameter_importance(self, ensemble_results: Dict):
        """Plots the importance of hyperparameters."""
        # Plot for XGBoost model
        if 'xgboost' in ensemble_results:
            study = ensemble_results['xgboost']['study']
            
            try:
                # Get parameter importances from Optuna
                param_importance = optuna.importance.get_param_importances(study)
                
                params = list(param_importance.keys())
                importance = list(param_importance.values())
                
                plt.figure(figsize=(10, 6))
                plt.barh(params, importance)
                plt.title('Hyperparameter Importance (XGBoost)')
                plt.xlabel('Importance')
                plt.ylabel('Parameters')
                plt.tight_layout()
                plt.savefig('data/visualizations/tuning/hyperparameter_importance.png', dpi=300)
                plt.close()
                
            except Exception as e:
                logger.warning(f"Could not create hyperparameter importance plot: {e}")
                
    def _plot_feature_importance(self, final_model: Dict):
        """Plots the importance of features."""
        feature_importance = final_model['feature_importance']
        
        if not feature_importance:
            return
            
        # Plot top 15 features
        top_features = dict(list(feature_importance.items())[:15])
        
        plt.figure(figsize=(10, 8))
        plt.barh(list(top_features.keys()), list(top_features.values()))
        plt.title('Top 15 Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('data/visualizations/tuning/feature_importance.png', dpi=300)
        plt.close()
        
    def _plot_training_convergence(self, ensemble_results: Dict):
        """Plots the training convergence for each model."""
        fig, axes = plt.subplots(1, len(ensemble_results), figsize=(15, 5))
        
        if len(ensemble_results) == 1:
            axes = [axes]
            
        for i, (model_name, model_info) in enumerate(ensemble_results.items()):
            study = model_info['study']
            
            # Get trial values
            trial_values = [trial.value for trial in study.trials if trial.value is not None]
            
            if trial_values:
                axes[i].plot(trial_values)
                axes[i].set_title(f'{model_name} - Convergence')
                axes[i].set_xlabel('Trial')
                axes[i].set_ylabel('RMSE')
                axes[i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig('data/visualizations/tuning/training_convergence.png', dpi=300)
        plt.close()
        
    def _log_to_mlflow(self, results: Dict, X_val: pd.DataFrame, y_val: pd.Series):
        """Logs tuning results to MLFlow."""
        logger.info("Logging to MLflow...")
        
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        
        with mlflow.start_run(run_name="Advanced_Hyperparameter_Tuning") as run:
            # Log tuning configuration
            mlflow.log_params({
                'n_trials': self.tuning_config.n_trials,
                'optimization_strategy': self.tuning_config.optimization_strategy,
                'multi_objective': self.tuning_config.multi_objective,
                'ensemble_tuning': self.tuning_config.ensemble_tuning
            })
            
            # Log final model details
            if 'final_model' in results:
                final_model = results['final_model']
                
                # Log best parameters
                mlflow.log_params(final_model['best_params'])
                
                # Log performance metrics
                for metric_name, metric_value in final_model['train_metrics'].items():
                    mlflow.log_metric(f"train_{metric_name}", metric_value)
                    
                for metric_name, metric_value in final_model['val_metrics'].items():
                    mlflow.log_metric(f"val_{metric_name}", metric_value)
                    
                # Log the model to MLFlow
                if final_model['model_name'].startswith('xgboost'):
                    mlflow.xgboost.log_model(
                        final_model['model'],
                        "model",
                        registered_model_name=f"{self.config['model']['name']}_advanced_tuned"
                    )
                else:
                    mlflow.sklearn.log_model(
                        final_model['model'],
                        "model",
                        registered_model_name=f"{self.config['model']['name']}_advanced_tuned"
                    )
                    
            # Log visualization artifacts
            mlflow.log_artifacts("data/visualizations/tuning")
            
            logger.info(f"Results logged to MLflow. Run ID: {run.info.run_id}")


def run_hyperparameter_tuning():
    """Runs the advanced hyperparameter tuning pipeline."""
    tuner = HyperparameterTuner()
    results = tuner.run_advanced_tuning()
    
    logger.info("=== Advanced Hyperparameter Tuning Completed ===")
    if 'final_model' in results:
        final_model = results['final_model']
        logger.info(f"Best model: {final_model['model_name']}")
        logger.info(f"Validation RMSE: {final_model['val_metrics']['rmse']:.4f}")
        logger.info(f"Validation R²: {final_model['val_metrics']['r2']:.4f}")


if __name__ == "__main__":
    run_hyperparameter_tuning()
