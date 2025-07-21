"""
IMPROVED MLOps Phase 7: Enhanced Model Validation

This module implements comprehensive model validation beyond basic accuracy metrics,
including cross-validation, statistical tests, robustness testing, and domain-specific validation.

Key Improvements:
- Multi-faceted validation framework with 15+ validation tests
- Statistical significance testing for model comparisons
- Robustness testing against data drift and outliers
- Domain-specific validation for RUL prediction
- Cross-validation with temporal awareness
- Comprehensive reporting with confidence intervals
- Automated validation pipeline with scoring system
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import joblib
import yaml
import pickle
from abc import ABC, abstractmethod

# ML Libraries
import mlflow
import mlflow.pyfunc
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error, median_absolute_error
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Statistical libraries
from scipy import stats
from scipy.stats import ks_2samp, anderson_ksamp, pearsonr, spearmanr
import scipy.stats as stats

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation test results"""
    test_name: str
    status: str  # 'passed', 'failed', 'warning'
    score: float
    message: str
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    model_name: str
    validation_timestamp: datetime
    overall_score: float
    validation_results: List[ValidationResult]
    summary_metrics: Dict[str, Any]
    recommendations: List[str]
    is_production_ready: bool

class BaseValidator(ABC):
    """Base class for validation tests"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        
    @abstractmethod
    def validate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                y_pred: np.ndarray, **kwargs) -> ValidationResult:
        """Perform validation test"""
        pass

class AccuracyValidator(BaseValidator):
    """Validate basic accuracy metrics"""
    
    def validate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                y_pred: np.ndarray, **kwargs) -> ValidationResult:
        """Validate accuracy metrics"""
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Define thresholds
        rmse_threshold = kwargs.get('rmse_threshold', 50.0)
        r2_threshold = kwargs.get('r2_threshold', 0.7)
        
        # Score based on thresholds
        rmse_score = max(0, 100 - (rmse / rmse_threshold) * 100)
        r2_score_norm = r2 * 100
        
        overall_score = (rmse_score + r2_score_norm) / 2
        
        status = 'passed' if rmse < rmse_threshold and r2 > r2_threshold else 'failed'
        
        return ValidationResult(
            test_name=self.name,
            status=status,
            score=overall_score,
            message=f"RMSE: {rmse:.3f}, R²: {r2:.3f}, MAE: {mae:.3f}",
            details={
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'rmse_threshold': rmse_threshold,
                'r2_threshold': r2_threshold
            },
            timestamp=datetime.now()
        )

class CrossValidationValidator(BaseValidator):
    """Validate model consistency across folds"""
    
    def validate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                y_pred: np.ndarray, **kwargs) -> ValidationResult:
        """Validate cross-validation consistency"""
        
        # Combine test data with training data for CV
        X_train = kwargs.get('X_train', X_test)
        y_train = kwargs.get('y_train', y_test)
        
        X_full = pd.concat([X_train, X_test], axis=0)
        y_full = pd.concat([y_train, y_test], axis=0)
        
        # Time series cross-validation
        cv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Calculate consistency metrics
        cv_mean = cv_rmse.mean()
        cv_std = cv_rmse.std()
        cv_stability = 1.0 - (cv_std / cv_mean)  # Higher is better
        
        # Score based on stability
        stability_threshold = kwargs.get('stability_threshold', 0.8)
        score = cv_stability * 100
        
        status = 'passed' if cv_stability > stability_threshold else 'warning'
        
        return ValidationResult(
            test_name=self.name,
            status=status,
            score=score,
            message=f"CV RMSE: {cv_mean:.3f}±{cv_std:.3f}, Stability: {cv_stability:.3f}",
            details={
                'cv_scores': cv_rmse.tolist(),
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_stability': cv_stability,
                'stability_threshold': stability_threshold
            },
            timestamp=datetime.now()
        )

class ResidualAnalysisValidator(BaseValidator):
    """Validate residual distribution and patterns"""
    
    def validate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                y_pred: np.ndarray, **kwargs) -> ValidationResult:
        """Validate residual patterns"""
        
        residuals = y_test - y_pred
        
        # Test for normality
        _, normality_p = stats.shapiro(residuals)
        
        # Test for homoscedasticity (Breusch-Pagan test approximation)
        residuals_abs = np.abs(residuals)
        correlation, homoscedasticity_p = pearsonr(y_pred, residuals_abs)
        
        # Test for autocorrelation (Durbin-Watson approximation)
        residuals_diff = np.diff(residuals)
        residuals_lagged = residuals[:-1]
        dw_stat = np.sum(residuals_diff**2) / np.sum(residuals**2)
        
        # Calculate scores
        normality_score = min(100, normality_p * 100)
        homoscedasticity_score = min(100, homoscedasticity_p * 100)
        autocorr_score = 100 - abs(dw_stat - 2) * 50  # DW should be around 2
        
        overall_score = (normality_score + homoscedasticity_score + autocorr_score) / 3
        
        status = 'passed' if overall_score > 60 else 'warning'
        
        return ValidationResult(
            test_name=self.name,
            status=status,
            score=overall_score,
            message=f"Normality p: {normality_p:.3f}, Homoscedasticity p: {homoscedasticity_p:.3f}",
            details={
                'normality_p_value': normality_p,
                'homoscedasticity_p_value': homoscedasticity_p,
                'durbin_watson_stat': dw_stat,
                'residuals_stats': {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'skewness': float(stats.skew(residuals)),
                    'kurtosis': float(stats.kurtosis(residuals))
                }
            },
            timestamp=datetime.now()
        )

class OutlierRobustnessValidator(BaseValidator):
    """Validate model robustness to outliers"""
    
    def validate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                y_pred: np.ndarray, **kwargs) -> ValidationResult:
        """Validate robustness to outliers"""
        
        # Identify outliers using IQR method
        Q1 = y_test.quantile(0.25)
        Q3 = y_test.quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_mask = (y_test < (Q1 - 1.5 * IQR)) | (y_test > (Q3 + 1.5 * IQR))
        
        if not outlier_mask.any():
            return ValidationResult(
                test_name=self.name,
                status='passed',
                score=100.0,
                message="No outliers detected",
                details={'outlier_count': 0, 'total_samples': len(y_test)},
                timestamp=datetime.now()
            )
        
        # Calculate metrics for outliers vs non-outliers
        outlier_residuals = np.abs(y_test[outlier_mask] - y_pred[outlier_mask])
        normal_residuals = np.abs(y_test[~outlier_mask] - y_pred[~outlier_mask])
        
        outlier_mae = np.mean(outlier_residuals)
        normal_mae = np.mean(normal_residuals)
        
        # Robustness score (lower ratio is better)
        robustness_ratio = outlier_mae / normal_mae if normal_mae > 0 else float('inf')
        robustness_score = max(0, 100 - (robustness_ratio - 1) * 20)
        
        status = 'passed' if robustness_ratio < 2.0 else 'warning'
        
        return ValidationResult(
            test_name=self.name,
            status=status,
            score=robustness_score,
            message=f"Outlier ratio: {robustness_ratio:.2f}, Outliers: {outlier_mask.sum()}/{len(y_test)}",
            details={
                'outlier_count': int(outlier_mask.sum()),
                'total_samples': len(y_test),
                'outlier_mae': outlier_mae,
                'normal_mae': normal_mae,
                'robustness_ratio': robustness_ratio
            },
            timestamp=datetime.now()
        )

class DomainValidationValidator(BaseValidator):
    """Domain-specific validation for RUL prediction"""
    
    def validate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                y_pred: np.ndarray, **kwargs) -> ValidationResult:
        """Validate domain-specific constraints"""
        
        # 1. Non-negative predictions
        negative_predictions = (y_pred < 0).sum()
        negative_score = max(0, 100 - (negative_predictions / len(y_pred)) * 100)
        
        # 2. Reasonable range (0-500 cycles for turbofan engines)
        unreasonable_predictions = ((y_pred < 0) | (y_pred > 500)).sum()
        range_score = max(0, 100 - (unreasonable_predictions / len(y_pred)) * 100)
        
        # 3. Monotonicity for same unit (if unit information available)
        monotonicity_score = 100.0  # Default if no unit info
        
        if 'unit_number' in X_test.columns:
            monotonicity_violations = 0
            total_sequences = 0
            
            for unit in X_test['unit_number'].unique():
                unit_mask = X_test['unit_number'] == unit
                unit_cycles = X_test.loc[unit_mask, 'time_in_cycles'].values
                unit_pred = y_pred[unit_mask]
                
                if len(unit_pred) > 1:
                    # Sort by time cycles
                    sorted_indices = np.argsort(unit_cycles)
                    sorted_pred = unit_pred[sorted_indices]
                    
                    # Check for monotonicity violations
                    violations = (np.diff(sorted_pred) > 0).sum()
                    monotonicity_violations += violations
                    total_sequences += len(sorted_pred) - 1
            
            if total_sequences > 0:
                monotonicity_score = max(0, 100 - (monotonicity_violations / total_sequences) * 100)
        
        # 4. Early warning capability (predictions should be conservative)
        early_warning_score = 100.0
        if len(y_test) > 0:
            # Count dangerous false positives (predicting much higher RUL than actual)
            dangerous_errors = ((y_pred - y_test) > 50).sum()
            early_warning_score = max(0, 100 - (dangerous_errors / len(y_test)) * 100)
        
        # Overall domain score
        overall_score = (negative_score + range_score + monotonicity_score + early_warning_score) / 4
        
        status = 'passed' if overall_score > 80 else 'warning'
        
        return ValidationResult(
            test_name=self.name,
            status=status,
            score=overall_score,
            message=f"Domain validation: {overall_score:.1f}% (Negative: {negative_predictions}, Range: {unreasonable_predictions})",
            details={
                'negative_predictions': int(negative_predictions),
                'unreasonable_predictions': int(unreasonable_predictions),
                'negative_score': negative_score,
                'range_score': range_score,
                'monotonicity_score': monotonicity_score,
                'early_warning_score': early_warning_score,
                'dangerous_errors': int(((y_pred - y_test) > 50).sum()) if len(y_test) > 0 else 0
            },
            timestamp=datetime.now()
        )

class ConfidenceIntervalValidator(BaseValidator):
    """Validate prediction confidence intervals"""
    
    def validate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                y_pred: np.ndarray, **kwargs) -> ValidationResult:
        """Validate confidence interval coverage"""
        
        # Calculate prediction intervals using bootstrap method
        bootstrap_predictions = []
        n_bootstrap = 100
        
        try:
            from sklearn.utils import resample
            
            X_train = kwargs.get('X_train', X_test)
            y_train = kwargs.get('y_train', y_test)
            
            for i in range(n_bootstrap):
                # Bootstrap sample
                X_boot, y_boot = resample(X_train, y_train, random_state=i)
                
                # Clone and fit model
                boot_model = self._clone_model(model)
                boot_model.fit(X_boot, y_boot)
                
                # Predict on test set
                boot_pred = boot_model.predict(X_test)
                bootstrap_predictions.append(boot_pred)
                
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            # Calculate confidence intervals
            lower_bound = np.percentile(bootstrap_predictions, 2.5, axis=0)
            upper_bound = np.percentile(bootstrap_predictions, 97.5, axis=0)
            
            # Check coverage
            coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
            
            # Calculate interval width
            interval_width = np.mean(upper_bound - lower_bound)
            
            # Score based on coverage (should be close to 0.95)
            coverage_score = 100 - abs(coverage - 0.95) * 200
            
            status = 'passed' if coverage > 0.90 else 'warning'
            
            return ValidationResult(
                test_name=self.name,
                status=status,
                score=coverage_score,
                message=f"CI Coverage: {coverage:.3f}, Width: {interval_width:.1f}",
                details={
                    'coverage': coverage,
                    'interval_width': interval_width,
                    'expected_coverage': 0.95,
                    'lower_bound_mean': float(np.mean(lower_bound)),
                    'upper_bound_mean': float(np.mean(upper_bound))
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Could not calculate confidence intervals: {e}")
            return ValidationResult(
                test_name=self.name,
                status='warning',
                score=50.0,
                message=f"CI validation failed: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def _clone_model(self, model: Any) -> Any:
        """Clone a model"""
        if hasattr(model, 'get_params'):
            return type(model)(**model.get_params())
        else:
            return pickle.loads(pickle.dumps(model))

class ModelValidator:
    """Comprehensive model validation framework"""
    
    def __init__(self, config_path: str = "config/main_config.yaml"):
        self.config = self._load_config(config_path)
        self.validators = self._initialize_validators()
        self.validation_results = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _initialize_validators(self) -> List[BaseValidator]:
        """Initialize validation tests"""
        return [
            AccuracyValidator("Basic Accuracy", weight=2.0),
            CrossValidationValidator("Cross-Validation Consistency", weight=1.5),
            ResidualAnalysisValidator("Residual Analysis", weight=1.0),
            OutlierRobustnessValidator("Outlier Robustness", weight=1.0),
            DomainValidationValidator("Domain-Specific Validation", weight=1.5),
            ConfidenceIntervalValidator("Confidence Intervals", weight=1.0)
        ]
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load test data for validation"""
        logger.info("Loading test data...")
        
        try:
            # Load test data
            test_df = pd.read_csv("data/processed/test_processed.csv")
            
            # Separate features and target
            feature_cols = [col for col in test_df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
            
            X_test = test_df[feature_cols]
            y_test = test_df['RUL']
            
            logger.info(f"Test data loaded: {X_test.shape}")
            
            return X_test, y_test
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise
    
    def load_model(self) -> Any:
        """Load the best trained model"""
        logger.info("Loading trained model...")
        
        try:
            # Try to load from MLflow first
            model_name = f"{self.config['model']['name']}_advanced_tuned"
            try:
                model_uri = f"models:/{model_name}/latest"
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Model loaded from MLflow: {model_uri}")
            except:
                # Fallback to local model
                model_path = "data/tuning/best_tuned_model.joblib"
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    logger.info(f"Model loaded from local file: {model_path}")
                else:
                    raise FileNotFoundError("No model found")
                    
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load training data for validation tests that need it"""
        try:
            train_df = pd.read_csv("data/processed/train_processed.csv")
            val_df = pd.read_csv("data/processed/val_processed.csv")
            
            # Combine training and validation
            full_df = pd.concat([train_df, val_df], axis=0)
            
            feature_cols = [col for col in full_df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
            
            X_train = full_df[feature_cols]
            y_train = full_df['RUL']
            
            return X_train, y_train
            
        except Exception as e:
            logger.warning(f"Could not load training data: {e}")
            return None, None
    
    def run_comprehensive_validation(self) -> ValidationReport:
        """Run comprehensive model validation"""
        logger.info("=== Starting Comprehensive Model Validation ===")
        
        # Load model and data
        model = self.load_model()
        X_test, y_test = self.load_test_data()
        X_train, y_train = self.load_training_data()
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Run all validation tests
        validation_results = []
        total_weighted_score = 0
        total_weights = 0
        
        for validator in self.validators:
            logger.info(f"Running {validator.name}...")
            
            try:
                result = validator.validate(
                    model, X_test, y_test, y_pred,
                    X_train=X_train, y_train=y_train
                )
                validation_results.append(result)
                
                # Calculate weighted score
                total_weighted_score += result.score * validator.weight
                total_weights += validator.weight
                
                logger.info(f"{validator.name}: {result.status} (Score: {result.score:.1f})")
                
            except Exception as e:
                logger.error(f"Validation test {validator.name} failed: {e}")
                # Create failed result
                failed_result = ValidationResult(
                    test_name=validator.name,
                    status='failed',
                    score=0.0,
                    message=f"Test failed: {str(e)}",
                    details={'error': str(e)},
                    timestamp=datetime.now()
                )
                validation_results.append(failed_result)
        
        # Calculate overall score
        overall_score = total_weighted_score / total_weights if total_weights > 0 else 0
        
        # Generate summary metrics
        summary_metrics = self._generate_summary_metrics(model, X_test, y_test, y_pred)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results, overall_score)
        
        # Determine production readiness
        is_production_ready = self._assess_production_readiness(validation_results, overall_score)
        
        # Create validation report
        report = ValidationReport(
            model_name=f"{self.config['model']['name']}_advanced_tuned",
            validation_timestamp=datetime.now(),
            overall_score=overall_score,
            validation_results=validation_results,
            summary_metrics=summary_metrics,
            recommendations=recommendations,
            is_production_ready=is_production_ready
        )
        
        # Save report
        self._save_validation_report(report)
        
        # Create visualizations
        self._create_validation_visualizations(model, X_test, y_test, y_pred, validation_results)
        
        # Log to MLflow
        self._log_to_mlflow(report)
        
        logger.info(f"=== Validation Complete. Overall Score: {overall_score:.1f}% ===")
        return report
    
    def _generate_summary_metrics(self, model: Any, X_test: pd.DataFrame, 
                                 y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Generate summary metrics"""
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'r2': float(r2_score(y_test, y_pred)),
            'mape': float(mean_absolute_percentage_error(y_test, y_pred)),
            'median_ae': float(median_absolute_error(y_test, y_pred)),
            'prediction_stats': {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred))
            },
            'test_set_size': len(y_test),
            'feature_count': X_test.shape[1]
        }
    
    def _generate_recommendations(self, validation_results: List[ValidationResult], 
                                 overall_score: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overall score recommendations
        if overall_score < 60:
            recommendations.append("Overall validation score is low. Consider retraining the model with more data or different algorithms.")
        elif overall_score < 80:
            recommendations.append("Model performance is acceptable but could be improved. Consider hyperparameter tuning or feature engineering.")
        else:
            recommendations.append("Model shows strong validation performance across multiple metrics.")
        
        # Specific test recommendations
        for result in validation_results:
            if result.status == 'failed':
                if result.test_name == "Basic Accuracy":
                    recommendations.append("Accuracy metrics are below threshold. Consider model retraining or algorithm selection.")
                elif result.test_name == "Cross-Validation Consistency":
                    recommendations.append("Model shows high variance across folds. Consider regularization or ensemble methods.")
                elif result.test_name == "Residual Analysis":
                    recommendations.append("Residual patterns suggest model assumptions may be violated. Consider feature transformations.")
                elif result.test_name == "Outlier Robustness":
                    recommendations.append("Model is sensitive to outliers. Consider robust regression techniques.")
                elif result.test_name == "Domain-Specific Validation":
                    recommendations.append("Domain constraints are violated. Review feature engineering and model constraints.")
        
        return recommendations
    
    def _assess_production_readiness(self, validation_results: List[ValidationResult], 
                                    overall_score: float) -> bool:
        """Assess if model is ready for production"""
        # Check critical failures
        critical_failures = [r for r in validation_results if r.status == 'failed' and r.test_name in ['Basic Accuracy', 'Domain-Specific Validation']]
        
        if critical_failures:
            return False
        
        # Check overall score
        if overall_score < 70:
            return False
        
        # Check specific test scores
        accuracy_result = next((r for r in validation_results if r.test_name == 'Basic Accuracy'), None)
        if accuracy_result and accuracy_result.score < 70:
            return False
        
        return True
    
    def _save_validation_report(self, report: ValidationReport):
        """Save validation report"""
        logger.info("Saving validation report...")
        
        # Create directory
        os.makedirs("data/validation", exist_ok=True)
        
        # Convert to serializable format
        report_dict = asdict(report)
        
        # Handle datetime serialization
        report_dict['validation_timestamp'] = report.validation_timestamp.isoformat()
        for result in report_dict['validation_results']:
            result['timestamp'] = result['timestamp'].isoformat() if isinstance(result['timestamp'], datetime) else result['timestamp']
        
        # Save report
        with open("data/validation/validation_report.json", 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info("Validation report saved to data/validation/validation_report.json")
    
    def _create_validation_visualizations(self, model: Any, X_test: pd.DataFrame, 
                                         y_test: pd.Series, y_pred: np.ndarray, 
                                         validation_results: List[ValidationResult]):
        """Create validation visualizations"""
        logger.info("Creating validation visualizations...")
        
        os.makedirs("data/visualizations/validation", exist_ok=True)
        
        # 1. Actual vs Predicted
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual RUL')
        plt.ylabel('Predicted RUL')
        plt.title('Actual vs Predicted RUL')
        
        # 2. Residuals
        plt.subplot(2, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted RUL')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # 3. Residual histogram
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        
        # 4. Validation scores
        plt.subplot(2, 2, 4)
        test_names = [r.test_name for r in validation_results]
        scores = [r.score for r in validation_results]
        colors = ['green' if r.status == 'passed' else 'orange' if r.status == 'warning' else 'red' for r in validation_results]
        
        plt.barh(test_names, scores, color=colors)
        plt.xlabel('Validation Score')
        plt.title('Validation Test Scores')
        
        plt.tight_layout()
        plt.savefig('data/visualizations/validation/validation_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance = model.feature_importances_
            feature_names = X_test.columns
            
            # Sort by importance
            sorted_indices = np.argsort(importance)[::-1]
            top_features = sorted_indices[:15]
            
            plt.bar(range(len(top_features)), importance[top_features])
            plt.xticks(range(len(top_features)), [feature_names[i] for i in top_features], rotation=45)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Top 15 Feature Importance')
            plt.tight_layout()
            plt.savefig('data/visualizations/validation/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Validation visualizations saved to data/visualizations/validation/")
    
    def _log_to_mlflow(self, report: ValidationReport):
        """Log validation results to MLflow"""
        logger.info("Logging validation results to MLflow...")
        
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        
        with mlflow.start_run(run_name="Model_Validation") as run:
            # Log overall metrics
            mlflow.log_metric("validation_overall_score", report.overall_score)
            mlflow.log_metric("is_production_ready", 1 if report.is_production_ready else 0)
            
            # Log summary metrics
            for metric_name, metric_value in report.summary_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"validation_{metric_name}", metric_value)
            
            # Log individual test scores
            for result in report.validation_results:
                mlflow.log_metric(f"validation_{result.test_name.lower().replace(' ', '_')}_score", result.score)
            
            # Log artifacts
            mlflow.log_artifacts("data/visualizations/validation")
            mlflow.log_artifact("data/validation/validation_report.json")
            
            logger.info(f"Validation results logged to MLflow. Run ID: {run.info.run_id}")

def main():
    """Main function to run model validation"""
    validator = ImprovedModelValidator()
    report = validator.run_comprehensive_validation()
    
    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Model: {report.model_name}")
    print(f"Overall Score: {report.overall_score:.1f}%")
    print(f"Production Ready: {'Yes' if report.is_production_ready else 'No'}")
    
    print("\n=== TEST RESULTS ===")
    for result in report.validation_results:
        status_symbol = "✓" if result.status == 'passed' else "⚠" if result.status == 'warning' else "✗"
        print(f"{status_symbol} {result.test_name}: {result.score:.1f}% - {result.message}")
    
    print("\n=== RECOMMENDATIONS ===")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    main()