

import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import joblib
import yaml
import sqlite3
from abc import ABC, abstractmethod
from collections import deque
import threading
import time

# ml libraries
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# statistical libraries
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance
import scipy.stats as stats

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# For monitoring-specific tasks
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# For handling warnings
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DriftAlert:
    """Data class to hold information about drift alerts."""
    alert_id: str
    alert_type: str  # e.g., 'data_drift', 'concept_drift'
    severity: str  # e.g., 'low', 'medium', 'high'
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    details: Dict[str, Any]

@dataclass
class MonitoringMetrics:
    """Data class to hold all monitoring metrics for a given timestamp."""
    timestamp: datetime
    model_performance: Dict[str, float]
    data_quality: Dict[str, float]
    drift_scores: Dict[str, float]
    system_metrics: Dict[str, float]
    prediction_stats: Dict[str, float]
    alert_count: int
    model_health_score: float

class BaseMonitor(ABC):
    """Abstract base class for all monitoring components."""
    
    def __init__(self, name: str, threshold: float = 0.1):
        self.name = name
        self.threshold = threshold
        self.history = deque(maxlen=1000)
        
    @abstractmethod
    def monitor(self, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Abstract method to perform monitoring and return a score and details."""
        pass
    
    def add_to_history(self, value: float, timestamp: datetime = None):
        """Adds a monitoring value to the history."""
        if timestamp is None:
            timestamp = datetime.now()
        self.history.append((timestamp, value))

class DataDriftMonitor(BaseMonitor):
    """Monitors for data drift between reference and current data."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        super().__init__("Data Drift", threshold)
        self.reference_data = reference_data
        self.reference_stats = self._calculate_reference_stats()
        
    def _calculate_reference_stats(self) -> Dict[str, Any]:
        """Calculates summary statistics for the reference data."""
        stats = {}
        
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': self.reference_data[col].mean(),
                'std': self.reference_data[col].std(),
                'min': self.reference_data[col].min(),
                'max': self.reference_data[col].max(),
                'quartiles': self.reference_data[col].quantile([0.25, 0.5, 0.75]).tolist()
            }
        
        return stats
    
    def monitor(self, current_data: pd.DataFrame, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Monitors for data drift using statistical tests."""
        drift_results = {}
        drift_scores = []
        
        # Compare each numerical column against the reference data
        for col in self.reference_data.select_dtypes(include=[np.number]).columns:
            if col in current_data.columns:
                # Kolmogorov-Smirnov test for distribution similarity
                ks_stat, ks_p_value = ks_2samp(
                    self.reference_data[col].dropna(), 
                    current_data[col].dropna()
                )
                
                # Wasserstein distance to measure the distance between distributions
                wasserstein_dist = wasserstein_distance(
                    self.reference_data[col].dropna(), 
                    current_data[col].dropna()
                )
                
                # T-test for difference in means
                t_stat, t_p_value = stats.ttest_ind(
                    self.reference_data[col].dropna(), 
                    current_data[col].dropna()
                )
                
                # Normalize Wasserstein distance for better comparability
                ref_std = self.reference_stats[col]['std']
                normalized_distance = wasserstein_dist / ref_std if ref_std > 0 else 0
                
                drift_results[col] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'wasserstein_distance': wasserstein_dist,
                    'normalized_distance': normalized_distance,
                    't_statistic': t_stat,
                    't_p_value': t_p_value,
                    'drift_detected': ks_p_value < self.threshold or normalized_distance > 1.0
                }
                
                # Use normalized distance as the drift score for this feature
                drift_scores.append(normalized_distance)
        
        # Calculate the overall drift score as the mean of individual scores
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        
        self.add_to_history(overall_drift_score)
        
        return overall_drift_score, drift_results

class ConceptDriftMonitor(BaseMonitor):
    """Monitors for concept drift by tracking model performance over time."""
    
    def __init__(self, threshold: float = 0.1):
        super().__init__("Concept Drift", threshold)
        self.performance_history = deque(maxlen=50)
        
    def monitor(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Monitors for concept drift by detecting changes in model performance."""
        
        # Calculate current model performance
        current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        current_mae = mean_absolute_error(y_true, y_pred)
        current_r2 = r2_score(y_true, y_pred)
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'rmse': current_rmse,
            'mae': current_mae,
            'r2': current_r2
        })
        
        drift_score = 0.0
        drift_details = {}
        
        if len(self.performance_history) >= 10:
            # Compare recent performance with historical performance
            recent_rmse = np.mean([p['rmse'] for p in list(self.performance_history)[-10:]])
            historical_rmse = np.mean([p['rmse'] for p in list(self.performance_history)[:-10]])
            
            recent_r2 = np.mean([p['r2'] for p in list(self.performance_history)[-10:]])
            historical_r2 = np.mean([p['r2'] for p in list(self.performance_history)[:-10]])
            
            # Calculate relative changes in performance metrics
            rmse_change = (recent_rmse - historical_rmse) / historical_rmse if historical_rmse > 0 else 0
            r2_change = (historical_r2 - recent_r2) / abs(historical_r2) if historical_r2 != 0 else 0
            
            # Drift score is the maximum of the relative changes
            drift_score = max(rmse_change, r2_change)
            
            drift_details = {
                'current_rmse': current_rmse,
                'recent_avg_rmse': recent_rmse,
                'historical_avg_rmse': historical_rmse,
                'rmse_change': rmse_change,
                'current_r2': current_r2,
                'recent_avg_r2': recent_r2,
                'historical_avg_r2': historical_r2,
                'r2_change': r2_change,
                'drift_detected': drift_score > self.threshold
            }
        
        self.add_to_history(drift_score)
        
        return drift_score, drift_details

class PredictionDriftMonitor(BaseMonitor):
    """Monitors for drift in the distribution of model predictions."""
    
    def __init__(self, threshold: float = 0.1):
        super().__init__("Prediction Drift", threshold)
        self.prediction_history = deque(maxlen=1000)
        
    def monitor(self, predictions: np.ndarray, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Monitors for drift in the prediction distribution."""
        
        # Calculate statistics for the current batch of predictions
        current_stats = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'median': np.median(predictions),
            'q25': np.percentile(predictions, 25),
            'q75': np.percentile(predictions, 75)
        }
        
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'predictions': predictions.copy(),
            'stats': current_stats
        })
        
        drift_score = 0.0
        drift_details = current_stats.copy()
        
        if len(self.prediction_history) >= 10:
            # Compare current predictions with historical predictions
            historical_predictions = []
            for hist_entry in list(self.prediction_history)[:-5]:  # Exclude recent entries
                historical_predictions.extend(hist_entry['predictions'])
            
            if len(historical_predictions) > 0:
                # Use KS test to compare distributions
                ks_stat, ks_p_value = ks_2samp(historical_predictions, predictions)
                
                # Use KS statistic as the drift score
                drift_score = ks_stat
                
                drift_details.update({
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'historical_mean': np.mean(historical_predictions),
                    'historical_std': np.std(historical_predictions),
                    'drift_detected': ks_p_value < self.threshold
                })
        
        self.add_to_history(drift_score)
        
        return drift_score, drift_details

class DataQualityMonitor(BaseMonitor):
    """Monitors the quality of incoming data."""
    
    def __init__(self, threshold: float = 0.1):
        super().__init__("Data Quality", threshold)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.fitted = False
        
    def monitor(self, data: pd.DataFrame, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Monitors data quality aspects like missing values, duplicates, and outliers."""
        
        quality_issues = {}
        quality_scores = []
        
        # 1. Missing values
        missing_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        missing_score = max(0, 100 - missing_percentage)
        quality_scores.append(missing_score)
        
        quality_issues['missing_values'] = {
            'percentage': missing_percentage,
            'score': missing_score,
            'issue_detected': missing_percentage > 5
        }
        
        # 2. Duplicate rows
        duplicate_percentage = (data.duplicated().sum() / len(data)) * 100
        duplicate_score = max(0, 100 - duplicate_percentage)
        quality_scores.append(duplicate_score)
        
        quality_issues['duplicates'] = {
            'percentage': duplicate_percentage,
            'score': duplicate_score,
            'issue_detected': duplicate_percentage > 1
        }
        
        # 3. Outliers (using IQR method)
        numeric_data = data.select_dtypes(include=[np.number])
        total_outliers = 0
        
        for col in numeric_data.columns:
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_data[col] < (Q1 - 1.5 * IQR)) | 
                       (numeric_data[col] > (Q3 + 1.5 * IQR))).sum()
            total_outliers += outliers
        
        outlier_percentage = (total_outliers / (len(data) * len(numeric_data.columns))) * 100
        outlier_score = max(0, 100 - outlier_percentage)
        quality_scores.append(outlier_score)
        
        quality_issues['outliers'] = {
            'percentage': outlier_percentage,
            'score': outlier_score,
            'issue_detected': outlier_percentage > 5
        }
        
        # 4. Anomaly detection using Isolation Forest
        anomaly_score = 100.0  # Default score
        if len(numeric_data) > 0:
            try:
                if not self.fitted:
                    self.anomaly_detector.fit(numeric_data)
                    self.fitted = True
                
                anomaly_labels = self.anomaly_detector.predict(numeric_data)
                anomaly_percentage = (np.sum(anomaly_labels == -1) / len(anomaly_labels)) * 100
                anomaly_score = max(0, 100 - anomaly_percentage)
                quality_scores.append(anomaly_score)
                
                quality_issues['anomalies'] = {
                    'percentage': anomaly_percentage,
                    'score': anomaly_score,
                    'issue_detected': anomaly_percentage > 10
                }
            except Exception as e:
                quality_issues['anomalies'] = {
                    'error': str(e),
                    'score': 50.0,
                    'issue_detected': True
                }
        
        # Calculate overall data quality score
        overall_quality_score = np.mean(quality_scores)
        
        self.add_to_history(overall_quality_score)
        
        return overall_quality_score, quality_issues

class SystemMonitor(BaseMonitor):
    """Monitors system resources like CPU, memory, and disk."""
    
    def __init__(self, threshold: float = 80.0):
        super().__init__("System Performance", threshold)
        
    def monitor(self, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Monitors system resource usage."""
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Calculate system health score
        cpu_score = max(0, 100 - cpu_percent)
        memory_score = max(0, 100 - memory_percent)
        disk_score = max(0, 100 - disk_percent)
        
        system_health_score = (cpu_score + memory_score + disk_score) / 3
        
        system_metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent,
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'disk_score': disk_score,
            'critical_resource_usage': any([cpu_percent > 90, memory_percent > 90, disk_percent > 90])
        }
        
        self.add_to_history(system_health_score)
        
        return system_health_score, system_metrics

class AlertManager:
    """Manages the creation and sending of alerts."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_history = deque(maxlen=1000)
        self.alert_thresholds = {
            'data_drift': 0.1,
            'concept_drift': 0.1,
            'prediction_drift': 0.1,
            'data_quality': 70.0,
            'system_performance': 70.0
        }
        
    def evaluate_alerts(self, monitoring_metrics: MonitoringMetrics) -> List[DriftAlert]:
        """Evaluates monitoring metrics and generates alerts if thresholds are breached."""
        alerts = []
        
        # Check for data drift alerts
        if monitoring_metrics.drift_scores.get('data_drift', 0) > self.alert_thresholds['data_drift']:
            alerts.append(DriftAlert(
                alert_id=f"data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type='data_drift',
                severity='medium',
                message=f"Data drift detected: {monitoring_metrics.drift_scores['data_drift']:.3f}",
                metric_name='data_drift',
                current_value=monitoring_metrics.drift_scores['data_drift'],
                threshold=self.alert_thresholds['data_drift'],
                timestamp=datetime.now(),
                details={}
            ))
        
        # Check for concept drift alerts
        if monitoring_metrics.drift_scores.get('concept_drift', 0) > self.alert_thresholds['concept_drift']:
            alerts.append(DriftAlert(
                alert_id=f"concept_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type='concept_drift',
                severity='high',
                message=f"Concept drift detected: {monitoring_metrics.drift_scores['concept_drift']:.3f}",
                metric_name='concept_drift',
                current_value=monitoring_metrics.drift_scores['concept_drift'],
                threshold=self.alert_thresholds['concept_drift'],
                timestamp=datetime.now(),
                details={}
            ))
        
        # Check for data quality alerts
        if monitoring_metrics.data_quality.get('overall_score', 100) < self.alert_thresholds['data_quality']:
            alerts.append(DriftAlert(
                alert_id=f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type='data_quality',
                severity='medium',
                message=f"Data quality degraded: {monitoring_metrics.data_quality['overall_score']:.1f}%",
                metric_name='data_quality',
                current_value=monitoring_metrics.data_quality['overall_score'],
                threshold=self.alert_thresholds['data_quality'],
                timestamp=datetime.now(),
                details=monitoring_metrics.data_quality
            ))
        
        # Check for system performance alerts
        if monitoring_metrics.system_metrics.get('health_score', 100) < self.alert_thresholds['system_performance']:
            alerts.append(DriftAlert(
                alert_id=f"system_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                alert_type='system_performance',
                severity='low',
                message=f"System performance degraded: {monitoring_metrics.system_metrics['health_score']:.1f}%",
                metric_name='system_performance',
                current_value=monitoring_metrics.system_metrics['health_score'],
                threshold=self.alert_thresholds['system_performance'],
                timestamp=datetime.now(),
                details=monitoring_metrics.system_metrics
            ))
        
        self.alert_history.extend(alerts)
        
        return alerts
    
    def send_alert(self, alert: DriftAlert):
        """Sends an alert notification."""
        try:
            # Log the alert
            logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")
            
            # Send email notification if enabled
            if self.config.get('email_alerts', {}).get('enabled', False):
                self._send_email_alert(alert)
            
            # Send Slack notification if enabled
            if self.config.get('slack_alerts', {}).get('enabled', False):
                self._send_slack_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def _send_email_alert(self, alert: DriftAlert):
        """Sends an email alert (placeholder for actual implementation)."""
        # This method should be implemented based on the specific email configuration.
        pass
    
    def _send_slack_alert(self, alert: DriftAlert):
        """Sends a Slack alert (placeholder for actual implementation)."""
        # This method should be implemented based on the specific Slack configuration.
        pass

class ModelMonitor:
    """Main class for orchestrating model monitoring."""
    
    def __init__(self, config_path: str = "config/main_config.yaml"):
        self.config = self._load_config(config_path)
        self.monitors = {}
        self.alert_manager = AlertManager(self.config.get('monitoring', {}))
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_data = []
        self._initialize_monitors()
        
    def _load_config(self, config_path: str) -> Dict:
        """Loads the configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _initialize_monitors(self):
        """Initializes all monitoring components."""
        try:
            # Load reference data for drift monitoring
            reference_data = pd.read_csv("data/processed/train_processed.csv")
            feature_cols = [col for col in reference_data.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
            
            self.monitors = {
                'data_drift': DataDriftMonitor(reference_data[feature_cols]),
                'concept_drift': ConceptDriftMonitor(),
                'prediction_drift': PredictionDriftMonitor(),
                'data_quality': DataQualityMonitor(),
                'system_performance': SystemMonitor()
            }
            
            logger.info("Monitoring components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitors: {e}")
            raise
    
    def start_monitoring(self, interval_seconds: int = 300):
        """Starts continuous monitoring in a background thread."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval_seconds,)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stops the continuous monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """The main loop for continuous monitoring."""
        while self.monitoring_active:
            try:
                # In a real-world scenario, this would check for new data from a queue or stream.
                # For this example, we simulate with test data.
                self._run_monitoring_cycle()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _run_monitoring_cycle(self):
        """Runs a single monitoring cycle (placeholder)."""
        # This method would typically process new data from a stream or queue.
        # For this example, we use the `monitor_batch` method with test data.
        pass
    
    def monitor_batch(self, current_data: pd.DataFrame, model: Any = None, 
                     y_true: np.ndarray = None, y_pred: np.ndarray = None) -> MonitoringMetrics:
        """Monitors a batch of data and returns monitoring metrics."""
        logger.info("Running batch monitoring...")
        
        # Load the model if not provided
        if model is None:
            model = self._load_model()
        
        # Make predictions if not provided
        if y_pred is None and model is not None:
            feature_cols = [col for col in current_data.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
            y_pred = model.predict(current_data[feature_cols])
        
        # Extract true labels if not provided
        if y_true is None and 'RUL' in current_data.columns:
            y_true = current_data['RUL'].values
        
        # Run all monitoring components
        monitoring_results = {}
        
        # Data drift monitoring
        if 'data_drift' in self.monitors:
            feature_cols = [col for col in current_data.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
            drift_score, drift_details = self.monitors['data_drift'].monitor(current_data[feature_cols])
            monitoring_results['data_drift'] = {'score': drift_score, 'details': drift_details}
        
        # Concept drift monitoring
        if 'concept_drift' in self.monitors and y_true is not None and y_pred is not None:
            concept_score, concept_details = self.monitors['concept_drift'].monitor(y_true, y_pred)
            monitoring_results['concept_drift'] = {'score': concept_score, 'details': concept_details}
        
        # Prediction drift monitoring
        if 'prediction_drift' in self.monitors and y_pred is not None:
            pred_score, pred_details = self.monitors['prediction_drift'].monitor(y_pred)
            monitoring_results['prediction_drift'] = {'score': pred_score, 'details': pred_details}
        
        # Data quality monitoring
        if 'data_quality' in self.monitors:
            quality_score, quality_details = self.monitors['data_quality'].monitor(current_data)
            monitoring_results['data_quality'] = {'score': quality_score, 'details': quality_details}
        
        # System performance monitoring
        if 'system_performance' in self.monitors:
            system_score, system_details = self.monitors['system_performance'].monitor()
            monitoring_results['system_performance'] = {'score': system_score, 'details': system_details}
        
        # Calculate model performance metrics
        model_performance = {}
        if y_true is not None and y_pred is not None:
            model_performance = {
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2': float(r2_score(y_true, y_pred))
            }
        
        # Calculate overall model health score
        health_scores = [result['score'] for result in monitoring_results.values()]
        model_health_score = np.mean(health_scores) if health_scores else 100.0
        
        # Create a consolidated metrics object
        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            model_performance=model_performance,
            data_quality=monitoring_results.get('data_quality', {}).get('details', {}),
            drift_scores={
                'data_drift': monitoring_results.get('data_drift', {}).get('score', 0),
                'concept_drift': monitoring_results.get('concept_drift', {}).get('score', 0),
                'prediction_drift': monitoring_results.get('prediction_drift', {}).get('score', 0)
            },
            system_metrics=monitoring_results.get('system_performance', {}).get('details', {}),
            prediction_stats={
                'mean': float(np.mean(y_pred)) if y_pred is not None else 0,
                'std': float(np.std(y_pred)) if y_pred is not None else 0,
                'min': float(np.min(y_pred)) if y_pred is not None else 0,
                'max': float(np.max(y_pred)) if y_pred is not None else 0
            } if y_pred is not None else {},
            alert_count=0,
            model_health_score=model_health_score
        )
        
        # Evaluate and send alerts
        alerts = self.alert_manager.evaluate_alerts(metrics)
        metrics.alert_count = len(alerts)
        
        for alert in alerts:
            self.alert_manager.send_alert(alert)
        
        self.monitoring_data.append(metrics)
        
        # Save and visualize results
        self._save_monitoring_results(metrics, monitoring_results)
        self._create_monitoring_visualizations(metrics, monitoring_results)
        self._log_to_mlflow(metrics)
        
        logger.info(f"Monitoring completed. Health score: {model_health_score:.1f}%")
        
        return metrics
    
    def _load_model(self) -> Any:
        """Loads the latest trained model."""
        try:
            # Try loading from MLFlow first
            model_name = f"{self.config['model']['name']}_advanced_tuned"
            try:
                model_uri = f"models:/{model_name}/latest"
                model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Model loaded from MLflow: {model_uri}")
            except:
                # Fallback to local file if MLFlow fails
                model_path = "data/tuning/best_tuned_model.joblib"
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    logger.info(f"Model loaded from local file: {model_path}")
                else:
                    logger.warning("No model found")
                    return None
                    
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def _save_monitoring_results(self, metrics: MonitoringMetrics, monitoring_results: Dict):
        """Saves monitoring results to files."""
        logger.info("Saving monitoring results...")
        
        os.makedirs("data/monitoring", exist_ok=True)
        
        # Save the latest metrics as a JSON file
        metrics_dict = asdict(metrics)
        metrics_dict['timestamp'] = metrics.timestamp.isoformat()
        
        with open("data/monitoring/latest_metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        # Append to a historical log file
        history_file = "data/monitoring/metrics_history.jsonl"
        with open(history_file, 'a') as f:
            f.write(json.dumps(metrics_dict, default=str) + '\n')
        
        logger.info("Monitoring results saved")
    
    def _create_monitoring_visualizations(self, metrics: MonitoringMetrics, monitoring_results: Dict):
        """Creates and saves monitoring visualizations."""
        logger.info("Creating monitoring visualizations...")
        
        os.makedirs("data/visualizations/monitoring", exist_ok=True)
        
        # Create a monitoring dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Model health score
        axes[0, 0].bar(['Model Health'], [metrics.model_health_score], color='green' if metrics.model_health_score > 80 else 'orange' if metrics.model_health_score > 60 else 'red')
        axes[0, 0].set_title('Model Health Score')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 100)
        
        # 2. Drift scores
        drift_names = list(metrics.drift_scores.keys())
        drift_values = list(metrics.drift_scores.values())
        colors = ['red' if v > 0.1 else 'green' for v in drift_values]
        axes[0, 1].bar(drift_names, drift_values, color=colors)
        axes[0, 1].set_title('Drift Scores')
        axes[0, 1].set_ylabel('Drift Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. System metrics
        if metrics.system_metrics:
            system_names = ['CPU', 'Memory', 'Disk']
            system_values = [
                metrics.system_metrics.get('cpu_percent', 0),
                metrics.system_metrics.get('memory_percent', 0),
                metrics.system_metrics.get('disk_percent', 0)
            ]
            colors = ['red' if v > 80 else 'orange' if v > 60 else 'green' for v in system_values]
            axes[0, 2].bar(system_names, system_values, color=colors)
            axes[0, 2].set_title('System Usage (%)')
            axes[0, 2].set_ylabel('Usage %')
            axes[0, 2].set_ylim(0, 100)
        
        # 4. Model performance
        if metrics.model_performance:
            perf_names = list(metrics.model_performance.keys())
            perf_values = list(metrics.model_performance.values())
            axes[1, 0].bar(perf_names, perf_values)
            axes[1, 0].set_title('Model Performance')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Data quality
        if metrics.data_quality:
            quality_issues = ['Missing', 'Duplicates', 'Outliers']
            quality_scores = [
                metrics.data_quality.get('missing_values', {}).get('score', 100),
                metrics.data_quality.get('duplicates', {}).get('score', 100),
                metrics.data_quality.get('outliers', {}).get('score', 100)
            ]
            colors = ['red' if v < 70 else 'orange' if v < 90 else 'green' for v in quality_scores]
            axes[1, 1].bar(quality_issues, quality_scores, color=colors)
            axes[1, 1].set_title('Data Quality Scores')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_ylim(0, 100)
        
        # 6. Prediction statistics
        if metrics.prediction_stats:
            pred_names = list(metrics.prediction_stats.keys())
            pred_values = list(metrics.prediction_stats.values())
            axes[1, 2].bar(pred_names, pred_values)
            axes[1, 2].set_title('Prediction Statistics')
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/visualizations/monitoring/monitoring_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Monitoring visualizations created")
    
    def _log_to_mlflow(self, metrics: MonitoringMetrics):
        """Logs monitoring metrics to MLFlow."""
        logger.info("Logging monitoring metrics to MLflow...")
        
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        
        with mlflow.start_run(run_name="Model_Monitoring") as run:
            # Log summary metrics
            mlflow.log_metric("model_health_score", metrics.model_health_score)
            mlflow.log_metric("alert_count", metrics.alert_count)
            
            # Log drift scores
            for drift_type, score in metrics.drift_scores.items():
                mlflow.log_metric(f"drift_{drift_type}", score)
            
            # Log model performance
            for metric_name, value in metrics.model_performance.items():
                mlflow.log_metric(f"performance_{metric_name}", value)
            
            # Log system metrics
            for metric_name, value in metrics.system_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"system_{metric_name}", value)
            
            # Log visualization artifacts
            mlflow.log_artifacts("data/visualizations/monitoring")
            
            logger.info(f"Monitoring metrics logged to MLflow. Run ID: {run.info.run_id}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Returns a summary of the latest monitoring results."""
        if not self.monitoring_data:
            return {"message": "No monitoring data available"}
        
        latest_metrics = self.monitoring_data[-1]
        
        return {
            "model_health_score": latest_metrics.model_health_score,
            "last_update": latest_metrics.timestamp.isoformat(),
            "alerts_count": latest_metrics.alert_count,
            "drift_scores": latest_metrics.drift_scores,
            "model_performance": latest_metrics.model_performance,
            "system_health": latest_metrics.system_metrics,
            "recommendations": self._generate_recommendations(latest_metrics)
        }
    
    def _generate_recommendations(self, metrics: MonitoringMetrics) -> List[str]:
        """Generates recommendations based on monitoring metrics."""
        recommendations = []
        
        # Recommendations based on model health score
        if metrics.model_health_score < 60:
            recommendations.append("Critical: Model health is poor. Immediate attention required.")
        elif metrics.model_health_score < 80:
            recommendations.append("Warning: Model health is declining. Consider investigation.")
        
        # Recommendations based on drift scores
        if metrics.drift_scores.get('data_drift', 0) > 0.1:
            recommendations.append("Data drift detected. Consider retraining with recent data.")
        
        if metrics.drift_scores.get('concept_drift', 0) > 0.1:
            recommendations.append("Concept drift detected. Model performance is degrading.")
        
        # Recommendations based on system metrics
        if metrics.system_metrics.get('cpu_percent', 0) > 80:
            recommendations.append("High CPU usage detected. Consider scaling resources.")
        
        if metrics.system_metrics.get('memory_percent', 0) > 80:
            recommendations.append("High memory usage detected. Consider memory optimization.")
        
        return recommendations

def run_model_monitoring():
    """Runs the model monitoring pipeline."""
    monitor = ModelMonitor()
    
    # Load test data for monitoring
    try:
        test_data = pd.read_csv("data/processed/test_processed.csv")
        
        # Run the monitoring on the batch of test data
        metrics = monitor.monitor_batch(test_data)
        
        # Print a summary of the monitoring results
        print("\n=== MONITORING SUMMARY ===")
        print(f"Model Health Score: {metrics.model_health_score:.1f}%")
        print(f"Alerts Generated: {metrics.alert_count}")
        print(f"Timestamp: {metrics.timestamp}")
        
        print("\n=== DRIFT SCORES ===")
        for drift_type, score in metrics.drift_scores.items():
            print(f"{drift_type}: {score:.4f}")
        
        if metrics.model_performance:
            print("\n=== MODEL PERFORMANCE ===")
            for metric, value in metrics.model_performance.items():
                print(f"{metric}: {value:.4f}")
        
        # Get and print recommendations
        summary = monitor.get_monitoring_summary()
        if summary.get('recommendations'):
            print("\n=== RECOMMENDATIONS ===")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"{i}. {rec}")
        
    except Exception as e:
        print(f"Monitoring failed: {e}")

if __name__ == "__main__":
    run_model_monitoring()

