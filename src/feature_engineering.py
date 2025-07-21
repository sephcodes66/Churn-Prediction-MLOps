
import pandas as pd
import numpy as np
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from scipy import stats
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/feature_engineering.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FeatureEngineeringConfig:
    """Data class to hold feature engineering settings."""
    rolling_windows: List[int]
    statistical_features: List[str]
    domain_features: bool
    feature_selection: bool
    target_transformation: str
    scaling_method: str
    handle_missing: str
    min_feature_importance: float
    max_features: int

class FeatureValidator:
    """Validates the quality of features and the target variable."""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_features(self, df: pd.DataFrame, feature_names: List[str]) -> Dict:
        """Checks the quality of the generated features."""
        results = {}
        
        # Basic feature checks
        results['feature_count'] = len(feature_names)
        results['missing_values'] = df[feature_names].isnull().sum().to_dict()
        results['infinite_values'] = {}
        results['constant_features'] = []
        results['highly_correlated_features'] = []
        
        # Check for infinite values
        for feature in feature_names:
            if pd.api.types.is_numeric_dtype(df[feature]):
                inf_count = np.isinf(df[feature]).sum()
                results['infinite_values'][feature] = inf_count
                
        # Check for constant features
        for feature in feature_names:
            if df[feature].nunique() <= 1:
                results['constant_features'].append(feature)
                
        # Check for highly correlated features
        numeric_features = [f for f in feature_names if pd.api.types.is_numeric_dtype(df[f])]
        if len(numeric_features) > 1:
            corr_matrix = df[numeric_features].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]
                        ))
            
            results['highly_correlated_features'] = high_corr_pairs
            
        # Analyze feature distributions
        results['feature_distributions'] = {}
        for feature in numeric_features[:10]:  # Analyze the first 10 numeric features
            feature_data = df[feature].dropna()
            if len(feature_data) > 0:
                results['feature_distributions'][feature] = {
                    'skewness': stats.skew(feature_data),
                    'kurtosis': stats.kurtosis(feature_data),
                    'normality_test': stats.jarque_bera(feature_data)[1],  # p-value
                    'zero_percentage': (feature_data == 0).sum() / len(feature_data)
                }
                
        self.validation_results = results
        return results
        
    def validate_target_variable(self, target: pd.Series) -> Dict:
        """Checks the quality of the target variable."""
        results = {}
        
        target_clean = target.dropna()
        
        results['basic_stats'] = {
            'count': len(target_clean),
            'mean': target_clean.mean(),
            'std': target_clean.std(),
            'min': target_clean.min(),
            'max': target_clean.max(),
            'median': target_clean.median(),
            'skewness': stats.skew(target_clean),
            'kurtosis': stats.kurtosis(target_clean)
        }
        
        # Check for negative values
        results['negative_values'] = (target_clean < 0).sum()
        
        # Analyze the distribution
        results['distribution_analysis'] = {
            'normality_test_pvalue': stats.jarque_bera(target_clean)[1],
            'zero_values': (target_clean == 0).sum(),
            'outliers_iqr': self._count_outliers_iqr(target_clean)
        }
        
        return results
        
    def _count_outliers_iqr(self, data: pd.Series) -> int:
        """Counts outliers using the IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((data < lower_bound) | (data > upper_bound)).sum()

class DomainFeatureEngineer:
    """Creates domain-specific features."""
    
    def __init__(self):
        self.feature_names = []
        
    def create_degradation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates features related to equipment degradation."""
        logger.info("Creating degradation features...")
        
        # Degradation trend features
        sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        degradation_features = []
        
        for sensor in sensor_cols:
            if sensor in df.columns:
                # Rate of change of sensor readings
                degradation_col = f'{sensor}_degradation_rate'
                df[degradation_col] = df.groupby('unit_number')[sensor].diff()
                degradation_features.append(degradation_col)
                
                # Cumulative degradation
                cumulative_col = f'{sensor}_cumulative_degradation'
                df[cumulative_col] = df.groupby('unit_number')[degradation_col].cumsum()
                degradation_features.append(cumulative_col)
                
                # Deviation from a healthy baseline
                healthy_baseline = df[df['time_in_cycles'] <= 10].groupby('unit_number')[sensor].mean()
                healthy_baseline_col = f'{sensor}_deviation_from_healthy'
                df[healthy_baseline_col] = df.apply(
                    lambda row: row[sensor] - healthy_baseline.get(row['unit_number'], row[sensor]),
                    axis=1
                )
                degradation_features.append(healthy_baseline_col)
                
        self.feature_names.extend(degradation_features)
        return df
        
    def create_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates features related to operational conditions."""
        logger.info("Creating operational features...")
        
        operational_features = []
        
        # Operating regime classification
        if all(f'op_setting_{i}' in df.columns for i in range(1, 4)):
            # Create regime clusters based on operational settings
            op_conditions = df[['op_setting_1', 'op_setting_2', 'op_setting_3']].copy()
            
            # Normalize operational conditions
            op_conditions_norm = (op_conditions - op_conditions.mean()) / op_conditions.std()
            
            # Create a categorical regime indicator
            regime_parts = []
            for col in ['op_setting_1', 'op_setting_2', 'op_setting_3']:
                part = op_conditions_norm[col].round(1).fillna(0).astype(str)
                regime_parts.append(part)
            
            df['operating_regime'] = regime_parts[0] + '_' + regime_parts[1] + '_' + regime_parts[2]
            
            # Operating severity index
            severity_components = []
            for col in ['op_setting_1', 'op_setting_2', 'op_setting_3']:
                component = op_conditions_norm[col].fillna(0)**2
                severity_components.append(component)
            
            df['operating_severity'] = np.sqrt(
                severity_components[0] + severity_components[1] + severity_components[2]
            )
            
            operational_features.extend(['operating_regime', 'operating_severity'])
            
        self.feature_names.extend(operational_features)
        return df
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates features based on time and cycles."""
        logger.info("Creating temporal features...")
        
        temporal_features = []
        
        # Cycle-based features
        df['cycle_normalized'] = df.groupby('unit_number')['time_in_cycles'].transform(
            lambda x: x / x.max()
        )
        temporal_features.append('cycle_normalized')
        
        # Life stage indicators
        df['early_life'] = (df['cycle_normalized'] <= 0.3).astype(int)
        df['mid_life'] = ((df['cycle_normalized'] > 0.3) & (df['cycle_normalized'] <= 0.7)).astype(int)
        df['end_life'] = (df['cycle_normalized'] > 0.7).astype(int)
        
        temporal_features.extend(['early_life', 'mid_life', 'end_life'])
        
        # Remaining life percentage
        if 'RUL' in df.columns:
            df['remaining_life_percentage'] = df['RUL'] / (df['RUL'] + df['time_in_cycles'])
            temporal_features.append('remaining_life_percentage')
            
        self.feature_names.extend(temporal_features)
        return df

class FeatureEngineer:
    """Creates and transforms features for the model."""
    
    def __init__(self, config_path: str = "config/main_config.yaml"):
        self.config = self._load_config(config_path)
        self.fe_config = self._create_fe_config()
        self.validator = FeatureValidator()
        self.domain_engineer = DomainFeatureEngineer()
        self.fitted_transformers = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Loads the configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    def _create_fe_config(self) -> FeatureEngineeringConfig:
        """Creates a FeatureEngineeringConfig object from the main config."""
        return FeatureEngineeringConfig(
            rolling_windows=[5, 10, 20, 30],
            statistical_features=['mean', 'std', 'min', 'max', 'skew', 'kurtosis'],
            domain_features=True,
            feature_selection=True,
            target_transformation='yeo-johnson',
            scaling_method='robust',
            handle_missing='interpolate',
            min_feature_importance=0.001,
            max_features=200
        )
        
    def calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the Remaining Useful Life (RUL) for each cycle."""
        logger.info("Calculating RUL...")
        
        if 'RUL' not in df.columns:
            # Get the maximum number of cycles for each unit
            max_cycles = df.groupby('unit_number')['time_in_cycles'].max()
            
            # Validate max cycles
            if max_cycles.min() < 10:
                logger.warning(f"Some units have few cycles: min={max_cycles.min()}")
                
            # Calculate RUL
            df['RUL'] = df.apply(
                lambda row: max_cycles[row['unit_number']] - row['time_in_cycles'], 
                axis=1
            )
            
            # Validate RUL
            if (df['RUL'] < 0).any():
                logger.error("Negative RUL values detected!")
                raise ValueError("RUL calculation resulted in negative values")
                
            logger.info(f"RUL calculated: mean={df['RUL'].mean():.2f}, std={df['RUL'].std():.2f}")
            
        return df
        
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates rolling window features for sensor data."""
        logger.info("Creating rolling window features...")
        
        sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        available_sensors = [col for col in sensor_cols if col in df.columns]
        
        new_features = []
        
        for window in self.fe_config.rolling_windows:
            for sensor in available_sensors:
                for stat in self.fe_config.statistical_features:
                    feature_name = f'{sensor}_{stat}_{window}'
                    
                    if stat == 'mean':
                        feature_series = df.groupby('unit_number')[sensor].rolling(window).mean()
                    elif stat == 'std':
                        feature_series = df.groupby('unit_number')[sensor].rolling(window).std()
                    elif stat == 'min':
                        feature_series = df.groupby('unit_number')[sensor].rolling(window).min()
                    elif stat == 'max':
                        feature_series = df.groupby('unit_number')[sensor].rolling(window).max()
                    elif stat == 'skew':
                        feature_series = df.groupby('unit_number')[sensor].rolling(window).skew()
                    elif stat == 'kurtosis':
                        feature_series = df.groupby('unit_number')[sensor].rolling(window).kurt()
                    else:
                        continue
                        
                    # Align the feature series with the DataFrame index
                    feature_series = feature_series.reset_index(level=0, drop=True)
                    df[feature_name] = feature_series
                    new_features.append(feature_name)
                    
        logger.info(f"Created {len(new_features)} rolling window features")
        return df, new_features
        
    def handle_missing_values(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Handles missing values in the feature set."""
        logger.info("Handling missing values...")
        
        for col in feature_cols:
            if df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                missing_percentage = missing_count / len(df) * 100
                
                if missing_percentage > 50:
                    logger.warning(f"High missing percentage for {col}: {missing_percentage:.2f}%")
                    
                if self.fe_config.handle_missing == 'interpolate':
                    # Interpolate within each unit
                    df[col] = df.groupby('unit_number')[col].transform(
                        lambda x: x.interpolate(method='linear').fillna(x.mean())
                    )
                elif self.fe_config.handle_missing == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif self.fe_config.handle_missing == 'median':
                    df[col] = df[col].fillna(df[col].median())
                    
        return df
        
    def perform_feature_selection(self, X: pd.DataFrame, y: pd.Series, is_training: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """Selects the most relevant features."""
        logger.info("Performing feature selection...")
        
        if is_training:
            # Separate feature types
            numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
            
            # Remove complex categorical features
            logger.info(f"Removing {len(categorical_features)} categorical features: {categorical_features}")
            X = X[numerical_features]
            
            # Remove constant features
            constant_features = [col for col in X.columns if X[col].nunique() <= 1]
            X = X.drop(columns=constant_features)
            
            # Remove highly correlated features
            corr_matrix = X.corr()
            high_corr_features = set()
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.95:
                        high_corr_features.add(corr_matrix.columns[j])
                        
            X = X.drop(columns=high_corr_features)
            
            # Statistical feature selection
            if len(X.columns) > self.fe_config.max_features:
                selector = SelectKBest(score_func=f_regression, k=self.fe_config.max_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                
                # Save the selector
                self.fitted_transformers['feature_selector'] = selector
                
            # Save the list of selected features
            self.fitted_transformers['selected_features'] = X.columns.tolist()
            
        else:
            # Apply the saved feature selection
            if 'selected_features' in self.fitted_transformers:
                available_features = [col for col in self.fitted_transformers['selected_features'] if col in X.columns]
                X = X[available_features]
                
        logger.info(f"Feature selection completed: {len(X.columns)} features selected")
        return X, X.columns.tolist()
        
    def transform_target(self, y: pd.Series, is_training: bool = True) -> pd.Series:
        """Transforms the target variable to be more normally distributed."""
        logger.info("Transforming target variable...")
        
        if is_training:
            # Validate the target variable first
            target_validation = self.validator.validate_target_variable(y)
            
            if target_validation['negative_values'] > 0:
                raise ValueError(f"Target has {target_validation['negative_values']} negative values")
                
            # Apply the specified transformation
            if self.fe_config.target_transformation == 'yeo-johnson':
                transformer = PowerTransformer(method='yeo-johnson')
                y_transformed = transformer.fit_transform(y.values.reshape(-1, 1)).flatten()
                
            elif self.fe_config.target_transformation == 'log':
                # Add a small constant to handle zero values
                y_transformed = np.log1p(y)
                transformer = None
                
            else:
                y_transformed = y
                transformer = None
                
            if transformer is not None:
                self.fitted_transformers['target_transformer'] = transformer
                
            y_transformed = pd.Series(y_transformed, index=y.index)
            
        else:
            # Apply the saved transformation
            if 'target_transformer' in self.fitted_transformers:
                transformer = self.fitted_transformers['target_transformer']
                y_transformed = transformer.transform(y.values.reshape(-1, 1)).flatten()
                y_transformed = pd.Series(y_transformed, index=y.index)
            else:
                y_transformed = y
                
        logger.info(f"Target transformation completed: mean={y_transformed.mean():.3f}, std={y_transformed.std():.3f}")
        return y_transformed
        
    def scale_features(self, X: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Scales the features using the specified method."""
        logger.info("Scaling features...")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if is_training:
            if self.fe_config.scaling_method == 'robust':
                scaler = RobustScaler()
            elif self.fe_config.scaling_method == 'standard':
                scaler = StandardScaler()
            else:
                return X
                
            X_scaled = X.copy()
            X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            
            self.fitted_transformers['feature_scaler'] = scaler
            
        else:
            if 'feature_scaler' in self.fitted_transformers:
                scaler = self.fitted_transformers['feature_scaler']
                X_scaled = X.copy()
                X_scaled[numeric_cols] = scaler.transform(X[numeric_cols])
            else:
                X_scaled = X
                
        return X_scaled
        
    def save_transformers(self, output_dir: str = "data/processed"):
        """Saves the fitted transformers to disk."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for name, transformer in self.fitted_transformers.items():
            if transformer is not None:
                joblib.dump(transformer, f"{output_dir}/{name}.joblib")
                
        logger.info(f"Transformers saved to {output_dir}")
        
    def load_transformers(self, input_dir: str = "data/processed"):
        """Loads fitted transformers from disk."""
        transformer_files = {
            'target_transformer': 'target_transformer.joblib',
            'feature_scaler': 'feature_scaler.joblib',
            'feature_selector': 'feature_selector.joblib'
        }
        
        for name, filename in transformer_files.items():
            filepath = Path(input_dir) / filename
            if filepath.exists():
                self.fitted_transformers[name] = joblib.load(filepath)
                
        # Load selected features from JSON
        features_file = Path(input_dir) / "selected_features.json"
        if features_file.exists():
            with open(features_file, 'r') as f:
                self.fitted_transformers['selected_features'] = json.load(f)
                
    def create_feature_report(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Creates a report summarizing the feature engineering process."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'feature_engineering_config': {
                'rolling_windows': self.fe_config.rolling_windows,
                'statistical_features': self.fe_config.statistical_features,
                'domain_features': self.fe_config.domain_features,
                'target_transformation': self.fe_config.target_transformation,
                'scaling_method': self.fe_config.scaling_method
            },
            'feature_summary': {
                'total_features': len(feature_cols),
                'original_sensors': len([col for col in feature_cols if col.startswith('sensor_') and '_' not in col[7:]]),
                'rolling_features': len([col for col in feature_cols if any(f'_{stat}_' in col for stat in self.fe_config.statistical_features)]),
                'domain_features': len([col for col in feature_cols if col in self.domain_engineer.feature_names]),
                'temporal_features': len([col for col in feature_cols if col in ['cycle_normalized', 'early_life', 'mid_life', 'end_life']])
            },
            'data_quality': {
                'missing_values': df[feature_cols].isnull().sum().sum(),
                'infinite_values': sum(np.isinf(df[col]).sum() for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])),
                'constant_features': len([col for col in feature_cols if df[col].nunique() <= 1])
            },
            'feature_validation': self.validator.validation_results
        }
        
        return report
        
    def run_feature_engineering(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Runs the full feature engineering pipeline."""
        logger.info(f"=== Starting Feature Engineering (Training: {is_training}) ===")
        
        # 1. Calculate RUL
        df = self.calculate_rul(df)
        
        # 2. Create rolling window features
        df, rolling_features = self.create_rolling_features(df)
        
        # 3. Create domain-specific features
        if self.fe_config.domain_features:
            df = self.domain_engineer.create_degradation_features(df)
            df = self.domain_engineer.create_operational_features(df)
            df = self.domain_engineer.create_temporal_features(df)
            
        # 4. Handle missing values
        feature_cols = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
        df = self.handle_missing_values(df, feature_cols)
        
        # 5. Validate features
        self.validator.validate_features(df, feature_cols)
        
        # 6. Separate features and target
        X = df[feature_cols]
        y = df['RUL']
        
        # 7. Perform feature selection
        if self.fe_config.feature_selection:
            X, selected_features = self.perform_feature_selection(X, y, is_training)
            
        # 8. Scale features
        X = self.scale_features(X, is_training)
        
        # 9. Transform the target variable
        y = self.transform_target(y, is_training)
        
        # 10. Save transformers if in training mode
        if is_training:
            self.save_transformers()
            
            # Save the list of selected features
            with open("data/processed/selected_features.json", 'w') as f:
                json.dump(X.columns.tolist(), f)
                
        # 11. Create a feature engineering report
        report = self.create_feature_report(df, X.columns.tolist())
        
        os.makedirs("data/validation", exist_ok=True)
        with open(f"data/validation/feature_engineering_report_{'train' if is_training else 'inference'}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # 12. Combine results into a final DataFrame
        result_df = X.copy()
        result_df['RUL'] = y
        result_df['unit_number'] = df['unit_number']
        result_df['time_in_cycles'] = df['time_in_cycles']
        
        logger.info("=== Feature Engineering Completed ===")
        logger.info(f"Features: {len(X.columns)}, Samples: {len(X)}")
        logger.info(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
        
        return result_df, y


def run_feature_engineering():
    """Main function to run the feature engineering module."""
    # This function is intended to be called from other scripts, not run directly.
    fe = FeatureEngineer()
    
    # Example usage
    logger.info("Feature engineering module loaded successfully")


if __name__ == "__main__":
    run_feature_engineering()
