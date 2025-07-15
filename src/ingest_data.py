"""
IMPROVED MLOps Phase 1: Data Ingestion with Validation

This module implements robust data ingestion with comprehensive validation,
data profiling, and quality checks following software engineering best practices.

Key Improvements:
- Comprehensive data validation and schema enforcement
- Data profiling and quality metrics
- Configurable data sources and robust error handling
- Output verification and integrity checks
- Detailed logging and monitoring
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from kaggle.api.kaggle_api_extended import KaggleApi
import hashlib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion parameters"""
    dataset_name: str
    download_path: str
    database_path: str
    target_file: str
    expected_columns: int
    expected_min_rows: int
    expected_max_rows: int
    
class DataValidator:
    """Comprehensive data validation and quality checks"""
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.validation_results = {}
        
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validate data schema and structure"""
        results = {}
        
        # Check column count
        results['column_count'] = len(df.columns) == self.config.expected_columns
        
        # Check row count
        results['row_count'] = (
            self.config.expected_min_rows <= len(df) <= self.config.expected_max_rows
        )
        
        # Check for required columns
        expected_cols = (
            ['unit_number', 'time_in_cycles'] + 
            [f'op_setting_{i}' for i in range(1, 4)] + 
            [f'sensor_{i}' for i in range(1, 22)]
        )
        results['required_columns'] = all(col in df.columns for col in expected_cols)
        
        # Check data types
        results['unit_number_type'] = pd.api.types.is_integer_dtype(df['unit_number'])
        results['time_cycles_type'] = pd.api.types.is_integer_dtype(df['time_in_cycles'])
        
        # Check for sensor data types
        sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        results['sensor_types'] = all(
            pd.api.types.is_numeric_dtype(df[col]) for col in sensor_cols 
            if col in df.columns
        )
        
        self.validation_results['schema'] = results
        return results
        
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate data quality metrics"""
        results = {}
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        results['missing_values'] = {
            'total_missing': missing_counts.sum(),
            'missing_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict()
        }
        
        # Duplicate rows
        results['duplicates'] = {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
        
        # Outlier detection (using IQR method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts[col] = outliers
            
        results['outliers'] = outlier_counts
        
        # Data consistency checks
        results['consistency'] = {
            'negative_time_cycles': (df['time_in_cycles'] < 0).sum(),
            'zero_unit_numbers': (df['unit_number'] <= 0).sum(),
            'time_sequence_issues': self._check_time_sequence(df)
        }
        
        self.validation_results['data_quality'] = results
        return results
        
    def _check_time_sequence(self, df: pd.DataFrame) -> int:
        """Check for time sequence consistency within each unit"""
        issues = 0
        for unit in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit].sort_values('time_in_cycles')
            time_cycles = unit_data['time_in_cycles'].values
            
            # Check for non-consecutive time cycles
            if not np.array_equal(time_cycles, np.arange(1, len(time_cycles) + 1)):
                issues += 1
                
        return issues
        
    def generate_data_profile(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate comprehensive data profile"""
        profile = {
            'basic_stats': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'creation_time': datetime.now().isoformat()
            },
            'column_info': {},
            'summary_stats': df.describe().to_dict(),
            'unique_engines': df['unit_number'].nunique(),
            'total_cycles': df['time_in_cycles'].max(),
            'avg_cycles_per_engine': df.groupby('unit_number')['time_in_cycles'].max().mean()
        }
        
        # Column-specific information
        for col in df.columns:
            profile['column_info'][col] = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'min_value': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'max_value': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None
            }
            
        return profile
        
    def save_validation_report(self, output_path: str):
        """Save validation results to JSON file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'summary': {
                'schema_valid': all(self.validation_results.get('schema', {}).values()),
                'data_quality_score': self._calculate_quality_score()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        if 'data_quality' not in self.validation_results:
            return 0.0
            
        dq = self.validation_results['data_quality']
        
        # Scoring criteria
        missing_score = max(0, 100 - dq['missing_values']['missing_percentage'])
        duplicate_score = max(0, 100 - dq['duplicates']['duplicate_percentage'])
        consistency_score = 100 if sum(dq['consistency'].values()) == 0 else 80
        
        return (missing_score + duplicate_score + consistency_score) / 3


class DataIngestor:
    """Improved data ingestion with comprehensive validation"""
    
    def __init__(self, config_path: str = "config/main_config.yaml"):
        self.config = self._load_config(config_path)
        self.ingestion_config = self._create_ingestion_config()
        self.validator = DataValidator(self.ingestion_config)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration with validation"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    def _create_ingestion_config(self) -> DataIngestionConfig:
        """Create ingestion configuration from main config"""
        return DataIngestionConfig(
            dataset_name="behrad3d/nasa-cmaps",
            download_path="data/raw",
            database_path=self.config["data"]["database_path"],
            target_file="CMaps/train_FD001.txt",
            expected_columns=26,
            expected_min_rows=10000,
            expected_max_rows=50000
        )
        
    def download_data(self) -> bool:
        """Download data from Kaggle with robust error handling"""
        try:
            logger.info("Initializing Kaggle API...")
            api = KaggleApi()
            api.authenticate()
            logger.info("Kaggle API authentication successful")
            
            # Create download directory
            Path(self.ingestion_config.download_path).mkdir(parents=True, exist_ok=True)
            
            # Download dataset
            logger.info(f"Downloading dataset: {self.ingestion_config.dataset_name}")
            api.dataset_download_files(
                self.ingestion_config.dataset_name,
                path=self.ingestion_config.download_path,
                unzip=True
            )
            
            # Verify download
            target_path = Path(self.ingestion_config.download_path) / self.ingestion_config.target_file
            if not target_path.exists():
                raise FileNotFoundError(f"Expected file not found: {target_path}")
                
            logger.info("Data download completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data download failed: {e}")
            return False
            
    def load_and_validate_data(self) -> Optional[pd.DataFrame]:
        """Load data with comprehensive validation"""
        try:
            # Load data
            file_path = Path(self.ingestion_config.download_path) / self.ingestion_config.target_file
            logger.info(f"Loading data from: {file_path}")
            
            # Read with robust parsing
            df = pd.read_csv(
                file_path,
                sep=r'\s+',
                header=None,
                engine='python',
                na_values=['', 'NULL', 'null', 'NaN']
            )
            
            # Assign column names
            column_names = (
                ['unit_number', 'time_in_cycles'] +
                [f'op_setting_{i}' for i in range(1, 4)] +
                [f'sensor_{i}' for i in range(1, 22)]
            )
            
            # Ensure we have the right number of columns
            if len(df.columns) > len(column_names):
                df = df.iloc[:, :len(column_names)]
            elif len(df.columns) < len(column_names):
                raise ValueError(f"Insufficient columns: expected {len(column_names)}, got {len(df.columns)}")
                
            df.columns = column_names
            
            logger.info(f"Data loaded successfully: {df.shape}")
            
            # Validate schema
            schema_results = self.validator.validate_schema(df)
            if not all(schema_results.values()):
                logger.warning(f"Schema validation issues: {schema_results}")
                
            # Validate data quality
            quality_results = self.validator.validate_data_quality(df)
            quality_score = self.validator._calculate_quality_score()
            logger.info(f"Data quality score: {quality_score:.2f}/100")
            
            # Generate data profile
            profile = self.validator.generate_data_profile(df)
            
            # Save validation report
            os.makedirs("data/validation", exist_ok=True)
            self.validator.save_validation_report("data/validation/ingestion_validation.json")
            
            # Save data profile
            with open("data/validation/data_profile.json", 'w') as f:
                json.dump(profile, f, indent=2, default=str)
                
            return df
            
        except Exception as e:
            logger.error(f"Data loading and validation failed: {e}")
            return None
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with type conversions and cleaning"""
        logger.info("Starting data preprocessing...")
        
        # Convert data types
        df['unit_number'] = df['unit_number'].astype(int)
        df['time_in_cycles'] = df['time_in_cycles'].astype(int)
        
        # Convert sensor columns to numeric
        sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        for col in sensor_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Handle specific sensors that should be integers
        for sensor_id in [15, 16, 17, 18]:
            col = f'sensor_{sensor_id}'
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
                
        # Fill remaining NaN values with column means
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        logger.info("Data preprocessing completed")
        return df
        
    def save_to_database(self, df: pd.DataFrame) -> bool:
        """Save data to SQLite database with integrity checks"""
        try:
            logger.info(f"Saving data to database: {self.ingestion_config.database_path}")
            
            # Create database connection
            conn = sqlite3.connect(self.ingestion_config.database_path)
            
            # Save data
            df.to_sql("train_fd001", conn, if_exists="replace", index=False)
            
            # Verify data was saved correctly
            verification_df = pd.read_sql("SELECT COUNT(*) as count FROM train_fd001", conn)
            saved_count = verification_df['count'].iloc[0]
            
            if saved_count != len(df):
                raise ValueError(f"Data integrity check failed: expected {len(df)}, saved {saved_count}")
                
            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_unit_number ON train_fd001 (unit_number)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_time_cycles ON train_fd001 (time_in_cycles)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_unit_time ON train_fd001 (unit_number, time_in_cycles)")
            
            conn.close()
            
            logger.info(f"Data saved successfully: {saved_count} rows")
            return True
            
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            return False
            
    def create_data_summary(self, df: pd.DataFrame) -> Dict:
        """Create comprehensive data summary for output verification"""
        return {
            'ingestion_timestamp': datetime.now().isoformat(),
            'data_shape': df.shape,
            'total_engines': df['unit_number'].nunique(),
            'total_cycles': df['time_in_cycles'].max(),
            'avg_cycles_per_engine': df.groupby('unit_number')['time_in_cycles'].max().mean(),
            'data_hash': hashlib.md5(df.to_string().encode()).hexdigest(),
            'column_info': {
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'null_counts': df.isnull().sum().to_dict()
            },
            'sample_data': df.head().to_dict('records')
        }
        
    def run_ingestion(self) -> bool:
        """Run complete data ingestion pipeline"""
        logger.info("=== Starting Improved Data Ingestion ===")
        
        # Step 1: Download data
        if not self.download_data():
            logger.error("Data download failed")
            return False
            
        # Step 2: Load and validate data
        df = self.load_and_validate_data()
        if df is None:
            logger.error("Data loading and validation failed")
            return False
            
        # Step 3: Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Step 4: Save to database
        if not self.save_to_database(df_processed):
            logger.error("Database save failed")
            return False
            
        # Step 5: Create and save summary
        summary = self.create_data_summary(df_processed)
        os.makedirs("data/summaries", exist_ok=True)
        with open("data/summaries/ingestion_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info("=== Data Ingestion Completed Successfully ===")
        logger.info(f"Summary: {summary['data_shape'][0]} rows, {summary['data_shape'][1]} columns")
        logger.info(f"Engines: {summary['total_engines']}, Max cycles: {summary['total_cycles']}")
        
        return True


def main():
    """Main function to run improved data ingestion"""
    ingestor = DataIngestor()
    success = ingestor.run_ingestion()
    
    if not success:
        logger.error("Data ingestion failed")
        exit(1)
    else:
        logger.info("Data ingestion completed successfully")


if __name__ == "__main__":
    main()