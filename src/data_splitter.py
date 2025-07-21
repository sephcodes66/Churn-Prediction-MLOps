
import pandas as pd
import numpy as np
import sqlite3
import yaml
import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SplitConfig:
    """Data class to hold data splitting settings."""
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_state: int
    min_cycles_per_unit: int
    max_cycles_per_unit: int
    
class TimeSerisSplitter:
    """Splits time-series data into training, validation, and test sets."""
    
    def __init__(self, config_path: str = "config/main_config.yaml"):
        self.config = self._load_config(config_path)
        self.split_config = self._create_split_config()
        self.split_results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Loads the main configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    def _create_split_config(self) -> SplitConfig:
        """Creates a SplitConfig object from the main config."""
        train_ratio = 1 - self.config["data"]["test_size"] - self.config["data"]["val_size"]
        val_ratio = self.config["data"]["val_size"]
        test_ratio = self.config["data"]["test_size"]
        
        # Ensure that the split ratios sum to 1
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")
            
        return SplitConfig(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=self.config["data"]["random_state"],
            min_cycles_per_unit=10,  # Minimum cycles per engine
            max_cycles_per_unit=500  # Maximum cycles per engine
        )
        
    def load_data(self) -> pd.DataFrame:
        """Loads data from the SQLite database."""
        try:
            db_path = self.config["data"]["database_path"]
            logger.info(f"Loading data from: {db_path}")
            
            conn = sqlite3.connect(db_path)
            df = pd.read_sql("SELECT * FROM train_fd001", conn)
            conn.close()
            
            logger.info(f"Data loaded successfully: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
            
    def analyze_data_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyzes the distribution of the data."""
        analysis = {}
        
        # Analyze statistics for each unit
        unit_stats = df.groupby('unit_number').agg({
            'time_in_cycles': ['count', 'max'],
            'sensor_1': ['mean', 'std']  # Using a sample sensor for analysis
        }).round(2)
        
        unit_stats.columns = ['total_cycles', 'max_cycle', 'sensor_mean', 'sensor_std']
        
        analysis['unit_statistics'] = {
            'total_units': len(unit_stats),
            'cycles_per_unit': {
                'mean': unit_stats['total_cycles'].mean(),
                'std': unit_stats['total_cycles'].std(),
                'min': unit_stats['total_cycles'].min(),
                'max': unit_stats['total_cycles'].max(),
                'median': unit_stats['total_cycles'].median()
            },
            'lifecycle_distribution': {
                'mean': unit_stats['max_cycle'].mean(),
                'std': unit_stats['max_cycle'].std(),
                'min': unit_stats['max_cycle'].min(),
                'max': unit_stats['max_cycle'].max(),
                'median': unit_stats['max_cycle'].median()
            }
        }
        
        # Filter units based on cycle count
        valid_units = unit_stats[
            (unit_stats['total_cycles'] >= self.split_config.min_cycles_per_unit) &
            (unit_stats['total_cycles'] <= self.split_config.max_cycles_per_unit)
        ]
        
        analysis['filtering'] = {
            'valid_units': len(valid_units),
            'filtered_out': len(unit_stats) - len(valid_units),
            'retention_rate': len(valid_units) / len(unit_stats)
        }
        
        # Analyze operational settings
        op_settings = df[['op_setting_1', 'op_setting_2', 'op_setting_3']].describe()
        analysis['operational_settings'] = op_settings.to_dict()
        
        return analysis, valid_units
        
    def create_balanced_splits(self, df: pd.DataFrame, valid_units: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Creates balanced data splits based on lifecycle quartiles."""
        
        # Sort units by total cycles
        sorted_units = valid_units.sort_values('total_cycles')
        unit_list = sorted_units.index.tolist()
        
        # Calculate split sizes
        total_units = len(unit_list)
        train_size = int(total_units * self.split_config.train_ratio)
        val_size = int(total_units * self.split_config.val_ratio)
        test_size = total_units - train_size - val_size
        
        logger.info(f"Split sizes: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Use stratified sampling based on lifecycle quartiles
        lifecycle_quartiles = pd.qcut(sorted_units['total_cycles'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Prepare the splits
        train_units = []
        val_units = []
        test_units = []
        
        # Distribute units evenly across splits
        np.random.seed(self.split_config.random_state)
        
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            quartile_units = sorted_units[lifecycle_quartiles == quartile].index.tolist()
            np.random.shuffle(quartile_units)
            
            # Get proportional sizes for each quartile
            q_total = len(quartile_units)
            q_train = int(q_total * self.split_config.train_ratio)
            q_val = int(q_total * self.split_config.val_ratio)
            q_test = q_total - q_train - q_val
            
            train_units.extend(quartile_units[:q_train])
            val_units.extend(quartile_units[q_train:q_train + q_val])
            test_units.extend(quartile_units[q_train + q_val:])
            
        # Create DataFrames for each split
        train_df = df[df['unit_number'].isin(train_units)].copy()
        val_df = df[df['unit_number'].isin(val_units)].copy()
        test_df = df[df['unit_number'].isin(test_units)].copy()
        
        # Check for data leakage between splits
        self._validate_no_overlap(train_units, val_units, test_units)
        
        # Store split information
        self.split_results = {
            'train_units': train_units,
            'val_units': val_units,
            'test_units': test_units,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        }
        
        return train_df, val_df, test_df
        
    def _validate_no_overlap(self, train_units: List, val_units: List, test_units: List):
        """Ensures there is no overlap of units between splits."""
        train_set = set(train_units)
        val_set = set(val_units)
        test_set = set(test_units)
        
        # Find any overlaps
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            raise ValueError(f"Unit overlap detected: "
                           f"Train-Val: {train_val_overlap}, "
                           f"Train-Test: {train_test_overlap}, "
                           f"Val-Test: {val_test_overlap}")
                           
        logger.info("✅ No unit overlap between splits - Data leakage prevented")
        
    def validate_temporal_consistency(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Checks for temporal consistency within each unit's data."""
        validation_results = {}
        
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            issues = 0
            total_units = df['unit_number'].nunique()
            
            for unit in df['unit_number'].unique():
                unit_data = df[df['unit_number'] == unit].sort_values('time_in_cycles')
                time_cycles = unit_data['time_in_cycles'].values
                
                # Check for jumps or missing cycles
                expected_cycles = np.arange(1, len(time_cycles) + 1)
                if not np.array_equal(time_cycles, expected_cycles):
                    issues += 1
                    
            validation_results[name] = {
                'total_units': total_units,
                'temporal_issues': issues,
                'temporal_consistency_rate': (total_units - issues) / total_units
            }
            
        return validation_results
        
    def analyze_split_balance(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Analyzes the balance of the data splits."""
        balance_analysis = {}
        
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            # Get basic statistics for each split
            unit_cycles = df.groupby('unit_number')['time_in_cycles'].max()
            
            balance_analysis[name] = {
                'units': len(unit_cycles),
                'total_samples': len(df),
                'avg_cycles_per_unit': unit_cycles.mean(),
                'lifecycle_distribution': {
                    'mean': unit_cycles.mean(),
                    'std': unit_cycles.std(),
                    'min': unit_cycles.min(),
                    'max': unit_cycles.max(),
                    'median': unit_cycles.median()
                },
                'sensor_statistics': {
                    'sensor_1_mean': df['sensor_1'].mean(),
                    'sensor_1_std': df['sensor_1'].std(),
                    'sensor_2_mean': df['sensor_2'].mean(),
                    'sensor_2_std': df['sensor_2'].std()
                }
            }
            
        # Compare distributions between splits
        train_lifecycle = train_df.groupby('unit_number')['time_in_cycles'].max()
        val_lifecycle = val_df.groupby('unit_number')['time_in_cycles'].max()
        test_lifecycle = test_df.groupby('unit_number')['time_in_cycles'].max()
        
        # Can add statistical tests for more rigorous comparison
        balance_analysis['distribution_comparison'] = {
            'train_val_lifecycle_diff': abs(train_lifecycle.mean() - val_lifecycle.mean()),
            'train_test_lifecycle_diff': abs(train_lifecycle.mean() - test_lifecycle.mean()),
            'val_test_lifecycle_diff': abs(val_lifecycle.mean() - test_lifecycle.mean())
        }
        
        return balance_analysis
        
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        """Saves the data splits to CSV files."""
        try:
            processed_dir = Path(self.config["data"]["processed_data_dir"])
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            train_path = processed_dir / "train.csv"
            val_path = processed_dir / "val.csv"
            test_path = processed_dir / "test.csv"
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            # Verify that files were saved correctly
            for path, expected_len in [(train_path, len(train_df)), (val_path, len(val_df)), (test_path, len(test_df))]:
                if not path.exists():
                    raise FileNotFoundError(f"Failed to save {path}")
                    
                # Quick integrity check
                saved_df = pd.read_csv(path)
                if len(saved_df) != expected_len:
                    raise ValueError(f"Data integrity check failed for {path}")
                    
            logger.info(f"Split data saved successfully to {processed_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save split data: {e}")
            return False
            
    def create_split_visualization(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Creates and saves a visualization of the data splits."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Split Analysis', fontsize=16)
        
        # 1. Show data distribution by split size
        sizes = [len(train_df), len(val_df), len(test_df)]
        labels = ['Train', 'Validation', 'Test']
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        axes[0, 0].set_title('Data Distribution by Split')
        
        # 2. Show lifecycle distribution across splits
        train_cycles = train_df.groupby('unit_number')['time_in_cycles'].max()
        val_cycles = val_df.groupby('unit_number')['time_in_cycles'].max()
        test_cycles = test_df.groupby('unit_number')['time_in_cycles'].max()
        
        axes[0, 1].hist(train_cycles, bins=20, alpha=0.7, label='Train', color='lightblue')
        axes[0, 1].hist(val_cycles, bins=20, alpha=0.7, label='Validation', color='lightgreen')
        axes[0, 1].hist(test_cycles, bins=20, alpha=0.7, label='Test', color='lightcoral')
        axes[0, 1].set_title('Lifecycle Distribution Across Splits')
        axes[0, 1].set_xlabel('Max Cycles per Unit')
        axes[0, 1].set_ylabel('Number of Units')
        axes[0, 1].legend()
        
        # 3. Show number of units per split
        unit_counts = [train_df['unit_number'].nunique(), 
                      val_df['unit_number'].nunique(), 
                      test_df['unit_number'].nunique()]
        
        axes[1, 0].bar(labels, unit_counts, color=colors)
        axes[1, 0].set_title('Number of Units per Split')
        axes[1, 0].set_ylabel('Number of Units')
        
        # 4. Show sensor distribution across splits
        sensor_data = []
        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            sensor_data.extend([(name, value) for value in df['sensor_1'].values])
            
        sensor_df = pd.DataFrame(sensor_data, columns=['Split', 'Sensor_1'])
        sns.boxplot(data=sensor_df, x='Split', y='Sensor_1', ax=axes[1, 1])
        axes[1, 1].set_title('Sensor 1 Distribution Across Splits')
        
        plt.tight_layout()
        
        # Save the visualization
        os.makedirs("data/validation", exist_ok=True)
        plt.savefig("data/validation/split_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_split_report(self, 
                            analysis: Dict, 
                            temporal_validation: Dict, 
                            balance_analysis: Dict) -> Dict:
        """Creates a comprehensive report of the data splitting process."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'split_configuration': {
                'train_ratio': self.split_config.train_ratio,
                'val_ratio': self.split_config.val_ratio,
                'test_ratio': self.split_config.test_ratio,
                'random_state': self.split_config.random_state
            },
            'data_analysis': analysis,
            'split_results': self.split_results,
            'temporal_validation': temporal_validation,
            'balance_analysis': balance_analysis,
            'quality_metrics': {
                'data_leakage_prevented': True,
                'temporal_consistency': all(
                    val['temporal_consistency_rate'] > 0.95 
                    for val in temporal_validation.values()
                ),
                'balanced_distribution': all(
                    abs(balance_analysis['train']['lifecycle_distribution']['mean'] - 
                        balance_analysis[split]['lifecycle_distribution']['mean']) < 50
                    for split in ['val', 'test']
                )
            }
        }
        
        return report
        
    def run_splitting(self) -> bool:
        """Runs the full data splitting pipeline."""
        logger.info("=== Starting Improved Data Splitting ===")
        
        try:
            # 1. Load the data
            df = self.load_data()
            
            # 2. Analyze the data distribution
            analysis, valid_units = self.analyze_data_distribution(df)
            logger.info(f"Data analysis complete: {analysis['filtering']['valid_units']} valid units")
            
            # 3. Create balanced splits
            train_df, val_df, test_df = self.create_balanced_splits(df, valid_units)
            
            # 4. Validate temporal consistency
            temporal_validation = self.validate_temporal_consistency(train_df, val_df, test_df)
            
            # 5. Analyze split balance
            balance_analysis = self.analyze_split_balance(train_df, val_df, test_df)
            
            # 6. Save the splits
            if not self.save_splits(train_df, val_df, test_df):
                return False
                
            # 7. Create a visualization
            self.create_split_visualization(train_df, val_df, test_df)
            
            # 8. Create a report
            report = self.generate_split_report(analysis, temporal_validation, balance_analysis)
            
            os.makedirs("data/validation", exist_ok=True)
            with open("data/validation/split_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            # Show summary
            logger.info("=== Data Splitting Completed Successfully ===")
            logger.info(f"Train: {len(train_df)} samples, {train_df['unit_number'].nunique()} units")
            logger.info(f"Validation: {len(val_df)} samples, {val_df['unit_number'].nunique()} units")
            logger.info(f"Test: {len(test_df)} samples, {test_df['unit_number'].nunique()} units")
            
            return True
            
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            return False


def run_data_splitting():
    """Main function to run the data splitting pipeline."""
    splitter = TimeSerisSplitter()
    success = splitter.run_splitting()
    
    if not success:
        logger.error("Data splitting failed")
        exit(1)
    else:
        logger.info("Data splitting completed successfully")


if __name__ == "__main__":
    run_data_splitting()
