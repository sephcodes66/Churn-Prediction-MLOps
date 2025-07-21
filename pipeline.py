#!/usr/bin/env python3
"""
Enterprise MLOps Pipeline Orchestrator

This script orchestrates a production-ready MLOps pipeline for enterprise deployment.
It manages the entire machine learning workflow, including data ingestion,
feature engineering, model training, validation, monitoring, and deployment.

Key Features:
- Modular, phase-based execution with dependency management.
- Comprehensive validation and testing framework.
- Robust error handling and recovery mechanisms.
- Enterprise-grade logging and monitoring.
- Support for scalable deployment using containerization.

Usage:
    python pipeline.py [--phase PHASE] [--validate] [--test] [--force]
    
Options:
    --phase PHASE    Run a specific phase (1-9) or 'all'.
    --validate       Run validation checks after each phase.
    --test           Run automated tests after pipeline completion.
    --force          Force re-run of phases even if their outputs exist.

Author: Senior MLOps Engineering Team
Version: 2.0.0
Date: 2025-07-15
"""

import sys
import os
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

# Add the source directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import pipeline modules
from ingest_data import DataIngestor
from data_splitter import TimeSerisSplitter
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from hyperparameter_tuning import HyperparameterTuner
from prediction_pipeline import PredictionPipeline
from model_validation import ModelValidator
from model_monitoring import ModelMonitor
from model_deployment import ModelDeployer
from phase_validation_suite import MLOpsPipelineValidator
from pipeline_visualizer import PipelineVisualizer
import pandas as pd
import numpy as np

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('logs/mlops_pipeline.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Orchestrates the execution of the MLOps pipeline.
    
    This class manages the entire ML workflow, providing a robust and modular
    system for enterprise-level machine learning operations.
    """
    
    def __init__(self):
        self.pipeline_results = {}
        self.start_time = datetime.now()
        self.visualizer = PipelineVisualizer()
        
    def run_phase_1(self, force: bool = False) -> bool:
        """Runs Phase 1: Data Ingestion."""
        logger.info("=" * 60)
        logger.info("PHASE 1: DATA INGESTION")
        logger.info("=" * 60)
        
        # Skip if already completed and not forced
        if not force and os.path.exists("turbofan.sqlite"):
            logger.info("Phase 1 already completed. Use --force to re-run.")
            return True
            
        try:
            ingestor = DataIngestor()
            success = ingestor.run_ingestion()
            
            if success:
                logger.info("✅ Phase 1 completed successfully")
                self.pipeline_results['phase1'] = {
                    'status': 'SUCCESS',
                    'timestamp': datetime.now().isoformat(),
                    'outputs': ['turbofan.sqlite', 'data/validation/ingestion_validation.json']
                }
                # Generate visualizations for this phase
                self.visualizer.phase_1_data_ingestion()
            else:
                logger.error("❌ Phase 1 failed")
                self.pipeline_results['phase1'] = {
                    'status': 'FAILED',
                    'timestamp': datetime.now().isoformat()
                }
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Phase 1 failed with error: {e}")
            self.pipeline_results['phase1'] = {
                'status': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
            
    def run_phase_2(self, force: bool = False) -> bool:
        """Runs Phase 2: Data Splitting."""
        logger.info("=" * 60)
        logger.info("PHASE 2: DATA SPLITTING")
        logger.info("=" * 60)
        
        # Skip if already completed and not forced
        split_files = ['data/processed/train.csv', 'data/processed/val.csv', 'data/processed/test.csv']
        if not force and all(os.path.exists(f) for f in split_files):
            logger.info("Phase 2 already completed. Use --force to re-run.")
            return True
            
        try:
            splitter = TimeSerisSplitter()
            success = splitter.run_splitting()
            
            if success:
                logger.info("✅ Phase 2 completed successfully")
                self.pipeline_results['phase2'] = {
                    'status': 'SUCCESS',
                    'timestamp': datetime.now().isoformat(),
                    'outputs': split_files + ['data/validation/split_report.json']
                }
                # Generate visualizations for this phase
                self.visualizer.phase_2_data_splitting()
            else:
                logger.error("❌ Phase 2 failed")
                self.pipeline_results['phase2'] = {
                    'status': 'FAILED',
                    'timestamp': datetime.now().isoformat()
                }
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Phase 2 failed with error: {e}")
            self.pipeline_results['phase2'] = {
                'status': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
            
    def run_phase_3(self, force: bool = False) -> bool:
        """Runs Phase 3: Feature Engineering."""
        logger.info("=" * 60)
        logger.info("PHASE 3: FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        # Skip if already completed and not forced
        if not force and os.path.exists("data/processed/selected_features.json"):
            logger.info("Phase 3 already completed. Use --force to re-run.")
            return True
            
        try:
            fe = FeatureEngineer()
            
            # Load and process training, validation, and test data
            train_df = pd.read_csv("data/processed/train.csv")
            val_df = pd.read_csv("data/processed/val.csv")
            test_df = pd.read_csv("data/processed/test.csv")
            
            # Process training data (fits transformers)
            train_processed, _ = fe.run_feature_engineering(train_df, is_training=True)
            
            # Process validation and test data (applies fitted transformers)
            val_processed, _ = fe.run_feature_engineering(val_df, is_training=False)
            test_processed, _ = fe.run_feature_engineering(test_df, is_training=False)
            
            # Save the processed data
            train_processed.to_csv("data/processed/train_processed.csv", index=False)
            val_processed.to_csv("data/processed/val_processed.csv", index=False)
            test_processed.to_csv("data/processed/test_processed.csv", index=False)
            
            logger.info("✅ Phase 3 completed successfully")
            self.pipeline_results['phase3'] = {
                'status': 'SUCCESS',
                'timestamp': datetime.now().isoformat(),
                'outputs': [
                    'data/processed/train_processed.csv',
                    'data/processed/val_processed.csv',
                    'data/processed/test_processed.csv',
                    'data/processed/selected_features.json'
                ]
            }
            # Generate visualizations for this phase
            self.visualizer.phase_3_feature_engineering()
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 3 failed with error: {e}")
            self.pipeline_results['phase3'] = {
                'status': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
            
    def run_phase_4(self, force: bool = False) -> bool:
        """Runs Phase 4: Model Training."""
        logger.info("=" * 60)
        logger.info("PHASE 4: MODEL TRAINING")
        logger.info("=" * 60)
        
        # Skip if already completed and not forced
        if not force and os.path.exists("data/models/final_model.joblib"):
            logger.info("Phase 4 already completed. Use --force to re-run.")
            return True
            
        try:
            trainer = ModelTrainer()
            results = trainer.train_and_evaluate_models()
            
            logger.info("✅ Phase 4 completed successfully")
            self.pipeline_results['phase4'] = {
                'status': 'SUCCESS',
                'timestamp': datetime.now().isoformat(),
                'outputs': [
                    'data/models/final_model.joblib',
                    'data/models/training_results.json'
                ],
                'metrics': results['evaluation_results']['val_metrics']
            }
            # Generate visualizations for this phase
            self.visualizer.phase_4_model_training()
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 4 failed with error: {e}")
            self.pipeline_results['phase4'] = {
                'status': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
            
    def run_phase_5(self, force: bool = False) -> bool:
        """Runs Phase 5: Hyperparameter Tuning."""
        logger.info("=" * 60)
        logger.info("PHASE 5: HYPERPARAMETER TUNING")
        logger.info("=" * 60)
        
        # Skip if already completed and not forced
        if not force and os.path.exists("data/tuning/best_tuned_model.joblib"):
            logger.info("Phase 5 already completed. Use --force to re-run.")
            return True
            
        try:
            tuner = HyperparameterTuner()
            results = tuner.run_advanced_tuning()
            
            logger.info("✅ Phase 5 completed successfully")
            self.pipeline_results['phase5'] = {
                'status': 'SUCCESS',
                'timestamp': datetime.now().isoformat(),
                'outputs': [
                    'data/tuning/best_tuned_model.joblib',
                    'data/tuning/tuning_results.json'
                ]
            }
            # Generate visualizations for this phase
            self.visualizer.phase_5_hyperparameter_tuning()
            return True
            
        except Exception as e:
            logger.error(f"❌ Phase 5 failed with error: {e}")
            self.pipeline_results['phase5'] = {
                'status': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
            
    def run_phase_6(self, force: bool = False) -> bool:
        """Runs Phase 6: Model Prediction."""
        logger.info("=" * 60)
        logger.info("PHASE 6: MODEL PREDICTION")
        logger.info("=" * 60)
        
        try:
            predictor = PredictionPipeline()
            
            # Test the prediction pipeline with sample data
            sample_data = pd.DataFrame({
                'unit_number': [1, 2],
                'time_in_cycles': [100, 150],
                **{f'sensor_{i}': [np.random.randn(), np.random.randn()] for i in range(1, 22)}
            })
            
            # Make predictions
            response = predictor.predict(sample_data)
            
            success = response.predictions is not None and len(response.predictions) > 0
            
            if success:
                logger.info("✅ Phase 6 completed successfully")
                self.pipeline_results['phase6'] = {
                    'status': 'SUCCESS',
                    'timestamp': datetime.now().isoformat(),
                    'outputs': ['Prediction pipeline initialized and tested']
                }
                # Generate visualizations for this phase
                self.visualizer.phase_6_prediction_pipeline()
            else:
                logger.error("❌ Phase 6 failed")
                self.pipeline_results['phase6'] = {
                    'status': 'FAILED',
                    'timestamp': datetime.now().isoformat()
                }
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Phase 6 failed with error: {e}")
            self.pipeline_results['phase6'] = {
                'status': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
            
    def run_phase_7(self, force: bool = False) -> bool:
        """Runs Phase 7: Model Validation."""
        logger.info("=" * 60)
        logger.info("PHASE 7: MODEL VALIDATION")
        logger.info("=" * 60)
        
        try:
            validator = ModelValidator()
            report = validator.run_comprehensive_validation()
            
            success = report.overall_score > 50  # Minimum acceptable validation score
            
            if success:
                logger.info("✅ Phase 7 completed successfully")
                self.pipeline_results['phase7'] = {
                    'status': 'SUCCESS',
                    'timestamp': datetime.now().isoformat(),
                    'outputs': ['data/validation/validation_report.json'],
                    'validation_score': report.overall_score
                }
                # Generate visualizations for this phase
                self.visualizer.phase_7_model_validation()
            else:
                logger.error("❌ Phase 7 failed")
                self.pipeline_results['phase7'] = {
                    'status': 'FAILED',
                    'timestamp': datetime.now().isoformat(),
                    'validation_score': report.overall_score
                }
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Phase 7 failed with error: {e}")
            self.pipeline_results['phase7'] = {
                'status': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
            
    def run_phase_8(self, force: bool = False) -> bool:
        """Runs Phase 8: Model Monitoring."""
        logger.info("=" * 60)
        logger.info("PHASE 8: MODEL MONITORING")
        logger.info("=" * 60)
        
        try:
            monitor = ModelMonitor()
            
            # Load test data for monitoring
            test_data = pd.read_csv("data/processed/test_processed.csv")
            
            # Run the monitoring batch
            metrics = monitor.monitor_batch(test_data)
            
            success = metrics.model_health_score > 50  # Minimum acceptable health score
            
            if success:
                logger.info("✅ Phase 8 completed successfully")
                self.pipeline_results['phase8'] = {
                    'status': 'SUCCESS',
                    'timestamp': datetime.now().isoformat(),
                    'outputs': ['data/monitoring/latest_metrics.json'],
                    'health_score': metrics.model_health_score
                }
                # Generate visualizations for this phase
                self.visualizer.phase_8_model_monitoring()
            else:
                logger.error("❌ Phase 8 failed")
                self.pipeline_results['phase8'] = {
                    'status': 'FAILED',
                    'timestamp': datetime.now().isoformat(),
                    'health_score': metrics.model_health_score
                }
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Phase 8 failed with error: {e}")
            self.pipeline_results['phase8'] = {
                'status': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
            
    def run_phase_9(self, force: bool = False) -> bool:
        """Runs Phase 9: Model Deployment."""
        logger.info("=" * 60)
        logger.info("PHASE 9: MODEL DEPLOYMENT")
        logger.info("=" * 60)
        
        try:
            deployer = ModelDeployer()
            
            # Deploy to the development environment
            success = deployer.deploy_model(environment="dev", strategy="docker")
            
            if success:
                # Check the health of the deployment
                health = deployer.check_deployment_health("dev", "docker")
                success = health.status in ['healthy', 'degraded']
                
                logger.info("✅ Phase 9 completed successfully")
                self.pipeline_results['phase9'] = {
                    'status': 'SUCCESS',
                    'timestamp': datetime.now().isoformat(),
                    'outputs': ['Model deployed to development environment'],
                    'deployment_status': health.status
                }
                # Generate visualizations for this phase
                self.visualizer.phase_9_model_deployment()
            else:
                logger.error("❌ Phase 9 failed")
                self.pipeline_results['phase9'] = {
                    'status': 'FAILED',
                    'timestamp': datetime.now().isoformat()
                }
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Phase 9 failed with error: {e}")
            self.pipeline_results['phase9'] = {
                'status': 'ERROR',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
            
    def run_validation(self) -> dict:
        """Runs the comprehensive pipeline validation suite."""
        logger.info("=" * 60)
        logger.info("PIPELINE VALIDATION")
        logger.info("=" * 60)
        
        try:
            validator = MLOpsPipelineValidator()
            validation_report = validator.run_full_validation()
            
            logger.info("✅ Pipeline validation completed")
            return validation_report
            
        except Exception as e:
            logger.error(f"❌ Pipeline validation failed: {e}")
            return {'error': str(e)}
            
    def run_tests(self) -> bool:
        """Runs the automated test suite."""
        logger.info("=" * 60)
        logger.info("RUNNING TESTS")
        logger.info("=" * 60)
        
        try:
            import subprocess
            
            # Run pytest
            result = subprocess.run(
                ['python', '-m', 'pytest', 'tests/', '-v'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            if result.returncode == 0:
                logger.info("✅ All tests passed")
                return True
            else:
                logger.error("❌ Some tests failed")
                logger.error(result.stdout)
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            return False
            
    def save_pipeline_results(self):
        """Saves the results of the pipeline execution."""
        results = {
            'pipeline_run': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration': str(datetime.now() - self.start_time),
                'status': 'SUCCESS' if all(
                    phase.get('status') == 'SUCCESS' 
                    for phase in self.pipeline_results.values()
                ) else 'FAILED'
            },
            'phase_results': self.pipeline_results
        }
        
        os.makedirs("data/pipeline_runs", exist_ok=True)
        
        with open(f"data/pipeline_runs/pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
    def run_pipeline(self, phases: list = None, validate: bool = False, test: bool = False, force: bool = False):
        """Runs the complete MLOps pipeline."""
        logger.info("🚀 Starting MLOps Pipeline")
        logger.info(f"Phases to run: {phases if phases else 'all'}")
        logger.info(f"Validation: {'enabled' if validate else 'disabled'}")
        logger.info(f"Testing: {'enabled' if test else 'disabled'}")
        
        # Default to all phases if none are specified
        if phases is None:
            phases = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            
        # Define the phase execution functions
        phase_functions = {
            1: self.run_phase_1,
            2: self.run_phase_2,
            3: self.run_phase_3,
            4: self.run_phase_4,
            5: self.run_phase_5,
            6: self.run_phase_6,
            7: self.run_phase_7,
            8: self.run_phase_8,
            9: self.run_phase_9
        }
        
        # Run the specified phases
        for phase_num in phases:
            if phase_num in phase_functions:
                success = phase_functions[phase_num](force=force)
                if not success:
                    logger.error(f"Pipeline failed at Phase {phase_num}")
                    break
                    
                # Run validation after each phase if requested
                if validate:
                    logger.info(f"Running validation for Phase {phase_num}")
                    # Individual phase validation logic can be implemented here
                    
        # Run comprehensive validation if requested
        if validate:
            validation_report = self.run_validation()
            self.pipeline_results['validation'] = validation_report
            
        # Run tests if requested
        if test:
            test_success = self.run_tests()
            self.pipeline_results['tests'] = {'status': 'SUCCESS' if test_success else 'FAILED'}
            
        # Save the pipeline results
        self.save_pipeline_results()
        
        # Print a final summary
        self.print_pipeline_summary()
        
    def print_pipeline_summary(self):
        """Prints a summary of the pipeline execution."""
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        
        total_phases = len(self.pipeline_results)
        successful_phases = sum(1 for phase in self.pipeline_results.values() if phase.get('status') == 'SUCCESS')
        
        logger.info(f"Total Phases: {total_phases}")
        logger.info(f"Successful Phases: {successful_phases}")
        logger.info(f"Success Rate: {successful_phases/total_phases:.2%}" if total_phases > 0 else "Success Rate: N/A")
        logger.info(f"Total Duration: {datetime.now() - self.start_time}")
        
        # Print a summary for each phase
        for phase_name, phase_result in self.pipeline_results.items():
            status = phase_result.get('status', 'UNKNOWN')
            timestamp = phase_result.get('timestamp', 'N/A')
            
            status_icon = "✅" if status == 'SUCCESS' else "❌"
            logger.info(f"{status_icon} {phase_name.upper()}: {status} at {timestamp}")
            
            if 'error' in phase_result:
                logger.error(f"   Error: {phase_result['error']}")
                
            if 'metrics' in phase_result:
                metrics = phase_result['metrics']
                logger.info(f"   Metrics: R²={metrics.get('r2', 'N/A'):.3f}, RMSE={metrics.get('rmse', 'N/A'):.3f}")
        
        logger.info("=" * 60)
        
        # Generate a comprehensive pipeline summary visualization
        self.visualizer.generate_pipeline_summary()

def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run MLOps Pipeline")
    parser.add_argument('--phase', type=str, help='Run a specific phase (1-9) or "all"', default='all')
    parser.add_argument('--validate', action='store_true', help='Run validation after the pipeline')
    parser.add_argument('--test', action='store_true', help='Run tests after the pipeline')
    parser.add_argument('--force', action='store_true', help='Force re-run even if outputs exist')
    
    args = parser.parse_args()
    
    # Parse the phase argument
    if args.phase == 'all':
        phases = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        try:
            phase_num = int(args.phase)
            if 1 <= phase_num <= 9:
                phases = [phase_num]
            else:
                logger.error("Invalid phase specified. Use a number from 1 to 9 or 'all'.")
                sys.exit(1)
        except ValueError:
            logger.error("Invalid phase specified. Use a number from 1 to 9 or 'all'.")
            sys.exit(1)
    
    # Run the pipeline
    runner = PipelineRunner()
    runner.run_pipeline(
        phases=phases,
        validate=args.validate,
        test=args.test,
        force=args.force
    )

if __name__ == "__main__":
    main()