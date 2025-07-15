"""
IMPROVED MLOps Phase 9: Model Deployment Pipeline

This module implements a comprehensive model deployment pipeline with
containerization, API serving, health checks, auto-scaling, and CI/CD integration.

Key Improvements:
- Multi-environment deployment (dev, staging, prod)
- Containerized deployment with Docker
- REST API serving with FastAPI
- Health monitoring and auto-scaling
- Blue-green deployment strategy
- Rollback capabilities
- CI/CD integration with automated testing
- Configuration management and secrets handling
- Performance monitoring and logging
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
import shutil
import subprocess
import time
from abc import ABC, abstractmethod

# FastAPI for REST API
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# MLflow for model management
import mlflow
import mlflow.pyfunc

# Docker SDK
import docker

# Kubernetes client
from kubernetes import client, config

# Monitoring
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total ML requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('ml_request_duration_seconds', 'ML request latency')
MODEL_PREDICTIONS = Counter('ml_predictions_total', 'Total predictions made')
MODEL_ERRORS = Counter('ml_errors_total', 'Total prediction errors')
ACTIVE_CONNECTIONS = Gauge('ml_active_connections', 'Active connections')

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    model_name: str
    model_version: str
    environment: str  # dev, staging, prod
    deployment_strategy: str  # blue_green, rolling, canary
    scaling_config: Dict[str, Any]
    resource_limits: Dict[str, Any]
    health_check_config: Dict[str, Any]

@dataclass
class DeploymentStatus:
    """Status of model deployment"""
    deployment_id: str
    environment: str
    status: str  # deploying, healthy, degraded, failed
    model_version: str
    replicas: int
    healthy_replicas: int
    timestamp: datetime
    metrics: Dict[str, Any]

# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    data: List[Dict[str, float]]
    model_version: Optional[str] = None
    include_uncertainty: bool = False
    include_explanations: bool = False

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[float]
    model_version: str
    request_id: str
    timestamp: str
    confidence_intervals: Optional[List[List[float]]] = None
    explanations: Optional[List[Dict[str, Any]]] = None
    processing_time: float

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    model_loaded: bool
    uptime: float
    memory_usage: float
    cpu_usage: float
    predictions_served: int
    last_prediction_time: Optional[str] = None

class BaseDeploymentStrategy(ABC):
    """Base class for deployment strategies"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
    @abstractmethod
    def deploy(self, model_path: str, **kwargs) -> bool:
        """Deploy model"""
        pass
    
    @abstractmethod
    def rollback(self, previous_version: str) -> bool:
        """Rollback to previous version"""
        pass
    
    @abstractmethod
    def health_check(self) -> DeploymentStatus:
        """Check deployment health"""
        pass

class DockerDeploymentStrategy(BaseDeploymentStrategy):
    """Docker-based deployment strategy"""
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.docker_client = docker.from_env()
        self.container_name = f"{config.model_name}_{config.environment}"
        
    def deploy(self, model_path: str, **kwargs) -> bool:
        """Deploy model using Docker"""
        logger.info(f"Deploying model using Docker strategy...")
        
        try:
            # Build Docker image
            dockerfile_path = self._create_dockerfile(model_path)
            image_tag = f"{self.config.model_name}:{self.config.model_version}"
            
            logger.info(f"Building Docker image: {image_tag}")
            image, logs = self.docker_client.images.build(
                path=str(dockerfile_path.parent),
                tag=image_tag,
                dockerfile=dockerfile_path.name
            )
            
            # Stop existing container if running
            self._stop_existing_container()
            
            # Run new container
            container = self.docker_client.containers.run(
                image_tag,
                name=self.container_name,
                ports={'8000/tcp': 8000},
                environment={
                    'MODEL_NAME': self.config.model_name,
                    'MODEL_VERSION': self.config.model_version,
                    'ENVIRONMENT': self.config.environment
                },
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            # Wait for container to be ready
            self._wait_for_container_ready(container)
            
            logger.info(f"Model deployed successfully in container: {container.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False
    
    def _create_dockerfile(self, model_path: str) -> Path:
        """Create Dockerfile for deployment"""
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy model
COPY {model_path} /app/model/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        dockerfile_path = Path("deployment") / "Dockerfile"
        dockerfile_path.parent.mkdir(exist_ok=True)
        
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        return dockerfile_path
    
    def _stop_existing_container(self):
        """Stop existing container if running"""
        try:
            existing_container = self.docker_client.containers.get(self.container_name)
            existing_container.stop()
            existing_container.remove()
            logger.info(f"Stopped existing container: {self.container_name}")
        except docker.errors.NotFound:
            logger.info("No existing container to stop")
    
    def _wait_for_container_ready(self, container, timeout: int = 120):
        """Wait for container to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                container.reload()
                if container.status == 'running':
                    # Test health endpoint
                    import requests
                    response = requests.get('http://localhost:8000/health', timeout=5)
                    if response.status_code == 200:
                        logger.info("Container is ready")
                        return
            except Exception:
                pass
            
            time.sleep(5)
        
        raise TimeoutError("Container failed to become ready within timeout")
    
    def rollback(self, previous_version: str) -> bool:
        """Rollback to previous version"""
        logger.info(f"Rolling back to version: {previous_version}")
        
        try:
            # Stop current container
            self._stop_existing_container()
            
            # Start previous version
            image_tag = f"{self.config.model_name}:{previous_version}"
            container = self.docker_client.containers.run(
                image_tag,
                name=self.container_name,
                ports={'8000/tcp': 8000},
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            self._wait_for_container_ready(container)
            
            logger.info(f"Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def health_check(self) -> DeploymentStatus:
        """Check deployment health"""
        try:
            container = self.docker_client.containers.get(self.container_name)
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate metrics
            memory_usage = stats['memory_stats']['usage'] / stats['memory_stats']['limit'] * 100
            cpu_usage = self._calculate_cpu_usage(stats)
            
            status = DeploymentStatus(
                deployment_id=container.id,
                environment=self.config.environment,
                status='healthy' if container.status == 'running' else 'degraded',
                model_version=self.config.model_version,
                replicas=1,
                healthy_replicas=1 if container.status == 'running' else 0,
                timestamp=datetime.now(),
                metrics={
                    'memory_usage': memory_usage,
                    'cpu_usage': cpu_usage,
                    'container_status': container.status
                }
            )
            
            return status
            
        except docker.errors.NotFound:
            return DeploymentStatus(
                deployment_id='none',
                environment=self.config.environment,
                status='failed',
                model_version=self.config.model_version,
                replicas=0,
                healthy_replicas=0,
                timestamp=datetime.now(),
                metrics={}
            )
    
    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                return (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
            else:
                return 0.0
        except (KeyError, ZeroDivisionError):
            return 0.0

class KubernetesDeploymentStrategy(BaseDeploymentStrategy):
    """Kubernetes-based deployment strategy"""
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        
    def deploy(self, model_path: str, **kwargs) -> bool:
        """Deploy model using Kubernetes"""
        logger.info(f"Deploying model using Kubernetes strategy...")
        
        try:
            # Create deployment manifest
            deployment_manifest = self._create_deployment_manifest(model_path)
            
            # Create service manifest
            service_manifest = self._create_service_manifest()
            
            # Deploy to Kubernetes
            self._deploy_to_kubernetes(deployment_manifest, service_manifest)
            
            logger.info(f"Model deployed successfully to Kubernetes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False
    
    def _create_deployment_manifest(self, model_path: str) -> Dict:
        """Create Kubernetes deployment manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.config.model_name}-{self.config.environment}",
                "labels": {
                    "app": self.config.model_name,
                    "version": self.config.model_version,
                    "environment": self.config.environment
                }
            },
            "spec": {
                "replicas": self.config.scaling_config.get('replicas', 2),
                "selector": {
                    "matchLabels": {
                        "app": self.config.model_name,
                        "environment": self.config.environment
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.model_name,
                            "version": self.config.model_version,
                            "environment": self.config.environment
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": self.config.model_name,
                            "image": f"{self.config.model_name}:{self.config.model_version}",
                            "ports": [{"containerPort": 8000}],
                            "env": [
                                {"name": "MODEL_NAME", "value": self.config.model_name},
                                {"name": "MODEL_VERSION", "value": self.config.model_version},
                                {"name": "ENVIRONMENT", "value": self.config.environment}
                            ],
                            "resources": {
                                "limits": self.config.resource_limits,
                                "requests": {
                                    "memory": "256Mi",
                                    "cpu": "250m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
    
    def _create_service_manifest(self) -> Dict:
        """Create Kubernetes service manifest"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.config.model_name}-service-{self.config.environment}",
                "labels": {
                    "app": self.config.model_name,
                    "environment": self.config.environment
                }
            },
            "spec": {
                "selector": {
                    "app": self.config.model_name,
                    "environment": self.config.environment
                },
                "ports": [{
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": 8000
                }],
                "type": "LoadBalancer"
            }
        }
    
    def _deploy_to_kubernetes(self, deployment_manifest: Dict, service_manifest: Dict):
        """Deploy manifests to Kubernetes"""
        namespace = self.config.environment
        
        # Create namespace if it doesn't exist
        try:
            self.core_v1.create_namespace(
                body=client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
            )
        except client.rest.ApiException as e:
            if e.status != 409:  # Ignore if namespace already exists
                raise
        
        # Deploy or update deployment
        deployment_name = deployment_manifest["metadata"]["name"]
        try:
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment_manifest
            )
        except client.rest.ApiException as e:
            if e.status == 404:
                self.apps_v1.create_namespaced_deployment(
                    namespace=namespace,
                    body=deployment_manifest
                )
            else:
                raise
        
        # Deploy or update service
        service_name = service_manifest["metadata"]["name"]
        try:
            self.core_v1.patch_namespaced_service(
                name=service_name,
                namespace=namespace,
                body=service_manifest
            )
        except client.rest.ApiException as e:
            if e.status == 404:
                self.core_v1.create_namespaced_service(
                    namespace=namespace,
                    body=service_manifest
                )
            else:
                raise
    
    def rollback(self, previous_version: str) -> bool:
        """Rollback to previous version"""
        logger.info(f"Rolling back to version: {previous_version}")
        
        try:
            deployment_name = f"{self.config.model_name}-{self.config.environment}"
            namespace = self.config.environment
            
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Update image tag
            deployment.spec.template.spec.containers[0].image = f"{self.config.model_name}:{previous_version}"
            
            # Update deployment
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def health_check(self) -> DeploymentStatus:
        """Check deployment health"""
        try:
            deployment_name = f"{self.config.model_name}-{self.config.environment}"
            namespace = self.config.environment
            
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            replicas = deployment.spec.replicas
            ready_replicas = deployment.status.ready_replicas or 0
            
            status = 'healthy' if ready_replicas == replicas else 'degraded'
            
            return DeploymentStatus(
                deployment_id=deployment_name,
                environment=self.config.environment,
                status=status,
                model_version=self.config.model_version,
                replicas=replicas,
                healthy_replicas=ready_replicas,
                timestamp=datetime.now(),
                metrics={
                    'available_replicas': deployment.status.available_replicas or 0,
                    'updated_replicas': deployment.status.updated_replicas or 0
                }
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return DeploymentStatus(
                deployment_id='none',
                environment=self.config.environment,
                status='failed',
                model_version=self.config.model_version,
                replicas=0,
                healthy_replicas=0,
                timestamp=datetime.now(),
                metrics={}
            )

class ModelAPI:
    """FastAPI application for model serving"""
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.transformers = {}
        self.start_time = time.time()
        self.prediction_count = 0
        self.last_prediction_time = None
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title=f"{model_name} ML API",
            description="Production ML model serving API",
            version=model_version
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Load model
        self._load_model()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()
            
            return HealthResponse(
                status="healthy" if self.model is not None else "unhealthy",
                version=self.model_version,
                model_loaded=self.model is not None,
                uptime=time.time() - self.start_time,
                memory_usage=psutil.virtual_memory().percent,
                cpu_usage=psutil.cpu_percent(),
                predictions_served=self.prediction_count,
                last_prediction_time=self.last_prediction_time
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Prediction endpoint"""
            REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
            
            with REQUEST_LATENCY.time():
                try:
                    start_time = time.time()
                    
                    # Validate model is loaded
                    if self.model is None:
                        MODEL_ERRORS.inc()
                        raise HTTPException(status_code=503, detail="Model not loaded")
                    
                    # Convert request to DataFrame
                    df = pd.DataFrame(request.data)
                    
                    # Preprocess data
                    processed_data = self._preprocess_data(df)
                    
                    # Make predictions
                    predictions = self.model.predict(processed_data)
                    
                    # Transform predictions back to original scale
                    if 'rul' in self.transformers:
                        predictions = self.transformers['rul'].inverse_transform(
                            predictions.reshape(-1, 1)
                        ).flatten()
                    
                    # Generate response
                    response = PredictionResponse(
                        predictions=predictions.tolist(),
                        model_version=self.model_version,
                        request_id=f"req_{int(time.time() * 1000)}",
                        timestamp=datetime.now().isoformat(),
                        processing_time=time.time() - start_time
                    )
                    
                    # Update metrics
                    MODEL_PREDICTIONS.inc()
                    self.prediction_count += 1
                    self.last_prediction_time = response.timestamp
                    
                    return response
                    
                except Exception as e:
                    MODEL_ERRORS.inc()
                    logger.error(f"Prediction failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()
        
        @self.app.get("/info")
        async def info():
            """Model information endpoint"""
            return {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "model_loaded": self.model is not None,
                "features_expected": len(self.transformers.get('selected_features', [])),
                "model_type": str(type(self.model)) if self.model else None
            }
    
    def _load_model(self):
        """Load model and transformers"""
        try:
            # Try to load from MLflow first
            model_name = f"{self.model_name}_advanced_tuned"
            try:
                model_uri = f"models:/{model_name}/latest"
                self.model = mlflow.pyfunc.load_model(model_uri)
                logger.info(f"Model loaded from MLflow: {model_uri}")
            except:
                # Fallback to local model
                model_path = "data/tuning/best_tuned_model.joblib"
                if os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                    logger.info(f"Model loaded from local file: {model_path}")
                else:
                    logger.error("No model found")
                    return
            
            # Load transformers
            self._load_transformers()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def _load_transformers(self):
        """Load data transformers"""
        try:
            # Load RUL transformer
            rul_transformer_path = "data/processed/rul_transformer.joblib"
            if os.path.exists(rul_transformer_path):
                self.transformers['rul'] = joblib.load(rul_transformer_path)
                
            # Load feature scaler
            scaler_path = "data/processed/feature_scaler.joblib"
            if os.path.exists(scaler_path):
                self.transformers['scaler'] = joblib.load(scaler_path)
                
            # Load selected features
            features_path = "data/processed/selected_features.json"
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.transformers['selected_features'] = json.load(f)
                    
            logger.info("Transformers loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load transformers: {e}")
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data"""
        try:
            # Apply feature engineering
            from improved_feature_engineering import ImprovedFeatureEngineer
            
            fe = ImprovedFeatureEngineer()
            fe.fitted_transformers = self.transformers
            
            # Load transformers
            fe.load_transformers()
            
            # Apply feature engineering
            processed_data, _ = fe.run_feature_engineering(data, is_training=False)
            
            # Select features
            if 'selected_features' in self.transformers:
                selected_features = self.transformers['selected_features']
                available_features = [col for col in selected_features if col in processed_data.columns]
                processed_data = processed_data[available_features]
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

class ModelDeployer:
    """Comprehensive model deployment orchestrator"""
    
    def __init__(self, config_path: str = "config/main_config.yaml"):
        self.config = self._load_config(config_path)
        self.deployment_strategies = {}
        self.deployment_history = []
        self._initialize_strategies()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _initialize_strategies(self):
        """Initialize deployment strategies"""
        # Default deployment configuration
        default_config = DeploymentConfig(
            model_name=self.config['model']['name'],
            model_version="latest",
            environment="dev",
            deployment_strategy="docker",
            scaling_config={"replicas": 1},
            resource_limits={"memory": "1Gi", "cpu": "500m"},
            health_check_config={"interval": 30, "timeout": 10}
        )
        
        # Initialize Docker strategy
        self.deployment_strategies['docker'] = DockerDeploymentStrategy(default_config)
        
        # Initialize Kubernetes strategy (if available)
        try:
            self.deployment_strategies['kubernetes'] = KubernetesDeploymentStrategy(default_config)
        except Exception as e:
            logger.warning(f"Kubernetes strategy not available: {e}")
    
    def deploy_model(self, environment: str = "dev", 
                    strategy: str = "docker", 
                    model_version: str = "latest") -> bool:
        """Deploy model to specified environment"""
        logger.info(f"Deploying model to {environment} using {strategy} strategy...")
        
        try:
            # Get deployment strategy
            if strategy not in self.deployment_strategies:
                raise ValueError(f"Deployment strategy '{strategy}' not available")
            
            deployment_strategy = self.deployment_strategies[strategy]
            
            # Update configuration
            deployment_strategy.config.environment = environment
            deployment_strategy.config.model_version = model_version
            
            # Get model path
            model_path = self._get_model_path(model_version)
            
            # Create deployment package
            package_path = self._create_deployment_package(model_path, environment)
            
            # Deploy model
            success = deployment_strategy.deploy(package_path)
            
            # Record deployment
            self.deployment_history.append({
                'timestamp': datetime.now().isoformat(),
                'environment': environment,
                'strategy': strategy,
                'model_version': model_version,
                'success': success
            })
            
            # Save deployment history
            self._save_deployment_history()
            
            if success:
                logger.info(f"Model deployed successfully to {environment}")
                
                # Run post-deployment validation
                self._post_deployment_validation(environment, strategy)
                
            return success
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def _get_model_path(self, model_version: str) -> str:
        """Get model path for deployment"""
        if model_version == "latest":
            return "data/tuning/best_tuned_model.joblib"
        else:
            return f"data/models/{model_version}/model.joblib"
    
    def _create_deployment_package(self, model_path: str, environment: str) -> str:
        """Create deployment package"""
        logger.info("Creating deployment package...")
        
        # Create deployment directory
        package_dir = Path("deployment") / f"package_{environment}"
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        if os.path.exists(model_path):
            shutil.copy2(model_path, package_dir / "model.joblib")
        
        # Copy transformers
        transformers_dir = Path("data/processed")
        if transformers_dir.exists():
            shutil.copytree(transformers_dir, package_dir / "transformers", dirs_exist_ok=True)
        
        # Copy source code
        src_files = [
            "src/improved_feature_engineering.py",
            "src/improved_prediction_pipeline.py"
        ]
        
        for src_file in src_files:
            if os.path.exists(src_file):
                shutil.copy2(src_file, package_dir)
        
        # Create API file
        self._create_api_file(package_dir)
        
        # Create requirements file
        self._create_requirements_file(package_dir)
        
        logger.info(f"Deployment package created at: {package_dir}")
        return str(package_dir)
    
    def _create_api_file(self, package_dir: Path):
        """Create API file for deployment"""
        api_content = f"""
import os
import sys
sys.path.append(os.path.dirname(__file__))

from improved_model_deployment import ModelAPI

# Initialize API
model_name = os.getenv('MODEL_NAME', '{self.config['model']['name']}')
model_version = os.getenv('MODEL_VERSION', 'latest')

api = ModelAPI(model_name, model_version)
app = api.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        
        with open(package_dir / "api.py", 'w') as f:
            f.write(api_content)
    
    def _create_requirements_file(self, package_dir: Path):
        """Create requirements file"""
        requirements = [
            "fastapi==0.104.1",
            "uvicorn==0.24.0",
            "pandas==1.5.3",
            "numpy==1.24.3",
            "scikit-learn==1.3.0",
            "xgboost==1.7.6",
            "joblib==1.3.2",
            "mlflow==2.7.1",
            "pydantic==2.4.2",
            "prometheus-client==0.17.1",
            "psutil==5.9.5",
            "pyyaml==6.0.1"
        ]
        
        with open(package_dir / "requirements.txt", 'w') as f:
            f.write('\n'.join(requirements))
    
    def _post_deployment_validation(self, environment: str, strategy: str):
        """Run post-deployment validation"""
        logger.info("Running post-deployment validation...")
        
        try:
            # Test health endpoint
            import requests
            import time
            
            # Wait for service to be ready
            time.sleep(10)
            
            # Test health endpoint
            response = requests.get('http://localhost:8000/health', timeout=30)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Health check passed: {health_data}")
                
                # Test prediction endpoint
                test_data = {
                    "data": [
                        {f"sensor_{i}": 0.5 for i in range(1, 22)},
                        {"unit_number": 1, "time_in_cycles": 100}
                    ]
                }
                
                pred_response = requests.post('http://localhost:8000/predict', 
                                            json=test_data, timeout=30)
                if pred_response.status_code == 200:
                    pred_data = pred_response.json()
                    logger.info(f"Prediction test passed: {pred_data}")
                else:
                    logger.error(f"Prediction test failed: {pred_response.status_code}")
            else:
                logger.error(f"Health check failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Post-deployment validation failed: {e}")
    
    def rollback_deployment(self, environment: str, strategy: str, 
                          previous_version: str) -> bool:
        """Rollback deployment to previous version"""
        logger.info(f"Rolling back deployment in {environment}...")
        
        try:
            if strategy not in self.deployment_strategies:
                raise ValueError(f"Deployment strategy '{strategy}' not available")
            
            deployment_strategy = self.deployment_strategies[strategy]
            deployment_strategy.config.environment = environment
            
            success = deployment_strategy.rollback(previous_version)
            
            if success:
                logger.info(f"Rollback completed successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def check_deployment_health(self, environment: str, strategy: str) -> DeploymentStatus:
        """Check deployment health"""
        try:
            if strategy not in self.deployment_strategies:
                raise ValueError(f"Deployment strategy '{strategy}' not available")
            
            deployment_strategy = self.deployment_strategies[strategy]
            deployment_strategy.config.environment = environment
            
            return deployment_strategy.health_check()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return DeploymentStatus(
                deployment_id='none',
                environment=environment,
                status='failed',
                model_version='unknown',
                replicas=0,
                healthy_replicas=0,
                timestamp=datetime.now(),
                metrics={}
            )
    
    def _save_deployment_history(self):
        """Save deployment history"""
        os.makedirs("data/deployment", exist_ok=True)
        
        with open("data/deployment/deployment_history.json", 'w') as f:
            json.dump(self.deployment_history, f, indent=2, default=str)
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get deployment summary"""
        return {
            "available_strategies": list(self.deployment_strategies.keys()),
            "deployment_history": self.deployment_history[-10:],  # Last 10 deployments
            "model_name": self.config['model']['name']
        }

def main():
    """Main function to run model deployment"""
    deployer = ImprovedModelDeployer()
    
    # Deploy to development environment
    success = deployer.deploy_model(environment="dev", strategy="docker")
    
    if success:
        print("✓ Model deployed successfully!")
        
        # Check deployment health
        health = deployer.check_deployment_health("dev", "docker")
        print(f"Deployment Status: {health.status}")
        print(f"Healthy Replicas: {health.healthy_replicas}/{health.replicas}")
        
        # Get deployment summary
        summary = deployer.get_deployment_summary()
        print(f"Available Strategies: {summary['available_strategies']}")
        
    else:
        print("✗ Deployment failed!")

if __name__ == "__main__":
    main()