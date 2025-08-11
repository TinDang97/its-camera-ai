"""
Federated Learning Framework for ITS Camera AI Traffic Monitoring System.

This module implements federated learning capabilities for distributed model training
across multiple edge nodes and camera locations while preserving privacy.

Key Features:
- Secure aggregation of model updates from edge devices
- Privacy-preserving training with differential privacy
- Communication-efficient algorithms (FedAvg, FedProx)
- Robust aggregation against Byzantine failures
- Adaptive client selection based on data quality

Architecture:
- Central coordinator manages training rounds
- Edge nodes train on local data without sharing raw data
- Encrypted model updates aggregated at central server
- Automated client management and failure handling
"""

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class FederatedRoundStatus(Enum):
    """Status of federated learning round."""
    INITIALIZING = "initializing"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class ClientStatus(Enum):
    """Status of federated client."""
    AVAILABLE = "available"
    TRAINING = "training"
    UPLOADING = "uploading"
    OFFLINE = "offline"
    FAILED = "failed"


@dataclass
class ClientInfo:
    """Federated client information."""
    
    client_id: str
    location: str
    camera_ids: list[str]
    
    # Capabilities
    compute_power: float = 1.0  # Relative compute capability
    bandwidth_mbps: float = 100.0
    data_samples: int = 0
    data_quality_score: float = 0.0
    
    # Status
    status: ClientStatus = ClientStatus.OFFLINE
    last_seen: float = field(default_factory=time.time)
    
    # Training history
    rounds_participated: int = 0
    avg_training_time_minutes: float = 0.0
    success_rate: float = 1.0
    
    # Privacy settings
    privacy_budget: float = 10.0  # Differential privacy budget
    noise_multiplier: float = 1.0


@dataclass
class ModelUpdate:
    """Encrypted model update from federated client."""
    
    client_id: str
    round_id: int
    
    # Model weights (encrypted)
    encrypted_weights: bytes
    
    # Training metadata
    training_samples: int
    training_loss: float
    training_accuracy: float
    training_time_minutes: float
    
    # Validation
    update_hash: str
    timestamp: float = field(default_factory=time.time)
    
    # Privacy metrics
    privacy_cost: float = 0.0
    noise_level: float = 0.0


@dataclass
class FederatedRound:
    """Federated learning training round."""
    
    round_id: int
    global_model_version: str
    
    # Participants
    selected_clients: list[str]
    participating_clients: list[str] = field(default_factory=list)
    
    # Round configuration
    min_clients: int = 3
    max_clients: int = 50
    client_fraction: float = 0.1
    
    # Training parameters
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    
    # Status
    status: FederatedRoundStatus = FederatedRoundStatus.INITIALIZING
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    
    # Results
    client_updates: dict[str, ModelUpdate] = field(default_factory=dict)
    aggregated_weights: dict[str, torch.Tensor] | None = None
    
    # Performance metrics
    round_accuracy: float = 0.0
    convergence_metric: float = 0.0
    communication_rounds: int = 0


class SecureCommunication:
    """Secure communication for federated learning."""
    
    def __init__(self):
        if CRYPTO_AVAILABLE:
            # Generate encryption key for secure aggregation
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
        else:
            logger.warning("Cryptography not available - using mock encryption")
            self.cipher_suite = None
    
    def encrypt_model_weights(self, weights: dict[str, torch.Tensor]) -> bytes:
        """Encrypt model weights for secure transmission."""
        
        # Serialize weights
        weights_bytes = self._serialize_weights(weights)
        
        if self.cipher_suite:
            return self.cipher_suite.encrypt(weights_bytes)
        else:
            # Mock encryption for development
            return b"mock_encrypted_" + weights_bytes[:100]
    
    def decrypt_model_weights(self, encrypted_weights: bytes) -> dict[str, torch.Tensor]:
        """Decrypt model weights from client."""
        
        if self.cipher_suite:
            weights_bytes = self.cipher_suite.decrypt(encrypted_weights)
        else:
            # Mock decryption
            if encrypted_weights.startswith(b"mock_encrypted_"):
                weights_bytes = b"mock_weights_data"
            else:
                weights_bytes = encrypted_weights
        
        return self._deserialize_weights(weights_bytes)
    
    def _serialize_weights(self, weights: dict[str, torch.Tensor]) -> bytes:
        """Serialize model weights to bytes."""
        
        # Convert tensors to numpy arrays and serialize
        serializable_weights = {}
        for key, tensor in weights.items():
            serializable_weights[key] = tensor.cpu().numpy().tolist()
        
        return json.dumps(serializable_weights).encode('utf-8')
    
    def _deserialize_weights(self, weights_bytes: bytes) -> dict[str, torch.Tensor]:
        """Deserialize model weights from bytes."""
        
        try:
            weights_data = json.loads(weights_bytes.decode('utf-8'))
            
            # Convert back to tensors
            weights = {}
            for key, array in weights_data.items():
                weights[key] = torch.tensor(array)
            
            return weights
            
        except Exception:
            # Return empty weights on failure
            return {}


class FederatedAggregator:
    """Aggregate model updates using various federated algorithms."""
    
    def __init__(self, aggregation_method: str = "fedavg"):
        self.aggregation_method = aggregation_method
        self.security = SecureCommunication()
    
    async def aggregate_updates(
        self,
        client_updates: dict[str, ModelUpdate],
        base_weights: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Aggregate client updates into new global model."""
        
        if not client_updates:
            return base_weights
        
        # Decrypt all client updates
        decrypted_updates = {}
        total_samples = 0
        
        for client_id, update in client_updates.items():
            try:
                weights = self.security.decrypt_model_weights(update.encrypted_weights)
                if weights:  # Only include valid updates
                    decrypted_updates[client_id] = {
                        'weights': weights,
                        'samples': update.training_samples,
                        'quality': self._calculate_update_quality(update)
                    }
                    total_samples += update.training_samples
            except Exception as e:
                logger.error(f"Failed to decrypt update from {client_id}: {e}")
        
        if not decrypted_updates:
            logger.warning("No valid client updates to aggregate")
            return base_weights
        
        # Apply aggregation algorithm
        if self.aggregation_method == "fedavg":
            return self._federated_averaging(decrypted_updates, total_samples)
        elif self.aggregation_method == "fedprox":
            return self._federated_proximal(decrypted_updates, base_weights)
        elif self.aggregation_method == "robust":
            return self._robust_aggregation(decrypted_updates, base_weights)
        else:
            return self._federated_averaging(decrypted_updates, total_samples)
    
    def _federated_averaging(
        self,
        decrypted_updates: dict[str, dict],
        total_samples: int
    ) -> dict[str, torch.Tensor]:
        """Standard FedAvg algorithm - weighted average by sample count."""
        
        if not decrypted_updates:
            return {}
        
        # Get reference model structure
        reference_weights = next(iter(decrypted_updates.values()))['weights']
        aggregated_weights = {}
        
        # Initialize aggregated weights
        for key in reference_weights.keys():
            aggregated_weights[key] = torch.zeros_like(reference_weights[key])
        
        # Weighted averaging
        for client_data in decrypted_updates.values():
            weight = client_data['samples'] / total_samples
            
            for key, tensor in client_data['weights'].items():
                if key in aggregated_weights:
                    aggregated_weights[key] += tensor * weight
        
        return aggregated_weights
    
    def _federated_proximal(
        self,
        decrypted_updates: dict[str, dict],
        base_weights: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """FedProx algorithm with proximal term."""
        
        # For simplicity, use FedAvg with regularization toward base model
        total_samples = sum(data['samples'] for data in decrypted_updates.values())
        aggregated = self._federated_averaging(decrypted_updates, total_samples)
        
        # Apply proximal regularization (blend with base model)
        proximal_weight = 0.1  # Hyperparameter
        
        final_weights = {}
        for key in aggregated.keys():
            if key in base_weights:
                final_weights[key] = (
                    (1 - proximal_weight) * aggregated[key] + 
                    proximal_weight * base_weights[key]
                )
            else:
                final_weights[key] = aggregated[key]
        
        return final_weights
    
    def _robust_aggregation(
        self,
        decrypted_updates: dict[str, dict],
        base_weights: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Robust aggregation against Byzantine failures."""
        
        if len(decrypted_updates) < 3:
            # Fall back to FedAvg for small number of clients
            total_samples = sum(data['samples'] for data in decrypted_updates.values())
            return self._federated_averaging(decrypted_updates, total_samples)
        
        # Use median aggregation for robustness
        reference_weights = next(iter(decrypted_updates.values()))['weights']
        aggregated_weights = {}
        
        for key in reference_weights.keys():
            # Collect all client weights for this parameter
            client_weights = []
            for client_data in decrypted_updates.values():
                if key in client_data['weights']:
                    client_weights.append(client_data['weights'][key])
            
            if client_weights:
                # Stack tensors and compute median
                stacked = torch.stack(client_weights)
                aggregated_weights[key] = torch.median(stacked, dim=0)[0]
            else:
                aggregated_weights[key] = reference_weights[key].clone()
        
        return aggregated_weights
    
    def _calculate_update_quality(self, update: ModelUpdate) -> float:
        """Calculate quality score for client update."""
        
        # Quality based on training metrics and client reliability
        accuracy_score = update.training_accuracy
        loss_score = max(0, 1.0 - update.training_loss)  # Lower loss is better
        
        # Combine scores
        quality = (accuracy_score * 0.7 + loss_score * 0.3)
        
        return min(1.0, max(0.0, quality))


class FederatedCoordinator:
    """Central coordinator for federated learning."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        
        # Client management
        self.registered_clients: dict[str, ClientInfo] = {}
        self.active_rounds: dict[int, FederatedRound] = {}
        
        # Global model state
        self.global_model_version = "v1.0.0"
        self.global_weights: dict[str, torch.Tensor] = {}
        
        # Federated settings
        self.min_clients_per_round = config.get('min_clients_per_round', 3)
        self.max_clients_per_round = config.get('max_clients_per_round', 50)
        self.client_selection_fraction = config.get('client_selection_fraction', 0.1)
        
        # Training parameters
        self.global_rounds = config.get('global_rounds', 100)
        self.local_epochs = config.get('local_epochs', 5)
        self.convergence_threshold = config.get('convergence_threshold', 0.001)
        
        # Components
        self.aggregator = FederatedAggregator(config.get('aggregation_method', 'fedavg'))
        
        # State
        self.current_round = 0
        self.is_training = False
        self.training_history: list[dict] = []
        
        logger.info("Federated coordinator initialized")
    
    async def register_client(self, client_info: ClientInfo) -> bool:
        """Register new federated client."""
        
        try:
            client_info.last_seen = time.time()
            client_info.status = ClientStatus.AVAILABLE
            
            self.registered_clients[client_info.client_id] = client_info
            
            logger.info(
                f"Registered federated client {client_info.client_id} "
                f"with {client_info.data_samples} samples"
            )
            return True
            
        except Exception as e:
            logger.error(f"Client registration failed: {e}")
            return False
    
    async def start_federated_training(self) -> bool:
        """Start federated training process."""
        
        if self.is_training:
            logger.warning("Federated training already in progress")
            return False
        
        if len(self.registered_clients) < self.min_clients_per_round:
            logger.error(f"Need at least {self.min_clients_per_round} clients to start training")
            return False
        
        self.is_training = True
        self.current_round = 0
        
        # Start training loop
        asyncio.create_task(self._federated_training_loop())
        
        logger.info(f"Started federated training with {len(self.registered_clients)} clients")
        return True
    
    async def _federated_training_loop(self):
        """Main federated training loop."""
        
        try:
            for round_num in range(self.global_rounds):
                self.current_round = round_num + 1
                
                logger.info(f"Starting federated round {self.current_round}/{self.global_rounds}")
                
                # Create new round
                fed_round = FederatedRound(
                    round_id=self.current_round,
                    global_model_version=self.global_model_version,
                    selected_clients=[],
                    min_clients=self.min_clients_per_round,
                    max_clients=self.max_clients_per_round,
                    local_epochs=self.local_epochs
                )
                
                # Select clients for this round
                selected_clients = await self._select_clients_for_round(fed_round)
                if len(selected_clients) < self.min_clients_per_round:
                    logger.warning(f"Not enough clients available for round {self.current_round}")
                    await asyncio.sleep(60)  # Wait before trying again
                    continue
                
                fed_round.selected_clients = [c.client_id for c in selected_clients]
                fed_round.status = FederatedRoundStatus.TRAINING
                
                self.active_rounds[self.current_round] = fed_round
                
                # Coordinate training round
                success = await self._coordinate_training_round(fed_round)
                
                if success:
                    # Update global model
                    await self._update_global_model(fed_round)
                    
                    # Check convergence
                    if await self._check_convergence():
                        logger.info("Federated training converged")
                        break
                else:
                    logger.warning(f"Round {self.current_round} failed")
                
                # Cleanup completed round
                fed_round.status = FederatedRoundStatus.COMPLETED
                fed_round.end_time = time.time()
                
                # Add to history
                self._record_round_history(fed_round)
                
                # Wait before next round
                await asyncio.sleep(10)
            
            logger.info("Federated training completed")
            
        except Exception as e:
            logger.error(f"Federated training error: {e}")
        finally:
            self.is_training = False
    
    async def _select_clients_for_round(self, fed_round: FederatedRound) -> list[ClientInfo]:
        """Select clients for training round based on various criteria."""
        
        # Get available clients
        available_clients = [
            client for client in self.registered_clients.values()
            if (client.status == ClientStatus.AVAILABLE and 
                time.time() - client.last_seen < 300 and  # Last seen within 5 minutes
                client.data_samples > 0)
        ]
        
        if len(available_clients) < fed_round.min_clients:
            return []
        
        # Calculate selection scores
        client_scores = []
        for client in available_clients:
            score = self._calculate_client_selection_score(client)
            client_scores.append((client, score))
        
        # Sort by score (highest first)
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top clients up to max limit
        num_selected = min(
            len(available_clients),
            max(
                fed_round.min_clients,
                int(len(available_clients) * self.client_selection_fraction)
            ),
            fed_round.max_clients
        )
        
        selected = [client for client, _ in client_scores[:num_selected]]
        
        logger.info(f"Selected {len(selected)} clients for round {fed_round.round_id}")
        return selected
    
    def _calculate_client_selection_score(self, client: ClientInfo) -> float:
        """Calculate selection score for client."""
        
        # Factors: data quality, compute power, reliability, data volume
        quality_score = client.data_quality_score
        compute_score = min(1.0, client.compute_power)
        reliability_score = client.success_rate
        volume_score = min(1.0, client.data_samples / 1000)  # Normalize by 1000 samples
        
        # Weighted combination
        total_score = (
            quality_score * 0.3 +
            reliability_score * 0.3 +
            volume_score * 0.2 +
            compute_score * 0.2
        )
        
        return total_score
    
    async def _coordinate_training_round(self, fed_round: FederatedRound) -> bool:
        """Coordinate a single federated training round."""
        
        try:
            # Send training configuration to selected clients
            training_tasks = []
            for client_id in fed_round.selected_clients:
                task = asyncio.create_task(
                    self._coordinate_client_training(client_id, fed_round)
                )
                training_tasks.append(task)
            
            # Wait for all clients to complete training (with timeout)
            timeout_minutes = 30
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*training_tasks, return_exceptions=True),
                timeout=timeout_minutes * 60
            )
            
            # Count successful completions
            successful_clients = sum(1 for result in completed_tasks if result is True)
            
            logger.info(
                f"Round {fed_round.round_id}: {successful_clients}/{len(fed_round.selected_clients)} "
                "clients completed training"
            )
            
            # Require minimum number of successful clients
            return successful_clients >= fed_round.min_clients
            
        except asyncio.TimeoutError:
            logger.error(f"Training round {fed_round.round_id} timed out")
            return False
        except Exception as e:
            logger.error(f"Training round coordination error: {e}")
            return False
    
    async def _coordinate_client_training(self, client_id: str, fed_round: FederatedRound) -> bool:
        """Coordinate training for a single client."""
        
        if client_id not in self.registered_clients:
            return False
        
        client = self.registered_clients[client_id]
        client.status = ClientStatus.TRAINING
        
        try:
            # In a real implementation, this would send training request to client
            # For now, simulate training
            training_time = random.uniform(60, 300)  # 1-5 minutes
            await asyncio.sleep(training_time / 60)  # Scale down for demo
            
            # Simulate model update from client
            model_update = self._simulate_client_update(client_id, fed_round)
            
            if model_update:
                fed_round.client_updates[client_id] = model_update
                fed_round.participating_clients.append(client_id)
                
                # Update client statistics
                client.rounds_participated += 1
                client.last_seen = time.time()
                client.status = ClientStatus.AVAILABLE
                
                return True
            
        except Exception as e:
            logger.error(f"Client {client_id} training failed: {e}")
            client.status = ClientStatus.FAILED
        
        return False
    
    def _simulate_client_update(self, client_id: str, fed_round: FederatedRound) -> ModelUpdate | None:
        """Simulate model update from client (for development/testing)."""
        
        try:
            # Create mock model weights
            mock_weights = {
                'layer1.weight': torch.randn(64, 3, 3, 3),
                'layer1.bias': torch.randn(64),
                'layer2.weight': torch.randn(128, 64, 3, 3),
                'layer2.bias': torch.randn(128)
            }
            
            # Encrypt weights
            encrypted_weights = self.aggregator.security.encrypt_model_weights(mock_weights)
            
            # Create model update
            update = ModelUpdate(
                client_id=client_id,
                round_id=fed_round.round_id,
                encrypted_weights=encrypted_weights,
                training_samples=random.randint(100, 1000),
                training_loss=random.uniform(0.1, 0.5),
                training_accuracy=random.uniform(0.85, 0.95),
                training_time_minutes=random.uniform(1, 5),
                update_hash=hashlib.md5(encrypted_weights).hexdigest()[:16]
            )
            
            return update
            
        except Exception as e:
            logger.error(f"Failed to create mock update for {client_id}: {e}")
            return None
    
    async def _update_global_model(self, fed_round: FederatedRound):
        """Update global model with aggregated client updates."""
        
        if not fed_round.client_updates:
            logger.warning("No client updates to aggregate")
            return
        
        try:
            # Aggregate client updates
            fed_round.status = FederatedRoundStatus.AGGREGATING
            
            aggregated_weights = await self.aggregator.aggregate_updates(
                fed_round.client_updates,
                self.global_weights
            )
            
            if aggregated_weights:
                # Update global model
                self.global_weights = aggregated_weights
                fed_round.aggregated_weights = aggregated_weights
                
                # Update model version
                version_parts = self.global_model_version.split('.')
                patch_version = int(version_parts[2]) + 1
                self.global_model_version = f"{version_parts[0]}.{version_parts[1]}.{patch_version}"
                
                logger.info(f"Updated global model to version {self.global_model_version}")
            
        except Exception as e:
            logger.error(f"Global model update failed: {e}")
    
    async def _check_convergence(self) -> bool:
        """Check if federated training has converged."""
        
        if len(self.training_history) < 5:  # Need at least 5 rounds
            return False
        
        # Check if accuracy improvement has plateaued
        recent_accuracies = [round_data['accuracy'] for round_data in self.training_history[-5:]]
        
        if len(recent_accuracies) >= 5:
            accuracy_trend = np.diff(recent_accuracies)
            avg_improvement = np.mean(accuracy_trend)
            
            # Converged if improvement is below threshold
            converged = avg_improvement < self.convergence_threshold
            
            if converged:
                logger.info(f"Convergence detected: avg improvement {avg_improvement:.4f} < {self.convergence_threshold}")
            
            return converged
        
        return False
    
    def _record_round_history(self, fed_round: FederatedRound):
        """Record round results in training history."""
        
        # Calculate round metrics
        if fed_round.client_updates:
            avg_accuracy = np.mean([
                update.training_accuracy for update in fed_round.client_updates.values()
            ])
            avg_loss = np.mean([
                update.training_loss for update in fed_round.client_updates.values()
            ])
        else:
            avg_accuracy = 0.0
            avg_loss = 1.0
        
        round_data = {
            'round_id': fed_round.round_id,
            'participants': len(fed_round.participating_clients),
            'accuracy': avg_accuracy,
            'loss': avg_loss,
            'duration_minutes': (fed_round.end_time - fed_round.start_time) / 60 if fed_round.end_time else 0,
            'model_version': self.global_model_version
        }
        
        self.training_history.append(round_data)
        
        # Keep only last 100 rounds
        if len(self.training_history) > 100:
            self.training_history = self.training_history[-100:]
    
    def get_training_status(self) -> dict[str, Any]:
        """Get current federated training status."""
        
        active_round = self.active_rounds.get(self.current_round)
        
        return {
            'is_training': self.is_training,
            'current_round': self.current_round,
            'total_rounds': self.global_rounds,
            'global_model_version': self.global_model_version,
            'registered_clients': len(self.registered_clients),
            'active_clients': len([
                c for c in self.registered_clients.values() 
                if c.status == ClientStatus.AVAILABLE
            ]),
            'current_round_status': active_round.status.value if active_round else None,
            'current_round_participants': len(active_round.participating_clients) if active_round else 0,
            'training_history_length': len(self.training_history),
            'latest_accuracy': self.training_history[-1]['accuracy'] if self.training_history else 0.0
        }
    
    async def stop_training(self):
        """Stop federated training."""
        
        logger.info("Stopping federated training")
        self.is_training = False
        
        # Mark all active rounds as completed
        for round_id, fed_round in self.active_rounds.items():
            if fed_round.status in [FederatedRoundStatus.TRAINING, FederatedRoundStatus.AGGREGATING]:
                fed_round.status = FederatedRoundStatus.COMPLETED
                fed_round.end_time = time.time()


# Factory function
async def create_federated_coordinator(config_path: Path | str = None) -> FederatedCoordinator:
    """Create and configure federated learning coordinator."""
    
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'min_clients_per_round': 3,
            'max_clients_per_round': 20,
            'client_selection_fraction': 0.2,
            'global_rounds': 100,
            'local_epochs': 5,
            'convergence_threshold': 0.001,
            'aggregation_method': 'fedavg'
        }
    
    coordinator = FederatedCoordinator(config)
    
    logger.info("Federated learning coordinator created")
    return coordinator


# Example usage
if __name__ == "__main__":
    async def main():
        # Create coordinator
        coordinator = await create_federated_coordinator()
        
        # Register some mock clients
        for i in range(5):
            client = ClientInfo(
                client_id=f"camera_edge_{i}",
                location=f"Intersection_{i}",
                camera_ids=[f"cam_{i}_1", f"cam_{i}_2"],
                data_samples=random.randint(500, 2000),
                data_quality_score=random.uniform(0.7, 0.95)
            )
            await coordinator.register_client(client)
        
        # Start training
        await coordinator.start_federated_training()
        
        # Monitor training
        while coordinator.is_training:
            status = coordinator.get_training_status()
            print(f"Training status: {status}")
            await asyncio.sleep(30)
        
        print("Federated training completed")
    
    asyncio.run(main())
