"""
Simulation-Based Inference for Model Rocket Aerodynamics
========================================================
FIXED VERSION: Corrected normalization bug + parallel data generation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Number of parallel workers for data generation
N_WORKERS = min(8, multiprocessing.cpu_count())

# =============================================================================
# ROCKET & MOTOR DEFINITIONS
# =============================================================================

@dataclass
class RocketConfig:
    name: str
    dry_mass: float  # kg
    diameter: float = 0.066  # m
    
    @property
    def area(self):
        return np.pi * (self.diameter / 2) ** 2

@dataclass
class MotorConfig:
    name: str
    total_impulse: float  # N·s
    avg_thrust: float     # N
    max_thrust: float     # N
    burn_time: float      # s
    propellant_mass: float  # kg

# Rocket configurations
CONFIG_A = RocketConfig("Config A (minimal payload)", dry_mass=0.322)
CONFIG_B = RocketConfig("Config B (full payload)", dry_mass=0.448)

CONFIGS = {"A": CONFIG_A, "B": CONFIG_B}

# Motor data from ThrustCurve.org
MOTORS = {
    "E35-5W": MotorConfig("E35-5W", 39.4, 33.8, 44.0, 1.1, 0.025),
    "F24-4W": MotorConfig("F24-4W", 47.3, 22.2, 41.0, 2.1, 0.019),
    "F39": MotorConfig("F39", 49.7, 37.3, 59.5, 1.3, 0.023),
}

MOTOR_NAMES = list(MOTORS.keys())
MOTOR_TO_IDX = {name: idx for idx, name in enumerate(MOTOR_NAMES)}

# Physical constants
G = 9.81
RHO_0 = 1.225
H_SCALE = 8500.0

# =============================================================================
# PHYSICS SIMULATOR
# =============================================================================

def thrust_curve(t: float, motor: MotorConfig, thrust_factor: float = 1.0) -> float:
    """Model thrust curve with ramp-up and decay."""
    if t < 0 or t >= motor.burn_time:
        return 0.0
    t_peak = 0.08 * motor.burn_time
    if t < t_peak:
        return motor.max_thrust * (t / t_peak) * thrust_factor
    else:
        progress = (t - t_peak) / (motor.burn_time - t_peak)
        return (motor.max_thrust - (motor.max_thrust - motor.avg_thrust) * progress) * thrust_factor


def rocket_ode(t: float, state: List[float], cd: float, motor: MotorConfig, 
               rocket: RocketConfig, thrust_factor: float) -> List[float]:
    """Equations of motion for vertical rocket flight."""
    h, v = state
    h = max(0, h)
    
    # Mass (decreases during burn)
    m_prop = motor.propellant_mass * max(0, 1 - t/motor.burn_time) if t < motor.burn_time else 0
    m = rocket.dry_mass + m_prop
    
    # Forces
    T = thrust_curve(t, motor, thrust_factor)
    rho = RHO_0 * np.exp(-h / H_SCALE)
    D = 0.5 * rho * v * abs(v) * cd * rocket.area
    
    # Acceleration
    a = (T - D - m * G) / m
    
    return [v, a]


def simulate_flight(cd: float, motor: MotorConfig, rocket: RocketConfig, 
                    thrust_factor: float = 1.0) -> float:
    """Simulate a flight and return apogee altitude."""
    sol = solve_ivp(
        rocket_ode, 
        [0, 60], 
        [0.0, 0.0],
        args=(cd, motor, rocket, thrust_factor),
        method='RK45',
        max_step=0.02
    )
    return np.max(sol.y[0])


# =============================================================================
# SYNTHETIC DATA GENERATION (PARALLELIZED)
# =============================================================================

def _simulate_single_flight(args: Tuple) -> Dict:
    """Worker function for parallel simulation."""
    idx, cd_range, thrust_factor_range, noise_std, seed = args
    
    # Set seed for this worker (ensures reproducibility + variety)
    np.random.seed(seed + idx)
    
    # Sample ground truth parameters
    true_cd = np.random.uniform(*cd_range)
    true_thrust_factor = np.random.uniform(*thrust_factor_range)
    
    # Random flight configuration
    motor_name = np.random.choice(MOTOR_NAMES)
    config_name = np.random.choice(["A", "B"])
    
    motor = MOTORS[motor_name]
    rocket = CONFIGS[config_name]
    
    # Simulate flight
    apogee = simulate_flight(true_cd, motor, rocket, true_thrust_factor)
    
    # Add measurement noise
    apogee_noisy = apogee + np.random.normal(0, noise_std)
    apogee_noisy = max(0, apogee_noisy)
    
    return {
        'feature': [
            apogee_noisy,
            MOTOR_TO_IDX[motor_name],
            rocket.dry_mass,
            motor.total_impulse,
            motor.burn_time,
        ],
        'label': [true_cd, true_thrust_factor],
        'metadata': {
            'motor': motor_name,
            'config': config_name,
            'true_apogee': apogee,
            'noisy_apogee': apogee_noisy,
            'true_cd': true_cd,
            'true_thrust_factor': true_thrust_factor,
        }
    }


def generate_synthetic_dataset(n_samples: int, 
                                cd_range: Tuple[float, float] = (0.3, 0.9),
                                thrust_factor_range: Tuple[float, float] = (0.8, 1.2),
                                noise_std: float = 3.0,
                                parallel: bool = True) -> Dict:
    """
    Generate synthetic flight data with known ground truth parameters.
    Uses parallel processing for speed.
    """
    print(f"Generating {n_samples} synthetic flights...")
    
    if parallel and N_WORKERS > 1:
        # Parallel generation
        args_list = [(i, cd_range, thrust_factor_range, noise_std, SEED) 
                     for i in range(n_samples)]
        
        results = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = [executor.submit(_simulate_single_flight, args) for args in args_list]
            for future in tqdm(as_completed(futures), total=n_samples):
                results.append(future.result())
    else:
        # Sequential generation (fallback)
        results = []
        for i in tqdm(range(n_samples)):
            result = _simulate_single_flight((i, cd_range, thrust_factor_range, noise_std, SEED))
            results.append(result)
    
    # Unpack results
    features = np.array([r['feature'] for r in results], dtype=np.float32)
    labels = np.array([r['label'] for r in results], dtype=np.float32)
    metadata = [r['metadata'] for r in results]
    
    return {
        'features': features,
        'labels': labels,
        'metadata': metadata
    }


# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================

class InferenceNetwork(nn.Module):
    """
    Neural network that predicts aerodynamic parameters from flight observations.
    
    Input: [apogee, motor_idx, mass, impulse, burn_time]
    Output: [cd, thrust_factor]
    """
    
    def __init__(self, input_dim: int = 5, hidden_dims: List[int] = [128, 256, 128], 
                 output_dim: int = 2, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Output activation to constrain predictions to valid ranges
        self.cd_range = (0.2, 1.5)
        self.thrust_range = (0.5, 1.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_output = self.network(x)
        
        # Apply sigmoid and scale to valid ranges
        cd = torch.sigmoid(raw_output[:, 0]) * (self.cd_range[1] - self.cd_range[0]) + self.cd_range[0]
        thrust = torch.sigmoid(raw_output[:, 1]) * (self.thrust_range[1] - self.thrust_range[0]) + self.thrust_range[0]
        
        return torch.stack([cd, thrust], dim=1)


class EnsembleInferenceNetwork(nn.Module):
    """
    Ensemble of networks for uncertainty estimation.
    """
    
    def __init__(self, n_models: int = 5, **kwargs):
        super().__init__()
        self.models = nn.ModuleList([InferenceNetwork(**kwargs) for _ in range(n_models)])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and std of predictions across ensemble."""
        predictions = torch.stack([model(x) for model in self.models], dim=0)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        return mean, std
    
    def predict_single(self, x: torch.Tensor, model_idx: int) -> torch.Tensor:
        """Predict with a single model (for training)."""
        return self.models[model_idx](x)


# =============================================================================
# TRAINING
# =============================================================================

def normalize_features(features: np.ndarray, stats: Dict = None) -> Tuple[np.ndarray, Dict]:
    """Normalize features to zero mean, unit variance."""
    if stats is None:
        stats = {
            'mean': features.mean(axis=0),
            'std': features.std(axis=0) + 1e-8
        }
    normalized = (features - stats['mean']) / stats['std']
    return normalized, stats


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                n_epochs: int = 100, lr: float = 1e-3, 
                device: torch.device = DEVICE) -> Dict:
    """
    Train the inference network.
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_cd_mae': [], 'val_thrust_mae': []}
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_cd_errors = []
        val_thrust_errors = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                predictions = model(features)
                loss = criterion(predictions, labels)
                val_losses.append(loss.item())
                
                val_cd_errors.append(torch.abs(predictions[:, 0] - labels[:, 0]).mean().item())
                val_thrust_errors.append(torch.abs(predictions[:, 1] - labels[:, 1]).mean().item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_cd_mae = np.mean(val_cd_errors)
        val_thrust_mae = np.mean(val_thrust_errors)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cd_mae'].append(val_cd_mae)
        history['val_thrust_mae'].append(val_thrust_mae)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Cd MAE: {val_cd_mae:.4f} | α MAE: {val_thrust_mae:.4f}")
    
    model.load_state_dict(best_state)
    return history


def train_ensemble(ensemble: EnsembleInferenceNetwork, 
                   train_features: np.ndarray, train_labels: np.ndarray,
                   val_features: np.ndarray, val_labels: np.ndarray,
                   n_epochs: int = 100, lr: float = 1e-3, batch_size: int = 256,
                   device: torch.device = DEVICE) -> List[Dict]:
    """Train each model in the ensemble on bootstrapped data."""
    
    histories = []
    
    for i, model in enumerate(ensemble.models):
        print(f"\n--- Training Ensemble Model {i+1}/{len(ensemble.models)} ---")
        
        # Bootstrap sample
        n_samples = len(train_features)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        boot_features = train_features[indices]
        boot_labels = train_labels[indices]
        
        train_dataset = TensorDataset(
            torch.FloatTensor(boot_features),
            torch.FloatTensor(boot_labels)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_features),
            torch.FloatTensor(val_labels)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        history = train_model(model, train_loader, val_loader, n_epochs, lr, device)
        histories.append(history)
    
    return histories


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_on_synthetic(model: nn.Module, 
                          features: np.ndarray,  # Already normalized!
                          labels: np.ndarray,
                          device: torch.device = DEVICE) -> Dict:
    """Evaluate model on synthetic test data with known ground truth."""
    model.eval()
    
    features_tensor = torch.FloatTensor(features).to(device)
    
    with torch.no_grad():
        if isinstance(model, EnsembleInferenceNetwork):
            pred_mean, pred_std = model(features_tensor)
            predictions = pred_mean.cpu().numpy()
            uncertainties = pred_std.cpu().numpy()
        else:
            predictions = model(features_tensor).cpu().numpy()
            uncertainties = None
    
    cd_mae = np.mean(np.abs(predictions[:, 0] - labels[:, 0]))
    cd_rmse = np.sqrt(np.mean((predictions[:, 0] - labels[:, 0])**2))
    thrust_mae = np.mean(np.abs(predictions[:, 1] - labels[:, 1]))
    thrust_rmse = np.sqrt(np.mean((predictions[:, 1] - labels[:, 1])**2))
    
    results = {
        'predictions': predictions,
        'uncertainties': uncertainties,
        'labels': labels,
        'cd_mae': cd_mae,
        'cd_rmse': cd_rmse,
        'thrust_mae': thrust_mae,
        'thrust_rmse': thrust_rmse,
    }
    
    print(f"\nSynthetic Test Results:")
    print(f"  Cd     - MAE: {cd_mae:.4f}, RMSE: {cd_rmse:.4f}")
    print(f"  Thrust - MAE: {thrust_mae:.4f}, RMSE: {thrust_rmse:.4f}")
    
    return results


def evaluate_on_real_flights(model: nn.Module, norm_stats: Dict,
                              device: torch.device = DEVICE) -> Dict:
    """Evaluate model on real flight data."""
    
    REAL_FLIGHTS = [
        ("E35-5W", 169.8, "B", "Good launch", True),
        ("F24-4W", 281.3, "A", "Excellent flight", True),
        ("F24-4W", 246.6, "A", "Good flight", True),
        ("E35-5W", 174.7, "A", "Ended in tree", True),
        ("E35-5W", 154.5, "A", "Weak launch", False),
        ("E35-5W", 131.1, "A", "Very weak", False),
        ("E35-5W", 166.1, "A", "Extra drag", False),
        ("F24-4W", 199.9, "A", "Off-axis flight", False),
        ("F24-4W", 241.4, "A", "Good flight", True),
        ("F39", 185.9, "B", "Qual flight 1", True),
        ("F39", 196.3, "B", "Qual flight 2", True),
        ("F39", 198.1, "B", "Qual flight 3", True),
    ]
    
    model.eval()
    results = []
    
    print("\nReal Flight Evaluation:")
    print("-" * 85)
    print(f"{'#':<3} {'Motor':<8} {'Cfg':<4} {'Measured':>10} {'Predicted':>10} {'Error':>8} "
          f"{'Cd':>7} {'±':>5} {'α':>7} {'±':>5}")
    print("-" * 85)
    
    for i, (motor_name, measured_apogee, config_name, notes, include) in enumerate(REAL_FLIGHTS):
        motor = MOTORS[motor_name]
        rocket = CONFIGS[config_name]
        
        # Create feature vector (RAW, then normalize)
        feature = np.array([[
            measured_apogee,
            MOTOR_TO_IDX[motor_name],
            rocket.dry_mass,
            motor.total_impulse,
            motor.burn_time,
        ]], dtype=np.float32)
        
        feature_norm = (feature - norm_stats['mean']) / norm_stats['std']
        feature_tensor = torch.FloatTensor(feature_norm).to(device)
        
        with torch.no_grad():
            if isinstance(model, EnsembleInferenceNetwork):
                pred_mean, pred_std = model(feature_tensor)
                pred_cd = pred_mean[0, 0].item()
                pred_thrust = pred_mean[0, 1].item()
                cd_std = pred_std[0, 0].item()
                thrust_std = pred_std[0, 1].item()
            else:
                pred = model(feature_tensor)
                pred_cd = pred[0, 0].item()
                pred_thrust = pred[0, 1].item()
                cd_std = thrust_std = 0
        
        # Simulate with predicted parameters
        pred_apogee = simulate_flight(pred_cd, motor, rocket, pred_thrust)
        error = pred_apogee - measured_apogee
        
        flag = "" if include else " [excl]"
        print(f"{i+1:<3} {motor_name:<8} {config_name:<4} {measured_apogee:>10.1f} {pred_apogee:>10.1f} "
              f"{error:>+8.1f} {pred_cd:>7.3f} {cd_std:>5.3f} {pred_thrust:>7.3f} {thrust_std:>5.3f}{flag}")
        
        results.append({
            'flight_num': i + 1,
            'motor': motor_name,
            'config': config_name,
            'measured': measured_apogee,
            'predicted': pred_apogee,
            'error': error,
            'pred_cd': pred_cd,
            'pred_thrust': pred_thrust,
            'cd_std': cd_std,
            'thrust_std': thrust_std,
            'include': include,
            'notes': notes,
        })
    
    included = [r for r in results if r['include']]
    errors = [abs(r['error']) for r in included]
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    
    print("-" * 85)
    print(f"Included flights ({len(included)}): MAE = {mae:.1f}m, RMSE = {rmse:.1f}m")
    
    return {'flights': results, 'mae': mae, 'rmse': rmse}


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_history(history: Dict, save_path: str = None):
    """Plot training curves."""
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    ax.plot(history['train_loss'], label='Train', linewidth=2)
    ax.plot(history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Progress', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax = axes[1]
    ax.plot(history['val_cd_mae'], label='Cd MAE', linewidth=2)
    ax.plot(history['val_thrust_mae'], label='Thrust Factor MAE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Parameter Estimation Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_synthetic_results(results: Dict, save_path: str = None):
    """Plot predicted vs true parameters on synthetic data."""
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Cd parity plot
    ax = axes[0]
    ax.scatter(results['labels'][:, 0], results['predictions'][:, 0], 
               alpha=0.3, s=15, c='#2E86AB', edgecolors='none')
    lims = [0.25, 0.95]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='y = x')
    ax.set_xlabel('True $C_d$')
    ax.set_ylabel('Predicted $C_d$')
    ax.set_title(f'Drag Coefficient (MAE = {results["cd_mae"]:.3f})', fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thrust factor parity plot
    ax = axes[1]
    ax.scatter(results['labels'][:, 1], results['predictions'][:, 1], 
               alpha=0.3, s=15, c='#E94F37', edgecolors='none')
    lims = [0.75, 1.25]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='y = x')
    ax.set_xlabel(r'True Thrust Factor $\alpha$')
    ax.set_ylabel(r'Predicted Thrust Factor $\alpha$')
    ax.set_title(f'Thrust Factor (MAE = {results["thrust_mae"]:.3f})', fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_real_flight_results(results: Dict, save_path: str = None):
    """Plot results on real flights."""
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    
    flights = results['flights']
    included = [f for f in flights if f['include']]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Parity plot
    ax = axes[0]
    measured = [f['measured'] for f in included]
    predicted = [f['predicted'] for f in included]
    configs = [f['config'] for f in included]
    
    for m, p, c in zip(measured, predicted, configs):
        color = '#2E86AB' if c == 'B' else '#F6AE2D'
        marker = 'o' if c == 'B' else 's'
        ax.scatter(m, p, c=color, marker=marker, s=120, edgecolors='white', linewidths=2, zorder=3)
    
    lims = [130, 300]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.7)
    ax.fill_between(lims, [l-20 for l in lims], [l+20 for l in lims], alpha=0.1, color='gray')
    
    ax.set_xlabel('Measured Apogee (m)')
    ax.set_ylabel('Predicted Apogee (m)')
    ax.set_title(f'Real Flight Predictions (MAE = {results["mae"]:.1f}m)', fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', 
               markersize=10, markeredgecolor='white', markeredgewidth=1.5, label='Config B'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#F6AE2D', 
               markersize=10, markeredgecolor='white', markeredgewidth=1.5, label='Config A'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Inferred parameters bar chart
    ax = axes[1]
    flight_nums = [f['flight_num'] for f in included]
    pred_cds = [f['pred_cd'] for f in included]
    pred_thrusts = [f['pred_thrust'] for f in included]
    cd_stds = [f['cd_std'] for f in included]
    thrust_stds = [f['thrust_std'] for f in included]
    
    x = np.arange(len(included))
    width = 0.35
    
    ax.bar(x - width/2, pred_cds, width, yerr=cd_stds, capsize=3,
           label='Predicted $C_d$', color='#2E86AB', alpha=0.85, edgecolor='white')
    ax.bar(x + width/2, pred_thrusts, width, yerr=thrust_stds, capsize=3,
           label=r'Predicted $\alpha$', color='#E94F37', alpha=0.85, edgecolor='white')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Flight')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Inferred Parameters (with Uncertainty)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'#{n}' for n in flight_nums])
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("SIMULATION-BASED INFERENCE FOR MODEL ROCKET AERODYNAMICS")
    print("=" * 70)
    
    os.makedirs("outputs", exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # -------------------------------------------------------------------------
    print("\n[1/5] Generating synthetic datasets...")
    
    train_data_raw = generate_synthetic_dataset(n_samples=10000, noise_std=3.0, parallel=True)
    val_data_raw = generate_synthetic_dataset(n_samples=2000, noise_std=3.0, parallel=True)
    test_data_raw = generate_synthetic_dataset(n_samples=2000, noise_std=3.0, parallel=True)
    
    # Normalize features (compute stats from training data only)
    train_features_norm, norm_stats = normalize_features(train_data_raw['features'])
    val_features_norm, _ = normalize_features(val_data_raw['features'], norm_stats)
    test_features_norm, _ = normalize_features(test_data_raw['features'], norm_stats)
    
    print(f"  Train: {len(train_features_norm)} samples")
    print(f"  Val:   {len(val_features_norm)} samples")
    print(f"  Test:  {len(test_features_norm)} samples")
    print(f"  Norm stats - mean: {norm_stats['mean']}")
    print(f"  Norm stats - std:  {norm_stats['std']}")
    
    # -------------------------------------------------------------------------
    # Step 2: Train ensemble model
    # -------------------------------------------------------------------------
    print("\n[2/5] Training ensemble model...")
    
    ensemble = EnsembleInferenceNetwork(n_models=5)
    histories = train_ensemble(
        ensemble, 
        train_features_norm, train_data_raw['labels'],
        val_features_norm, val_data_raw['labels'],
        n_epochs=100, lr=1e-3, batch_size=256, device=DEVICE
    )
    
    plot_training_history(histories[0], save_path="outputs/training_history.png")
    
    # -------------------------------------------------------------------------
    # Step 3: Evaluate on synthetic test data
    # -------------------------------------------------------------------------
    print("\n[3/5] Evaluating on synthetic test data...")
    
    # Pass already-normalized features (NO double normalization!)
    synthetic_results = evaluate_on_synthetic(
        ensemble, 
        test_features_norm,  # Already normalized
        test_data_raw['labels'],
        DEVICE
    )
    plot_synthetic_results(synthetic_results, save_path="outputs/synthetic_results.png")
    
    # -------------------------------------------------------------------------
    # Step 4: Evaluate on real flights
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating on real flights...")
    
    real_results = evaluate_on_real_flights(ensemble, norm_stats, DEVICE)
    plot_real_flight_results(real_results, save_path="outputs/real_flight_results.png")
    
    # -------------------------------------------------------------------------
    # Step 5: Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
Synthetic Data Performance:
  Cd MAE:      {synthetic_results['cd_mae']:.4f}
  Cd RMSE:     {synthetic_results['cd_rmse']:.4f}
  Thrust MAE:  {synthetic_results['thrust_mae']:.4f}
  Thrust RMSE: {synthetic_results['thrust_rmse']:.4f}

Real Flight Performance:
  Apogee MAE:  {real_results['mae']:.1f}m
  Apogee RMSE: {real_results['rmse']:.1f}m

Outputs saved to: outputs/
""")
    
    # Save model and normalization stats
    torch.save({
        'model_state': ensemble.state_dict(),
        'norm_stats': norm_stats,
    }, "outputs/model_checkpoint.pt")
    print("Model saved to: outputs/model_checkpoint.pt")


if __name__ == "__main__":
    main()