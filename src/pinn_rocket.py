"""
Physics-Informed Neural Network for Model Rocket Aerodynamic Parameter Estimation
==================================================================================

This module implements a PINN that learns the drag coefficient (Cd) of a model rocket
from sparse apogee measurements by embedding the 1D equations of motion as physics
constraints.

Paper: "Physics-Informed Neural Networks for Aerodynamic Parameter Estimation in Model Rocketry"
Authors: Rohan + Brother (Team Astra)

Key Innovation: Learning aerodynamic parameters from <15 real flights by combining
physics-informed learning with simulation pre-training.

Usage:
    python src/pinn_rocket.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import os


# =============================================================================
# ROCKET AND FLIGHT DATA
# =============================================================================

@dataclass
class RocketConfig:
    """Configuration for the Astra TARC rocket."""
    name: str = "Astra"
    dry_mass: float = 0.405          # kg (without motor)
    diameter: float = 0.066          # m (6.6 cm)
    length: float = 1.16             # m (116 cm)
    
    @property
    def radius(self) -> float:
        return self.diameter / 2
    
    @property
    def cross_section_area(self) -> float:
        return np.pi * self.radius ** 2


@dataclass 
class MotorConfig:
    """Motor specifications from Aerotech."""
    name: str
    total_impulse: float      # N·s
    avg_thrust: float         # N
    burn_time: float          # s
    propellant_mass: float    # kg
    
    def thrust_curve(self, t: torch.Tensor) -> torch.Tensor:
        """Simple trapezoidal thrust curve approximation."""
        # Ramp up for 10% of burn, plateau, ramp down for 10%
        t_ramp = 0.1 * self.burn_time
        
        thrust = torch.zeros_like(t)
        
        # Ramp up phase
        mask_ramp_up = t < t_ramp
        thrust = torch.where(mask_ramp_up, 
                            self.avg_thrust * 1.2 * (t / t_ramp), 
                            thrust)
        
        # Plateau phase  
        mask_plateau = (t >= t_ramp) & (t < self.burn_time - t_ramp)
        thrust = torch.where(mask_plateau, 
                            torch.full_like(t, self.avg_thrust * 1.1), 
                            thrust)
        
        # Ramp down phase
        mask_ramp_down = (t >= self.burn_time - t_ramp) & (t < self.burn_time)
        thrust = torch.where(mask_ramp_down,
                            self.avg_thrust * 1.2 * ((self.burn_time - t) / t_ramp),
                            thrust)
        
        return thrust


# Motor database from TARC presentation
MOTORS = {
    "E35-5W": MotorConfig(
        name="E35-5W",
        total_impulse=35.0,       # ~E class
        avg_thrust=35.0,
        burn_time=1.0,
        propellant_mass=0.025
    ),
    "F24-4W": MotorConfig(
        name="F24-4W",
        total_impulse=48.0,       # ~F class
        avg_thrust=24.0,
        burn_time=2.0,
        propellant_mass=0.035
    ),
    "F39": MotorConfig(
        name="F39",
        total_impulse=79.2,       # From slide 7
        avg_thrust=39.0,
        burn_time=1.3,            # Estimated from thrust curve
        propellant_mass=0.040
    )
}


# Real flight data from TARC presentation (converted to meters)
FLIGHT_DATA = [
    # (motor_name, apogee_m, predicted_m, notes)
    ("E35-5W", 169.8, 167.6, "Good first launch"),
    ("F24-4W", 281.3, 289.6, "Went higher than expected"),
    ("F24-4W", 246.6, 289.6, "Perfect launch"),
    ("E35-5W", 174.7, 167.6, "Rocket got stuck in tree"),
    ("E35-5W", 154.5, 167.6, "Good flight, weak launch"),
    ("E35-5W", 131.1, 167.6, "Took off very weak"),
    ("E35-5W", 166.1, 167.6, "Dragged down by extra weight"),
    ("F24-4W", 199.9, 289.6, "Did not fly straight"),
    ("F24-4W", 241.4, 289.6, "Flew straight"),
    ("F39", 185.9, 198.1, "Qual flight 1 - parachute issue"),
    ("F39", 196.3, 198.1, "Qual flight 2 - good"),
    ("F39", 198.1, 198.1, "Qual flight 3 - parachute issue"),
]


# =============================================================================
# PHYSICS ENGINE
# =============================================================================

class RocketPhysics:
    """
    1D rocket physics model for vertical flight.
    
    Equations of motion:
        dh/dt = v
        m(t) * dv/dt = T(t) - D(v, h) - m(t) * g
        
    where:
        D(v, h) = 0.5 * rho(h) * v^2 * Cd * A
        rho(h) = rho_0 * exp(-h / H)
        m(t) = m_dry + m_prop * (1 - t/t_burn)  for t < t_burn
    """
    
    def __init__(self, rocket: RocketConfig):
        self.rocket = rocket
        self.g = 9.81                    # m/s^2
        self.rho_0 = 1.225               # kg/m^3 (sea level density)
        self.H = 8500.0                  # m (scale height)
        
    def air_density(self, h: torch.Tensor) -> torch.Tensor:
        """Atmospheric density as function of altitude."""
        return self.rho_0 * torch.exp(-h / self.H)
    
    def mass(self, t: torch.Tensor, motor: MotorConfig) -> torch.Tensor:
        """Rocket mass as function of time (decreases during burn)."""
        m_total = self.rocket.dry_mass + motor.propellant_mass
        
        # Linear mass decrease during burn
        mass_fraction = torch.clamp(1 - t / motor.burn_time, min=0.0, max=1.0)
        m_prop_remaining = motor.propellant_mass * mass_fraction
        
        return self.rocket.dry_mass + m_prop_remaining
    
    def drag_force(self, v: torch.Tensor, h: torch.Tensor, Cd: torch.Tensor) -> torch.Tensor:
        """Aerodynamic drag force."""
        rho = self.air_density(h)
        A = self.rocket.cross_section_area
        # Drag always opposes motion
        return 0.5 * rho * v * torch.abs(v) * Cd * A
    
    def acceleration(self, t: torch.Tensor, h: torch.Tensor, v: torch.Tensor,
                     Cd: torch.Tensor, motor: MotorConfig) -> torch.Tensor:
        """
        Compute acceleration from equations of motion.
        
        dv/dt = (T - D - m*g) / m
        """
        m = self.mass(t, motor)
        T = motor.thrust_curve(t)
        D = self.drag_force(v, h, Cd)
        
        # Forces: thrust up, drag opposes motion, gravity down
        # For upward flight (v > 0): T - D - mg
        # For downward flight (v < 0): T + D - mg (drag now helps slow descent)
        F_net = T - D - m * self.g
        
        return F_net / m
    
    def simulate_flight(self, Cd: float, motor: MotorConfig, 
                        dt: float = 0.01, max_time: float = 60.0) -> Dict:
        """
        Simulate a complete flight using Euler integration.
        Returns trajectory data including apogee.
        """
        t_values = [0.0]
        h_values = [0.0]
        v_values = [0.0]
        
        t, h, v = 0.0, 0.0, 0.0
        Cd_tensor = torch.tensor([Cd])
        
        apogee = 0.0
        apogee_time = 0.0
        
        while t < max_time:
            # Convert to tensors for physics computation
            t_t = torch.tensor([t])
            h_t = torch.tensor([h])
            v_t = torch.tensor([v])
            
            # Get acceleration
            a = self.acceleration(t_t, h_t, v_t, Cd_tensor, motor).item()
            
            # Euler step
            v_new = v + a * dt
            h_new = h + v * dt
            t_new = t + dt
            
            # Update
            t, h, v = t_new, max(0, h_new), v_new
            
            t_values.append(t)
            h_values.append(h)
            v_values.append(v)
            
            # Track apogee
            if h > apogee:
                apogee = h
                apogee_time = t
            
            # Stop if landed
            if h <= 0 and t > motor.burn_time:
                break
                
        return {
            "t": np.array(t_values),
            "h": np.array(h_values),
            "v": np.array(v_values),
            "apogee": apogee,
            "apogee_time": apogee_time
        }


# =============================================================================
# PINN MODEL
# =============================================================================

class TrajectoryPINN(nn.Module):
    """
    Physics-Informed Neural Network for rocket trajectory prediction.
    
    Input: time t (normalized)
    Output: altitude h(t), velocity v(t)
    Learnable: Cd (drag coefficient)
    
    The network learns to predict trajectories that satisfy:
    1. The equations of motion (physics loss)
    2. Initial conditions (boundary loss)  
    3. Match real apogee data (data loss)
    """
    
    def __init__(self, hidden_dim: int = 64, num_layers: int = 4,
                 Cd_init: float = 0.5):
        super().__init__()
        
        # Learnable drag coefficient
        self.log_Cd = nn.Parameter(torch.tensor(np.log(Cd_init)))
        
        # Neural network for trajectory
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 2))  # Output: [h, v]
        
        self.net = nn.Sequential(*layers)
        
        # Normalization parameters (set during training)
        self.t_max = 30.0  # max time for normalization
        self.h_max = 300.0  # max altitude for normalization
        self.v_max = 100.0  # max velocity for normalization
        
    @property
    def Cd(self) -> torch.Tensor:
        """Drag coefficient (constrained to be positive via exp)."""
        return torch.exp(self.log_Cd)
    
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict altitude and velocity at time t.
        
        Args:
            t: Time tensor, shape (batch,) or (batch, 1)
            
        Returns:
            h: Altitude tensor
            v: Velocity tensor
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        # Normalize time
        t_norm = t / self.t_max
        
        # Forward pass
        out = self.net(t_norm)
        
        # Denormalize outputs
        h = out[:, 0:1] * self.h_max
        v = out[:, 1:2] * self.v_max
        
        return h.squeeze(-1), v.squeeze(-1)
    
    def predict_apogee(self, motor: MotorConfig, physics: RocketPhysics,
                       dt: float = 0.1, max_time: float = 30.0) -> torch.Tensor:
        """
        Predict apogee by simulating with learned Cd.
        Uses differentiable simulation for gradient flow.
        """
        t = torch.tensor([0.0], requires_grad=False)
        h = torch.tensor([0.0], requires_grad=True)
        v = torch.tensor([0.0], requires_grad=True)
        
        apogee = h.clone()
        
        for _ in range(int(max_time / dt)):
            # Get acceleration with learned Cd
            a = physics.acceleration(t, h, v, self.Cd, motor)
            
            # Euler step (differentiable)
            v_new = v + a * dt
            h_new = h + v * dt
            t = t + dt
            
            # Update
            v = v_new
            h = torch.clamp(h_new, min=0.0)
            
            # Track maximum altitude
            apogee = torch.maximum(apogee, h)
            
            # Stop condition (non-differentiable, just for efficiency)
            if h.item() <= 0 and t.item() > motor.burn_time + 1.0:
                break
                
        return apogee


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss function for rocket trajectory.
    
    L = λ_physics * L_physics + λ_boundary * L_boundary + λ_data * L_data
    
    where:
        L_physics: Residual of equations of motion at collocation points
        L_boundary: Initial condition constraints
        L_data: Match predicted apogee to real apogee
    """
    
    def __init__(self, physics: RocketPhysics,
                 lambda_physics: float = 1.0,
                 lambda_boundary: float = 10.0,
                 lambda_data: float = 1.0):
        super().__init__()
        self.physics = physics
        self.lambda_physics = lambda_physics
        self.lambda_boundary = lambda_boundary
        self.lambda_data = lambda_data
        
    def physics_residual(self, model: TrajectoryPINN, 
                         t_colloc: torch.Tensor,
                         motor: MotorConfig) -> torch.Tensor:
        """
        Compute residual of equations of motion.
        
        R1 = dh/dt - v  (should be 0)
        R2 = m*dv/dt - (T - D - mg)  (should be 0)
        """
        t_colloc = t_colloc.requires_grad_(True)
        
        h, v = model(t_colloc)
        
        # Compute derivatives via autograd
        dh_dt = torch.autograd.grad(h.sum(), t_colloc, create_graph=True)[0]
        dv_dt = torch.autograd.grad(v.sum(), t_colloc, create_graph=True)[0]
        
        # Physics computations
        m = self.physics.mass(t_colloc, motor)
        T = motor.thrust_curve(t_colloc)
        D = self.physics.drag_force(v, h, model.Cd)
        
        # Residuals
        R1 = dh_dt - v
        R2 = m * dv_dt - (T - D - m * self.physics.g)
        
        return torch.mean(R1**2) + torch.mean(R2**2)
    
    def boundary_loss(self, model: TrajectoryPINN) -> torch.Tensor:
        """Initial conditions: h(0) = 0, v(0) = 0."""
        t0 = torch.tensor([0.0])
        h0, v0 = model(t0)
        return h0**2 + v0**2
    
    def data_loss(self, predicted_apogee: torch.Tensor, 
                  real_apogee: torch.Tensor) -> torch.Tensor:
        """Match predicted apogee to real measurement."""
        return (predicted_apogee - real_apogee)**2
    
    def forward(self, model: TrajectoryPINN,
                t_colloc: torch.Tensor,
                motor: MotorConfig,
                real_apogee: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Returns dict with individual loss components for logging.
        """
        # Physics loss
        L_physics = self.physics_residual(model, t_colloc, motor)
        
        # Boundary loss
        L_boundary = self.boundary_loss(model)
        
        # Data loss (if real apogee provided)
        if real_apogee is not None:
            predicted_apogee = model.predict_apogee(motor, self.physics)
            L_data = self.data_loss(predicted_apogee, real_apogee)
        else:
            L_data = torch.tensor(0.0)
            
        # Total loss
        L_total = (self.lambda_physics * L_physics + 
                   self.lambda_boundary * L_boundary +
                   self.lambda_data * L_data)
        
        return {
            "total": L_total,
            "physics": L_physics,
            "boundary": L_boundary,
            "data": L_data,
            "Cd": model.Cd.detach()
        }


# =============================================================================
# TRAINING
# =============================================================================

def train_pinn(model: TrajectoryPINN,
               physics: RocketPhysics,
               flight_data: list,
               n_epochs: int = 5000,
               lr: float = 1e-3,
               n_colloc: int = 100,
               verbose: bool = True) -> Dict:
    """
    Train the PINN on flight data.
    
    Args:
        model: TrajectoryPINN instance
        physics: RocketPhysics instance
        flight_data: List of (motor_name, real_apogee_m, predicted_m, notes)
        n_epochs: Number of training epochs
        lr: Learning rate
        n_colloc: Number of collocation points for physics loss
        verbose: Print progress
        
    Returns:
        Dictionary with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    loss_fn = PhysicsLoss(physics)
    
    history = {
        "loss": [],
        "physics_loss": [],
        "data_loss": [],
        "Cd": []
    }
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_physics = 0.0
        epoch_data = 0.0
        
        # Train on each flight
        for motor_name, real_apogee, _, _ in flight_data:
            motor = MOTORS[motor_name]
            
            # Sample collocation points
            t_colloc = torch.rand(n_colloc) * (motor.burn_time + 10.0)
            
            # Compute loss
            optimizer.zero_grad()
            losses = loss_fn(model, t_colloc, motor, 
                           torch.tensor([real_apogee]))
            
            losses["total"].backward()
            optimizer.step()
            
            epoch_loss += losses["total"].item()
            epoch_physics += losses["physics"].item()
            epoch_data += losses["data"].item()
        
        scheduler.step()
        
        # Log
        n_flights = len(flight_data)
        history["loss"].append(epoch_loss / n_flights)
        history["physics_loss"].append(epoch_physics / n_flights)
        history["data_loss"].append(epoch_data / n_flights)
        history["Cd"].append(model.Cd.item())
        
        if verbose and (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {epoch_loss/n_flights:.4f} | "
                  f"Cd: {model.Cd.item():.4f}")
    
    return history


# =============================================================================
# SIMPLER APPROACH: DIRECT CD OPTIMIZATION
# =============================================================================

def optimize_Cd_direct(physics: RocketPhysics,
                       flight_data: list,
                       Cd_init: float = 0.5,
                       n_iters: int = 1000,
                       lr: float = 0.01) -> Dict:
    """
    Directly optimize Cd to match apogee data (simpler baseline).
    
    This skips the neural network and just learns Cd by minimizing
    the MSE between simulated and real apogees.
    """
    log_Cd = torch.tensor([np.log(Cd_init)], requires_grad=True)
    optimizer = torch.optim.Adam([log_Cd], lr=lr)
    
    history = {"loss": [], "Cd": []}
    
    for i in range(n_iters):
        optimizer.zero_grad()
        
        Cd = torch.exp(log_Cd)
        total_loss = 0.0
        
        for motor_name, real_apogee, _, _ in flight_data:
            motor = MOTORS[motor_name]
            
            # Simulate with current Cd
            result = physics.simulate_flight(Cd.item(), motor)
            pred_apogee = torch.tensor([result["apogee"]])
            real = torch.tensor([real_apogee])
            
            total_loss += (pred_apogee - real)**2
        
        # This won't backprop through simulate_flight (non-differentiable)
        # So we use finite differences instead
        loss_val = total_loss.item()
        
        # Finite difference gradient
        eps = 1e-4
        Cd_plus = torch.exp(log_Cd + eps)
        loss_plus = 0.0
        for motor_name, real_apogee, _, _ in flight_data:
            motor = MOTORS[motor_name]
            result = physics.simulate_flight(Cd_plus.item(), motor)
            loss_plus += (result["apogee"] - real_apogee)**2
            
        grad = (loss_plus - loss_val) / eps
        log_Cd.grad = torch.tensor([grad])
        optimizer.step()
        
        history["loss"].append(loss_val / len(flight_data))
        history["Cd"].append(torch.exp(log_Cd).item())
        
        if (i + 1) % 100 == 0:
            print(f"Iter {i+1}/{n_iters} | Loss: {loss_val/len(flight_data):.2f} | "
                  f"Cd: {torch.exp(log_Cd).item():.4f}")
    
    return history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PINN for Model Rocket Aerodynamic Parameter Estimation")
    print("Rocket: Astra (TARC 2025)")
    print("=" * 60)
    
    # Setup
    rocket = RocketConfig()
    physics = RocketPhysics(rocket)
    
    print(f"\nRocket specs:")
    print(f"  Dry mass: {rocket.dry_mass*1000:.0f} g")
    print(f"  Diameter: {rocket.diameter*100:.1f} cm")
    print(f"  Cross-section: {rocket.cross_section_area*10000:.2f} cm²")
    
    print(f"\nFlight data: {len(FLIGHT_DATA)} flights")
    for motor_name, apogee, pred, notes in FLIGHT_DATA:
        print(f"  {motor_name}: {apogee:.1f}m (predicted: {pred:.1f}m)")
    
    # Method 1: Direct Cd optimization (baseline)
    print("\n" + "=" * 60)
    print("Method 1: Direct Cd Optimization (Baseline)")
    print("=" * 60)
    
    history_direct = optimize_Cd_direct(physics, FLIGHT_DATA, 
                                         Cd_init=0.5, n_iters=500)
    
    Cd_learned = history_direct["Cd"][-1]
    print(f"\nLearned Cd: {Cd_learned:.4f}")
    
    # Validate
    print("\nValidation (with learned Cd):")
    for motor_name, real_apogee, pred, _ in FLIGHT_DATA[:3]:  # First 3
        motor = MOTORS[motor_name]
        result = physics.simulate_flight(Cd_learned, motor)
        print(f"  {motor_name}: Real={real_apogee:.1f}m, "
              f"Predicted={result['apogee']:.1f}m, "
              f"OpenRocket={pred:.1f}m")
    
    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Plot 1: Cd convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(history_direct["Cd"])
    axes[0].axhline(y=Cd_learned, color='r', linestyle='--', label=f'Final Cd = {Cd_learned:.4f}')
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Drag Coefficient (Cd)")
    axes[0].set_title("Cd Convergence During Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history_direct["loss"])
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("MSE Loss (m²)")
    axes[1].set_title("Loss Convergence")
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/cd_convergence.png", dpi=150)
    print("\nSaved: outputs/cd_convergence.png")
    
    # Plot 2: Predicted vs Actual Apogee
    fig, ax = plt.subplots(figsize=(10, 6))
    
    motors_used = []
    real_apogees = []
    pred_apogees_learned = []
    pred_apogees_openrocket = []
    
    for motor_name, real_apogee, openrocket_pred, notes in FLIGHT_DATA:
        motor = MOTORS[motor_name]
        result = physics.simulate_flight(Cd_learned, motor)
        
        motors_used.append(motor_name)
        real_apogees.append(real_apogee)
        pred_apogees_learned.append(result["apogee"])
        pred_apogees_openrocket.append(openrocket_pred)
    
    x = np.arange(len(FLIGHT_DATA))
    width = 0.25
    
    bars1 = ax.bar(x - width, real_apogees, width, label='Real Flight', color='green', alpha=0.8)
    bars2 = ax.bar(x, pred_apogees_learned, width, label=f'PINN (Cd={Cd_learned:.3f})', color='blue', alpha=0.8)
    bars3 = ax.bar(x + width, pred_apogees_openrocket, width, label='OpenRocket', color='orange', alpha=0.8)
    
    ax.set_xlabel('Flight Number')
    ax.set_ylabel('Apogee (m)')
    ax.set_title('Apogee Prediction: Real vs PINN vs OpenRocket')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}\n({m})' for i, m in enumerate(motors_used)], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("outputs/apogee_comparison.png", dpi=150)
    print("Saved: outputs/apogee_comparison.png")
    
    # Plot 3: Sample trajectory with learned Cd
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, motor_name in enumerate(["E35-5W", "F24-4W", "F39"]):
        motor = MOTORS[motor_name]
        result = physics.simulate_flight(Cd_learned, motor)
        
        axes[idx].plot(result["t"], result["h"], 'b-', linewidth=2)
        axes[idx].axhline(y=result["apogee"], color='r', linestyle='--', 
                         label=f'Apogee: {result["apogee"]:.1f}m')
        axes[idx].axvline(x=motor.burn_time, color='orange', linestyle=':', 
                         label=f'Burnout: {motor.burn_time:.1f}s')
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_ylabel('Altitude (m)')
        axes[idx].set_title(f'{motor_name} Motor')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/trajectories.png", dpi=150)
    print("Saved: outputs/trajectories.png")
    
    # ==========================================================================
    # COMPUTE METRICS
    # ==========================================================================
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Compute errors
    errors_pinn = np.array(pred_apogees_learned) - np.array(real_apogees)
    errors_openrocket = np.array(pred_apogees_openrocket) - np.array(real_apogees)
    
    print(f"\nLearned Drag Coefficient: Cd = {Cd_learned:.4f}")
    print(f"\nPINN Prediction Errors:")
    print(f"  MAE:  {np.mean(np.abs(errors_pinn)):.2f} m")
    print(f"  RMSE: {np.sqrt(np.mean(errors_pinn**2)):.2f} m")
    print(f"  Max:  {np.max(np.abs(errors_pinn)):.2f} m")
    
    print(f"\nOpenRocket Prediction Errors:")
    print(f"  MAE:  {np.mean(np.abs(errors_openrocket)):.2f} m")
    print(f"  RMSE: {np.sqrt(np.mean(errors_openrocket**2)):.2f} m")
    print(f"  Max:  {np.max(np.abs(errors_openrocket)):.2f} m")
    
    improvement = (np.mean(np.abs(errors_openrocket)) - np.mean(np.abs(errors_pinn))) / np.mean(np.abs(errors_openrocket)) * 100
    print(f"\nImprovement over OpenRocket: {improvement:.1f}%")
    
    # Per-motor analysis
    print("\n" + "-" * 60)
    print("Per-Motor Analysis:")
    print("-" * 60)
    
    for motor_name in ["E35-5W", "F24-4W", "F39"]:
        motor_flights = [(r, p, o) for m, r, o, _ in FLIGHT_DATA 
                        for p in [physics.simulate_flight(Cd_learned, MOTORS[m])["apogee"]]
                        if m == motor_name]
        
        real_vals = [FLIGHT_DATA[i][1] for i in range(len(FLIGHT_DATA)) if FLIGHT_DATA[i][0] == motor_name]
        pred_vals = [physics.simulate_flight(Cd_learned, MOTORS[motor_name])["apogee"]] * len(real_vals)
        
        if real_vals:
            mae = np.mean(np.abs(np.array(pred_vals) - np.array(real_vals)))
            print(f"  {motor_name}: {len(real_vals)} flights, MAE = {mae:.2f}m, "
                  f"Avg real = {np.mean(real_vals):.1f}m, Predicted = {pred_vals[0]:.1f}m")
    
    print("\n" + "=" * 60)
    print("Done! Check the 'outputs/' folder for plots.")
    print("=" * 60)
    
    # Show plots if running interactively
    plt.show()