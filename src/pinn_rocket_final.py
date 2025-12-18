"""
Physics-Informed Parameter Estimation for Model Rockets
Team Astra - TARC 2025

Key finding: OpenRocket incorrectly predicted Config A would be unstable.
Real flights showed Config A flew successfully - our method learns from this real data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from dataclasses import dataclass
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RocketConfig:
    name: str
    dry_mass: float  # kg
    diameter: float = 0.066  # m
    Cd_openrocket: float = 0.411
    
    @property
    def area(self):
        return np.pi * (self.diameter / 2) ** 2

CONFIG_A = RocketConfig("No payload", dry_mass=0.322)
CONFIG_B = RocketConfig("With eggs", dry_mass=0.448)

@dataclass 
class MotorConfig:
    name: str
    total_impulse: float  # N·s
    avg_thrust: float     # N
    max_thrust: float     # N
    burn_time: float      # s
    propellant_mass: float  # kg

# Motor data from thrustcurve.org
MOTORS = {
    "E35-5W": MotorConfig("E35-5W", 39.4, 33.8, 44.0, 1.1, 0.025),
    "F24-4W": MotorConfig("F24-4W", 47.3, 22.2, 41.0, 2.1, 0.019),
    "F39": MotorConfig("F39", 49.7, 37.3, 59.5, 1.3, 0.023),
}

# OpenRocket predictions - Config B only 
# (Config A: OpenRocket incorrectly predicted instability, but flights were successful)
OPENROCKET = {
    ("E35-5W", "B"): 231.3,  # 759 ft
    ("F24-4W", "B"): 288.0,  # 945 ft
    ("F39", "B"): 259.7,     # 852 ft
}

# Flight data: (motor, apogee_m, config, notes, include)
FLIGHT_DATA = [
    ("E35-5W", 169.8, "B", "Good launch", True),
    ("F24-4W", 281.3, "A", "Excellent flight (923ft)", True),
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

G = 9.81
RHO_0 = 1.225
H_SCALE = 8500.0

# =============================================================================
# PHYSICS SIMULATION
# =============================================================================

def thrust_curve(t, motor, thrust_factor=1.0):
    if t < 0 or t >= motor.burn_time:
        return 0.0
    t_peak = 0.08 * motor.burn_time
    if t < t_peak:
        return motor.max_thrust * (t / t_peak) * thrust_factor
    else:
        progress = (t - t_peak) / (motor.burn_time - t_peak)
        return (motor.max_thrust - (motor.max_thrust - motor.avg_thrust) * progress) * thrust_factor

def rocket_ode(t, state, Cd, motor, rocket, thrust_factor):
    h, v = state
    h = max(0, h)
    m_prop = motor.propellant_mass * max(0, 1 - t/motor.burn_time) if t < motor.burn_time else 0
    m = rocket.dry_mass + m_prop
    T = thrust_curve(t, motor, thrust_factor)
    rho = RHO_0 * np.exp(-h / H_SCALE)
    D = 0.5 * rho * v * abs(v) * Cd * rocket.area
    return [v, (T - D - m * G) / m]

def simulate(Cd, motor, rocket, thrust_factor=1.0):
    sol = solve_ivp(rocket_ode, [0, 60], [0.0, 0.0], 
                    args=(Cd, motor, rocket, thrust_factor), method='RK45', max_step=0.02)
    return np.max(sol.y[0])

def get_config(cfg):
    return CONFIG_A if cfg == "A" else CONFIG_B

# =============================================================================
# OPTIMIZATION
# =============================================================================

def objective(params, flight_data):
    Cd = params[0]
    thrust_factors = {"E35-5W": params[1], "F24-4W": params[2], "F39": params[3]}
    
    total_error, count = 0, 0
    for motor_name, real_apogee, cfg, _, include in flight_data:
        if include:
            motor = MOTORS[motor_name]
            rocket = get_config(cfg)
            pred = simulate(Cd, motor, rocket, thrust_factors[motor_name])
            total_error += (pred - real_apogee) ** 2
            count += 1
    return total_error / count if count > 0 else float('inf')

def optimize(flight_data):
    x0 = [0.5, 1.0, 1.0, 1.0]
    bounds = [(0.2, 1.5), (0.3, 1.5), (0.3, 1.5), (0.3, 1.5)]
    result = minimize(objective, x0, args=(flight_data,), method='L-BFGS-B', bounds=bounds)
    return {
        "Cd": result.x[0],
        "thrust_factors": {"E35-5W": result.x[1], "F24-4W": result.x[2], "F39": result.x[3]}
    }

def predict(params, flight_data):
    return np.array([
        simulate(params["Cd"], MOTORS[f[0]], get_config(f[2]), params["thrust_factors"][f[0]])
        for f in flight_data
    ])

# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def leave_one_out_cv(flight_data):
    included_idx = [i for i, f in enumerate(flight_data) if f[4]]
    errors = []
    
    for test_idx in included_idx:
        train_data = [(f[0], f[1], f[2], f[3], False if i == test_idx else f[4]) 
                      for i, f in enumerate(flight_data)]
        params = optimize(train_data)
        f = flight_data[test_idx]
        pred = simulate(params["Cd"], MOTORS[f[0]], get_config(f[2]), params["thrust_factors"][f[0]])
        errors.append((test_idx, abs(pred - f[1]), pred, f[1]))
    
    return errors

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("   PHYSICS-INFORMED PARAMETER ESTIMATION FOR MODEL ROCKETS")
    print("   Team Astra - TARC 2025")
    print("=" * 70)
    
    # Dataset info
    n_total = len(FLIGHT_DATA)
    n_train = sum(1 for f in FLIGHT_DATA if f[4])
    n_config_a = sum(1 for f in FLIGHT_DATA if f[2] == "A")
    n_config_b = sum(1 for f in FLIGHT_DATA if f[2] == "B")
    
    print(f"""
Dataset Summary:
  Total flights: {n_total} ({n_train} used for training)
  Config A (no payload): {n_config_a} flights
  Config B (with eggs):  {n_config_b} flights
  
Rocket Specifications:
  Config A mass: {CONFIG_A.dry_mass*1000:.0f}g
  Config B mass: {CONFIG_B.dry_mass*1000:.0f}g
  Diameter: {CONFIG_A.diameter*100:.1f}cm
  Theoretical Cd: {CONFIG_A.Cd_openrocket} (from OpenRocket)
  
OpenRocket Baselines (Config B only):
  E35-5W: {OPENROCKET[('E35-5W','B')]:.1f}m (759ft)
  F24-4W: {OPENROCKET[('F24-4W','B')]:.1f}m (945ft)  
  F39:    {OPENROCKET[('F39','B')]:.1f}m (852ft)

Note: OpenRocket incorrectly predicted Config A would be unstable.
      Real flights showed Config A flew successfully.
""")
    
    # Optimize
    print("-" * 70)
    print("OPTIMIZATION")
    print("-" * 70)
    
    params = optimize(FLIGHT_DATA)
    predictions = predict(params, FLIGHT_DATA)
    real_arr = np.array([f[1] for f in FLIGHT_DATA])
    
    print(f"""
Learned Parameters:
  Cd = {params['Cd']:.3f} (vs theoretical 0.411)
  
Thrust Correction Factors:
  E35-5W: {params['thrust_factors']['E35-5W']:.3f} (effective impulse: {MOTORS['E35-5W'].total_impulse * params['thrust_factors']['E35-5W']:.1f} Ns)
  F24-4W: {params['thrust_factors']['F24-4W']:.3f} (effective impulse: {MOTORS['F24-4W'].total_impulse * params['thrust_factors']['F24-4W']:.1f} Ns)
  F39:    {params['thrust_factors']['F39']:.3f} (effective impulse: {MOTORS['F39'].total_impulse * params['thrust_factors']['F39']:.1f} Ns)
""")
    
    # Results table
    print("-" * 70)
    print("FLIGHT-BY-FLIGHT RESULTS")
    print("-" * 70)
    print(f"{'#':<3} {'Motor':<8} {'Cfg':<4} {'Real':>7} {'Ours':>7} {'OpenRocket':>11} {'Error':>7} {'Notes'}")
    print("-" * 70)
    
    for i, (motor, real, cfg, notes, inc) in enumerate(FLIGHT_DATA):
        or_pred = OPENROCKET.get((motor, cfg), None)
        or_str = f"{or_pred:.1f}" if or_pred else "N/A"
        err = predictions[i] - real
        flag = "" if inc else " [excl]"
        print(f"{i+1:<3} {motor:<8} {cfg:<4} {real:>7.1f} {predictions[i]:>7.1f} {or_str:>11} {err:>+7.1f} {notes}{flag}")
    
    # Metrics
    print("\n" + "-" * 70)
    print("ERROR METRICS")
    print("-" * 70)
    
    # All flights - our method
    our_mae_all = np.mean(np.abs(predictions - real_arr))
    our_rmse_all = np.sqrt(np.mean((predictions - real_arr)**2))
    
    # Config B only (where OpenRocket baseline exists)
    config_b_idx = [i for i, f in enumerate(FLIGHT_DATA) if f[2] == "B"]
    our_errors_b = predictions[config_b_idx] - real_arr[config_b_idx]
    or_preds_b = np.array([OPENROCKET[(FLIGHT_DATA[i][0], "B")] for i in config_b_idx])
    or_errors_b = or_preds_b - real_arr[config_b_idx]
    
    our_mae_b = np.mean(np.abs(our_errors_b))
    or_mae_b = np.mean(np.abs(or_errors_b))
    our_rmse_b = np.sqrt(np.mean(our_errors_b**2))
    or_rmse_b = np.sqrt(np.mean(or_errors_b**2))
    
    # Config A only - our method (no baseline available)
    config_a_idx = [i for i, f in enumerate(FLIGHT_DATA) if f[2] == "A"]
    our_mae_a = np.mean(np.abs(predictions[config_a_idx] - real_arr[config_a_idx]))
    
    print(f"""
All {n_total} Flights (Our Method):
  MAE:  {our_mae_all:.1f}m
  RMSE: {our_rmse_all:.1f}m

Config A ({len(config_a_idx)} flights - no OpenRocket baseline available):
  Our MAE: {our_mae_a:.1f}m

Config B ({len(config_b_idx)} flights - has OpenRocket baseline):
  {'Metric':<12} {'Ours':>10} {'OpenRocket':>12} {'Improvement':>12}
  {'-'*48}
  {'MAE':<12} {our_mae_b:>10.1f}m {or_mae_b:>11.1f}m {(or_mae_b-our_mae_b)/or_mae_b*100:>+11.1f}%
  {'RMSE':<12} {our_rmse_b:>10.1f}m {or_rmse_b:>11.1f}m {(or_rmse_b-our_rmse_b)/or_rmse_b*100:>+11.1f}%
""")
    
    # Cross-validation
    print("-" * 70)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print("-" * 70)
    
    loo_results = leave_one_out_cv(FLIGHT_DATA)
    
    for idx, err, pred, real in loo_results:
        f = FLIGHT_DATA[idx]
        print(f"  Hold out #{idx+1} ({f[0]}, Cfg {f[2]}): Real={real:.0f}m, Pred={pred:.0f}m, Error={err:.0f}m")
    
    loo_errors = [r[1] for r in loo_results]
    loo_mae = np.mean(loo_errors)
    loo_std = np.std(loo_errors)
    print(f"\n  LOO-CV MAE: {loo_mae:.1f} ± {loo_std:.1f}m")
    
    # Summary
    print("\n" + "=" * 70)
    print("PAPER SUMMARY")
    print("=" * 70)
    print(f"""
We present a physics-informed method for learning aerodynamic parameters
from sparse flight data. Using {n_train} flights from a TARC competition rocket
across two mass configurations, we learn:
  • Drag coefficient: Cd = {params['Cd']:.3f} (vs theoretical 0.411)
  • Per-motor thrust corrections accounting for real-world deviations

Key Results:
  • Overall MAE: {our_mae_all:.1f}m across all {n_total} flights
  • Config B: {(or_mae_b-our_mae_b)/or_mae_b*100:.0f}% improvement over OpenRocket
  • LOO Cross-Validation: {loo_mae:.1f} ± {loo_std:.1f}m (generalization error)

Notable: OpenRocket's stability model incorrectly predicted Config A would
be unstable, preventing simulation. Real flights proved otherwise — Config A
flew successfully, with flight #2 reaching 923ft in one of our best launches.
Our data-driven method learns from actual flight performance, bypassing
limitations of theoretical stability predictions.
""")
    
    # ==========================================================================
    # PLOTS
    # ==========================================================================
    
    os.makedirs("outputs", exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: All flights comparison
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(n_total)
    width = 0.35
    
    ax1.bar(x - width/2, real_arr, width, label='Real', color='green', alpha=0.8)
    ax1.bar(x + width/2, predictions, width, label=f'Ours (Cd={params["Cd"]:.2f})', color='blue', alpha=0.8)
    
    # Add OpenRocket where available
    for i, f in enumerate(FLIGHT_DATA):
        or_pred = OPENROCKET.get((f[0], f[2]), None)
        if or_pred:
            ax1.scatter(i, or_pred, color='orange', s=100, zorder=5, marker='D')
    ax1.scatter([], [], color='orange', s=100, marker='D', label='OpenRocket')
    
    # Shade excluded flights
    for i, f in enumerate(FLIGHT_DATA):
        if not f[4]:
            ax1.axvspan(i-0.4, i+0.4, alpha=0.15, color='gray')
    
    ax1.set_xlabel('Flight Number')
    ax1.set_ylabel('Apogee (m)')
    ax1.set_title(f'All Flights: Real vs Predicted (MAE={our_mae_all:.1f}m)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i+1}\n({'A' if FLIGHT_DATA[i][2]=='A' else 'B'})" for i in range(n_total)], fontsize=8)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Config B comparison (where we have baseline)
    ax2 = fig.add_subplot(2, 2, 2)
    
    config_b_flights = [(i, f) for i, f in enumerate(FLIGHT_DATA) if f[2] == "B"]
    x_b = np.arange(len(config_b_flights))
    
    real_b = [f[1] for _, f in config_b_flights]
    pred_b = [predictions[i] for i, _ in config_b_flights]
    or_b = [OPENROCKET[(f[0], "B")] for _, f in config_b_flights]
    labels_b = [f"#{i+1}\n{f[0]}" for i, f in config_b_flights]
    
    width = 0.25
    ax2.bar(x_b - width, real_b, width, label='Real', color='green', alpha=0.8)
    ax2.bar(x_b, pred_b, width, label=f'Ours (MAE={our_mae_b:.1f}m)', color='blue', alpha=0.8)
    ax2.bar(x_b + width, or_b, width, label=f'OpenRocket (MAE={or_mae_b:.1f}m)', color='orange', alpha=0.8)
    
    ax2.set_xlabel('Flight')
    ax2.set_ylabel('Apogee (m)')
    improvement_pct = (or_mae_b-our_mae_b)/or_mae_b*100
    ax2.set_title(f'Config B: {improvement_pct:.0f}% Improvement over OpenRocket')
    ax2.set_xticks(x_b)
    ax2.set_xticklabels(labels_b)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: LOO Cross-validation
    ax3 = fig.add_subplot(2, 2, 3)
    
    loo_x = range(len(loo_errors))
    colors = ['steelblue' if FLIGHT_DATA[r[0]][2] == 'B' else 'coral' for r in loo_results]
    ax3.bar(loo_x, loo_errors, color=colors, alpha=0.8)
    ax3.axhline(loo_mae, color='red', linestyle='--', linewidth=2, label=f'Mean = {loo_mae:.1f}m')
    ax3.fill_between([-0.5, len(loo_errors)-0.5], loo_mae-loo_std, loo_mae+loo_std, 
                     color='red', alpha=0.15)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='coral', alpha=0.8, label='Config A'),
        Patch(facecolor='steelblue', alpha=0.8, label='Config B'),
        plt.Line2D([0], [0], color='red', linestyle='--', label=f'Mean = {loo_mae:.1f}m'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    ax3.set_xlabel('Held-out Flight Index')
    ax3.set_ylabel('Absolute Error (m)')
    ax3.set_title(f'Leave-One-Out CV: {loo_mae:.1f} ± {loo_std:.1f}m')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Learned parameters visualization
    ax4 = fig.add_subplot(2, 2, 4)
    
    motors = list(MOTORS.keys())
    nominal_impulse = [MOTORS[m].total_impulse for m in motors]
    effective_impulse = [MOTORS[m].total_impulse * params['thrust_factors'][m] for m in motors]
    
    x_m = np.arange(len(motors))
    width = 0.35
    ax4.bar(x_m - width/2, nominal_impulse, width, label='Nominal (thrustcurve.org)', color='gray', alpha=0.7)
    ax4.bar(x_m + width/2, effective_impulse, width, label='Learned Effective', color='blue', alpha=0.8)
    
    for i, m in enumerate(motors):
        tf = params['thrust_factors'][m]
        ax4.annotate(f'{tf:.2f}x', (i + width/2, effective_impulse[i] + 1), ha='center', fontsize=10)
    
    ax4.set_xlabel('Motor')
    ax4.set_ylabel('Total Impulse (Ns)')
    ax4.set_title(f'Learned Thrust Corrections (Cd={params["Cd"]:.3f})')
    ax4.set_xticks(x_m)
    ax4.set_xticklabels(motors)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("outputs/paper_results.png", dpi=200, bbox_inches='tight')
    print("\nSaved: outputs/paper_results.png")
    
    plt.show()