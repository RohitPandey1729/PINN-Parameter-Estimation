"""
Rocket Parameter Estimation v3
- Learn thrust correction factor per motor (absorbs motor data errors)
- Single global Cd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from dataclasses import dataclass
import os

@dataclass
class RocketConfig:
    name: str = "Astra"
    dry_mass: float = 0.405
    diameter: float = 0.066
    
    @property
    def area(self):
        return np.pi * (self.diameter / 2) ** 2

@dataclass 
class MotorConfig:
    name: str
    total_impulse: float
    avg_thrust: float
    burn_time: float
    propellant_mass: float

G = 9.81
RHO_0 = 1.225
H_SCALE = 8500.0

# Original motor params (we'll learn corrections)
MOTORS = {
    "E35-5W": MotorConfig("E35-5W", 35.0, 35.0, 1.0, 0.025),
    "F24-4W": MotorConfig("F24-4W", 48.0, 24.0, 2.0, 0.035),
    "F39": MotorConfig("F39", 79.2, 39.0, 1.3, 0.040),
}

FLIGHT_DATA = [
    ("E35-5W", 169.8, 167.6, True),
    ("F24-4W", 281.3, 289.6, True),
    ("F24-4W", 246.6, 289.6, True),
    ("E35-5W", 174.7, 167.6, True),
    ("E35-5W", 154.5, 167.6, False),  # weak
    ("E35-5W", 131.1, 167.6, False),  # weak
    ("E35-5W", 166.1, 167.6, False),  # different config
    ("F24-4W", 199.9, 289.6, False),  # crooked
    ("F24-4W", 241.4, 289.6, True),
    ("F39", 185.9, 198.1, True),
    ("F39", 196.3, 198.1, True),
    ("F39", 198.1, 198.1, True),
]

def thrust_curve(t, motor, thrust_factor=1.0):
    t_burn = motor.burn_time
    if t < 0 or t >= t_burn:
        return 0.0
    t_ramp = 0.1 * t_burn
    if t < t_ramp:
        return motor.avg_thrust * 1.2 * (t / t_ramp) * thrust_factor
    elif t < t_burn - t_ramp:
        return motor.avg_thrust * 1.1 * thrust_factor
    else:
        return motor.avg_thrust * 1.2 * ((t_burn - t) / t_ramp) * thrust_factor

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
                    args=(Cd, motor, rocket, thrust_factor), method='RK45', max_step=0.05)
    return np.max(sol.y[0])

def objective(params, flight_data, rocket):
    """params = [Cd, thrust_E35, thrust_F24, thrust_F39]"""
    Cd = params[0]
    thrust_factors = {"E35-5W": params[1], "F24-4W": params[2], "F39": params[3]}
    
    total_error = 0
    count = 0
    for motor_name, real_apogee, _, include in flight_data:
        if include:
            motor = MOTORS[motor_name]
            pred = simulate(Cd, motor, rocket, thrust_factors[motor_name])
            total_error += (pred - real_apogee) ** 2
            count += 1
    return total_error / count

if __name__ == "__main__":
    print("=" * 60)
    print("Rocket Parameter Estimation v3")
    print("Learning: Cd + thrust correction per motor")
    print("=" * 60)
    
    rocket = RocketConfig()
    
    # Initial guess: Cd=0.5, thrust_factors=1.0
    x0 = [0.5, 1.0, 1.0, 1.0]
    bounds = [(0.3, 0.8), (0.5, 2.0), (0.5, 2.0), (0.5, 2.0)]
    
    print("\nOptimizing...")
    result = minimize(objective, x0, args=(FLIGHT_DATA, rocket), 
                     method='L-BFGS-B', bounds=bounds)
    
    Cd = result.x[0]
    thrust_factors = {"E35-5W": result.x[1], "F24-4W": result.x[2], "F39": result.x[3]}
    
    print(f"\nLearned parameters:")
    print(f"  Cd = {Cd:.3f}")
    for m, tf in thrust_factors.items():
        print(f"  {m} thrust factor = {tf:.3f}")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    real_arr, pred_arr, or_arr = [], [], []
    for motor_name, real_apogee, or_pred, include in FLIGHT_DATA:
        motor = MOTORS[motor_name]
        pred = simulate(Cd, motor, rocket, thrust_factors[motor_name])
        real_arr.append(real_apogee)
        pred_arr.append(pred)
        or_arr.append(or_pred)
        flag = "" if include else " [EXCLUDED]"
        print(f"{motor_name}: Real={real_apogee:.0f}m, Ours={pred:.0f}m, OR={or_pred:.0f}m{flag}")
    
    real_arr, pred_arr, or_arr = np.array(real_arr), np.array(pred_arr), np.array(or_arr)
    our_mae = np.mean(np.abs(pred_arr - real_arr))
    or_mae = np.mean(np.abs(or_arr - real_arr))
    
    print(f"\nMAE - Ours: {our_mae:.1f}m, OpenRocket: {or_mae:.1f}m")
    improvement = (or_mae - our_mae) / or_mae * 100
    print(f"Improvement over OpenRocket: {improvement:.1f}%")
    
    # Plot
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(FLIGHT_DATA))
    width = 0.25
    ax.bar(x - width, real_arr, width, label='Real', color='green', alpha=0.8)
    ax.bar(x, pred_arr, width, label=f'Ours (Cd={Cd:.2f})', color='blue', alpha=0.8)
    ax.bar(x + width, or_arr, width, label='OpenRocket', color='orange', alpha=0.8)
    for i, f in enumerate(FLIGHT_DATA):
        if not f[3]:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color='gray')
    ax.set_xlabel('Flight')
    ax.set_ylabel('Apogee (m)')
    ax.set_title(f'v3: Learned Cd={Cd:.2f} + per-motor thrust corrections')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}\n({f[0]})' for i, f in enumerate(FLIGHT_DATA)], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("outputs/apogee_comparison_v3.png", dpi=150)
    print("\nSaved: outputs/apogee_comparison_v3.png")
    plt.show()