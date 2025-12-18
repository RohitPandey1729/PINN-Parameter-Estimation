# Physics-Informed Parameter Estimation for Model Rockets

**Authors:** Rohan & Rohit

## Overview

This project presents a physics-informed method for learning aerodynamic parameters from sparse flight data in model rocketry. We estimate the drag coefficient (Cd) and per-motor thrust corrections from ~12 real competition flights across two rocket configurations.

**Key Innovation:** OpenRocket's theoretical stability model incorrectly predicted our Configuration A (no payload) would be unstable, preventing simulation. Real flights proved otherwise—our data-driven method learns directly from actual flight performance, bypassing limitations of theoretical predictions.

## The Problem

Traditional rocket simulation tools (OpenRocket, RocketPy) require pre-defined aerodynamic parameters. Issues:
- **OpenRocket limitations:** Cannot simulate all configurations (e.g., stability model rejected our Config A)
- **Parameter uncertainty:** Default Cd values often inaccurate for custom rockets
- **Few flights to learn from:** Competition constraints limit data (we had 12 flights)
- **Motor variations:** Real motor thrust curves differ from nominal specifications

**Our Solution:** Optimize both drag coefficient and per-motor thrust corrections directly from flight data using physics-informed inverse modeling.

## Data

Flight data from TARC 2025 competition:
- **Rocket Configs:** 
  - Config A: No payload (~322g dry mass)
  - Config B: With egg payload (~448g dry mass)
- **Diameter:** 6.6cm (constant)
- **Motors tested:** E35-5W, F24-4W, F39 (Aerotech)
- **Total flights:** 12 with recorded apogees
- **Best flight:** Config A with F24-4W reached 923ft

| Motor | Config A | Config B |
|-------|----------|----------|
| E35-5W | 5 flights | 1 flight |
| F24-4W | 4 flights | 4 flights |
| F39 | 0 flights | 3 flights |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the physics-informed parameter estimation
python src/pinn_rocket_final.py

# 3. Check outputs/ folder for visualizations
```

## Code Files

- **`pinn_rocket_final.py`** (Main): Physics-informed optimization of Cd and thrust corrections
  - Implements 1D ballistic flight equations with atmospheric drag
  - Optimizes parameters using scipy.optimize.minimize
  - Includes leave-one-out cross-validation
  - Compares results against OpenRocket baselines
  - Generates comprehensive visualization plots

- **`pinn_rocket.py`** (Development): Original PINN-based approach with neural networks
  - Implements full Physics-Informed Neural Network with trajectory prediction
  - Uses automatic differentiation for physics loss
  - Includes sample trajectory predictions

## Project Structure

```
PINN-Parameter-Estimation/
├── src/
│   ├── pinn_rocket_final.py     # Main: Physics-informed optimization
│   └── pinn_rocket.py           # Alternative: Full PINN approach
├── outputs/                      # Generated plots and results
│   ├── paper_results.png        # Comprehensive results visualization
│   ├── cd_convergence.png       # Optimization convergence
│   ├── apogee_comparison.png    # Prediction comparison
│   └── trajectories.png         # Sample flight trajectories
├── requirements.txt
└── README.md
```

## Methods

### Physics-Informed Optimization (pinn_rocket_final.py)

**Approach:** Direct optimization of parameters to minimize prediction error while respecting physics.

**Learned Parameters:**
1. **Drag coefficient (Cd):** Single value for both rocket configurations
2. **Per-motor thrust corrections:** Scaling factors for E35-5W, F24-4W, F39
   - Accounts for real-world thrust curve deviations from nominal specifications

**Loss Function:** 
```
MSE = (1/N) * Σ(predicted_apogee - real_apogee)²
```

**Optimization:** L-BFGS-B with bounds
- Cd ∈ [0.2, 1.5]
- Thrust factors ∈ [0.3, 1.5]

**Validation:** Leave-one-out cross-validation to assess generalization error

### Alternative: Full PINN (pinn_rocket.py)

Uses neural networks to learn trajectory functions h(t), v(t) with:
- **Physics Loss:** Enforces equations of motion at collocation points via automatic differentiation
- **Boundary Loss:** Initial conditions h(0)=0, v(0)=0
- **Data Loss:** Matches predicted apogee to real measurements

## Physics Model

1D vertical flight with atmospheric drag:
```
dh/dt = v
m(t) * dv/dt = T(t) - D(v,h) - m(t)*g

where:
  D(v,h) = 0.5 * ρ(h) * |v| * v * Cd * A      (drag force, opposes motion)
  ρ(h) = ρ₀ * exp(-h/H)                        (exponential atmosphere)
  m(t) = m_dry + m_prop * (1 - t/t_burn)       (mass depletion)
  T(t) = motor thrust curve (piecewise)
  
Constants:
  g = 9.81 m/s²
  ρ₀ = 1.225 kg/m³ (sea level density)
  H = 8500 m (scale height)
```

**Integration:** RK45 (scipy.integrate.solve_ivp) with step size 0.02s

## Results

### Learned Parameters
- **Drag Coefficient:** Cd ≈ **0.3-0.4** (learned from data)
- **Theoretical Cd:** 0.411 (from OpenRocket)
- **Per-motor thrust corrections:** Motor-specific deviations from nominal impulse

### Performance

**All 12 Flights:**
- MAE: ~22-25m across all configurations

**Config B (where OpenRocket baseline exists):**
- Our MAE: ~20m
- OpenRocket MAE: ~30m
- **Improvement: +30-35%** over OpenRocket

**Leave-One-Out Cross-Validation:**
- LOO-CV error: ~23-25m ± 8m
- Demonstrates generalization across held-out flights

**Config A Notes:**
- OpenRocket could not predict Config A (stability model rejected it)
- Our method learns from 9 Config A flights
- Average prediction error: ~10-15m

### Key Findings
1. **Drag variations:** Observed Cd differs slightly from theoretical predictions
2. **Motor deviations:** Real motors deviate 10-20% from nominal thrust specs  
3. **Config-independent Cd:** Single Cd value works for both mass configurations
4. **Robustness:** Method generalizes well despite limited training data (~12 flights)
<!-- 
## Next Steps

1. [x] Implement baseline Cd optimization
2. [x] Add cross-validation analysis
3. [ ] Add full PINN training with neural networks
4. [ ] Generate synthetic training data for pre-training
5. [ ] Run ablation studies (data efficiency, parameter coupling)
6. [ ] Compare alternative optimization methods (genetic algorithms, Bayesian optimization)
7. [ ] Write paper for publication
-->

## Citation

If you use this code, please cite:
```bibtex
@article{astra2025param,
  title={Physics-Informed Parameter Estimation for Model Rockets: 
         Learning from Flight Data When Simulations Fail},
  author={P. Rohan, P. Rohit},
  year={2025},
  note={TARC Competition 2025}
}
```

## References

- OpenRocket: https://openrocket.info/
- Thrustcurve.org: Motor thrust curves and impulse data
- Rocketry Physics: Barrowman and Barrowman (1966), Aerodynamic Prediction Method for Rockets