# Physics-Informed Neural Networks for Model Rocket Aerodynamic Parameter Estimation

**Authors:** Rohit and Rohan

<!-- **Target:** arXiv publication (Winter Break 2025) -->

## Overview

This project uses Physics-Informed Neural Networks (PINNs) to learn the drag coefficient (Cd) of a model rocket from sparse apogee measurements (~12 flights). The key innovation is combining physics constraints with real flight data to enable accurate parameter estimation with minimal data.

## The Problem

Traditional rocket simulation tools (OpenRocket, RocketPy) require manually tuning aerodynamic parameters like Cd. This is time-consuming and often inaccurate because:
- Real-world drag differs from theoretical predictions
- Each rocket has manufacturing variations
- Environmental conditions vary between flights

**Our Solution:** Embed the equations of motion directly into a neural network's loss function, allowing it to learn Cd from real flight data while respecting physics.

## Data

Flight data from launches:
- **Rocket:** 535g, 6.6cm diameter, 116cm length
- **Motors:** E35-5W, F24-4W, F39 (Aerotech)
- **Flights:** 12 total with recorded apogee altitudes

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main script
python src/pinn_rocket.py

# 3. Check outputs/ folder for results
```

## Project Structure

```
pinn_rocket/
├── src/
│   └── pinn_rocket.py      # Main PINN implementation
├── outputs/                 # Generated plots and results
├── requirements.txt
└── README.md
```

## Methods

### Method 1: Direct Cd Optimization (Baseline)
Directly optimize Cd to minimize the MSE between simulated and real apogees using gradient descent with finite differences.

### Method 2: Full PINN (Coming Soon)
Train a neural network to predict trajectories h(t), v(t) with:
- **Physics Loss:** Enforce equations of motion at collocation points
- **Boundary Loss:** Initial conditions h(0)=0, v(0)=0
- **Data Loss:** Match predicted apogee to real measurements

## Physics Model

1D vertical flight equations:
```
dh/dt = v
m(t) * dv/dt = T(t) - D(v,h) - m(t)*g

where:
  D(v,h) = 0.5 * ρ(h) * v² * Cd * A      (drag force)
  ρ(h) = ρ₀ * exp(-h/H)                   (atmospheric density)
  m(t) = m_dry + m_prop * (1 - t/t_burn)  (mass decreases during burn)
```

## Expected Results

- **Learned Cd:** ~0.4-0.6 (typical for model rockets)
- **Prediction Error:** Lower than OpenRocket's default predictions
- **Generalization:** Better predictions for held-out flights
<!-- 
## Next Steps

1. [x] Implement baseline Cd optimization
2. [ ] Add full PINN training
3. [ ] Generate RocketPy simulation dataset for pre-training
4. [ ] Run ablation studies (data efficiency, physics loss weight)
5. [ ] Write paper -->

## Citation

If you use this code, please cite:
```
@article{astra2025pinn,
  title={Physics-Informed Neural Networks for Aerodynamic Parameter Estimation in Model Rocketry},
  author={[Names]},
  journal={arXiv preprint},
  year={2025}
}
```

## Acknowledgments
- OpenRocket for baseline predictions