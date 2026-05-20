# Asteroid Detection, Tracking & Risk Assessment System

An end-to-end pipeline for detecting moving asteroids in telescope survey data, estimating orbital paths, assessing impact risk, and visualizing results through an interactive Streamlit dashboard.

## Overview

This project is designed to detect moving objects from astronomical image data, estimate orbital parameters, compute risk metrics such as MOID (Minimum Orbit Intersection Distance), and present results in a user-friendly web dashboard. It combines classical astronomy libraries, computer vision, machine learning, and interactive visualization to create a practical asteroid monitoring workflow.

## Features

- Detect moving objects in telescope survey images or simulated datasets using OpenCV-based streak and motion detection.
- Estimate orbital trajectories and propagate orbits using astronomy libraries such as Astropy, poliastro, and Skyfield.
- Calculate asteroid risk indicators, including MOID and orbit intersection analysis.
- Visualize asteroid paths in 2D and 3D using Plotly charts.
- Upload images or data files through a Streamlit dashboard and view detection outputs, risk scores, and orbit plots in real time.
- Optional advanced modules for anomaly detection, LSTM/Transformer forecasting, synthetic data generation, or collision-avoidance optimization.

## Tech Stack
- Python
- Astropy
- poliastro
- Skyfield
- OpenCV
- NumPy, Pandas
- scikit-learn
- TensorFlow or PyTorch
- Plotly
- Streamlit

## Project Workflow

1. **Input data**  
   Upload telescope survey images, time-series observations, or simulated asteroid data.

2. **Object detection**  
   Identify moving points or streaks across frames using OpenCV-based detection methods.

3. **Orbit estimation**  
   Convert detections into orbital parameters and propagate trajectories with astronomy libraries.

4. **Risk assessment**  
   Compute MOID and related orbital risk measures to estimate potential collision threat.

5. **Visualization**  
   Display orbits, paths, and confidence/risk metrics in interactive plots.

6. **Dashboard**  
   Use Streamlit to make the workflow accessible through a clean web interface.

## Example Use Cases
- Academic or portfolio project for astronomy and AI.
- Prototype for automated asteroid screening.
- Interactive demo for orbital mechanics and space safety.
- Research sandbox for synthetic asteroid data and detection experiments.

## Folder Structure

```bash
asteroidwatch/
├── dashboard/
│   └── app.py                  # Streamlit User Interface
├── data/
│   ├── __init__.py             # Mark package namespace
│   └── simulator.py            # Synthetic Image Generator
├── detection/
│   ├── __init__.py             
│   └── streak_detector.py      # OpenCV Extraction Engine
├── ml/
│   ├── __init__.py             
│   └── models.py               # PyTorch CNN & LSTM Architectures
├── orbits/
│   ├── __init__.py             
│   └── orbit_engine.py         # Orbital Mechanics & MOID Optimizer
└── utils/
    ├── __init__.py             
    ├── constants.py            # Astrophysical Constants & Thresholds
    └── helpers.py              # FITS I/O & Percentile Stretches
```

## Installation

```bash
git clone https://github.com/your-username/asteroid-detection-system.git
cd asteroid-detection-system
pip install -r requirements.txt
```

## Usage

### Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

### Run detection pipeline

```bash
python src/main.py --input data/sample_frame.fits
```

## Future Improvements
- Add reinforcement learning for collision-avoidance optimization.
- Generate synthetic asteroid datasets with generative models.
- Improve detection accuracy with CNN-based streak classification.
- Add database support for storing asteroid observations and orbit history.
- Integrate public ephemeris or survey feeds for live updates.

## Contributing
- Contributions are welcome. Feel free to improve detection methods, orbit estimation, visualization, or dashboard UX.

## License
- MIT License
```
Copyright (c) 2026 Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
