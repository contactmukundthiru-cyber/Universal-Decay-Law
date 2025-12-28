# The Universal Decay Law of Human Engagement

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive computational framework for discovering and validating a universal law governing the decline of human digital engagement across platforms.

## Overview

This research project demonstrates that engagement decay across diverse digital platforms—from Reddit and GitHub to Strava and Wikipedia—follows a **universal two-parameter scaling law**:

$$E(t) = E_0 \cdot f\left(\frac{t}{\tau(\alpha)}\right)$$

Where:
- $E(t)$ is engagement at time $t$
- $E_0$ is initial engagement
- $\tau(\alpha) = \tau_0 \cdot \alpha^{-\beta}$ relates the characteristic timescale to motivation
- $f(x) = \exp(-x^\gamma)$ is the universal master curve (stretched exponential)

### Key Findings

1. **Universality**: All engagement curves collapse onto a single master curve when rescaled by user-specific timescales
2. **Stretched Exponential Form**: The universal curve follows $f(x) = \exp(-x^\gamma)$ with $\gamma \approx 0.5-0.7$
3. **Motivation Scaling**: Characteristic decay time scales as $\tau \propto \alpha^{-\beta}$ with motivation parameter $\alpha$
4. **Cross-Platform Validity**: The law holds across social, learning, fitness, and creative platforms

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL (optional, SQLite supported for development)
- Node.js 18+ (for dashboard)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/contactmukundthiru-cyber/Universal-Decay-Law.git
cd Universal-Decay-Law

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys and database settings

# Initialize database
python run.py init-db
```

### Dashboard Setup

```bash
cd dashboard
npm install
```

## Quick Start

### Check Environment Configuration

```bash
python run.py check-env
```

This verifies your API credentials and database configuration.

### Start the Server

```bash
# Development mode with auto-reload
python run.py server --reload

# Production mode
python run.py server --host 0.0.0.0 --port 8000
```

### Start the Dashboard

```bash
cd dashboard
npm run dev
```

Access the dashboard at http://localhost:3000

## Project Structure

```
THE_UNIVERSAL_DECAY_LAW/
├── config/                     # Configuration management
│   └── settings.py             # Pydantic settings
├── database/                   # Database layer
│   ├── models.py               # SQLAlchemy ORM models
│   ├── connection.py           # Async database connection
│   └── crud.py                 # CRUD operations
├── src/
│   ├── models/                 # Mathematical decay models
│   │   ├── base.py             # Abstract base class and registry
│   │   ├── stretched_exponential.py
│   │   ├── power_law.py
│   │   ├── weibull.py
│   │   ├── double_exponential.py
│   │   ├── mechanistic.py      # SDE-based model
│   │   └── motivation.py       # Motivation parameter estimation
│   ├── data/                   # Data connectors
│   │   ├── base.py             # Abstract connector interface
│   │   ├── reddit.py
│   │   ├── github.py
│   │   ├── wikipedia.py
│   │   ├── strava.py
│   │   ├── lastfm.py
│   │   ├── synthetic.py        # Synthetic data generation
│   │   └── loader.py           # Unified data loading
│   ├── analysis/               # Analysis pipeline
│   │   ├── preprocessing.py    # Data cleaning and normalization
│   │   ├── fitting.py          # Model fitting
│   │   ├── universality.py     # Master curve collapse
│   │   ├── validation.py       # Cross-validation
│   │   └── statistics.py       # Statistical tests
│   ├── visualization/          # Visualization system
│   │   ├── style.py            # Publication styling
│   │   ├── figures.py          # Static figures (matplotlib)
│   │   └── interactive.py      # Interactive plots (plotly)
│   └── api/                    # FastAPI backend
│       ├── main.py
│       └── routes/
│           ├── datasets.py
│           ├── trials.py
│           ├── analysis.py
│           └── visualization.py
├── dashboard/                  # React frontend
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── api/
│   └── package.json
├── run.py                      # Entry point
└── pyproject.toml
```

## Mathematical Framework

### Decay Models

| Model | Equation | Parameters | Physical Interpretation |
|-------|----------|------------|------------------------|
| Stretched Exponential | $f(x) = e^{-x^\gamma}$ | $\tau, \gamma$ | Heterogeneous relaxation |
| Power Law | $f(x) = (1+x)^{-\gamma}$ | $\tau, \gamma$ | Scale-free dynamics |
| Weibull | $f(x) = e^{-(x/\lambda)^\kappa}$ | $\lambda, \kappa$ | Survival analysis |
| Double Exponential | $f(t) = Ae^{-t/\tau_1} + (1-A)e^{-t/\tau_2}$ | $A, \tau_1, \tau_2$ | Two-timescale decay |
| Mechanistic SDE | $dE = -kE^\gamma dt + \sigma E^\beta dW$ | $k, \gamma, \sigma, \beta$ | Stochastic dynamics |

### Motivation Parameter

The motivation parameter $\alpha$ quantifies the intrinsic/extrinsic motivation balance:

$$\alpha = \frac{\text{intrinsic signals}}{\text{extrinsic signals}}$$

Higher $\alpha$ → longer engagement ($\tau \propto \alpha^{-\beta}$ with $\beta > 0$)

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/datasets/` | List all datasets |
| POST | `/api/datasets/` | Create new dataset |
| POST | `/api/datasets/synthetic` | Generate synthetic data |
| GET | `/api/trials/` | List all trials |
| POST | `/api/trials/` | Create new trial |
| POST | `/api/trials/{id}/run` | Start trial execution |
| GET | `/api/visualization/collapse/{id}` | Get collapse data |
| GET | `/api/visualization/scaling/{id}` | Get scaling data |
| GET | `/api/visualization/dashboard/{id}` | Get full dashboard data |

### Example: Python Client

```python
import requests

# Create synthetic dataset
resp = requests.post("http://localhost:8000/api/datasets/synthetic", json={
    "n_users": 200,
    "platforms": ["reddit", "github", "strava"],
    "tau_distribution": {"mean": 30, "std": 15}
})
dataset = resp.json()

# Create and run trial
trial = requests.post("http://localhost:8000/api/trials/", json={
    "name": "Universality Test",
    "dataset_id": dataset["id"],
    "config": {
        "models": ["stretched_exponential", "power_law"],
        "min_data_points": 20,
        "cross_validate": True
    }
}).json()

requests.post(f"http://localhost:8000/api/trials/{trial['id']}/run")
```

## Visualization Gallery

The framework generates six publication-quality figures:

1. **Raw Decay Curves**: Individual engagement trajectories across platforms
2. **Master Curve Collapse**: All curves rescaled to universal form
3. **Scaling Relationship**: Log-log plot of $\tau$ vs $\alpha$
4. **Platform Comparison**: Cross-platform validation
5. **Model Comparison**: AIC/BIC comparison across models
6. **Deviant Analysis**: Identifying non-universal behaviors

## Data Sources

| Platform | Engagement Metric | API |
|----------|------------------|-----|
| Reddit | Posts, comments, karma | PRAW |
| GitHub | Commits, issues, PRs | PyGithub |
| Wikipedia | Edits, bytes added | mwclient |
| Strava | Activities, distance | Strava API |
| Last.fm | Scrobbles | Last.fm API |
| Duolingo | XP, streaks | (web scraping) |
| Khan Academy | Mastery points | (web scraping) |

## Configuration

Key configuration options in `.env`:

```bash
# Analysis parameters
MIN_DATA_POINTS=20          # Minimum observations per user
MAX_USERS_PER_TRIAL=10000   # Maximum users to process
FIT_METHOD=L-BFGS-B         # Optimization algorithm
DEVIATION_THRESHOLD=2.0     # σ threshold for deviants

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/decay_law
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{thiru2025universal,
  title={The Universal Decay Law of Human Digital Engagement},
  author={Thiru, Mukund},
  journal={Preprint},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
