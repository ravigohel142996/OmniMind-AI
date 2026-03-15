# 🧠 OmniMind AI — Autonomous Decision Intelligence System

> A production-quality AI analytics platform that analyses companies, simulates markets, forecasts outcomes using machine learning, and recommends optimal business strategies.

---

## 🚀 Features

| Feature | Description |
|---|---|
| **Synthetic Company Dataset** | 120 realistic companies across 10 industries |
| **ML Growth Model** | Gradient Boosting regressor predicts company growth scores |
| **ML Risk Model** | Gradient Boosting classifier estimates risk probability |
| **ML Hiring Model** | Forecasts hiring likelihood based on growth and market signals |
| **Monte Carlo Engine** | 100–1 000 stochastic future simulations per company |
| **Market Simulator** | 12-round market evolution with economic shocks |
| **Multi-Agent System** | Growth, Risk, Innovation & Finance agents |
| **Consensus Engine** | Weighted voting across agents → single recommended strategy |
| **Interactive Dashboard** | 5-section Streamlit dashboard with Plotly charts |
| **Animated Visuals** | Animated bubble chart showing market evolution |

---

## 🏗️ Architecture

```
omnimind-ai/
├── app.py                     # Streamlit entry point
├── config.py                  # Platform configuration & constants
├── requirements.txt
│
├── data/
│   ├── company_generator.py   # Synthetic company dataset (120 companies)
│   └── market_environment.py  # Market macro-parameters dataclass
│
├── models/
│   ├── growth_model.py        # GBR — growth score prediction
│   ├── risk_model.py          # GBC — risk probability estimation
│   └── hiring_model.py        # GBC — hiring forecast
│
├── simulation/
│   ├── monte_carlo_engine.py  # 500-run stochastic simulation
│   ├── market_simulator.py    # 12-round market evolution
│   └── future_scenarios.py    # Named strategic scenarios
│
├── agents/
│   ├── growth_agent.py        # Maximise expansion
│   ├── risk_agent.py          # Minimise risk exposure
│   ├── innovation_agent.py    # Drive R&D investment
│   ├── finance_agent.py       # Control costs
│   └── strategy_consensus.py  # Weighted voting consensus
│
├── analytics/
│   ├── company_analysis.py    # Per-company enrichment & profiling
│   └── market_analysis.py     # Cross-industry aggregations
│
├── ui/
│   ├── dashboard.py           # Dashboard section renderers
│   ├── charts.py              # Plotly chart factory functions
│   ├── controls.py            # Sidebar controls
│   └── animations.py          # Animated Plotly visuals
│
└── utils/
    └── helpers.py             # Shared formatting & math utilities
```

---

## 🖥️ Dashboard Sections

1. **🌐 Global Market Overview** — Market growth, economic pressure, tech disruption, global demand
2. **🏢 Company Intelligence** — Per-company radar chart, revenue trend, risk heatmap
3. **🤖 Strategy Advisor** — AI-recommended strategy, agent breakdown, vote scores
4. **🔮 Future Simulation** — Monte Carlo distributions, growth forecast cone, scenario comparison
5. **📊 Market Intelligence** — Industry bubble charts, innovation vs revenue, opportunity heatmap

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — web dashboard
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — ML models (GradientBoosting)
- **Plotly** — interactive and animated charts
- **SciPy** — statistical utilities
- **NetworkX** — graph utilities (available for extension)
- **Altair** — supplementary charts (available for extension)

---

## ⚙️ Setup & Running

### Prerequisites

```bash
python -m pip install -r requirements.txt
```

### Launch

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** by default.

### Streamlit Cloud Deployment

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set the main file to `app.py`.
4. Deploy — no additional secrets required.

---

## 🎛️ Sidebar Controls

| Control | Description |
|---|---|
| Company Size | Filter companies by employee count |
| Market Growth Rate | Annual market growth (0–30%) |
| Competition Level | Market competition intensity (0–1) |
| Technology Disruption | Speed of tech change (0–1) |
| Economic Pressure | Macro-economic headwinds (0–1) |
| Monte Carlo Runs | Number of simulation runs (100–1 000) |
| Select Company | Choose a specific company to analyse |
| Generate AI Strategy | Trigger full analysis pipeline |

---

## 🔮 Future Improvements

- Real company data integration via financial APIs (Alpha Vantage, Yahoo Finance)
- LSTM / Transformer time-series forecasting
- Reinforcement learning agents for adaptive strategy
- PDF strategy report export
- Multi-user authentication
- Real-time market data streaming

---

## 📄 License

MIT — see `LICENSE` for details.
