# FLUX (Forecasting Liquidity Using eXternal signals)

FLUX is a data-driven simulation engine designed to model and forecast the short-term behavior of commodity markets—starting with oil. By leveraging external macroeconomic signals such as currency strength, geopolitical risk, and inventory levels, FLUX goes beyond traditional static models to capture real-world market dynamics. It combines Monte Carlo simulation with adaptive volatility modeling to provide probabilistic insights into price trajectories, enabling better risk assessment, scenario planning, and decision-making for traders, analysts, and energy sector stakeholders.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run

```bash
python main.py
``` 