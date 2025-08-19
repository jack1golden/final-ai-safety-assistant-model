# Pharma Safety HMI — Replica (No Plotly)

A self-contained Streamlit app that visually replicates the OBW/Atvise HMI feel:
- Facility overview with clickable room hotspots (pure HTML/CSS)
- Room pages with dynamic detector tile and trends
- Auto-generated 2.5D-style backgrounds (no external server assets)
- Simulator for live readings + spike demo

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
You can replace `assets/facility.png` and each `assets/<room>.png` with your own renders at any time.
