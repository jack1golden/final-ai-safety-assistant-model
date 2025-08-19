
# Pharma Facility — 2.5D Interactive (Buttons Overlay)

- Facility uses a 2.5D image background with clickable overlay "buttons" per room.
- Room view uses a faux‑3D image with a detector button.
- AI analytics, thresholds, and Simulation Center are included.
- Renamed evac flag to `evac_route_enabled` to avoid state conflicts.

## Run
pip install -r requirements.txt
streamlit run tppr_ai_demo_app.py
