# Pharma Safety HMI — AI First (Streamlit)

Pitch‑ready, single‑repo Streamlit app with:
- 2.5D facility view (Plotly) with clickable rooms
- Room pages: detector list, live trend, and an AI Safety Assistant panel
- Demo controls to simulate spikes
- Optional Modbus TCP polling path (UI unchanged; simulator by default)
- Auto‑generated placeholder images on first run (swap later with your real renders)

## Layout
```
your-repo/
├─ app.py
├─ requirements.txt
└─ assets/
   ├─ facility.png            # auto-created on first run if missing
   ├─ room_1.png              # "
   ├─ room_2.png              # "
   ├─ room_3.png              # "
   ├─ room_12.png             # "
   ├─ production_1.png        # "
   └─ production_2.png        # "
```
Rooms/gases default map:
- Room 1 → NH₃
- Room 2 → CO
- Room 3 → O₂
- Room 12 → Ethanol
- Production 1 → O₂, CH₄
- Production 2 → O₂, H₂S

## Quickstart
```bash
git clone <this-repo-url>
cd this-repo
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
Open the local URL Streamlit prints (typically http://localhost:8501).

## Streamlit Cloud
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo, pick `app.py` as the entrypoint.
3. Set Python version (3.11+) and let it install from `requirements.txt`.

## Modbus (optional)
When you're ready to connect to your TPPR via Modbus TCP, we'll add a small `pymodbus` poller that feeds the same in‑memory buffers used by the simulator. No UI changes required.

## Notes
- Placeholder images are generated with Pillow on first run if missing.
- Use **Simulate Spike** on a detector to demo alarm transitions and guidance.
