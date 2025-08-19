import os, time, threading, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ----------- CONFIG -----------
st.set_page_config(page_title="Pharma Safety HMI — AI First", layout="wide")

ROOMS = [
    "Room 1", "Room 2", "Room 3", "Room 12",
    "Production 1", "Production 2"
]
ROOM_DETECTORS = {
    "Room 1": ["Room 1: NH3"],
    "Room 2": ["Room 2: CO"],
    "Room 3": ["Room 3: O2"],
    "Room 12": ["Room 12: Ethanol"],
    "Production 1": ["Production 1: O2", "Production 1: CH4"],
    "Production 2": ["Production 2: O2", "Production 2: H2S"],
}
DEFAULT_THR = {
    "O2":      {"mode":"low",  "warn":19.5, "alarm":18.0, "units":"%vol"},
    "CO":      {"mode":"high", "warn":35.0, "alarm":50.0, "units":"ppm"},
    "H2S":     {"mode":"high", "warn":10.0, "alarm":15.0, "units":"ppm"},
    "CH4":     {"mode":"high", "warn":10.0, "alarm":20.0, "units":"%LEL"},
    "NH3":     {"mode":"high", "warn":25.0, "alarm":35.0, "units":"ppm"},
    "Ethanol": {"mode":"high", "warn":300.0, "alarm":500.0, "units":"ppm"},
}
FACILITY_SIZE = (1400, 800)   # width, height for plotting canvas

# Clickable room rectangles (approx coords in facility canvas)
ROOM_RECTS = {
    "Room 1":        (100, 120, 360, 270),
    "Room 2":        (400, 120, 660, 270),
    "Room 3":        (700, 120, 960, 270),
    "Room 12":       (1000, 120, 1260, 270),
    "Production 1":  (250, 360, 700, 650),
    "Production 2":  (780, 360, 1230, 650),
}

# Evac graph (optional visual)
NODES = {"ENTRY": (80, 740), "C1": (560, 320), "C2": (900, 320), "EXIT_W": (40, 420), "EXIT_E": (1360, 420)}
EDGES = [("ENTRY","C1"),("C1","C2"),("C2","EXIT_E"),("C1","EXIT_W")]

# ----------- IMAGE HELPERS (robust) -----------
ASSETS = "assets"
os.makedirs(ASSETS, exist_ok=True)

def have_image(name: str) -> bool:
    """Return True only if a real, valid image file exists."""
    path = os.path.join(ASSETS, name)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()  # validate header/structure
        return True
    except Exception:
        return False

def mk_facility():
    W, H = FACILITY_SIZE
    img = Image.new("RGB", (W, H), (18, 26, 38))
    d = ImageDraw.Draw(img)

    def rr(box, r, fill, outline, w):
        d.rounded_rectangle(box, r, fill=fill, outline=outline, width=w)

    rr((10, 10, W - 10, H - 10), 30, (30, 41, 59), (71, 85, 105), 6)
    d.rectangle((80, 100, W - 80, 160), fill=(41, 54, 78))
    d.rectangle((W // 2 - 20, 160, W // 2 + 20, H - 100), fill=(41, 54, 78))
    for box in ROOM_RECTS.values():
        rr(box, 18, (31, 41, 55), (90, 104, 120), 3)

    vign = Image.new("L", (W, H), 0)
    ImageDraw.Draw(vign).ellipse((-250, -150, W + 250, H + 200), fill=180)
    img = Image.composite(img, Image.new("RGB", (W, H), (18, 26, 38)), vign.filter(ImageFilter.GaussianBlur(80)))
    img.save(os.path.join(ASSETS, "facility.png"))

def mk_room(name: str):
    img = Image.new("RGB", (1000, 600), (24, 33, 46))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((15, 15, 985, 585), 24, fill=(32, 43, 60), outline=(70, 84, 100), width=3)
    floor = Image.new("L", (1000, 600), 0)
    ImageDraw.Draw(floor).rectangle((30, 300, 970, 560), fill=180)
    img = Image.composite(Image.new("RGB", (1000, 600), (29, 38, 53)), img, floor.filter(ImageFilter.GaussianBlur(60)))
    # props
    blocks = [
        (120, 360, 250, 60), (420, 360, 250, 60), (720, 360, 250, 60),
        (170, 260, 80, 120), (500, 240, 120, 140), (840, 240, 100, 160)
    ]
    cols = [(56,189,248),(99,102,241),(34,197,94),(249,115,22),(239,68,68)]
    for i, (x, y, w, h) in enumerate(blocks):
        d.rounded_rectangle((x, y, x + w, y + h), 12, fill=cols[i % len(cols)], outline=(20, 20, 30), width=2)
    try:
        fnt = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        fnt = None
    d.text((40, 30), name, fill=(220, 230, 240), font=fnt)
    img = img.filter(ImageFilter.GaussianBlur(0.3))
    out = os.path.join(ASSETS, f"{name.replace(' ','_').lower()}.png")
    img.save(out)

def ensure_facility_image() -> Image.Image:
    """Return a valid facility PIL image, regenerating if needed."""
    if not have_image("facility.png"):
        mk_facility()
    try:
        return Image.open(os.path.join(ASSETS, "facility.png"))
    except Exception:
        mk_facility()
        return Image.open(os.path.join(ASSETS, "facility.png"))

def ensure_room_image_path(room_name: str) -> str:
    """Return a valid path for the room image, regenerating if needed."""
    fname = f"{room_name.replace(' ','_').lower()}.png"
    if not have_image(fname):
        mk_room(room_name)
    path = os.path.join(ASSETS, fname)
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception:
        mk_room(room_name)
    return path

# ----------- DATA BUFFERS -----------
if "buffers" not in st.session_state:
    st.session_state.buffers = {}  # key -> list[(ts,val)]
if "latest" not in st.session_state:
    st.session_state.latest = {}   # key -> (ts,val)

def push_point(key: str, val: float):
    ts = time.time()
    buf = st.session_state.buffers.setdefault(key, [])
    buf.append((ts, float(val)))
    # keep last ~30 minutes @1Hz
    if len(buf) > 1800:
        del buf[:len(buf) - 1800]
    st.session_state.latest[key] = (ts, float(val))

def get_series(key: str) -> pd.DataFrame:
    buf = st.session_state.buffers.get(key, [])
    if not buf:
        return pd.DataFrame(columns=["ts", "value"])
    ts, vs = zip(*buf)
    return pd.DataFrame({"ts": ts, "value": vs})

# simulator thread (optional if no Modbus)
class Simulator(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.stop_flag = False
        self.dt = interval
    def run(self):
        t = 0
        while no
