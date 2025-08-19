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
        while not self.stop_flag:
            t += 1
            vals = {
                "Room 1: NH3":          5 + 3*math.sin(t/30) + max(0, 10*math.sin(t/12)),
                "Room 2: CO":           12 + 8*math.sin(t/25) + max(0, 20*math.sin(t/15)),
                "Room 3: O2":           20.7 + 0.2*math.sin(t/35),
                "Room 12: Ethanol":     250 + 150*math.sin(t/40) + 80,
                "Production 1: O2":     20.8 + 0.12*math.sin(t/28),
                "Production 1: CH4":    max(0, 5*math.sin(t/10)) + 4,
                "Production 2: O2":     20.9 + 0.1*math.sin(t/31),
                "Production 2: H2S":    max(0, 4*math.sin(t/9)) + 1,
            }
            for k, v in vals.items():
                push_point(k, v)
            time.sleep(self.dt)

# start simulator once
if "sim_started" not in st.session_state:
    st.session_state.sim_thread = Simulator(interval=1.0)
    st.session_state.sim_thread.start()
    st.session_state.sim_started = True

# ----------- UI -----------

with st.sidebar:
    st.header("Settings")
    st.caption("Optional: point to Modbus TCP (else simulator is used).")
    host = st.text_input("Modbus Host", value="")
    port = st.number_input("Modbus Port", value=502, step=1)
    unit = st.number_input("Unit ID", value=1, step=1)
    st.write("---")
    st.caption("Refresh rate")
    refresh = st.slider("Seconds", 1, 10, 2)

st.title("Pharma Safety HMI — AI First")

# Facility view or room view state
view = st.session_state.get("view", "facility")
room = st.session_state.get("room", None)

def set_view(v, r=None):
    st.session_state.view = v
    st.session_state.room = r

def gas_from_label(key: str) -> str:
    k = key.lower()
    if "o2" in k: return "O2"
    if "h2s" in k: return "H2S"
    if "ch4" in k or "lel" in k or "methane" in k: return "CH4"
    if "nh3" in k or "ammonia" in k: return "NH3"
    if "ethanol" in k: return "Ethanol"
    if "co" in k and "co2" not in k: return "CO"
    return "CO"

def status_for_value(gas, val):
    thr = DEFAULT_THR.get(gas, {"mode": "high", "warn": 0, "alarm": 0, "units": ""})
    if thr["mode"] == "low":
        if val <= thr["alarm"]: return "ALARM"
        if val <= thr["warn"]:  return "WARN"
        return "HEALTHY"
    else:
        if val >= thr["alarm"]: return "ALARM"
        if val >= thr["warn"]:  return "WARN"
        return "HEALTHY"

# ---------- Facility ----------
if view == "facility":
    colL, colR = st.columns([3, 2], gap="large")

    with colL:
        # Plotly image background with clickable rectangles
        w, h = FACILITY_SIZE
        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=ensure_facility_image(),  # robust loader
                x=0, y=h, sizex=w, sizey=h, xref="x", yref="y",
                sizing="stretch", layer="below"
            )
        )
        # Draw room rectangles and labels
        scatter_x, scatter_y, texts, custom = [], [], [], []
        for rn, (x0, y0, x1, y1) in ROOM_RECTS.items():
            fig.add_shape(
                type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="#111827"), fillcolor="rgba(0,0,0,0.25)"
            )
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            scatter_x.append(cx); scatter_y.append(cy)
            # status rollup (worst detector)
            dets = ROOM_DETECTORS.get(rn, [])
            worst = "HEALTHY"
            for dkey in dets:
                live = st.session_state.latest.get(dkey, (0, np.nan))[1]
                if not np.isnan(live):
                    gas = gas_from_label(dkey)
                    s = status_for_value(gas, live)
                    if s == "ALARM": worst = "ALARM"; break
                    if s == "WARN" and worst != "ALARM": worst = "WARN"
            texts.append(f"{rn}\n{worst}")
            custom.append(rn)

        fig.add_trace(go.Scatter(
            x=scatter_x, y=scatter_y, mode="markers+text",
            text=texts, textposition="middle center",
            marker=dict(size=20, color="#22c55e"),
            customdata=custom,
            hoverinfo="text"
        ))
        fig.update_xaxes(visible=False, range=[0, w])
        fig.update_yaxes(visible=False, range=[h, 0])
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))

        clicked = plotly_events(
            fig, click_event=True, hover_event=False, select_event=False,
            override_height=600, override_width=None, key="fac_click"
        )
        if clicked:
            rn = clicked[0]["customdata"]
            set_view("room", rn)
            st.experimental_rerun()

        st.caption("Click a room to open details. Status rollup = worst detector in the room.")

    with colR:
        st.subheader("AI Safety Assistant — Facility")
        parts = []
        for rn in ROOMS:
            dets = ROOM_DETECTORS.get(rn, [])
            worst = "HEALTHY"
            for dkey in dets:
                live = st.session_state.latest.get(dkey, (0, np.nan))[1]
                if not np.isnan(live):
                    gas = gas_from_label(dkey)
                    s = status_for_value(gas, live)
                    if s == "ALARM": worst = "ALARM"; break
                    if s == "WARN" and worst != "ALARM": worst = "WARN"
            parts.append(f"**{rn}**: {worst}")
        st.markdown(" • ".join(parts))
        st.write("---")
        st.caption("Legend: Green=Healthy • Orange=Warn • Red=Alarm")

    st.autorefresh(interval=refresh * 1000, key="fac_refresh")

# ---------- Room ----------
if view == "room" and room:
    st.markdown(f"### ← [{room}](#)")
    if st.button("Back to Facility"):
        set_view("facility")
        st.experimental_rerun()

    colA, colB = st.columns([2, 1], gap="large")

    with colA:
        img_path = ensure_room_image_path(room)  # robust ensure
        st.image(img_path, use_container_width=True, caption=f"{room} — realistic view")

        st.markdown("#### Detectors")
        for key in ROOM_DETECTORS.get(room, []):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.write(key)
            with c2:
                df = get_series(key)
                if df.empty:
                    st.line_chart(pd.DataFrame({"value": []}))
                else:
                    chart = pd.DataFrame({"value": df["value"].tail(180)})
                    st.line_chart(chart)
            with c3:
                gas = gas_from_label(key)
                thr = DEFAULT_THR[gas]
                live = st.session_state.latest.get(key, (0, np.nan))[1]
                if np.isnan(live):
                    st.info("No data yet")
                else:
                    if thr["mode"] == "low":
                        if live <= thr["alarm"]:
                            st.error(f"ALARM • {live:.2f}{thr['units']}")
                        elif live <= thr["warn"]:
                            st.warning(f"WARN • {live:.2f}{thr['units']}")
                        else:
                            st.success(f"Healthy • {live:.2f}{thr['units']}")
                    else:
                        if live >= thr["alarm"]:
                            st.error(f"ALARM • {live:.2f}{thr['units']}")
                        elif live >= thr["warn"]:
                            st.warning(f"WARN • {live:.2f}{thr['units']}")
                        else:
                            st.success(f"Healthy • {live:.2f}{thr['units']}")

                if st.button("Simulate Spike", key=f"spike_{key}"):
                    base = st.session_state.latest.get(key, (time.time(), 0.0))[1]
                    for i in range(8):
                        push_point(key, base + 5.0 + i * 0.5)
                    st.toast("Spike injected", icon="⚡")

    with colB:
        st.subheader("AI Safety Assistant")
        worst = ("HEALTHY", "No unusual readings.")
        for key in ROOM_DETECTORS.get(room, []):
            live = st.session_state.latest.get(key, (0, np.nan))[1]
            if np.isnan(live):
                continue
            gas = gas_from_label(key)
            thr = DEFAULT_THR[gas]
            if thr["mode"] == "low":
                if live <= thr["alarm"]:
                    worst = ("ALARM", f"{gas} critically low at {live:.2f}{thr['units']}. Evacuate and ventilate.")
                    break
                elif live <= thr["warn"] and worst[0] != "ALARM":
                    worst = ("WARN", f"{gas} trending low ({live:.2f}{thr['units']}). Investigate sources/consumption.")
            else:
                if live >= thr["alarm"]:
                    worst = ("ALARM", f"{gas} high at {live:.2f}{thr['units']}. Evacuate & isolate source.")
                    break
                elif live >= thr["warn"] and worst[0] != "ALARM":
                    worst = ("WARN", f"{gas} elevated ({live:.2f}{thr['units']}). Increase ventilation, check for leaks.")

        st.write(f"**Status:** {worst[0]}")
        st.write(f"**Advice:** {worst[1]}")
        st.info("Tip: use **Simulate Spike** to demo alarms changing the AI guidance.")

    st.autorefresh(interval=refresh * 1000, key=f"room_refresh_{room}")
