import os, io, time, math, threading, base64, urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import streamlit.components.v1 as components

# ===================== CONFIG =====================
st.set_page_config(page_title="Pharma Safety HMI — Replica", layout="wide")
ASSETS = "assets"
os.makedirs(ASSETS, exist_ok=True)

ROOMS = ["Room 1","Room 2","Room 3","Room 12","Production 1","Production 2"]
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

# Hotspots (as percent of facility image)
ROOM_RECTS_PCT = {
    "Room 1":       (72.0,  6.0, 18.0, 13.0),
    "Room 2":       (60.0, 34.0, 16.0, 13.0),
    "Room 3":       (54.0, 13.0, 16.0, 13.0),
    "Room 12":      (46.0,  1.0, 16.0, 13.0),
    "Production 1": (27.0, 17.0, 18.0, 15.0),
    "Production 2": (27.0, 34.0, 18.0, 15.0),
}

# ===================== IMAGE GEN =====================
def _font(size=24):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def make_facility(size=(1400, 820)) -> Image.Image:
    W, H = size
    img = Image.new("RGB", size, (225, 231, 234))  # light bg like grass/wall
    d = ImageDraw.Draw(img)

    # Main building slab
    d.rounded_rectangle((60, 60, W-60, H-80), 28, fill=(212,219,223), outline=(170,180,188), width=2)
    # Two production halls (pink-ish floor)
    d.rounded_rectangle((W*0.40, H*0.18, W*0.88, H*0.76), 22, fill=(236,210,210), outline=(190,170,170), width=2)
    # Side corridors
    d.rectangle((80, H*0.72, W-80, H-100), fill=(205,210,214))
    d.rectangle((W*0.63, 120, W*0.66, H-100), fill=(205,210,214))

    # Rooms (match hotspots roughly)
    def room_box(pct):
        l,t,w,h = pct
        x0 = int(60 + (W-120) * (l/100.0))
        y0 = int(60 + (H-140) * (t/100.0))
        x1 = int(x0 + (W-120) * (w/100.0))
        y1 = int(y0 + (H-140) * (h/100.0))
        return (x0,y0,x1,y1)

    for name,p in ROOM_RECTS_PCT.items():
        x0,y0,x1,y1 = room_box(p)
        d.rounded_rectangle((x0,y0,x1,y1), 14, fill=(218,226,232), outline=(100,110,120), width=2)
        # tiny 3D shadows
        d.rectangle((x0+6,y1-10,x1-6,y1-6), fill=(180,184,188))

    # Small equipment blocks
    for bx in range(6):
        x = int(W*0.47 + bx*90); y = int(H*0.32)
        d.rounded_rectangle((x,y,x+70,y+40), 8, fill=(118,180,118), outline=(70,110,70), width=1)
    for bx in range(4):
        x = int(W*0.47 + bx*100); y = int(H*0.56)
        d.rounded_rectangle((x,y,x+80,y+50), 8, fill=(118,180,118), outline=(70,110,70), width=1)

    # Legend box
    legend = Image.new("RGB", (220,120), (245,248,250))
    ld = ImageDraw.Draw(legend)
    ld.rectangle((0,0,219,119), outline=(180,190,200), width=1)
    ld.text((10,6), "Legend", fill=(20,28,38), font=_font(18))
    items = [("#16a34a","Healthy"),("#f59e0b","Inhibit"),("#ef4444","Alarm"),("#64748b","Fault"),("#0ea5e9","Activated")]
    for i,(col,label) in enumerate(items):
        y = 34 + i*16
        ld.rectangle((12,y,24,y+12), fill=col)
        ld.text((34,y-2), label, fill=(20,28,38), font=_font(14))
    img.paste(legend, (W-260, H-200))
    return img

def make_room(name: str, size=(1200, 700)) -> Image.Image:
    W,H = size
    img = Image.new("RGB", size, (214,224,230))
    d = ImageDraw.Draw(img)
    # Walls
    d.rounded_rectangle((10,10,W-10,H-10), 24, fill=(224,232,238), outline=(150,160,170), width=2)
    # Floor
    d.rounded_rectangle((40,120,W-40,H-40), 18, fill=(206,214,222), outline=(170,178,186), width=2)
    # Objects
    blocks = [(120,340,260,60),(420,340,260,60),(720,340,260,60),
              (160,230,100,90),(520,220,140,110),(860,220,120,130)]
    cols = [(90,180,250),(140,150,240),(110,200,140),(240,170,80),(230,110,110)]
    for i,(x,y,w,h) in enumerate(blocks):
        d.rounded_rectangle((x,y,x+w,y+h), 12, fill=cols[i%len(cols)], outline=(60,70,80), width=2)
    # Door sign
    d.rounded_rectangle((40,40,260,90), 8, fill=(32,43,60))
    d.text((60,55), name, fill=(235,240,246), font=_font(24))
    return img

def ensure_facility_bytes() -> bytes:
    path = os.path.join(ASSETS, "facility.png")
    if not os.path.exists(path):
        make_facility().save(path)
    with open(path, "rb") as f:
        return f.read()

def ensure_room_bytes(room: str) -> bytes:
    fn = f"{room.replace(' ','_').lower()}.png"
    path = os.path.join(ASSETS, fn)
    if not os.path.exists(path):
        make_room(room).save(path)
    with open(path, "rb") as f:
        return f.read()

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

# ===================== DATA + SIM =====================
if "buffers" not in st.session_state:
    st.session_state.buffers = {}
if "latest" not in st.session_state:
    st.session_state.latest = {}

def push_point(key: str, val: float):
    ts = time.time()
    buf = st.session_state.buffers.setdefault(key, [])
    buf.append((ts, float(val)))
    if len(buf) > 1800:
        del buf[:len(buf)-1800]
    st.session_state.latest[key] = (ts, float(val))

def get_series(key: str) -> pd.DataFrame:
    buf = st.session_state.buffers.get(key, [])
    if not buf:
        return pd.DataFrame(columns=["ts","value"])
    ts, vs = zip(*buf)
    return pd.DataFrame({"ts": ts, "value": vs})

class Simulator(threading.Thread):
    def __init__(self, dt=1.0):
        super().__init__(daemon=True)
        self.dt = dt; self.stop_flag = False
    def run(self):
        t=0
        while not self.stop_flag:
            t+=1
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
            for k,v in vals.items():
                push_point(k, v)
            time.sleep(self.dt)

if "sim" not in st.session_state:
    st.session_state.sim = Simulator(1.0); st.session_state.sim.start()

# ===================== HELPERS =====================
def auto_refresh(ms):
    st.markdown(f"<script>setTimeout(()=>window.location.reload(), {int(ms)});</script>", unsafe_allow_html=True)

def gas_from_label(key: str) -> str:
    k = key.lower()
    if "o2" in k: return "O2"
    if "h2s" in k: return "H2S"
    if "ch4" in k or "lel" in k or "methane" in k: return "CH4"
    if "nh3" in k or "ammonia" in k: return "NH3"
    if "ethanol" in k: return "Ethanol"
    if "co" in k and "co2" not in k: return "CO"
    return "CO"

def status_for_value(gas: str, val: float):
    thr = DEFAULT_THR.get(gas, {"mode":"high","warn":0,"alarm":0})
    if thr["mode"] == "low":
        if val <= thr["alarm"]: return "ALARM"
        if val <= thr["warn"]:  return "WARN"
        return "HEALTHY"
    else:
        if val >= thr["alarm"]: return "ALARM"
        if val >= thr["warn"]:  return "WARN"
        return "HEALTHY"

# ===================== TOP BAR =====================
def topbar(active_room: str|None):
    tabs = ["Entry","Room 1","Room 2","Room 3","Room 12","Production 1","Production 2"]
    links = []
    for t in tabs:
        if t=="Entry":
            url = "?view=facility"
        else:
            url = f"?view=room&room={urllib.parse.quote(t)}"
        cls = "tab active" if t==active_room else "tab"
        links.append(f'<a class="{cls}" href="{url}">{t}</a>')
    html = f"""
    <style>
      .topbar {{ background:#111827; color:#e5e7eb; padding:10px 14px; border-radius:8px; }}
      .brand {{ font-weight:700; margin-right:16px; display:inline-block; }}
      .tabs {{ display:inline-block; }}
      .tab {{ color:#e5e7eb; text-decoration:none; padding:6px 10px; margin:0 4px; border:1px solid #334155; border-radius:6px; }}
      .tab:hover {{ background:#1f2937; }}
      .active {{ background:#374151; }}
      .log {{ float:right; font-size:12px; color:#cbd5e1; }}
    </style>
    <div class="topbar">
      <span class="brand">OBW • Pharma HMI</span>
      <span class="tabs">{''.join(links)}</span>
      <span class="log">TPPR/OPC: Communication Error (simulated)</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ===================== ROUTING =====================
params = st.experimental_get_query_params()
view = params.get("view", ["facility"])[0]
room = params.get("room", [None])[0]

with st.sidebar:
    refresh = st.slider("Auto-refresh (s)", 1, 10, 3)

# ===================== FACILITY PAGE =====================
if view == "facility":
    topbar(None)
    st.title("Facility Overview")
    data = ensure_facility_bytes()
    datauri = "data:image/png;base64," + base64.b64encode(data).decode("ascii")

    # Build hotspots
    hs = []
    for rn,(l,t,w,h) in ROOM_RECTS_PCT.items():
        url = f"?view=room&room={urllib.parse.quote(rn)}"
        hs.append(f'<a class="hotspot" href="{url}" style="left:{l}%;top:{t}%;width:{w}%;height:{h}%"><span>{rn}</span></a>')

    html = f"""
    <style>
      .wrap {{ position:relative; width:100%; max-width:1400px; margin:0 auto; border-radius:16px; overflow:hidden; box-shadow:0 10px 30px rgba(0,0,0,.25);}}
      .wrap img {{ width:100%; height:auto; display:block; }}
      .hotspot {{ position:absolute; border:2px solid rgba(255,255,255,.8); border-radius:10px; background:rgba(0,0,0,.05); }}
      .hotspot:hover {{ box-shadow:0 0 0 4px rgba(34,197,94,.25) inset; border-color:#22c55e; }}
      .hotspot span {{ position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); color:#fff; font-weight:600; text-shadow:0 1px 2px rgba(0,0,0,.6); font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif; font-size:14px;}}
      .legend {{ margin-top:8px; float:right; background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:8px 12px; }}
      .dot {{ display:inline-block; width:12px; height:12px; margin-right:6px; border-radius:2px; }}
    </style>
    <div class="wrap">
      <img src="{datauri}" alt="facility"/>
      {''.join(hs)}
    </div>
    <div class="legend">
      <div><span class="dot" style="background:#16a34a"></span>Healthy</div>
      <div><span class="dot" style="background:#f59e0b"></span>Inhibit</div>
      <div><span class="dot" style="background:#ef4444"></span>Alarm</div>
      <div><span class="dot" style="background:#64748b"></span>Fault</div>
      <div><span class="dot" style="background:#0ea5e9"></span>Activated / Over Range</div>
    </div>
    """
    components.html(html, height=740, scrolling=False)
    auto_refresh(refresh*1000)

# ===================== ROOM PAGE =====================
if view == "room" and room:
    topbar(room)
    st.button("← Back", on_click=lambda: st.experimental_set_query_params(view="facility"))
    st.subheader(room)

    # Compose dynamic room image with a "value tile" overlay
    base_img = Image.open(io.BytesIO(ensure_room_bytes(room))).convert("RGB")
    draw = ImageDraw.Draw(base_img)
    # Draw a detector tile (yellow) like screenshot
    tile = (base_img.width//2 - 80, 120, base_img.width//2 + 80, 170)
    draw.rounded_rectangle(tile, 8, fill=(245, 205, 80), outline=(150,120,30), width=2)
    # Show primary detector readout (first in list)
    main_key = ROOM_DETECTORS.get(room, [""])[0]
    val = st.session_state.latest.get(main_key, (0, float("nan")))[1]
    text = f"{main_key.split(':')[-1].strip()}: {val:.1f}" if not np.isnan(val) else f"{main_key.split(':')[-1].strip()}: --"
    try:
        fnt = ImageFont.truetype("arial.ttf", 20)
    except:
        fnt = ImageFont.load_default()
    draw.text((tile[0]+10, tile[1]+12), text, fill=(20,20,20), font=fnt)

    # Render
    buf = io.BytesIO(); base_img.save(buf, format="PNG"); st.image(buf.getvalue(), use_container_width=True)

    st.markdown("### Detectors")
    for key in ROOM_DETECTORS.get(room, []):
        c1, c2, c3 = st.columns([3, 3, 2])
        with c1:
            st.write(key)
        with c2:
            df = get_series(key)
            if df.empty:
                st.line_chart(pd.DataFrame({"value":[]}))  # Altair-based
            else:
                st.line_chart(pd.DataFrame({"value": df["value"].tail(180)}))
        with c3:
            gas = key.split(":")[-1].strip()
            if gas not in DEFAULT_THR:  # basic parse fallback
                gas = "CO"
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
                base_v = st.session_state.latest.get(key, (time.time(), 0.0))[1]
                for i in range(8):
                    push_point(key, base_v + 5.0 + i*0.5)
                st.toast("Spike injected", icon="⚡")

    auto_refresh(refresh*1000)
