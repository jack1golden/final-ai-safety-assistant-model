
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="OBW AI Safety Assistant — 2.5D", layout="wide", page_icon="🧪")

ROOM_DISPLAY = {
    "entry": "Entry",
    "room1": "Room 1",
    "room2": "Room 2",
    "room3": "Room 3",
    "room12": "Room 12",
    "prod1": "Production 1",
    "prod2": "Production 2",
}

COLOR = {
    "healthy": "#10b981",
    "inhibit": "#f59e0b",
    "alarm":   "#ef4444",
    "fault":   "#fb923c",
    "over":    "#e5e7eb",
}

ROOMS_POLY = {
    "entry":  [(80, 600), (300, 600), (300, 740), (80, 740)],
    "prod1":  [(340, 80), (520, 80), (520, 260), (340, 260)],
    "room12": [(540, 80), (820, 80), (820, 260), (540, 260)],
    "room1":  [(860, 80), (1120, 80), (1120, 260), (860, 260)],
    "prod2":  [(340, 280), (520, 280), (520, 460), (340, 460)],
    "room3":  [(540, 280), (820, 280), (820, 460), (540, 460)],
    "room2":  [(860, 280), (1120, 280), (1120, 460), (860, 460)],
}

def _init_state():
    if "data" not in st.session_state:
        st.session_state.data = pd.read_csv("demo_data.csv", parse_dates=["timestamp"])
    st.session_state.setdefault("view","facility")
    st.session_state.setdefault("room_key", None)
    st.session_state.setdefault("detector_key", None)
    st.session_state.setdefault("demo_index", 0)
    st.session_state.setdefault("dev", False)
    st.session_state.setdefault("wireframe", False)
    st.session_state.setdefault("last_event", None)
    st.session_state.setdefault("evac_route_enabled", False)

_init_state()

def _room_centers_from_polys(rooms_poly: dict) -> dict:
    centers = {}
    for rk, poly in rooms_poly.items():
        xs = [float(p[0]) for p in poly]; ys = [float(p[1]) for p in poly]
        centers[rk] = (sum(xs)/len(xs), sum(ys)/len(ys))
    return centers

def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def resolve_room_click(payload: dict, rooms_poly: dict) -> str | None:
    if not isinstance(payload, dict):
        return None
    cd = payload.get("customdata")
    if isinstance(cd, str) and cd in rooms_poly:
        return cd
    x = _safe_float(payload.get("x")); y = _safe_float(payload.get("y"))
    if x is None or y is None:
        return None
    centers = _room_centers_from_polys(rooms_poly)
    best, bestd = None, float("inf")
    for rk,(cx,cy) in centers.items():
        d=(cx-x)**2+(cy-y)**2
        if d<bestd:
            bestd, best = d, rk
    return best

THRESH = {
    "prod1":  {"warn": 35.0, "danger": 45.0, "oxygen_mode": False},
    "prod2":  {"warn": 35.0, "danger": 45.0, "oxygen_mode": False},
    "room1":  {"warn": 30.0, "danger": 40.0, "oxygen_mode": False},
    "room2":  {"warn": 30.0, "danger": 40.0, "oxygen_mode": False},
    "room3":  {"warn": 30.0, "danger": 40.0, "oxygen_mode": False},
    "room12": {"warn": 30.0, "danger": 40.0, "oxygen_mode": False},
    "entry":  {"warn": 30.0, "danger": 40.0, "oxygen_mode": False},
}

def room_status(rk: str, idx: int) -> str:
    df = st.session_state.data
    sub = df[df["room"] == rk]
    if sub.empty:
        return "fault"
    local_idx = min(len(sub)-1, idx)
    val = float(sub.iloc[local_idx]["ppm"])
    thr = THRESH.get(rk, {"warn": 30.0, "danger": 40.0, "oxygen_mode": False})
    warn, danger, is_o2 = thr["warn"], thr["danger"], thr["oxygen_mode"]
    if is_o2:
        if val <= danger: return "alarm"
        if val <= warn:   return "inhibit"
        return "healthy"
    else:
        if val >= danger*1.15: return "over"
        if val >= danger:      return "alarm"
        if val >= warn:        return "inhibit"
        return "healthy"

def apply_simulation(room_key: str, mode: str, intensity: int, duration: int):
    df = st.session_state.data.copy()
    idx = st.session_state.demo_index
    sub_idx = df[df["room"]==room_key].index
    # choose duration rows from current point
    target = [i for i in sub_idx if i >= sub_idx.min()+idx][:duration]
    for j,row_i in enumerate(target):
        if mode=="Spike":
            df.loc[row_i,"ppm"]=float(df.loc[row_i,"ppm"])+float(intensity)
        elif mode=="Ramp":
            df.loc[row_i,"ppm"]=float(df.loc[row_i,"ppm"])+float(intensity)*(j+1)/max(1,duration)
        elif mode=="O₂ drop":
            df.loc[row_i,"ppm"]=float(df.loc[row_i,"ppm"])-float(intensity)*(j+1)/max(1,duration)
        elif mode=="CO spike":
            df.loc[row_i,"ppm"]=float(df.loc[row_i,"ppm"])+float(intensity)*0.4
    st.session_state.data = df

def safe_color(name_or_hex: str) -> str:
    # Ensure we never pass None/invalid to Plotly
    if not isinstance(name_or_hex, str) or not name_or_hex:
        return "#10b981"
    return name_or_hex

def build_facility():
    W,H = 1200,700
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False, range=[0,W]),
        yaxis=dict(visible=False, range=[H,0]),
        margin=dict(l=0,r=260,t=0,b=0),
        height=560,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
        dragmode=False,
        showlegend=False,
        images=[dict(source="assets/floorplan.png", xref="x", yref="y", x=0, y=0, sizex=W, sizey=H, sizing="stretch", opacity=1.0, layer="below")]
    )
    # Draw room overlays
    for rk, poly in ROOMS_POLY.items():
        status = room_status(rk, st.session_state.demo_index)
        fill = safe_color(COLOR.get(status, "#10b981"))
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor=fill,
            line=dict(color="#4b5563", width=2),
            hovertemplate=f"{ROOM_DISPLAY.get(rk,rk)}<extra></extra>",
            name=rk,
            customdata=[rk]*len(xs),
            showlegend=False,
        ))
        cx = sum(p[0] for p in poly)/len(poly)
        cy = sum(p[1] for p in poly)/len(poly)
        fig.add_annotation(x=cx, y=cy, text=f"<b>{ROOM_DISPLAY.get(rk,rk)}</b>", showarrow=False, font=dict(color="#e5e7eb", size=13))
    # Legend
    legend_items=[("Healthy","#10b981"),("Inhibit","#f59e0b"),("Alarm","#ef4444"),("Fault","#fb923c"),("Activated / Over Range","#e5e7eb")]
    y_cursor=80
    for label,color in legend_items:
        fig.add_shape(type="rect", x0=W+20, y0=y_cursor, x1=W+50, y1=y_cursor+20, fillcolor=safe_color(color), line=dict(color="#334155"))
        fig.add_annotation(x=W+60, y=y_cursor+10, text=label, showarrow=False, font=dict(color="#e5e7eb", size=12), xanchor="left", yanchor="middle")
        y_cursor+=30
    return fig

def render_facility():
    st.markdown("#### Facility Overview")
    col_map, col_side = st.columns([3,1])
    with col_map:
        fig = build_facility()
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="facility_click_2p5d", override_height=560)
        if clicked:
            payload = clicked[0] or {}
            st.session_state.last_event = payload
            rk = resolve_room_click(payload, ROOMS_POLY)
            if rk:
                st.session_state.room_key = rk
                st.session_state.view = "room"
    with col_side:
        st.markdown("### Rooms")
        for rk in ROOMS_POLY.keys():
            if st.button(ROOM_DISPLAY[rk], key=f"btn_{rk}"):
                st.session_state.room_key=rk
                st.session_state.view="room"
        st.toggle("Dev", key="dev")

def render_room():
    rk = st.session_state.room_key
    if rk is None or rk not in ROOMS_POLY:
        st.session_state.view="facility"; return
    st.markdown(f"### {ROOM_DISPLAY.get(rk,rk)} — Room")
    # Room image (placeholder using facility img cropped look)
    st.image("assets/floorplan.png", use_container_width=True, caption="Room view (placeholder)")
    # Detector button
    if st.button("Open Detector", key=f"det_{rk}"):
        st.session_state.view="detector"

    # Quick back
    if st.button("⬅️ Back to Facility", key=f"back_{rk}"):
        st.session_state.view="facility"

def render_detector():
    rk = st.session_state.room_key
    if rk is None:
        st.session_state.view="facility"; return
    st.markdown(f"### {ROOM_DISPLAY.get(rk,rk)} — Detector & AI")
    df = st.session_state.data
    sub = df[df["room"]==rk].iloc[:st.session_state.demo_index+1].copy()
    c1,c2,c3 = st.columns(3)
    if not sub.empty:
        # simple analytics
        last = float(sub["ppm"].iloc[-1])
        roll = sub["ppm"].tail(12).mean()
        std = sub["ppm"].tail(12).std() or 1.0
        z = (last-roll)/std
        slope = float(np.polyfit(np.arange(min(10,len(sub))), sub["ppm"].tail(10), 1)[0]) if len(sub)>=2 else 0.0
    else:
        last=0.0; z=0.0; slope=0.0
    c1.metric("Live", f"{last:.1f} ppm")
    c2.metric("Anomaly (z)", f"{z:.2f}")
    c3.metric("Slope Δppm/tick", f"{slope:.2f}")

    # chart
    fig = go.Figure()
    if not sub.empty:
        sub["roll"]=sub["ppm"].rolling(12, min_periods=3).mean()
        fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["ppm"], mode="lines+markers", name="Live", line=dict(width=3)))
        fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["roll"], mode="lines", name="Rolling mean", line=dict(dash="dot")))
    thr = THRESH.get(rk, {"warn":30.0, "danger":40.0, "oxygen_mode":False})
    warn, danger, is_o2 = thr["warn"], thr["danger"], thr["oxygen_mode"]
    if is_o2:
        fig.add_hrect(y0=-1e6, y1=danger, fillcolor=COLOR["alarm"], opacity=0.10, line_width=0)
        fig.add_hrect(y0=danger, y1=warn, fillcolor=COLOR["inhibit"], opacity=0.08, line_width=0)
    else:
        fig.add_hrect(y0=danger, y1=1e6, fillcolor=COLOR["alarm"], opacity=0.10, line_width=0)
        fig.add_hrect(y0=warn, y1=danger, fillcolor=COLOR["inhibit"], opacity=0.08, line_width=0)
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10,r=10,t=10,b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # back buttons
    b1,b2 = st.columns(2)
    if b1.button("⬅️ Room", key=f"det_back_room_{rk}"):
        st.session_state.view="room"
    if b2.button("⬅️ Facility", key=f"det_back_fac_{rk}"):
        st.session_state.view="facility"

def dev_sidebar():
    st.markdown("### Simulation Center")
    room_choice = st.selectbox("Room", list(ROOMS_POLY.keys()), key="sim_room")
    mode = st.selectbox("Mode", ["Spike","Ramp","O₂ drop","CO spike"], key="sim_mode")
    intensity = st.slider("Intensity", 5, 100, 50, 5, key="sim_intensity")
    duration = st.slider("Duration (ticks)", 3, 30, 10, key="sim_duration")
    if st.button("Run Simulation", key="sim_run"):
        apply_simulation(room_choice, mode, intensity, duration)
        st.toast(f"Simulated {mode} in {ROOM_DISPLAY.get(room_choice,room_choice)}")

# header
col1,col2,col3 = st.columns([1,3,1])
with col1: st.write("🧪")
with col2: st.markdown("<h2 style='text-align:center;margin-top:10px;'>Pharma Facility — 2.5D</h2>", unsafe_allow_html=True)
with col3:
    st.session_state.dev = st.checkbox("Dev", key="dev_hdr", value=st.session_state.get("dev", False))
    if st.session_state.dev:
        with st.sidebar:
            dev_sidebar()

# advance demo
if st.session_state.demo_index < len(st.session_state.data)-1:
    st.session_state.demo_index += 1

# route
if st.session_state.view=="facility":
    render_facility()
elif st.session_state.view=="room":
    render_room()
elif st.session_state.view=="detector":
    render_detector()
else:
    render_facility()
