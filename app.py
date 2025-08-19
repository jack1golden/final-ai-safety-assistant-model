import os, time, math, threading
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from PIL import Image, ImageDraw

st.set_page_config(page_title="Pharma HMI — OBW Skin", layout="wide")

# Rooms
ROOMS = ["Room 1","Room 2","Room 3","Room 12","Production 1","Production 2"]
ROOM_DETECTORS = {
    "Room 1":["Room 1: NH3"],
    "Room 2":["Room 2: CO"],
    "Room 3":["Room 3: O2"],
    "Room 12":["Room 12: Ethanol"],
    "Production 1":["Production 1: O2","Production 1: CH4"],
    "Production 2":["Production 2: O2","Production 2: H2S"],
}
ASSETS = "assets"

# Facility clickable rects
FACILITY_SIZE=(1200,700)
ROOM_RECTS = {
    "Room 1":(100,100,250,200),
    "Room 2":(300,100,450,200),
    "Room 3":(500,100,650,200),
    "Room 12":(700,100,850,200),
    "Production 1":(200,300,500,500),
    "Production 2":(600,300,900,500),
}

# Buffers
if "buffers" not in st.session_state:
    st.session_state.buffers={}
if "latest" not in st.session_state:
    st.session_state.latest={}

def push_point(k,v):
    ts=time.time()
    buf=st.session_state.buffers.setdefault(k,[])
    buf.append((ts,v))
    if len(buf)>1800: del buf[:len(buf)-1800]
    st.session_state.latest[k]=(ts,v)

# Simulator
class Simulator(threading.Thread):
    def __init__(self): super().__init__(daemon=True); self.stop=False
    def run(self):
        t=0
        while not self.stop:
            t+=1
            vals={
                "Room 1: NH3":10+5*math.sin(t/10),
                "Room 2: CO":20+10*math.sin(t/15),
                "Room 3: O2":20.8+0.1*math.sin(t/20),
                "Room 12: Ethanol":200+50*math.sin(t/18),
                "Production 1: O2":20.9+0.1*math.sin(t/22),
                "Production 1: CH4":5+2*math.sin(t/8),
                "Production 2: O2":20.7+0.1*math.sin(t/17),
                "Production 2: H2S":3+1.5*math.sin(t/7),
            }
            for k,v in vals.items(): push_point(k,v)
            time.sleep(1)

if "sim" not in st.session_state:
    st.session_state.sim=Simulator(); st.session_state.sim.start()

# Helpers
def auto_refresh(ms): st.markdown(f"<script>setTimeout(()=>window.location.reload(),{ms});</script>",unsafe_allow_html=True)
def get_series(k):
    buf=st.session_state.buffers.get(k,[])
    if not buf: return pd.DataFrame(columns=["ts","value"])
    ts,vs=zip(*buf)
    return pd.DataFrame({"ts":ts,"value":vs})

def status(val,warn,alarm,mode="high"):
    if mode=="low":
        if val<=alarm: return "ALARM","red"
        if val<=warn: return "WARN","orange"
        return "HEALTHY","green"
    else:
        if val>=alarm: return "ALARM","red"
        if val>=warn: return "WARN","orange"
        return "HEALTHY","green"

# UI
refresh=st.sidebar.slider("Refresh (s)",1,10,3)

view=st.session_state.get("view","facility")
room=st.session_state.get("room",None)
def set_view(v,r=None): st.session_state.view=v; st.session_state.room=r

if view=="facility":
    st.title("Facility Overview")
    fig=go.Figure()
    # background facility image
    fac_path=os.path.join(ASSETS,"facility.jpg")
    if os.path.exists(fac_path):
        img=Image.open(fac_path)
    else:
        img=Image.new("RGB",FACILITY_SIZE,(200,200,200)); ImageDraw.Draw(img).text((20,20),"FACILITY",(0,0,0))
    fig.add_layout_image(dict(source=img,x=0,y=FACILITY_SIZE[1],sizex=FACILITY_SIZE[0],sizey=FACILITY_SIZE[1],xref="x",yref="y",sizing="stretch",layer="below"))
    # clickable rooms
    xs,ys,texts,custom=[],[],[],[]
    for rn,(x0,y0,x1,y1) in ROOM_RECTS.items():
        fig.add_shape(type="rect",x0=x0,y0=y0,x1=x1,y1=y1,line=dict(color="black"),fillcolor="rgba(0,0,0,0.2)")
        xs.append((x0+x1)/2); ys.append((y0+y1)/2); texts.append(rn); custom.append(rn)
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="markers+text",text=texts,textposition="middle center",customdata=custom,marker=dict(size=20,color="green")))
    fig.update_xaxes(visible=False,range=[0,FACILITY_SIZE[0]])
    fig.update_yaxes(visible=False,range=[FACILITY_SIZE[1],0])
    fig.update_layout(height=600,margin=dict(l=0,r=0,t=0,b=0))
    click=plotly_events(fig,click_event=True,hover_event=False)
    if click: set_view("room",click[0]["customdata"]); st.experimental_rerun()
    auto_refresh(refresh*1000)

if view=="room" and room:
    st.button("← Back",on_click=lambda:set_view("facility"))
    st.subheader(room)
    img_path=os.path.join(ASSETS,f"{room.replace(' ','_').lower()}.jpg")
    if os.path.exists(img_path):
        st.image(img_path,use_container_width=True)
    else:
        st.info("No background image for this room.")
    for key in ROOM_DETECTORS.get(room,[]):
        st.write(key)
        df=get_series(key)
        if not df.empty: st.line_chart(df.tail(100),y="value")
        live=st.session_state.latest.get(key,(0,float("nan")))[1]
        if not np.isnan(live):
            st.success(f"Live: {live:.2f}")
    auto_refresh(refresh*1000)
