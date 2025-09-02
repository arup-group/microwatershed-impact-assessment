# home.py

import streamlit as st
import geopandas as gpd
import leafmap.foliumap as leafmap
import os

st.set_page_config(page_title="Microwatershed Viewer", layout="wide")
st.title("Microwatershed Impact Assessment")

# -------------------------------
# Welcome Message and Instructions
# -------------------------------
st.markdown("### 👋 Welcome")
st.markdown("""
This tool helps you assess the hydrological impact of microwatersheds.  
Start by defining your area of interest, then proceed to delineation and prioritization.
""")

# -------------------------------
# Session State Initialization
# -------------------------------
def init_session_state():
    keys = [
        "polygon", "conditioned_output_path", "grid", "dem", "fdir", "acc",
        "branches", "microwatersheds_gdf", "microwatersheds_all_gdf",
        "branches_gdf", "catchments_generated", "selected_attribute", "folium_map"
    ]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None
    if "catchments_generated" not in st.session_state:
        st.session_state["catchments_generated"] = False

init_session_state()

# -------------------------------
# Step 1: Define Area of Interest
# -------------------------------
st.markdown("### 🗺️ Step 1: Define Area of Interest")
input_method = st.radio("Input method:", ["Draw on map", "Upload Shapefile"])

if input_method == "Draw on map":
    st.markdown("Use the map below to draw your area of interest.")
    m = leafmap.Map(center=[40.7831, -73.9712], zoom=10, draw_export=True)
    m.add_draw_control()
    m.to_streamlit(height=500)

    st.info("After drawing, export the geometry and use it in the delineation step.")

elif input_method == "Upload Shapefile":
    uploaded_file = st.file_uploader("Upload zipped shapefile (.zip)", type=["zip"])
    if uploaded_file:
        try:
            gdf = gpd.read_file(f"zip://{uploaded_file.name}")
            st.session_state["polygon"] = gdf
            st.success("Shapefile uploaded successfully!")
            st.map(gdf)
        except Exception as e:
            st.error(f"Error reading shapefile: {e}")

# -------------------------------
# DEM Upload (Optional for Testing)
# -------------------------------
st.markdown("### 🌄 Optional: Upload Conditioned DEM")
dem_file = st.file_uploader("Upload conditioned DEM (.tif)", type=["tif"])
if dem_file:
    dem_path = os.path.join("temp_dem", dem_file.name)
    os.makedirs("temp_dem", exist_ok=True)
    with open(dem_path, "wb") as f:
        f.write(dem_file.read())
    st.session_state["conditioned_output_path"] = dem_path
    st.success("DEM uploaded and stored for delineation.")

# -------------------------------
# Utilities
# -------------------------------
st.markdown("### 🛠️ Utilities")
st.button("🔄 Clear Session", on_click=lambda: st.session_state.clear())
st.markdown("Use this to reset your inputs and start fresh.")

# -------------------------------
# Help Section
# -------------------------------
st.markdown("### ❓ Help")
with st.expander("How does this tool work?"):
    st.markdown("""
    - **Draw on map**: Use the interactive map to sketch your watershed boundary.
    - **Upload Shapefile**: Provide a zipped shapefile to define your area.
    - **Upload DEM**: Optionally upload a conditioned DEM for delineation.
    - After defining your area, go to the next page to run elevation and flow analysis.
    """)