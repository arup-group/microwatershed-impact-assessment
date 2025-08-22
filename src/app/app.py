# ─────────────────────────────────────────────────────────────
# Standard Library Imports
# ─────────────────────────────────────────────────────────────
import os
import sys
import subprocess
import platform
import stat
import tempfile
import atexit
import io
import contextlib
import shutil
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Third-Party Library Imports
# ─────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pysheds.grid import Grid
from rasterstats import zonal_stats
import folium
from folium.plugins import Draw
import leafmap
from shapely.geometry import Polygon
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────
# Local Application Imports
# ─────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core import make_catchments


# -------------------------------
# Helper: Get WhiteboxTools Path
# -------------------------------
def get_whitebox_binary_path():
    system = platform.system()

    if system == "Windows":
        binary_path = "tools/whitebox/whiteboxtools_binaries/WhiteboxTools_win_amd64/WBT/whitebox_tools.exe"
    elif system == "Linux":
        binary_path = "tools/whitebox/whiteboxtools_binaries/WhiteboxTools_linux_amd64/WBT/whitebox_tools"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    abs_path = os.path.abspath(binary_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"WhiteboxTools binary not found at: {abs_path}")

    if system != "Windows":
        os.chmod(abs_path, os.stat(abs_path).st_mode | 0o111)

    return abs_path

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(
    page_title="Microwatershed Impact Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

def clear_all():
    for key in st.session_state.keys():
        del st.session_state[key]

with st.sidebar:
    st.header("Utilities")
    st.button("🔄 Clear Session", on_click=clear_all)
    st.markdown("Use this to reset your inputs and start fresh.")

st.title("Microwatershed Impact Assessment")
st.markdown("""
Welcome to the Microwatershed Impact Assessment tool!  
Choose how you'd like to define your area of interest, either by drawing directly on the map or uploading a Shapefile.  
Once defined, the app will fetch elevation data and perform hydrological conditioning.
""")


# User choice
st.subheader("Step 1: Define Your Area of Interest")
option = st.radio("Choose input method:", ["Draw on map", "Upload Shapefile"])

polygon = None

if option == "Draw on map":
    with st.container():
        st.subheader("Step 2: Draw a Boundary")
        st.write("Use the map below to draw a polygon that defines your area of interest. Only polygon shapes are supported.")
        m = folium.Map(location=[27.6380, -80.3984], zoom_start=13)
        Draw(export=False).add_to(m)
        map_data = st_folium(m, width=None, height=400)

        if map_data["last_active_drawing"]:
            geometry = map_data["last_active_drawing"].get("geometry", {})
            coords = geometry.get("coordinates")
            geom_type = geometry.get("type")

            if coords and geom_type == "Polygon":
                polygon = Polygon(coords[0])
                st.success("Polygon captured successfully.")
                st.write(polygon)

elif option == "Upload Shapefile":
    with st.container():
        st.subheader("Step 2: Upload a Shapefile")
        uploaded_file = st.file_uploader("Upload a zipped Shapefile (.zip)", type=["zip"])

        if uploaded_file:
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = os.path.join(tmp_dir, "shapefile.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                shutil.unpack_archive(zip_path, tmp_dir)
                shp_files = [f for f in os.listdir(tmp_dir) if f.endswith(".shp")]
                if shp_files:
                    gdf = gpd.read_file(os.path.join(tmp_dir, shp_files[0])).to_crs(4326)
                    polygon = gdf.geometry.unary_union


if polygon:
    with st.spinner("Fetching DEM from 3DEP..."):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                dem_path = os.path.join(tmp_dir, "dem.tif")
                output_path = dem_path.replace(".tif", "_dem.tif")

                leafmap.get_3dep_dem(
                    polygon,
                    resolution=10,
                    output=dem_path,
                    dst_crs="EPSG:4326",
                    to_cog=True
                )

                if not os.path.exists(dem_path):
                    st.error("DEM file was not created. Check your polygon and internet connection.")
                    st.stop()

                binary_path = get_whitebox_binary_path()
                ## debug
                # st.sidebar.write("OS:", platform.system())
                # st.sidebar.write("WhiteboxTools path:", binary_path)

                st.header("DEM Conditioning")
                st.write("Executing WhiteboxTools: FeaturePreservingSmoothing...")

                cmd = [
                    binary_path,
                    "--run=FeaturePreservingSmoothing",
                    f"--input={dem_path}",
                    f"--output={output_path}",
                    "--filter=3"
                ]

                subprocess.run(cmd, capture_output=True, text=True)

                if not os.path.exists(output_path):
                    st.error("Smoothed DEM file was not created.")
                    st.stop()

                st.success("DEM smoothing complete!")

                # Run WhiteboxTools: Find No Flow Cells (Before Breaching)
                noflow_before_path = output_path.replace(".tif", "_noflow_before.tif")
                st.write("Executing WhiteboxTools: FindNoFlowCells...")

                cmd_noflow = [
                    binary_path,
                    "--run=FindNoFlowCells",
                    f"--dem={output_path}",
                    f"--output={noflow_before_path}"
                ]

                with st.spinner("Identifying no-flow cells before breaching"):
                    subprocess.run(cmd_noflow, capture_output=True, text=True)

                    if not os.path.exists(noflow_before_path):
                        st.error("No-flow cells raster was not created.")
                        st.stop()

                    st.success("No-flow cells identification complete!")

                # Run WhiteboxTools: Breach Depressions
                conditioned_output_path = output_path.replace(".tif", "_conditioned.tif")
                st.write("Executing WhiteboxTools: BreachDepressions...")

                cmd_breach = [
                    binary_path,
                    "--run=BreachDepressions",
                    f"--dem={output_path}",
                    f"--output={conditioned_output_path}"
                ]

                with st.spinner("Breaching depressions..."):
                    subprocess.run(cmd_breach, capture_output=True, text=True)

                    if not os.path.exists(conditioned_output_path):
                        st.error("Conditioned DEM file was not created.")
                        st.stop()

                    st.success("Depressions breached successfully!")

                # Run WhiteboxTools: Find No Flow Cells (After Breaching)
                noflow_after_path = output_path.replace(".tif", "_noflow_after.tif")
                st.write("Executing WhiteboxTools: FindNoFlowCells...")

                cmd_noflow_after = [
                    binary_path,
                    "--run=FindNoFlowCells",
                    f"--dem={conditioned_output_path}",
                    f"--output={noflow_after_path}"
                ]

                with st.spinner("Identifying no-flow cells after breaching..."):
                    subprocess.run(cmd_noflow_after, capture_output=True, text=True)

                    if not os.path.exists(noflow_after_path):
                        st.error("No-flow cells raster was not created.")
                        st.stop()

                    st.success("No-flow cells identification complete!")

                # Download buttons
                with open(output_path, "rb") as f:
                    st.download_button("Download Smoothed DEM", f, file_name="smoothed_dem.tif")

                with open(noflow_before_path, "rb") as f:
                    st.download_button("Download No-Flow Cells Raster", f, file_name="noflow_before_cells.tif")

                with open(conditioned_output_path, "rb") as f:
                    st.download_button("Download Conditioned DEM", f, file_name="conditioned_dem.tif")

                with open(noflow_after_path, "rb") as f:
                    st.download_button("Download No-Flow Cells Raster", f, file_name="noflow_after_cells.tif")

                # Plot smoothed DEM
                with rasterio.open(output_path) as src:
                    nodata_val = src.nodata or 9999
                    dem_data = src.read(1, masked=True)
                    masked = np.ma.masked_where(dem_data == nodata_val, dem_data)
                    extent = rasterio.plot.plotting_extent(src)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    cax = ax.imshow(masked, extent=extent, cmap='terrain')
                    ax.set_title("Smoothed DEM")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    fig.colorbar(cax, ax=ax, label="Elevation (m)")
                    st.pyplot(fig)

                # Plot no-flow cells before breaching
                with st.expander("No-Flow Cells Before Breaching"):
                    with rasterio.open(noflow_before_path) as src:
                        data = src.read(1)
                        extent = rasterio.plot.plotting_extent(src)
                        binary_mask = (data == 1).astype(int)
                        cmap = plt.cm.get_cmap("Greys", 2)

                        fig, ax = plt.subplots(figsize=(10, 8))
                        cax = ax.imshow(binary_mask, extent=extent, cmap=cmap)
                        ax.set_title("No-Flow Cells (Binary)")
                        ax.set_xlabel("Longitude")
                        ax.set_ylabel("Latitude")
                        fig.colorbar(cax, ax=ax, ticks=[0, 1], label="No-Flow")
                        st.pyplot(fig)

                # Plot conditioned DEM
                with rasterio.open(conditioned_output_path) as src:
                    nodata_val = src.nodata or 9999
                    dem_data = src.read(1, masked=True)
                    masked = np.ma.masked_where(dem_data == nodata_val, dem_data)
                    extent = rasterio.plot.plotting_extent(src)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    cax = ax.imshow(masked, extent=extent, cmap='terrain')
                    ax.set_title("Conditioned DEM")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    fig.colorbar(cax, ax=ax, label="Elevation (m)")
                    st.pyplot(fig)

                # Plot no-flow cells after breaching
                with st.expander("No-Flow Cells After Breaching"):
                    with rasterio.open(noflow_after_path) as src:
                        data = src.read(1)
                        extent = rasterio.plot.plotting_extent(src)

                        # Filter to binary mask: 1 = show, 9999 = skip
                        binary_mask = (data == 1).astype(int)
                        cmap = plt.cm.get_cmap("Greys", 2)

                        fig, ax = plt.subplots(figsize=(10, 8))
                        cax = ax.imshow(binary_mask, extent=extent, cmap=cmap)
                        ax.set_title("No-Flow Cells After Breaching")
                        ax.set_xlabel("Longitude")
                        ax.set_ylabel("Latitude")
                        fig.colorbar(cax, ax=ax, ticks=[0, 1], label="No-Flow")
                        st.pyplot(fig)





                # -------------------------------
                # PySheds Flow Accumulation & Direction
                # -------------------------------
                st.header("Microwatershed Deliniation")
                with st.spinner("Microwatershed Deliniation with PySheds"):

                    try:
                        # Initialize grid and read DEM once
                        grid = Grid.from_raster(conditioned_output_path)
                        dem = grid.read_raster(conditioned_output_path)

                        # Condition DEM
                        pit_filled = grid.fill_pits(dem)
                        depression_filled = grid.fill_depressions(pit_filled)
                        inflated = grid.resolve_flats(depression_filled)

                        # D8 direction mapping
                        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

                        # Compute flow direction
                        fdir = grid.flowdir(inflated, dirmap=dirmap, nodata_out=np.int32(-1))

                        fig = plt.figure(figsize=(8,6))
                        fig.patch.set_alpha(0)

                        plt.imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
                        boundaries = ([0] + sorted(list(dirmap)))
                        plt.colorbar(boundaries= boundaries,
                                    values=sorted(dirmap))
                        ax.set_xlabel("Longitude")
                        ax.set_ylabel("Latitude")
                        plt.title('Flow Direction Grid', size=14, fontsize=16, fontweight='bold')
                        plt.grid(zorder=-1)
                        plt.tight_layout()
                        st.pyplot(fig)

                        # Compute flow accumulation
                        acc = grid.accumulation(fdir, dirmap=dirmap, nodata_out=np.int32(-1))

                        fig, ax = plt.subplots(figsize=(8,6))
                        fig.patch.set_alpha(0)
                        plt.grid('on', zorder=0)
                        im = ax.imshow(acc, extent=grid.extent, zorder=2,
                                    cmap='cubehelix',
                                    norm=colors.LogNorm(1, acc.max()),
                                    interpolation='bilinear')
                        plt.colorbar(im, ax=ax, label='Upstream Cells')
                        plt.title('Flow Accumulation', fontsize=16, fontweight='bold')
                        ax.set_xlabel("Longitude")
                        ax.set_ylabel("Latitude")
                        plt.tight_layout()
                        st.pyplot(fig)

                        # Flatten and filter flow accumulation values
                        acc_values = acc.flatten()
                        acc_values = acc_values[acc_values > 0]  # Remove nodata or zero values
                        # Create the histogram figure
                        fig_hist, ax_hist = plt.subplots(figsize=(10, 6), dpi=100)
                        # Plot histogram
                        n, bins, patches = ax_hist.hist(
                            acc_values,
                            bins=60,
                            color='#1f77b4',
                            edgecolor='black',
                            linewidth=0.8
                    )
                        # Annotate the peak bin
                        max_bin_index = np.argmax(n)
                        peak_bin_center = 0.5 * (bins[max_bin_index] + bins[max_bin_index + 1])
                        peak_bin_height = n[max_bin_index]
                        ax_hist.annotate(
                            f'Peak: {int(peak_bin_height):,}',
                            xy=(peak_bin_center, peak_bin_height),
                            xytext=(peak_bin_center, peak_bin_height * 1.5),
                            arrowprops=dict(facecolor='black', arrowstyle='->'),
                            fontsize=10,
                            ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5)
                        )
                        # Axis labels and title
                        ax_hist.set_title('Distribution of Flow Accumulation Values', fontsize=16, fontweight='bold')
                        ax_hist.set_xlabel('Flow Accumulation Value', fontsize=13)
                        ax_hist.set_ylabel('Frequency (log scale)', fontsize=13)
                        # Log scale and grid
                        ax_hist.set_yscale('log')
                        ax_hist.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                        # Format x-axis ticks
                        ax_hist.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                        # Layout and display
                        plt.tight_layout()
                        st.pyplot(fig_hist)

                    except Exception as e:
                        st.error(f"PySheds processing failed: {e}")

                # Interactive stream network visualization
                with st.expander("Interactive Stream Network Visualization", expanded=True):
                    # Compute normalization for DEM
                    norm = colors.Normalize(vmin=np.nanmin(dem), vmax=np.nanmax(dem))
                    # Slider for flow accumulation threshold
                    channel_threshold = st.slider(
                        "Minimum Flow Accumulation for Stream Network",
                        min_value=100,
                        max_value=int(acc.max()),
                        value=1000,
                        step=100,
                        help="Adjust the threshold to control stream network density"
                    )

                    # Extract stream network
                    branches = grid.extract_river_network(
                        fdir,
                        acc > channel_threshold,
                        dirmap=dirmap,
                        nodata_out=np.int64(0)
                    )

                    fig_stream, ax_stream = plt.subplots(figsize=(8, 6))
                    fig_stream.patch.set_alpha(0)

                    ax_stream.set_xlim(grid.bbox[0], grid.bbox[2])
                    ax_stream.set_ylim(grid.bbox[1], grid.bbox[3])
                    ax_stream.set_aspect('equal')

                    for branch in branches['features']:
                        line = np.asarray(branch['geometry']['coordinates'])
                        ax_stream.plot(line[:, 0], line[:, 1])

                    ax_stream.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.4)
                    ax_stream.set_title(f'D8 Channels - Min Flow Acc of {channel_threshold}', fontsize=16, fontweight='bold')
                    ax_stream.set_xlabel("Longitude")
                    ax_stream.set_ylabel("Latitude")

                st.pyplot(fig_stream)


                # -------------------------------
                # Nested Catchment Deliniation
                # -------------------------------
                st.header("Nested Catchment Deliniation")
                with st.spinner("Nested Catchment Deliniation with Pysheds make_catchments wrapper"):

                    # Capture print output from generate_catchments
                    buffer = io.StringIO()
                    with contextlib.redirect_stdout(buffer):
                        basins, branches = make_catchments.generate_catchments(
                            conditioned_output_path,
                            acc_thresh=5000,
                            so_filter=4
                        )
                    output_text = buffer.getvalue()
                    st.text("Progress:")
                    st.code(output_text)

                # Print output
                microwatersheds_gdf = basins.copy()
                st.dataframe(microwatersheds_gdf)

                # Get the current datetime and format it
                now = datetime.now()
                datetime_str = now.strftime("%Y-%m-%d_%H%M")  # Format: 2024-11-22_0230

                # Define the output file path
                os.makedirs(rf'data/branches/{datetime_str}', exist_ok=True)
                output_file_path = f"data/branches/{datetime_str}/branches.shp"

                # Export the GeoDataFrame to a shapefile
                branches.to_file(output_file_path, driver='ESRI Shapefile')

                # Create a figure for plotting all catchments
                fig, ax = plt.subplots(figsize=(8, 6))

                # Plot conditioned DEM
                with rasterio.open(conditioned_output_path) as src:
                    conditioned_dem = src.read(1)
                    extent = rasterio.plot.plotting_extent(src)

                    cax = ax.imshow(conditioned_dem, extent=extent, cmap='terrain')
                    ax.set_title("Conditioned DEM (Depressions Breached)")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")

                microwatersheds_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='white', alpha=0.5)
                branches.plot(ax=ax, aspect=1, color='black', linewidth=0.3)

                # Add title and show the combined plot
                plt.title('Microwatersheds - Delineated from Channel Junction Points')
                st.pyplot(fig)


                # -------------------------------
                # Prioritization Attributes - Calculations
                # -------------------------------
                st.header("Prioritization Attributes - Calculations")
                with st.spinner("Prioritization Attributes - Calculations"):
                # Calculate areas - tabulate

                    # Add a simple numeric 'Microwatershed_ID' column starting at 1
                    microwatersheds_gdf['Microwatershed_ID'] = range(1, len(microwatersheds_gdf) + 1)

                    # Calculate the area of each polygon in acres (originally in square meters because of the CRS then divide by conversion factor)

                    # Reproject to a suitable projected CRS (e.g., UTM Zone 17N), so that the area has units (web mercator will be in degrees)
                    microwatersheds_gdf_projected = microwatersheds_gdf.to_crs(epsg=26917)

                    # Calculate area in square meters
                    microwatersheds_gdf_projected['Area_SqMeters'] = microwatersheds_gdf_projected['BasinGeo'].area

                    # Convert to acres
                    microwatersheds_gdf_projected['Area_Acres'] = microwatersheds_gdf_projected['Area_SqMeters'] / 4046.85642

                    # Create a new GeoDataFrame with the original ID and the calculated area
                    area_acres_gdf = microwatersheds_gdf_projected[['Microwatershed_ID', 'Area_Acres']]

                    # Join the 'Area_Acres' field back to the original GeoDataFrame using the original ID
                    microwatersheds_gdf = microwatersheds_gdf.merge(area_acres_gdf, on='Microwatershed_ID')

                ## optional
                # st.dataframe(microwatersheds_gdf)

                # Pull in ponds dataset and intersect
                st.header("Pull in ponds dataset and intersect")
                # Define the destination CRS
                destination_crs = 'EPSG:26917'

                # Load ponds data
                ponds = gpd.read_file(r'data\\IRL-Ponds-Export\\IRL-Ponds-Export_4269.shp')

                # NOTE Filter out ponds with an area less than 1 acre
                # NOTE: addition: could make this dynamic instead of relying on an existing area attribute
                ponds = ponds[ponds['Area_Acres'] >= 1]

                # Find intersecting ponds - all ponds that TOUCH a MWS boundary in addition to ponds completely within
                # Finding all ponds that touch MWS was chosen as most ponds are indeed fully within a MWS (from a watershed delineation perspective this makes sense)
                # And for ponds that are partly within a MWS, they will still influence the "control" of that MWS, so should be included
                # However, this does mean that the "Total Pond Area" will include ALL of the pond area, even pond area that falls outside of a MWS, if it touches one
                # But, for the downstream steps that involve pondSHEDs, the pondshed will be generated based on the CLIPPED pond.
                ponds_intersect = gpd.sjoin(ponds, microwatersheds_gdf, how='inner', predicate='intersects')

                # Count the number of intersecting ponds for each microwatershed
                pond_counts = ponds_intersect.groupby('index_right').size().reset_index(name='Pond_Count')

                # Sum the area of intersecting ponds for each microwatershed
                pond_area_sum = ponds_intersect.groupby('index_right')['Area_Acres_left'].sum().reset_index(name='Total_Pond_Area_Acres')

                # Calculate the average pond area within each microwatershed
                pond_area_avg = ponds_intersect.groupby('index_right')['Area_Acres_left'].mean().reset_index(name='Average_Pond_Area_Acres')

                # Combine pond_counts, pond_area_sum, and pond_area_avg into a single DataFrame
                pond_summary = pond_counts.merge(pond_area_sum, on='index_right').merge(pond_area_avg, on='index_right')

                # Merge the combined summary DataFrame back into the microwatersheds_gdf
                microwatersheds_all_gdf = microwatersheds_gdf.merge(pond_summary, left_index=True, right_on='index_right', how='left')

                # Fill NaN values with 0 (if there are microwatersheds with no intersecting ponds)
                microwatersheds_all_gdf['Pond_Count'] = microwatersheds_all_gdf['Pond_Count'].fillna(0)
                microwatersheds_all_gdf['Total_Pond_Area_Acres'] = microwatersheds_all_gdf['Total_Pond_Area_Acres'].fillna(0)
                microwatersheds_all_gdf['Average_Pond_Area_Acres'] = microwatersheds_all_gdf['Average_Pond_Area_Acres'].fillna(0)

                # Calculate the ratio of total pond area to the area of the microwatershed
                microwatersheds_all_gdf['Pond_Area_Percentage'] = microwatersheds_all_gdf['Total_Pond_Area_Acres'] / microwatersheds_all_gdf['Area_Acres'] *100

                # Calculate assumed pond volume
                # This equation is derived from a simple correlation between pond surface area and pond volume for a set of FDOT-controlled ponds
                microwatersheds_all_gdf['Pond_Controllable_Volume_Ac-Ft'] = 0.6431378064 + 2.5920596874*microwatersheds_all_gdf['Total_Pond_Area_Acres']

                # Select only the specified columns and order by Pond_Count
                columns_to_display = ['Microwatershed_ID', 'Area_Acres', 'Order', 'Pond_Count', 'Total_Pond_Area_Acres', 'Pond_Controllable_Volume_Ac-Ft', 'Average_Pond_Area_Acres', 'Pond_Area_Percentage']
                summary_df = microwatersheds_all_gdf[columns_to_display].sort_values(by='Pond_Count', ascending=False)

                # Print the DataFrame
                st.dataframe(summary_df)

                # Print MWS and intersecting ponds
                st.header("Plot MWS and intersecting ponds")

                # Create a figure for plotting all catchments
                fig, ax = plt.subplots(figsize=(8,6))
                plt.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.25)
                # Set the plot boundaries and aspect ratio
                plt.xlim(grid.bbox[0], grid.bbox[2])
                plt.ylim(grid.bbox[1], grid.bbox[3])
                plt.gca().set_aspect('equal')
                microwatersheds_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='white', alpha=0.5)
                # Plot ponds
                ponds_intersect.plot(ax=ax, aspect=1, color='blue', edgecolor='blue')
                branches.plot(ax=ax, aspect=1, color='black', linewidth=0.3)

                # Add title and show the combined plot
                plt.title('Microwatersheds - Pond Overlay')
                st.pyplot(fig)

                ## Plot MWS by order
                st.header("Plot MWS by order")

                # Optional: Add sidebar filters or controls
                selected_orders = st.multiselect(
                    "Select Stream Orders to Display:",
                    options=[1, 2, 3, 4, 5, 6],
                    default=[1, 2, 3, 4, 5, 6]
                )

                # Filter microwatersheds by selected orders
                orders_dict = {
                    1: microwatersheds_gdf[microwatersheds_gdf['Order'] == 1],
                    2: microwatersheds_gdf[microwatersheds_gdf['Order'] == 2],
                    3: microwatersheds_gdf[microwatersheds_gdf['Order'] == 3],
                    4: microwatersheds_gdf[microwatersheds_gdf['Order'] == 4],
                    5: microwatersheds_gdf[microwatersheds_gdf['Order'] == 5],
                    6: microwatersheds_gdf[microwatersheds_gdf['Order'] == 6]
                }

                # Sort highest to lowest order
                sorted_orders = sorted(selected_orders, reverse=True)

                for order_val in sorted_orders:
                    item = orders_dict[order_val]
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.25)
                        ax.set_xlim(grid.bbox[0], grid.bbox[2])
                        ax.set_ylim(grid.bbox[1], grid.bbox[3])
                        ax.set_aspect('equal')

                        item.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='white', alpha=0.5)
                        ponds_intersect.plot(ax=ax, aspect=1, color='blue', edgecolor='blue')
                        branches.plot(ax=ax, aspect=1, color='black', linewidth=0.3)

                        ax.set_title(f"Microwatersheds - Order {order_val} - Delineated from Channel Junction Points")
                        st.pyplot(fig)

                    except Exception as e:
                        st.warning(f"Failed to render Order {order_val}: {e}")


                ## Pondshed Buffer Calculator
                st.title('Pondshed Calculator')

                # Assume input_gdf and input_mws_gdf are already loaded or passed
                # You can wrap this in a file uploader and loader if needed

                input_gdf = ponds_intersect
                input_mws_gdf = microwatersheds_all_gdf

                def pondshed_buffer(ponds_gdf, mws_all_gdf, tolerance=1e-3):
                    st.info("📐 Calculating controllable volumes and buffer distances...")

                    ponds_gdf['Pond_Controllable_Volume_Ac-Ft'] = 0.6431378064 + 2.5920596874 * ponds_gdf['Area_Acres_left']
                    ponds_gdf['Pondshed_Area_Ac'] = ponds_gdf['Pond_Controllable_Volume_Ac-Ft'] / (1/3)
                    ponds_gdf = ponds_gdf.to_crs(epsg=26917)
                    ponds_gdf['Area_Acres_Temp'] = ponds_gdf.geometry.area / 4046.86

                    buffered_ponds = []
                    st.write('Calculating pondshed buffers...')
                    for idx, row in ponds_gdf.iterrows():
                        pond = row.geometry
                        controlled_area = row['Pondshed_Area_Ac']
                        buffer_distance = 0.0
                        step = 1

                        while True:
                            buffered_pond = pond.buffer(buffer_distance)
                            buffered_area = buffered_pond.area / 4046.86
                            if abs(buffered_area - controlled_area) < tolerance:
                                break
                            if buffered_area < controlled_area:
                                buffer_distance += step
                            else:
                                buffer_distance -= step
                                step /= 2
                        buffered_ponds.append(buffered_pond)

                    buffered_ponds_gdf = ponds_gdf.copy()
                    buffered_ponds_gdf['geometry'] = buffered_ponds

                    st.write('Calculating pondshed areas...')

                    mws_all_gdf = mws_all_gdf.to_crs(epsg=26917)
                    buff_dissolved = buffered_ponds_gdf.dissolve(by='Microwatershed_ID', as_index=False)

                    clipped_buffers_list = []
                    for idx, pond in buff_dissolved.iterrows():
                        mws_id = pond['Microwatershed_ID']
                        mws_geom = mws_all_gdf[mws_all_gdf['Microwatershed_ID'] == mws_id].geometry.iloc[0]
                        clipped_buff = gpd.clip(gpd.GeoDataFrame([pond], columns=buff_dissolved.columns), mws_geom)
                        clipped_buffers_list.append(clipped_buff)

                    clipped_buffers = pd.concat(clipped_buffers_list, ignore_index=True)
                    clipped_buffers = gpd.GeoDataFrame(clipped_buffers, geometry='geometry', crs=buff_dissolved.crs)

                    clipped_buffers['Clipped_Pondshed_Area_Acres'] = clipped_buffers.geometry.area / 4046.86
                    clipped_buffers['PondshedAreaSum'] = clipped_buffers.groupby('Microwatershed_ID')['Clipped_Pondshed_Area_Acres'].transform('sum')

                    pondsheds = clipped_buffers.to_crs(epsg=4269)
                    mws_all_gdf = mws_all_gdf.merge(pondsheds[['Microwatershed_ID', 'PondshedAreaSum']], on='Microwatershed_ID', how='left')

                    mws_all_gdf['Total_Pondshed_Area_Acres'] = mws_all_gdf['PondshedAreaSum']
                    mws_all_gdf['Pondshed_to_Pond_Ratio'] = (mws_all_gdf['Total_Pondshed_Area_Acres'] / mws_all_gdf['Total_Pond_Area_Acres']).round(2)
                    mws_all_gdf['Pondshed_to_MWS_Percentage'] = (mws_all_gdf['Total_Pondshed_Area_Acres'] / mws_all_gdf['Area_Acres'] * 100).round(2)
                    mws_all_gdf = mws_all_gdf.to_crs(epsg=4269)

                    st.write('Pondshed areas complete.')

                    return mws_all_gdf, pondsheds, ponds_gdf, buffered_ponds_gdf
                
                mws_buffer_sum, pondsheds_4269, pds, buff = pondshed_buffer(input_gdf, microwatersheds_all_gdf, tolerance=1e-6)

                # Set the gdf to be the output of the pondshed buffering
                microwatersheds_all_gdf = mws_buffer_sum

                impervious_raster = r'data\\ImperviousArea\\FlaImperviousArea_4326.tif'

                def calculate_impervious_percentage(raster_path, microwatersheds_gdf):
                    # Compute zonal statistics (sum of impervious pixels and total pixel count)
                    stats = zonal_stats(
                        microwatersheds_gdf,
                        raster_path,
                        stats=["sum", "count"],  # Sum of impervious pixels, and total pixel count
                        all_touched=True  # Ensures all intersecting pixels are included
                    )

                    # Extract values
                    impervious_pixel_sums = [stat["sum"] for stat in stats]  # Sum of pixels classified as 1
                    total_pixels = [stat["count"] for stat in stats]  # Total pixels in each polygon

                    # Calculate percentage of impervious area
                    microwatersheds_gdf["Percent_Impervious"] = [
                        (impervious / total * 100) if total > 0 else 0
                        for impervious, total in zip(impervious_pixel_sums, total_pixels)
                    ]

                    # Calculate impervious area in acres
                    microwatersheds_gdf["ImperviousAreaAcres"] = (
                        microwatersheds_gdf["Area_Acres"] * microwatersheds_gdf["Percent_Impervious"] / 100
                    )

                    return microwatersheds_gdf

                # Apply function
                microwatersheds_all_gdf = calculate_impervious_percentage(impervious_raster, microwatersheds_all_gdf)

                # NOTE: this file will not be included in the GitHub repo, as it's a large tif file. It can be downloaded from here:
                # https://coastalimagery.blob.core.windows.net/ccap-landcover/CCAP_bulk_download/High_Resolution_Land_Cover/Phase_1_Initial_Layers/Impervious/index.html
                # fl_2022_ccap_v2_hires_impervious_20231226.zip
                impervious_raster = r'data\\ImperviousArea\\FlaImperviousArea_4326.tif'

                def calculate_pondshed_impervious(raster_path, microwatersheds_gdf, pondsheds):
                    # Compute zonal statistics (sum of impervious pixels and total pixel count)
                    stats = zonal_stats(
                        pondsheds,
                        raster_path,
                        stats=["sum", "count"],  # Sum of impervious pixels, and total pixel count
                        all_touched=True  # Ensures all intersecting pixels are included
                    )

                    # Extract values
                    impervious_pixel_sums = [stat["sum"] for stat in stats]  # Sum of pixels classified as 1
                    total_pixels = [stat["count"] for stat in stats]  # Total pixels in each polygon

                    # Calculate percentage of impervious area
                    pondsheds["Percent_Impervious"] = [
                        (impervious / total * 100) if total > 0 else 0
                        for impervious, total in zip(impervious_pixel_sums, total_pixels)
                    ]

                    # Calculate impervious area in acres
                    pondsheds["PondshedImperviousAreaAcres"] = (
                        pondsheds["Pondshed_Area_Ac"] * pondsheds["Percent_Impervious"] / 100
                    )

                    # Sum pondshed impervious area for each microwatershed
                    pondshed_grouped = pondsheds.groupby('Microwatershed_ID', as_index=False).agg({
                        'PondshedImperviousAreaAcres': 'sum'
                    })

                    microwatersheds_gdf = microwatersheds_gdf.merge(pondshed_grouped, on='Microwatershed_ID', how='left')

                    return microwatersheds_gdf

                # Apply function
                microwatersheds_all_gdf = calculate_pondshed_impervious(impervious_raster, microwatersheds_all_gdf, pondsheds_4269)

                # Ensure rainfall raster is in web mercator

                import rasterio
                import numpy as np
                from rasterio.warp import calculate_default_transform, reproject, Resampling

                annual_rainfall_path = r'data\\Annual-Rainfall\\PRISM_ppt_30yr_normal_4kmM4_annual_bil.bil'

                # Open the source raster
                with rasterio.open(annual_rainfall_path) as src:
                    # Open the target raster (the one you want to match)
                    with rasterio.open("data\\streamorder\\colorado_sample_dem.tiff") as dst:
                        # Calculate the transformation parameters
                        print(src.crs)
                        print(dst.crs)
                        transform, width, height = calculate_default_transform(
                            src.crs, dst.crs, src.width, src.height, *src.bounds
                        )

                        # Create an empty array for the reprojected data
                        reprojected_data = np.empty((src.count, height, width), dtype=src.dtypes[0])

                        # Reproject the data
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=reprojected_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst.crs,
                            resampling=Resampling.nearest  # Choose a suitable resampling method
                        )

                        # Save the reprojected raster
                        with rasterio.open(
                            "data\\temp\\temp_reprojected_raster_rainfall.tif",
                            "w",
                            driver="GTiff",
                            width=width,
                            height=height,
                            count=src.count,
                            dtype=src.dtypes[0],
                            crs=dst.crs,
                            transform=transform,
                        ) as dst:
                            dst.write(reprojected_data)


                annual_rainfall_path = 'data\\temp\\temp_reprojected_raster_rainfall.tif'
                rain = Grid.from_raster(annual_rainfall_path)
                rainfall = rain.read_raster(annual_rainfall_path)

                import geopandas as gpd
                from rasterstats import zonal_stats

                raster_path = 'data\\temp\\temp_reprojected_raster_rainfall.tif'

                def get_annual_rainfall(gdf, raster_path):
                    # Compute zonal statistics (mean value of raster within each polygon)
                    stats = zonal_stats(
                        gdf,
                        raster_path,
                        stats="mean",  # You can use other statistics like "min", "max", etc.
                        all_touched=True # to ensure all pixels intersecting the polygon are included
                    )

                    # Extract the mean rainfall values and add them as a new column
                    gdf['AnnualRainfallMm'] = [stat['mean'] for stat in stats] # dataset has units of mm
                    gdf['AnnualRainfallInches'] = gdf['AnnualRainfallMm'] / 25.4

                    return gdf

                # Run function on the MWS dataset
                microwatersheds_all_gdf = get_annual_rainfall(microwatersheds_all_gdf, raster_path)

                # Run function on the pondsheds dataset
                pondsheds_4269 = get_annual_rainfall(pondsheds_4269, raster_path)


                # Annual runoff by land class

                land_cover = gpd.read_file(r'data\\LandCover\\Land_Cover_IRL_4326.shp')

                def calculate_runoff_lulc(land_use_gdf, microwatersheds_gdf, pondsheds):

                    # Create areas of intersection
                    land_use_intersect = gpd.overlay(pondsheds, land_use_gdf, how='intersection')
                
                    # Convert back to CRS with meters to calculate areas
                    land_use_intersect = land_use_intersect.to_crs(epsg=26917)
                    # PARAMETER: burn width
                    land_use_intersect[f'LULC_Area_Acres'] = land_use_intersect.area / 4046.85642
                    # Convert back to CRS with lat lon
                    land_use_intersect = land_use_intersect.to_crs(epsg=4269)
                    
                    # Read in lookup table
                    csv_path = r'data\LandCover\FlLandUseForNutrients.csv'
                    # Notable columns in lookup table: 'FlLandUse' (contains the classifications, but some instances have multiple classes separated by commas), 'RunnoffCoefficient', 'TotalNitrogenEMC_(mg/L)', 'TotalPhosEMC_(mg/L)'
                    lookup_df = pd.read_csv(csv_path)
                    # Explode lookup table so each row corresponds to a single land-use class
                    lookup_df['FlLandUse'] = lookup_df['FlLandUse'].str.split(' / ')  # Convert to list
                    lookup_df = lookup_df.explode('FlLandUse')  # Create separate rows for each class
                    lookup_df['FlLandUse'] = lookup_df['FlLandUse'].str.strip()  # Remove whitespace

                    # Merge with land_use_intersect
                    land_use_merge = land_use_intersect.merge(
                        lookup_df, left_on='LEVEL2_L_1', right_on='FlLandUse', how='left'
                    )

                    print(land_use_merge['AnnualRainfallInches'].head())
                    # Calculate Runoff for each classification
                    # Calculate Runoff_m3 = LULC_Area_Acres * RunoffCoefficient (from lookup table column RunoffCoefficient, for a given class) * annual rainfall value (in/year), 0.027154 MG = 1 acre-inch 
                    # Updated - use simple 'drainage acre * runoff coefficient * rainfall inches (* conversion)' method
                    land_use_merge[f'AnnualRunoffMGYr'] = land_use_merge[f'LULC_Area_Acres'] * land_use_merge['RunoffCoefficient'] * land_use_merge['AnnualRainfallInches'] * 0.027154

                    # Calculate weighted runoff coefficient
                    # Compute area-weighted RunoffCoefficient
                    land_use_merge['WeightedRunoff'] = land_use_merge['RunoffCoefficient'] * land_use_merge['LULC_Area_Acres']

                    # Sum runoff for each microwatershed
                    nutrient_summary = land_use_merge.groupby('Microwatershed_ID', as_index=False).agg({
                        'AnnualRunoffMGYr': 'sum',
                        'WeightedRunoff': 'sum',  # Sum of weighted runoff coefficients
                        'LULC_Area_Acres': 'sum'  # Sum of total land use areas
                    })

                    nutrient_summary['AreaWeightedRunoffCoeff'] = nutrient_summary['WeightedRunoff'] / nutrient_summary['LULC_Area_Acres']
                    
                    # Merge the Runoff_MG column back into the microwatersheds_gdf
                    microwatersheds_gdf = microwatersheds_gdf.merge(nutrient_summary, on='Microwatershed_ID', how='left')
                    print(microwatersheds_gdf.columns)
                    
                    return microwatersheds_gdf, land_use_merge
                
                # Consolidated function
                def annual_control_volume(gdf):
                    """
                    Processes a GeoDataFrame to calculate and add the following columns:
                    - 'AnnualRunoffMGYr': Annual runoff in MG/yr.
                    - 'Annual_Volume_Treated_MG/Yr': Incremental annual wet weather capture in MG/yr.
                    - 'Annual_Volume_Treated_MG/Yr_PerPond': Volume treated per pond.
                    
                    Parameters:
                        gdf (GeoDataFrame): Input GeoDataFrame with necessary columns.
                    
                    Returns:
                        GeoDataFrame: Updated GeoDataFrame with new calculated columns.
                    """
                    # SS - using above runoff LULC function
                    # def calculate_annual_runoff(pondshed_impervious_area_ac, annual_rainfall_in, runoff_coefficient=1.0):
                    #     """Calculate Annual Runoff (MG/yr)."""
                    #     return pondshed_impervious_area_ac * annual_rainfall_in * runoff_coefficient * 0.027154
                    
                    def calculate_incremental_wet_weather_capture(pondshed_impervious_area_ac, annual_rainfall_in, runoff_coefficient, annual_runoff_mgyr, passive_volume_acft, cmac_volume_acft):
                        """Calculate Incremental Annual Wet Weather Capture (MG/yr)."""
                        # Calculate Incremental Annual Wet Weather Capture (MG/yr).
                        

                        passive_volume_inIA = (passive_volume_acft / (pondshed_impervious_area_ac * runoff_coefficient)) * 12 if (pondshed_impervious_area_ac * runoff_coefficient) != 0 else 0
                        cmac_volume_inIA = (cmac_volume_acft / (pondshed_impervious_area_ac * runoff_coefficient)) * 12 if (pondshed_impervious_area_ac * runoff_coefficient) != 0 else 0
                        
                        A, B = 25.07, 20.864
                        intercept, precip_coef, ln_precip, ln_Vol = 124.24, 0.3415, -22.161, 27.417
                        
                        try:
                            passive_efficiency = max(min((A + B * np.log(passive_volume_inIA)) / 100, 0.99), 0.05)
                            cmac_efficiency = max(min((intercept + precip_coef * annual_rainfall_in + ln_precip * math.log(annual_rainfall_in) + ln_Vol * math.log(cmac_volume_inIA)) / 100, 0.99), 0.05)
                        except:
                            passive_efficiency = 0
                            cmac_efficiency = 0

                        return annual_runoff_mgyr * (cmac_efficiency - passive_efficiency)
                    
                    # Calculate and add columns
                    print('reading in landcover')
                    land_cover = gpd.read_file(r'data\\LandCover\\Land_Cover_IRL_4326.shp')
                    gdf, land_use_merge_pshed = calculate_runoff_lulc(land_cover, gdf, pondsheds_4269)
                    print('completed runoff lulc fxn')
                    # gdf['AnnualRunoffMGYr'] = gdf.apply(lambda row: calculate_annual_runoff(row['ImperviousAreaAcres'], row['AnnualRainfallInches']), axis=1)
                    gdf['Annual_Volume_Treated_MG/Yr'] = gdf.apply(
                        lambda row: calculate_incremental_wet_weather_capture(
                            row['ImperviousAreaAcres'],
                            row['AnnualRainfallInches'],
                            row['AreaWeightedRunoffCoeff'],
                            row['AnnualRunoffMGYr'],
                            row['Pond_Controllable_Volume_Ac-Ft'],
                            row['Pond_Controllable_Volume_Ac-Ft']
                        ),
                        axis=1
                    )
                    print('completed main fxn')
                    gdf['Annual_Volume_Treated_MG/Yr_PerPond'] = np.where(
                        gdf['Pond_Count'] == 0,
                        0,
                        gdf['Annual_Volume_Treated_MG/Yr'] / gdf['Pond_Count']
                    )
                    
                    return gdf
                

                microwatersheds_all_gdf = annual_control_volume(microwatersheds_all_gdf)

                # Land cover "urban area" percentage
                land_cover = gpd.read_file(r'data\\LandCover\\Land_Cover_IRL_4326.shp')

                def urban_area(overlay_gdf, microwatersheds_gdf):
                    # Ensure both GeoDataFrames use the same CRS
                    
                    # # MWS area
                    microwatersheds_gdf['MicrowshedArea_Unitless'] = microwatersheds_gdf.area
                    # print(microwatersheds_gdf.columns)

                    # Create areas of intersection
                    df_is2 = gpd.overlay(microwatersheds_gdf, overlay_gdf, how='intersection')
                    # print(df_is2.columns)
                    
                    urban = [
                        'Commercial and Services',
                        'Institutional',
                        'Industrial',
                        'Residential Low Density',
                        'Residential Medium Density', 
                        'Residential High Density', 
                        'Transportation',
                        'Communications',
                        'Utilities'
                    ]

                    # Filter intersections to only include impervious areas
                    df_is2 = df_is2[df_is2['LEVEL2_L_1'].isin(urban)]

                    # # Store the area size of intersections
                    df_is2['Urban_Area'] = df_is2.area
                    # print(df_is2.columns)

                    # Sum over microwatersheds
                    df_is2 = df_is2.groupby(['Microwatershed_ID', 'MicrowshedArea_Unitless'])[['Urban_Area']].sum().reset_index()

                    df_is2['Percent_Urban'] = df_is2['Urban_Area'] / df_is2['MicrowshedArea_Unitless'] * 100

                    # print(df_is2.head(20))
                    microwatersheds_gdf = microwatersheds_gdf.merge(df_is2, on='Microwatershed_ID', how='left')

                    return microwatersheds_gdf

                microwatersheds_all_gdf = urban_area(land_cover, microwatersheds_all_gdf)


                # Nutrient load calculation
                land_cover = gpd.read_file(r'data\\LandCover\\Land_Cover_IRL_4326.shp')


                def calculate_nutrients(overlay_gdf, microwatersheds_gdf, pondsheds):

                    # Create areas of intersection
                    land_use_intersect = gpd.overlay(pondsheds, overlay_gdf, how='intersection')
                
                    # Convert back to CRS with meters to calculate areas
                    land_use_intersect = land_use_intersect.to_crs(epsg=26917)
                    # PARAMETER: burn width
                    land_use_intersect[f'LULC_Area_m2'] = land_use_intersect.area
                    # Convert back to CRS with lat lon
                    land_use_intersect = land_use_intersect.to_crs(epsg=4269)
                    
                    # Read in lookup table
                    csv_path = r'data\LandCover\FlLandUseForNutrients.csv'
                    # Notable columns in lookup table: 'FlLandUse' (contains the classifications, but some instances have multiple classes separated by commas), 'RunnoffCoefficient', 'TotalNitrogenEMC_(mg/L)', 'TotalPhosEMC_(mg/L)'
                    lookup_df = pd.read_csv(csv_path)
                    # Explode lookup table so each row corresponds to a single land-use class
                    lookup_df['FlLandUse'] = lookup_df['FlLandUse'].str.split(' / ')  # Convert to list
                    lookup_df = lookup_df.explode('FlLandUse')  # Create separate rows for each class
                    lookup_df['FlLandUse'] = lookup_df['FlLandUse'].str.strip()  # Remove whitespace

                    # Merge with land_use_intersect
                    land_use_merge = land_use_intersect.merge(
                        lookup_df, left_on='LEVEL2_L_1', right_on='FlLandUse', how='left'
                    )


                    # Calculate Runoff for each classification
                    # Calculate Runoff_m3_per_year = LULC_Area_m2 * RunoffCoefficient (from lookup table column RunoffCoefficient, for a given class) * annual rainfall inches * 0.0254 (m/in)
                    land_use_merge[f'Runoff_m3_per_year'] = land_use_merge[f'LULC_Area_m2'] * land_use_merge['RunoffCoefficient'] * land_use_merge['AnnualRainfallInches'] * 0.0254

                    # Calculate N and P. 1000 to convert L to m3 and 0.000001 mg to kg and 2.2 is the factor for kg to lb
                    land_use_merge[f'Nitrogen_lb_per_year'] = land_use_merge[f'Runoff_m3_per_year'] * land_use_merge[f'TotalNitrogenEMC_(mg/L)'] * 1000 * 0.000001 * 2.20462262
                    land_use_merge[f'Phosphorous_lb_per_year'] = land_use_merge[f'Runoff_m3_per_year'] * land_use_merge[f'TotalPhosEMC_(mg/L)'] * 1000 * 0.000001 * 2.20462262

                    # Sum runoff, nitrogen, and phosphorus for each microwatershed
                    nutrient_summary = land_use_merge.groupby('Microwatershed_ID', as_index=False).agg({
                        f'LULC_Area_m2': 'sum',
                        f'Runoff_m3_per_year': 'sum',
                        f'Nitrogen_lb_per_year': 'sum',
                        f'Phosphorous_lb_per_year': 'sum'
                    })
                    # Nutrient to area ratio
                    
                    # Normalize to a 0–100 scale using percentile ranking
                    # nutrient_summary['Nitrogen_lb_Percentile'] = nutrient_summary['Nitrogen_lb'].rank(pct=True) * 100
                    # nutrient_summary['Phosphorous_lb_Percentile'] = nutrient_summary['Phosphorous_lb'].rank(pct=True) * 100

                    # Merge the Runoff_m3_per_year column back into the microwatersheds_gdf
                    microwatersheds_gdf = microwatersheds_gdf.merge(nutrient_summary, on='Microwatershed_ID', how='left')
                    # print(microwatersheds_gdf.columns)
                    microwatersheds_gdf[f'Nitrogen_lb_per_Pondshed_Area'] = microwatersheds_gdf[f'Nitrogen_lb_per_year']/microwatersheds_gdf[f'Total_Pondshed_Area_Acres']
                    microwatersheds_gdf[f'Phosphorous_lb_per_Pondshed_Area'] = microwatersheds_gdf[f'Phosphorous_lb_per_year']/microwatersheds_gdf[f'Total_Pondshed_Area_Acres']

                    return microwatersheds_gdf, land_use_merge

                microwatersheds_all_gdf, land_use_merge_pshed = calculate_nutrients(land_cover, microwatersheds_all_gdf, pondsheds_4269)


                import matplotlib.pyplot as plt
                import os
                from matplotlib import colors
                import os
                from datetime import datetime

                now = datetime.now()
                datetime_str = now.strftime("%Y-%m-%d_%H%M")  # Format: 2024-11-22_0230
                file = "table"

                # Filter MWS characteristics

                # Total Pond Area - likely the most important
                min_total_pond_area = 5
                microwatersheds_filter_gdf = microwatersheds_all_gdf[microwatersheds_all_gdf['Total_Pond_Area_Acres'] >= min_total_pond_area]

                # Pond Count - second of importance (likely won't implement the tech on a high number of ponds)
                max_pond_count = 25
                microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Pond_Count'] <= max_pond_count]

                # Filter out catchment of order 3+ (the largest orders)
                # microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Order'] >= order_filter_min]
                # microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Order'] <= order_filter_max]


                # MWS Area - less important (?) because a large area could still have favorable above characteristics
                max_mws_area = 1000
                microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Area_Acres'] <= max_mws_area]

                # if type(microwatersheds_filter_gdf) != 'geopandas.geodataframe.GeoDataFrame':
                #     microwatersheds_filter_gdf = gpd.GeoDataFrame(microwatersheds_filter_gdf, geometry='geometry', crs=4326)

                # filter_by = [175]
                # filter_by = [175, 130]
                # filter_by = [175, 130, 121]
                filter_by = [175, 130, 121, 95, 134]
                # microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Microwatershed_ID'].isin(filter_by)]
                # microwatersheds_filter_gdf = microwatersheds_all_gdf

                # Print MWS and intersecting ponds
                fig, ax = plt.subplots(figsize=(8, 6))
                norm = colors.Normalize(vmin=0, vmax=15)
                plt.xlim(grid.bbox[0], grid.bbox[2])
                plt.ylim(grid.bbox[1], grid.bbox[3])
                plt.gca().set_aspect('equal')
                plt.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.25)
                microwatersheds_filter_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='black', alpha=0.5)
                # pondsheds_4269.plot(ax=ax, aspect=1, alpha=0.25, color='blue')
                ponds_intersect.plot(ax=ax, aspect=1, color='blue', edgecolor='blue')
                # branches_all.plot(ax=ax, aspect=1, alpha=0.25, color='black', linewidth=0.3)

                # Add labels to the microwatersheds using 'BasinGeo' as the geometry column
                # for idx, row in microwatersheds_filter_gdf.iterrows():
                #     plt.annotate(text=row['Microwatershed_ID'], xy=(row['BasinGeo'].centroid.x, row['BasinGeo'].centroid.y),
                #                  horizontalalignment='center', verticalalignment='center', fontsize=6, color='black',fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.1'))

                plt.title(f'Microwatersheds - Minimum Total Pond Area {min_total_pond_area} Acres')
                plt.show()

                # Rename columns 
                # microwatersheds_filter_gdf.rename(columns={'Avg_SUM_Annu_5': 'Brevard_Total_Nitrogen_(Lb/Yr)'}, inplace=True)
                # microwatersheds_filter_gdf.rename(columns={'Avg_SUM_Annu_8': 'Brevard_Total_Phosphorous_(Lb/Yr)'}, inplace=True)
                # microwatersheds_filter_gdf.rename(columns={'Controllable_Nitrogen_(Lb/Yr)': 'Brevard_Controllable_Nitrogen_(Lb/Yr)'}, inplace=True)
                microwatersheds_filter_gdf.rename(columns={'Pond_Area_Percentage': 'Pond /_MWS Area_Percentage'}, inplace=True)
                microwatersheds_filter_gdf.rename(columns={'Pondshed_to_MWS_Percentage': 'Pondshed /_MWS Area_Percentage'}, inplace=True)
                microwatersheds_filter_gdf.rename(columns={'Microwatershed_ID': 'Microwshed_ID'}, inplace=True)

                # Select only the specified columns and order by Total_Pond_Area_Acres
                columns_to_display = ['Microwshed_ID', 
                                    'Pond_Count', 
                                    'Area_Acres', 
                                    #   'Average_Pond_Area_Acres', 
                                    'Total_Pond_Area_Acres', 
                                    'Total_Pondshed_Area_Acres',
                                    #   'Pondshed_to_Pond_Ratio',
                                    'Pond /_MWS Area_Percentage',
                                    'Pondshed /_MWS Area_Percentage',
                                    'Pond_Controllable_Volume_Ac-Ft', 
                                    'Annual_Volume_Treated_MG/Yr',
                                    'Nitrogen_lb_per_Pondshed_Area',
                                    #   'Nitrogen_lb_Percentile',
                                    #   'Brevard_Total_Nitrogen_(Lb/Yr)', 
                                    #   'Brevard_Controllable_Nitrogen_(Lb/Yr)', 
                                    #   'Brevard_Total_Phosphorous_(Lb/Yr)', 
                                    'Percent_Impervious', 
                                    'Percent_Urban']
                filter_df = microwatersheds_filter_gdf[columns_to_display].sort_values(by='Total_Pondshed_Area_Acres', ascending=False)
                # filter_df = filter_df[filter_df['Microwatershed_ID'] == 121]

                format_columns = {
                    'Microwshed_ID': '{:.0f}',
                    'Pond_Count': '{:.0f}',
                    'Area_Acres': '{:.2f}',
                    # 'Average_Pond_Area_Acres': '{:.2f}',
                    'Total_Pond_Area_Acres': '{:.2f}',
                    'Total_Pondshed_Area_Acres': '{:.2f}',
                    # 'Pondshed_to_Pond_Ratio': '{:.2f}',
                    'Pond /_MWS Area_Percentage': '{:.2f}',
                    'Pondshed /_MWS Area_Percentage': '{:.2f}',
                    'Pond_Controllable_Volume_Ac-Ft': '{:.2f}',
                    'Annual_Volume_Treated_MG/Yr': '{:.2f}',
                    'Nitrogen_lb_per_Pondshed_Area': '{:.4f}',
                    # 'Nitrogen_lb_Percentile': '{:.2f}',
                    # 'Brevard_Total_Nitrogen_(Lb/Yr)': '{:.2f}',
                    # 'Brevard_Controllable_Nitrogen_(Lb/Yr)': '{:.2f}',
                    # 'Brevard_Total_Phosphorous_(Lb/Yr)': '{:.2f}',
                    'Percent_Impervious': '{:.2f}',
                    'Percent_Urban': '{:.2f}'
                }

                for col, fmt in format_columns.items():
                    filter_df[col] = filter_df[col].map(fmt.format)

                # Print the DataFrame
                print(filter_df.head(20))

                st.header("MIcrowatershed Characteristics")
                st.dataframe(filter_df)

                # Save plot to pdf
                # pdf_pages.savefig(fig)

                # Function for coloring columns
                def create_color_map(df, column, cmap, alpha=0.5):
                    norm = plt.Normalize(df[column].astype(float).min(), df[column].astype(float).max())
                    colors = cmap(norm(df[column].astype(float)))
                    colors[:, -1] = alpha  # Set alpha transparency
                    return colors

                columns_to_color = {
                    'Pond_Count': plt.cm.plasma,
                    'Area_Acres': plt.cm.plasma,
                    'Total_Pond_Area_Acres': plt.cm.plasma,
                    'Total_Pondshed_Area_Acres': plt.cm.plasma,
                    # 'Pondshed_to_Pond_Ratio': plt.cm.plasma,
                    # 'Average_Pond_Area_Acres': plt.cm.plasma,
                    'Pond /_MWS Area_Percentage': plt.cm.plasma,
                    'Pondshed /_MWS Area_Percentage': plt.cm.plasma,
                    'Pond_Controllable_Volume_Ac-Ft': plt.cm.plasma,
                    'Annual_Volume_Treated_MG/Yr': plt.cm.plasma,
                    'Nitrogen_lb_per_Pondshed_Area': plt.cm.plasma,
                    # 'Nitrogen_lb_Percentile': plt.cm.plasma,
                    # 'Brevard_Total_Nitrogen_(Lb/Yr)': plt.cm.plasma,
                    # 'Brevard_Controllable_Nitrogen_(Lb/Yr)': plt.cm.plasma,
                    # 'Brevard_Total_Phosphorous_(Lb/Yr)': plt.cm.plasma,
                    'Percent_Impervious': plt.cm.plasma,
                    'Percent_Urban': plt.cm.plasma,
                }

                colors = {col: create_color_map(filter_df, col, cmap) for col, cmap in columns_to_color.items()}

                # Add DataFrame to the PDF
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.axis('tight')
                ax.axis('off')
                table_data = filter_df.head(20).values  # Assuming filter_df is your DataFrame
                columns = [col.replace('_', '\n') for col in filter_df.columns.tolist()]
                table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')

                # Apply colors to the columns
                for col, color_map in colors.items():
                    col_index = columns.index(col.replace('_', '\n'))
                    for i in range(len(table_data)):
                        cell = table[(i+1, col_index)]  # (row, column)
                        cell.set_facecolor(color_map[i])
                        cell.set_text_props(color='black')


                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1.2, 1.2)  # Adjust the size of the table



                # Adjust header cell height if needed
                for (i, j), cell in table.get_celld().items():
                    if i == 0:  # Header row
                        cell.set_height(0.12)  # Adjust height for better visibility

                # Also export to a csv
                os.makedirs(rf'outputs\\tables\\{datetime_str}', exist_ok=True)
                filter_df.to_csv(rf'outputs\\tables\\SummaryTable_{file}_{datetime_str}.csv', index=False)
                # also export the filtered df but with all the columns
                microwatersheds_filter_gdf.to_csv(rf'outputs\\tables\\SummaryTable_{file}_{datetime_str}_Full.csv', index=False)

                st.dataframe(microwatersheds_filter_gdf)

                # pdf_pages.savefig(fig)  # Save the DataFrame table to the PDF
                # pdf_pages.close()

                # # Open the PDF file
                # if os.path.exists(pdf_path):
                #     if os.name == 'nt':  # For Windows
                #         os.startfile(pdf_path)
                #     else:  # For macOS and Linux
                #         subprocess.call(['open', pdf_path])  # macOS
                #         # subprocess.call(['xdg-open', pdf_path])  # Linux


        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()
    # else:
    #     st.warning("Please draw a valid polygon.")
else:
    st.info("Waiting for user input...")


