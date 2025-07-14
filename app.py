import streamlit as st
import os
import subprocess
import stat
from tempfile import NamedTemporaryFile
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from pysheds.grid import Grid

st.title("DEM Hillshade, Flow Direction & Accumulation Visualizer")

# Upload DEM
uploaded_file = st.file_uploader("Upload a DEM file (GeoTIFF)", type=["tif", "tiff"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    output_path = input_path.replace(".tif", "_hillshade.tif")

    # Show input metadata
    try:
        with rasterio.open(input_path) as src:
            st.write("DEM metadata:", src.meta)
    except Exception as e:
        st.error(f"Failed to read DEM file: {e}")
        st.stop()

    # Ensure whitebox_tools is executable
    binary_path = "tools/WBT/whitebox_tools"
    if not os.access(binary_path, os.X_OK):
        try:
            os.chmod(binary_path, os.stat(binary_path).st_mode | stat.S_IEXEC)
        except Exception as e:
            st.error(f"Failed to set executable permission: {e}")
            st.stop()

    # Run WhiteboxTools via subprocess
    st.write("Generating hillshade...")
    cmd = [
        binary_path,
        "--run=Hillshade",
        f"--dem={input_path}",
        f"--output={output_path}",
        "--azimuth=315.0",
        "--altitude=45.0"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        st.text("WhiteboxTools stdout:")
        st.text(result.stdout)
        st.text("WhiteboxTools stderr:")
        st.text(result.stderr)
    except Exception as e:
        st.error(f"Failed to run whitebox_tools: {e}")
        st.stop()

    if not os.path.exists(output_path):
        st.error("Hillshade file was not created. Please check the input DEM file.")
        st.stop()

    st.success("Hillshade generated!")

    # Display hillshade
    with rasterio.open(output_path) as src:
        hillshade = src.read(1)

    fig, ax = plt.subplots()
    ax.imshow(hillshade, cmap="gray")
    ax.set_title("Hillshade Output")
    ax.axis("off")
    st.pyplot(fig)

    # Download link
    with open(output_path, "rb") as f:
        st.download_button("Download Hillshade", f, file_name="hillshade.tif")

    # -------------------------------
    # PySheds Flow Accumulation & Direction
    # -------------------------------
    st.write("Computing flow direction and accumulation with PySheds...")

    try:
        # Initialize grid and read DEM once
        grid = Grid.from_raster(input_path)
        dem = grid.read_raster(input_path)

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
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Flow direction grid', size=14)
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
        plt.title('Flow Accumulation', size=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"PySheds processing failed: {e}")