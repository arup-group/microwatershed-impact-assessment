import streamlit as st
import os
import subprocess
import platform
import stat
from tempfile import NamedTemporaryFile
import atexit
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from pysheds.grid import Grid

# -------------------------------
# Helper: Get WhiteboxTools Path
# -------------------------------
def get_whitebox_binary_path():
    system = platform.system()

    if system == "Windows":
        binary_path = "whitebox/whiteboxtools_binaries/WhiteboxTools_win_amd64/WBT/whitebox_tools.exe"
    elif system == "Linux":
        binary_path = "whitebox/whiteboxtools_binaries/WhiteboxTools_linux_amd64/WBT/whitebox_tools"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    abs_path = os.path.abspath(binary_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"WhiteboxTools binary not found at: {abs_path}")

    if system != "Windows":
        os.chmod(abs_path, os.stat(abs_path).st_mode | 0o111)

    return abs_path


st.title("DEM Hillshade, Flow Direction & Accumulation Visualizer")

# Upload DEM
uploaded_file = st.file_uploader("Upload a DEM file (GeoTIFF)", type=["tif", "tiff"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    output_path = input_path.replace(".tif", "_hillshade.tif")

    # Register cleanup of temporary files
    atexit.register(lambda: os.remove(input_path) if os.path.exists(input_path) else None)
    atexit.register(lambda: os.remove(output_path) if os.path.exists(output_path) else None)


    # Show input metadata
    # try:
    #     with rasterio.open(input_path) as src:
    #         st.write("DEM metadata:", src.meta)
    # except Exception as e:
    #     st.error(f"Failed to read DEM file: {e}")
    #     st.stop()


    # Get WhiteboxTools binary
    try:
        binary_path = get_whitebox_binary_path()
    except Exception as e:
        st.error(f"Error locating WhiteboxTools: {e}")
        st.stop()

    # Optional debug info
    st.sidebar.write("OS:", platform.system())
    st.sidebar.write("WhiteboxTools path:", binary_path)
    st.sidebar.write("Exists:", os.path.exists(binary_path))

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

    with st.spinner("Generating hillshade..."):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        ## uncomment to debug
        #     st.text("WhiteboxTools stdout:")
        #     st.text(result.stdout)
        #     st.text("WhiteboxTools stderr:")
        #     st.text(result.stderr)
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
    with st.spinner("Computing flow direction and accumulation with PySheds"):

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