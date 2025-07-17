import streamlit as st
import os
import subprocess
import platform
import stat
from tempfile import NamedTemporaryFile
import atexit
import rasterio
from rasterio.plot import show
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


st.title("Microwatershed Impact Assessment")

# Upload DEM
uploaded_file = st.file_uploader("Upload Digital Elevation Model (DEM) file", type=["tif", "tiff"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    output_path = input_path.replace(".tif", "_dem.tif")

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

    st.header("DEM Conditioning")
    # Run WhiteboxTools via subprocess for smoothing
    st.write("Executing WhiteboxTools: FeaturePreservingSmoothing...")
    cmd = [
        binary_path,
        "--run=FeaturePreservingSmoothing",
        f"--input={input_path}",
        f"--output={output_path}",
        "--filter=3"
    ]

    with st.spinner("Smoothing DEM..."):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except Exception as e:
            st.error(f"Failed to run whitebox_tools: {e}")
            st.stop()

        if not os.path.exists(output_path):
            st.error("Smoothed DEM file was not created. Please check the input DEM file.")
            st.stop()

        st.success("DEM smoothing complete!")

    # -------------------------------------
    # Run WhiteboxTools: Find No Flow Cells
    # -------------------------------------
    noflow_before_path = output_path.replace(".tif", "_noflow_before.tif")

    st.write("Executing WhiteboxTools: FindNoFlowCells...")
    cmd_noflow = [
        binary_path,
        "--run=FindNoFlowCells",
        f"--dem={output_path}",
        f"--output={noflow_before_path}"
    ]

    with st.spinner("Identifying no-flow cells before breaching"):
        try:
            result_noflow = subprocess.run(cmd_noflow, capture_output=True, text=True)
        except Exception as e:
            st.error(f"Failed to run FindNoFlowCells: {e}")
            st.stop()

        if not os.path.exists(noflow_before_path):
            st.error("No-flow cells raster was not created.")
            st.stop()

        st.success("No-flow cells identification complete!")

    # -------------------------------------
    # Run WhiteboxTools: Breach Depressions
    # -------------------------------------
    conditioned_output_path = output_path.replace(".tif", "_conditioned.tif")

    st.write("Executing WhiteboxTools: BreachDepressions...")
    cmd_breach = [
        binary_path,
        "--run=BreachDepressions",
        f"--dem={output_path}",
        f"--output={conditioned_output_path}"
    ]

    with st.spinner("Breaching depressions..."):
        try:
            result_breach = subprocess.run(cmd_breach, capture_output=True, text=True)
        except Exception as e:
            st.error(f"Failed to run BreachDepressions: {e}")
            st.stop()

        if not os.path.exists(conditioned_output_path):
            st.error("Conditioned DEM file was not created.")
            st.stop()

        st.success("Depressions breached successfully!")

    # -------------------------------------
    # Run WhiteboxTools: Find No Flow Cells
    # -------------------------------------
    noflow_after_path = output_path.replace(".tif", "_noflow_after.tif")

    st.write("Executing WhiteboxTools: FindNoFlowCells...")
    cmd_noflow = [
        binary_path,
        "--run=FindNoFlowCells",
        f"--dem={conditioned_output_path}",
        f"--output={noflow_after_path}"
    ]

    with st.spinner("Identifying no-flow cells after breaching..."):
        try:
            result_noflow = subprocess.run(cmd_noflow, capture_output=True, text=True)
        except Exception as e:
            st.error(f"Failed to run FindNoFlowCells: {e}")
            st.stop()

        if not os.path.exists(noflow_after_path):
            st.error("No-flow cells raster was not created.")
            st.stop()

        st.success("No-flow cells identification complete!")    

    # Download link for smoothed DEM
    with open(output_path, "rb") as f:
        st.download_button("Download Smoothed DEM", f, file_name="smoothed_dem.tif")

    # Download link for no-flow cells
    with open(noflow_before_path, "rb") as f:
        st.download_button("Download No-Flow Cells Raster", f, file_name="noflow_before_cells.tif")

    # Download link for conditioned DEM
    with open(conditioned_output_path, "rb") as f:
        st.download_button("Download Conditioned DEM", f, file_name="conditioned_dem.tif")

    # Download link for no-flow cells
    with open(noflow_after_path, "rb") as f:
        st.download_button("Download No-Flow Cells Raster", f, file_name="noflow_after_cells.tif")

    # Plot smoothed DEM with CRS and colorbar
    with rasterio.open(output_path) as src:
        dem_data = src.read(1)
        extent = rasterio.plot.plotting_extent(src)

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(dem_data, extent=extent, cmap='terrain')
        ax.set_title("Smoothed DEM")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(cax, ax=ax, label="Elevation (m)")
        st.pyplot(fig)


    # Plot no-flow cells before breaching
    with st.expander("No-Flow Cells Before Breaching"):
        with rasterio.open(noflow_before_path) as src:
            data = src.read(1)
            extent = rasterio.plot.plotting_extent(src)

            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.imshow(data, extent=extent, cmap='viridis')
            ax.set_title("No-Flow Cells Before Breaching")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.colorbar(cax, ax=ax, label="Value")
            st.pyplot(fig)


    # Plot conditioned DEM
    with rasterio.open(conditioned_output_path) as src:
        conditioned_dem = src.read(1)
        extent = rasterio.plot.plotting_extent(src)

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(conditioned_dem, extent=extent, cmap='terrain')
        ax.set_title("Conditioned DEM (Depressions Breached)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(cax, ax=ax, label="Elevation (m)")
        st.pyplot(fig)


    # Plot no-flow cells after breaching
    with st.expander("No-Flow Cells After Breaching"):
        with rasterio.open(noflow_after_path) as src:
            data = src.read(1)
            extent = rasterio.plot.plotting_extent(src)

            fig, ax = plt.subplots(figsize=(10, 8))
            cax = ax.imshow(data, extent=extent, cmap='viridis')
            ax.set_title("No-Flow Cells After Breaching")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.colorbar(cax, ax=ax, label="Value")
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
            plt.xlabel('X')
            plt.ylabel('Y')
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
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"PySheds processing failed: {e}")