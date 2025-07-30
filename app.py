import streamlit as st
import os
import subprocess
import platform
import stat
from tempfile import NamedTemporaryFile
import atexit
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import numpy as np
from pysheds.grid import Grid
import make_catchments
import folium
import io
import contextlib
from datetime import datetime

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

def transform_crs_to_4326(src_path):
    output_path = src_path.replace(".tif", "_4326.tif")
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": "EPSG:4326",
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": src.nodata
        })
        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear
                )
    return output_path

# Upload DEM
uploaded_file = st.file_uploader("Upload Digital Elevation Model (DEM) file", type=["tif", "tiff"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    reprojected_path = transform_crs_to_4326(input_path)
    output_path = reprojected_path.replace(".tif", "_dem.tif")
    input_path = reprojected_path  # for consistency with rest of script

    # Register cleanup of temporary files
    atexit.register(lambda: os.remove(input_path) if os.path.exists(input_path) else None)
    atexit.register(lambda: os.remove(reprojected_path) if os.path.exists(reprojected_path) else None)
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

            # Filter to binary mask: 1 = show, 9999 = skip
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
        ax.set_title("Conditioned DEM (Depressions Breached)")
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
            plt.xlabel('X')
            plt.ylabel('Y')
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
            plt.xlabel('X')
            plt.ylabel('Y')
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
        os.makedirs(rf'{datetime_str}', exist_ok=True)
        output_file_path = f"{datetime_str}/branches.shp"

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
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        microwatersheds_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='white', alpha=0.5)
        branches.plot(ax=ax, aspect=1, color='black', linewidth=0.3)

        # Add title and show the combined plot
        plt.title('Microwatersheds - Delineated from Channel Junction Points')
        st.pyplot(fig)

