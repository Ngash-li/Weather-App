# --- Imports ---
import streamlit as st
import geopandas as gpd
from netCDF4 import Dataset
from wrf import getvar, to_np, latlon_coords, extract_times, interplevel
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import io
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from adjustText import adjust_text
from matplotlib import patheffects

# --- Page Config ---
st.set_page_config(page_title="WRF Kenya Weather Map", layout="wide")
st.title("⛅ WRF Kenya Weather Map")
st.sidebar.subheader("📊 County Mean Summary")

# --- Constants ---
pressure_levels = {
    "1000 hPa (~100m)": 1000,
    "850 hPa (~1.5km)": 850,
    "700 hPa (~3km)": 700,
    "500 hPa (~5.5km)": 500
}

colormaps = {
    "Temperature (2m)": "coolwarm",
    "Humidity (2m)": "BrBG",
    "Surface Pressure": "viridis",
    "Geopotential Height": "viridis",
    "Temperature (Pressure Level)": "coolwarm",
    "Humidity (Pressure Level)": "BrBG",
    "Wind (Pressure Level)": "viridis",
}

labels = {
    "Rainfall": "Rainfall (mm)",
    "Temperature (2m)": "°C",
    "Humidity (2m)": "%",
    "Surface Pressure": "hPa",
    "Geopotential Height": "Geopotential Height (m)",
    "Temperature (Pressure Level)": "Temperature (°C)",
    "Humidity (Pressure Level)": "Humidity (%)",
    "Wind (Pressure Level)": "Wind Speed (m/s)",
    "Wind (10m)": "Wind Speed (m/s)"
}

varname_map = {
    "Temperature (2m)": "T2",
    "Rainfall": ("RAINC", "RAINNC"),
    "Humidity (2m)": "rh2",
    "Surface Pressure": "PSFC",
    "Geopotential Height": "z",
    "Temperature (Pressure Level)": "tk",
    "Humidity (Pressure Level)": "rh",
    "Wind (10m)": ("U10", "V10"),
    "Wind (Pressure Level)": ("ua", "va")
}

descriptions = {
    "Rainfall": "🌧️ Accumulated rainfall (mm) — hydrology, flood risk, agriculture.",
    "Temperature (2m)": "🌡️ 2m air temperature in °C — daily weather and planning.",
    "Humidity (2m)": "💧 Relative humidity at 2m (%) — dew point and comfort.",
    "Wind (10m)": "🌬️ Wind vectors at 10m — aviation, energy, dust, fire.",
    "Surface Pressure": "📉 Surface pressure (hPa) — indicates highs and lows.",
    "Geopotential Height": "📍 Altitude of pressure surfaces — used for ridges and troughs.",
    "Temperature (Pressure Level)": "🌡️ Temperature at altitude — identifies inversions and CAPE.",
    "Humidity (Pressure Level)": "💧 Moisture at altitude — helps locate clouds and instability.",
    "Wind (Pressure Level)": "🌬️ Wind vector map at height — jet stream, shear zones."
}

# --- Load Data ---
@st.cache_resource
def load_inputs():
    ncfile = Dataset(r"C:\Users\Admin\wrfout.nc")
    kenya = gpd.read_file(r"C:\Users\Admin\Desktop\Ken_Adm1")  # Your county shapefile path
    
    # Ensure CRS is set to EPSG:4326 for lat/lon
    if kenya.crs is None:
        kenya.set_crs(epsg=4326, inplace=True)
    else:
        kenya = kenya.to_crs(epsg=4326)

    times = extract_times(ncfile, timeidx=None, meta=False)

    rain_values = []
    for t in range(ncfile.dimensions["Time"].size):
        rainnc = getvar(ncfile, "RAINNC", timeidx=t)
        rainc = getvar(ncfile, "RAINC", timeidx=t)
        rain_values.append(to_np(rainnc + rainc))
    global_rain_max = np.nanpercentile(np.concatenate(rain_values), 99)
    if global_rain_max < 1:
        global_rain_max = 10

    sample = getvar(ncfile, "RAINNC", timeidx=0)
    lats, lons = latlon_coords(sample)

    return ncfile, kenya, times, to_np(lats), to_np(lons), global_rain_max

ncfile, kenya, times, lats, lons, global_rain_max = load_inputs()
time_labels = [pd.to_datetime(t).strftime("%Y-%m-%d %H:%M UTC") for t in times]

# Compute centroids in EPSG:4326 for plotting county labels
centroids = kenya.to_crs(epsg=21037).centroid.to_crs(epsg=4326)

# Show shapefile columns for debug (optional)
st.write("Shapefile columns:", kenya.columns.tolist())

# Commented out because your shapefile doesn't have "NAME_1"
# county_names = kenya["NAME_1"]

# --- Utility for Pressure Levels ---
@st.cache_data
def get_pressure_level_var(_ncfile, varname, time_idx, pressure, subtract_kelvin=False):
    pressure_3d = getvar(_ncfile, "pressure", timeidx=time_idx)
    var3d = getvar(_ncfile, varname, timeidx=time_idx)
    interp = interplevel(var3d, pressure_3d, pressure)
    return np.nan_to_num(to_np(interp) - 273.15 if subtract_kelvin else to_np(interp))

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🔧 Controls")
    var_choice = st.selectbox("Variable", list(varname_map.keys()))
    time_label_selected = st.selectbox("Time (UTC)", time_labels)
    time_idx = time_labels.index(time_label_selected)

    selected_pressure = None
    if "Pressure Level" in var_choice or var_choice == "Geopotential Height":
        selected_label = st.selectbox("Pressure Level", list(pressure_levels.keys()))
        selected_pressure = pressure_levels[selected_label]

    st.markdown(f"**Currently Visualizing:** `{time_labels[time_idx]}`")
    if selected_pressure:
        st.markdown(f"**Pressure Level:** `{selected_pressure} hPa`")

# --- Description ---
st.success(descriptions.get(var_choice, ""))

# --- County Summary ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 County Mean Summary")

try:
    field = None
    if var_choice == "Rainfall":
        field = to_np(getvar(ncfile, "RAINNC", timeidx=time_idx) + getvar(ncfile, "RAINC", timeidx=time_idx))
    elif var_choice in ["Temperature (2m)", "Humidity (2m)", "Surface Pressure"]:
        field = to_np(getvar(ncfile, varname_map[var_choice], timeidx=time_idx))
        if var_choice == "Temperature (2m)":
            field -= 273.15
        elif var_choice == "Surface Pressure":
            field /= 100.0
    elif var_choice == "Wind (10m)":
        u, v = [to_np(getvar(ncfile, var, timeidx=time_idx)) for var in varname_map[var_choice]]
        field = np.sqrt(u**2 + v**2)

    if field is not None:
        df = pd.DataFrame({"lat": lats.flatten(), "lon": lons.flatten(), "value": field.flatten()})
        gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
        joined = gpd.sjoin(gdf_points, kenya, how="inner", predicate='within')
        
        # Group by index of shapefile, not by any name column
        summary = joined.groupby(joined.index)["value"].mean().sort_values(ascending=False)
        st.sidebar.dataframe(summary.round(2).rename("Mean Value"), use_container_width=True)
except Exception:
    st.sidebar.warning("Summary not available for this variable.")

# --- Plot Generation ---
def generate_plot():
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    kenya.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), edgecolor="#0280FD", facecolor='#0280FD', alpha=0.5)

    try:
        field = None
        if var_choice == "Rainfall":
            field = to_np(getvar(ncfile, "RAINNC", timeidx=time_idx) + getvar(ncfile, "RAINC", timeidx=time_idx))
            cmap = LinearSegmentedColormap.from_list("rain", ["#f0f0f0", "#00ff00"])
            levels = np.linspace(0, global_rain_max, 20)
            norm = BoundaryNorm(levels, cmap.N)
            cs = ax.contourf(lons, lats, field, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        elif var_choice in ["Wind (10m)", "Wind (Pressure Level)"]:
            u, v = (
                [to_np(getvar(ncfile, var, timeidx=time_idx)) for var in varname_map[var_choice]]
                if "10m" in var_choice else
                [get_pressure_level_var(ncfile, var, time_idx, selected_pressure) for var in varname_map[var_choice]]
            )
            speed = np.sqrt(u**2 + v**2)
            cs = ax.quiver(lons[::5, ::5], lats[::5, ::5], u[::5, ::5], v[::5, ::5], speed[::5, ::5], cmap="viridis", scale=400, width=0.003)
        else:
            if "Pressure Level" in var_choice or var_choice == "Geopotential Height":
                field = get_pressure_level_var(ncfile, varname_map[var_choice], time_idx, selected_pressure, subtract_kelvin="Temperature" in var_choice)
            else:
                field = to_np(getvar(ncfile, varname_map[var_choice], timeidx=time_idx))
                if var_choice == "Temperature (2m)":
                    field -= 273.15
                elif var_choice == "Surface Pressure":
                    field /= 100.0
            cmap = colormaps.get(var_choice, "viridis")
            cs = ax.contourf(lons, lats, field, cmap=cmap, transform=ccrs.PlateCarree())

        if cs:
            cbar = plt.colorbar(cs, ax=ax, shrink=0.6, pad=0.03)
            cbar.set_label(labels.get(var_choice, "Value"))

        ax.set_title(f"{var_choice}\n{time_labels[time_idx]}")
    except Exception as e:
        st.error(f"Plotting failed: {e}")

    ax.coastlines(resolution='10m', linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linewidth=1)

    # County labels disabled because your shapefile lacks county names
    # texts = []
    # for i, row in kenya.iterrows():
    #     cx, cy = centroids[i].x, centroids[i].y
    #     texts.append(ax.text(
    #         cx, cy, county_names[i],
    #         fontsize=7.5, ha='center', va='center',
    #         transform=ccrs.PlateCarree(),
    #         path_effects=[patheffects.withStroke(linewidth=1.5, foreground='white')]
    #     ))
    # adjust_text(texts, ax=ax, expand_points=(1.2, 1.4))

    plt.tight_layout()
    return fig

# --- Display Plot ---
with st.spinner("⏳ Generating plot..."):
    fig = generate_plot()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    st.image(buf)
    st.markdown("</div>", unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download Plot Image",
        data=buf.getvalue(),
        file_name=f"{var_choice.replace(' ', '_')}_{time_labels[time_idx]}.png",
        mime="image/png"
    )

# --- Footer ---
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    font-size: 0.9em;
    padding: 10px 0;
    color: #555;
    background-color: #f9f9f9;
    border-top: 1px solid #ccc;
}
</style>
<div class="footer">
    🚀 Created by <a href='https://github.com/Samuel-Wanza' target='_blank' style='color: #1f77b4; text-decoration: none;'>Samuel Wanza</a> • 
    <strong>Version 1.1</strong> • <em>2025</em>
</div>
""", unsafe_allow_html=True)
