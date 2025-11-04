import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import io
import os
from datetime import datetime

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="PLQY Calculator", layout="wide", page_icon="💡")

st.markdown("""
    <style>
        .main { background-color: #f9fafb; }
        h1 {
            text-align: center;
            background: linear-gradient(90deg,#1e3a8a,#0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            margin-bottom: 0.2em;
        }
        h2, h3 { color: #1e3a8a; margin-top: 1.2em; }
        .stMetric {
            background-color: white;
            border-radius: 0.6rem;
            box-shadow: 0 0 8px rgba(0,0,0,0.08);
            padding: 0.6em;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Photoluminescence Quantum Yield (PLQY) Calculator")
st.caption("Estimate PLQY, spectral parameters, and Voc losses from laser and sample spectra.")

# ---------- FILE READER ----------
def read_measurement_file(file):
    """Read measurement file with optional header (#key;value) and return (DataFrame, params)."""
    header_params = {}

    # Decode bytes if necessary
    if isinstance(file, io.BytesIO):
        content = file.getvalue().decode("utf-8", errors="ignore")
    else:
        content = file.read()
    if isinstance(content, bytes):  # e.g. when using open() in binary mode
        content = content.decode("utf-8", errors="ignore")

    lines = content.splitlines()
    data_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            try:
                key, val = line[1:].split(";")
                header_params[key.strip().lower()] = float(val.strip())
            except Exception:
                pass
        elif line:
            data_lines.append(line)

    # Convert remaining lines to numeric DataFrame
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=";", header=None)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Reset file pointer for re-reads (important if reused)
    file.seek(0)
    return df, header_params



# ---------- SIDEBAR ----------
st.sidebar.header("📁 File Upload")

sample = st.sidebar.file_uploader("Sample measurement (.txt)", type="txt")
laser = st.sidebar.file_uploader("Laser measurement (.txt)", type="txt")
load_example = st.sidebar.button("🧪 Load Example Data")

# Load example data if selected or nothing uploaded
if load_example or (sample is None and laser is None):
    st.sidebar.info("Using bundled example data.")
    sample = open(os.path.join("data", "sample_example.txt"), "r")
    laser = open(os.path.join("data", "laser_example.txt"), "r")

# ---------- MAIN ----------
if sample and laser:
    st.sidebar.success(f"Loaded: {sample.name} and {laser.name}")

    df_sample, header_sample = read_measurement_file(sample)
    df_laser, header_laser = read_measurement_file(laser)

    x_sample = df_sample[0].dropna()
    y_sample = df_sample[1].dropna()
    x_laser = df_laser[0].dropna()
    y_laser = df_laser[1].dropna()

    # ---------- PARAMETER TAB ----------
    tab1, tab2, tab3 = st.tabs(["⚙️ Parameters", "📊 Results", "📈 Spectra"])

    # Helper for default or header
    def get_param(header_dict, key, default):
        return header_dict.get(key, default)

    with tab1:
        st.subheader("Measurement Parameters")

        col1, col2, col3 = st.columns(3)
        with col1:
            int_time_sample = st.number_input(
                "Integration time (sample, s)",
                value=get_param(header_sample, "integration_time", 1.0),
                format="%.3e",
            )
            int_time_laser = st.number_input(
                "Integration time (laser, s)",
                value=get_param(header_laser, "integration_time", 0.1e-3),
                format="%.3e",
            )
        with col2:
            spotsize_diameter = st.number_input(
                "Spot diameter (mm)",
                value=get_param(header_laser, "spot_diameter", 1.0),
                step=0.01,
            )
            power_laser = st.number_input(
                "Laser power (W)",
                value=get_param(header_laser, "power", 145e-6),
                format="%.2e",
            )
        with col3:
            power_sample = st.number_input(
                "Sample power (W)",
                value=get_param(header_sample, "power", 1.6e-3),
                format="%.2e",
            )
            background_laser = st.number_input(
                "Laser background (W)",
                value=get_param(header_laser, "background", 50e-6),
                format="%.2e",
            )

        # ---------- SHOW HEADER METADATA ----------
        st.markdown("### Parsed File Metadata")
        c1, c2 = st.columns(2)
        with c1:
            with st.expander("📄 Sample File Header Data", expanded=False):
                if header_sample:
                    st.json(header_sample)
                else:
                    st.info("No header metadata found in sample file.")
        with c2:
            with st.expander("🔦 Laser File Header Data", expanded=False):
                if header_laser:
                    st.json(header_laser)
                else:
                    st.info("No header metadata found in laser file.")

    # ---------- FITTING ----------
    def gaussian(x, a, mu, sigma):
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def fit_gaussian(x, y):
        try:
            p0 = [np.trapezoid(y, x), x[np.argmax(y)], np.std(x)]
            popt, _ = curve_fit(gaussian, x, y, p0=p0)
            return popt
        except RuntimeError:
            return [np.nan] * 3

    popt_sample = fit_gaussian(x_sample, y_sample)
    popt_laser = fit_gaussian(x_laser, y_laser)

    q = 1.602e-19
    wavelength_laser = popt_laser[1]
    wavelength_sample = popt_sample[1]
    fwhm_sample_nm = 2.355 * abs(popt_sample[2])

    # Energy conversion
    energy_sample = 1240 / wavelength_sample
    e1 = 1240 / (wavelength_sample - fwhm_sample_nm / 2)
    e2 = 1240 / (wavelength_sample + fwhm_sample_nm / 2)
    fwhm_sample_eV = abs(e1 - e2)

    energy_laser = 1240 / wavelength_laser * q
    nphotons_laser = power_laser / energy_laser
    area = (spotsize_diameter / 10) ** 2 * np.pi
    intensity = power_sample / area * 1e3
    attenuation = power_laser / power_sample

    intfitarea_sample = abs(popt_sample[0] * popt_sample[2] * np.sqrt(2 * np.pi))
    intfitarea_laser = abs(popt_laser[0] * popt_laser[2] * np.sqrt(2 * np.pi))
    plqy = (intfitarea_sample / intfitarea_laser) * (int_time_laser / int_time_sample) * attenuation
    voc_loss = abs(0.026 * np.log(plqy)) * 1000  # mV

    # ---------- RESULTS TAB ----------
    with tab2:
        st.subheader("Results Overview")

        st.markdown("### Excitation")
        col1, col2, col3 = st.columns(3)
        col1.metric("Excitation λ (nm)", f"{wavelength_laser:.2f}")
        col2.metric("Intensity (mW/cm²)", f"{intensity:.2f}")
        col3.metric("Photon Flux (cm⁻²·s⁻¹)", f"{nphotons_laser:.2e}")

        st.markdown("### Photoluminescence")
        col4, col5, col6 = st.columns(3)
        col4.metric("PL Peak Position (eV)", f"{energy_sample:.3f}")
        col5.metric("PL Peak FWHM (eV)", f"{fwhm_sample_eV:.3f}")
        col6.metric("PLQY (%)", f"{plqy * 100:.3f}")

        # Logarithmic PLQY gauge
        st.markdown("#### PLQY Gauge (logarithmic scale)")
        plqy_log = np.clip(np.log10(plqy * 100 + 1e-6), -3, 2)
        plqy_norm = (plqy_log + 3) / 5 * 100
        gauge_color = "#ef4444" if plqy < 0.01 else "#facc15" if plqy < 0.05 else "#22c55e"
        st.markdown(f"""
        <div style="height:30px;width:100%;background-color:#e5e7eb;border-radius:20px;">
            <div style="height:100%;width:{plqy_norm:.1f}%;background-color:{gauge_color};
                        border-radius:20px;text-align:center;color:white;font-weight:600;">
                {plqy*100:.3f} %
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Device Prediction")
        col7, _, _ = st.columns(3)
        col7.metric("Voc Loss (mV)", f"{voc_loss:.1f}")

        st.markdown("---")
        st.markdown("### Gaussian Fit Parameters")
        fit_results = pd.DataFrame({
            "Parameter": ["Amplitude (a)", "Center (μ)", "Sigma (σ)"],
            "Laser": np.round(popt_laser, 3),
            "Sample": np.round(popt_sample, 3)
        })
        st.dataframe(fit_results, use_container_width=True)
        st.markdown(f"**Attenuation factor:** {attenuation:.2e}")

        # ---------- CSV DOWNLOAD ----------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dict = {
            "Timestamp": timestamp,
            "Sample File": sample.name,
            "Laser File": laser.name,
            "Integration Time Sample (s)": int_time_sample,
            "Integration Time Laser (s)": int_time_laser,
            "Spot Diameter (mm)": spotsize_diameter,
            "Excitation λ (nm)": wavelength_laser,
            "Intensity (mW/cm²)": intensity,
            "Photon Flux (cm⁻²·s⁻¹)": nphotons_laser,
            "PL Peak Position (eV)": energy_sample,
            "PL Peak FWHM (eV)": fwhm_sample_eV,
            "PLQY (%)": plqy * 100,
            "Voc Loss (mV)": voc_loss,
            "Attenuation Factor": attenuation,
            "Laser a": popt_laser[0],
            "Laser μ": popt_laser[1],
            "Laser σ": popt_laser[2],
            "Sample a": popt_sample[0],
            "Sample μ": popt_sample[1],
            "Sample σ": popt_sample[2],
        }

        df_results = pd.DataFrame([results_dict])
        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        csv_filename = f"PLQY_full_results_{timestamp}.csv"

        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv_buffer.getvalue(),
            file_name=csv_filename,
            mime="text/csv",
        )

    # ---------- PLOT TAB ----------
    with tab3:
        st.subheader("Spectra and Gaussian Fits")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_laser, y_laser, color="#f97316", lw=1.5, label="Laser")
        ax.plot(x_laser, gaussian(x_laser, *popt_laser),
                color="#fb923c", lw=2.5, ls="--", label="Laser Fit")
        ax.plot(x_sample, y_sample, color="#2563eb", lw=1.5, label="Sample")
        ax.plot(x_sample, gaussian(x_sample, *popt_sample),
                color="#60a5fa", lw=2.5, ls="--", label="Sample Fit")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Photoluminescence and Laser Gaussian Fits", fontsize=11, pad=8)
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        st.pyplot(fig)

else:
    st.info("⬅️ Please upload both a **sample** and a **laser** measurement file, or click *Load Example Data*.")
