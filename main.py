
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Curve Fitter", layout="centered")
st.title("Cost vs Conversions Power Curve Fitter")

st.write("""
This app allows you to input data points (Cost, Conversions), fits a power curve to them, and interactively explore the curve using sliders. The selected point is shown on the chart.
""")

st.header("Step 1: Ingest Data Points")
upload = st.file_uploader("Upload CSV with columns Cost,Conversions, or enter points manually:", type=["csv"])

if upload is not None:
    df = pd.read_csv(upload)
    st.write("Uploaded Data:")
    st.dataframe(df)
else:
    st.write("Or enter data points below:")
    default = "10000,500\n15000,600\n20000,700\n25000,800\n30000,900"
    points_text = st.text_area("Enter Cost,Conversions pairs (comma separated, one per line):", value=default, height=120)
    try:
        data = [tuple(map(float, line.split(","))) for line in points_text.strip().split("\n") if line]
        df = pd.DataFrame(data, columns=["Cost", "Conversions"])
        st.write("Manual Data:")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        df = None

# Step 2: Fit power curve and plot
if df is not None and not df.empty:
	st.header("Step 2: Power Curve Fitting and Visualisation")
	x = df["Cost"].values
	y = df["Conversions"].values
	# Fit power curve: y = a * x^b
	# log(y) = log(a) + b*log(x)
	mask = (x > 0) & (y > 0)
	x_pos = x[mask]
	y_pos = y[mask]
	logx = np.log(x_pos)
	logy = np.log(y_pos)
	b, loga = np.polyfit(logx, logy, 1)
	a = np.exp(loga)
	def power_curve(x):
		return a * x ** b
	x_fit = np.linspace(min(x), max(x), 200)
	y_fit = power_curve(x_fit)

	# Interactive selection
	st.header("Step 3: Explore the Curve")
	col1, col2 = st.columns(2)
	with col1:
		x_val = st.slider("Move along Cost", float(min(x)), float(max(x)), float(np.median(x)), key="x_slider")
		y_from_x = float(power_curve(x_val))
		st.metric("Conversions at Cost", f"{y_from_x:.2f}")
	with col2:
		y_val = st.slider("Move along Conversions", float(min(y)), float(max(y)), float(np.median(y)), key="y_slider")
		# Find closest Cost for this Conversions on the curve
		idx = (np.abs(y_fit - y_val)).argmin()
		x_from_y = float(x_fit[idx])
		st.metric("Cost at Conversions", f"{x_from_y:.2f}")

	# Plot with selected point
	fig, ax = plt.subplots()
	ax.scatter(x, y, color="blue", label="Data Points")
	ax.plot(x_fit, y_fit, color="red", label="Power Fit")
	ax.scatter([x_val], [y_from_x], color="green", s=80, label="Selected (Cost→Conversions)")
	ax.scatter([x_from_y], [y_val], color="orange", s=80, label="Selected (Conversions→Cost)")
	ax.set_xlabel("Cost")
	ax.set_ylabel("Conversions")
	ax.legend()
	st.pyplot(fig)
