import pandas as pd
import numpy as np
from scipy.signal import spectrogram, get_window
import plotly.graph_objs as go
import plotly.subplots as sp

# --- Parameters ---
fd = 47.5e3         # Drive frequency
f1 = fd / 2         # 1fd
f2 = fd             # 2fd
window_type = 'hann'

# --- Load CSV ---
df = pd.read_csv("your_file.csv")  # Replace with your filename
time = df['time'].values
velocity = df['velocity'].values

# --- Sampling ---
dt = np.mean(np.diff(time))  # Time resolution
fs = 1 / dt                  # Sampling frequency

# --- Acceleration (a = dv/dt) ---
acceleration = np.diff(velocity) / dt
t_acc = time[:-1]  # Same length as acceleration

# --- FFT of acceleration ---
n = len(acceleration)
window = get_window(window_type, n)
fft_vals = np.fft.fft(acceleration * window)
fft_freqs = np.fft.fftfreq(n, d=dt)

# Only keep positive frequencies
fft_freqs = fft_freqs[:n // 2]
fft_vals = np.abs(fft_vals[:n // 2])

# --- Create Plotly Subplots ---
fig = sp.make_subplots(rows=3, cols=1, subplot_titles=[
    "Velocity vs Time",
    "Acceleration FFT around 1fd (≈23.75kHz)",
    "Acceleration FFT around 2fd (≈47.5kHz)"
])

# --- Plot 1: Velocity vs Time ---
fig.add_trace(go.Scatter(x=time, y=velocity, mode='lines', name='Velocity (m/s)'), row=1, col=1)

# --- Plot 2: FFT near 1fd ---
mask_1fd = (fft_freqs > f1 - 7500) & (fft_freqs < f1 + 7500)
fig.add_trace(go.Scatter(x=fft_freqs[mask_1fd], y=fft_vals[mask_1fd],
                         mode='lines', name='FFT around 1fd'), row=2, col=1)

# --- Plot 3: FFT near 2fd ---
mask_2fd = (fft_freqs > f2 - 7500) & (fft_freqs < f2 + 7500)
fig.add_trace(go.Scatter(x=fft_freqs[mask_2fd], y=fft_vals[mask_2fd],
                         mode='lines', name='FFT around 2fd'), row=3, col=1)

# --- Layout Settings ---
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=1)

fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
fig.update_yaxes(title_text="Amplitude (a in m/s²)", row=2, col=1)

fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=1)
fig.update_yaxes(title_text="Amplitude (a in m/s²)", row=3, col=1)

fig.update_layout(height=900, width=1000, title_text="Vibration Analysis Results")
fig.show()
