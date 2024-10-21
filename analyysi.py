import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
import streamlit as st
import folium
from streamlit_folium import st_folium

accel_data = pd.read_csv("Acceleration.csv")
gps_data = pd.read_csv("Location.csv")

# Lowpass-suodatin
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Näytteenottotaajuus
time_diffs = np.diff(accel_data["Time (s)"])
mean_time_diff = np.mean(time_diffs)
fs = 1 / mean_time_diff

# Suodatetaan kiihtyvyysdata, tähän on valittu y-akseli
cutoff = 3.0 
accel_y = accel_data["Linear Acceleration y (m/s^2)"]
accel_y_filt = butter_lowpass_filter(accel_y, cutoff, fs)

# Askelmäärän laskeminen suodatesta y-akselista
from scipy.signal import find_peaks
peaks, _ = find_peaks(accel_y_filt, height=0.3, distance=fs*0.5)
step_count = len(peaks)

# Fourier-analyysi
n = len(accel_y_filt)
yf = fft(accel_y_filt)
xf = np.fft.fftfreq(n, 1/fs)
power_spectrum = np.abs(yf[:n//2])  
xf = xf[:n//2] 

# Valitaan taajuudet väliltä 1.0 - 2.5 Hz
step_frequency_range = (1.0, 2.5)  # Oletetaan että askeleet löytyy tältä taajuusalueelta
valid_frequencies = (xf > step_frequency_range[0]) & (xf < step_frequency_range[1])

# Etsitään suurin huippu tehospektristä validilla taajuusalueella
if np.any(valid_frequencies):
    dominant_frequency = xf[valid_frequencies][np.argmax(power_spectrum[valid_frequencies])]
else:
    dominant_frequency = 0  # Jos ei löydy taajuutta, asetetaan nollaksi

# Lasketaan askelmäärä Fourier-analyysin perusteella
total_time = accel_data["Time (s)"].iloc[-1] - accel_data["Time (s)"].iloc[0]
step_count_fourier = dominant_frequency * total_time

# Kuljettu matka (Haversine-kaava)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

distances = [haversine(gps_data["Latitude (°)"][i], gps_data["Longitude (°)"][i], gps_data["Latitude (°)"][i+1], gps_data["Longitude (°)"][i+1]) for i in range(len(gps_data)-1)]
total_distance = sum(distances)

# Keskinopeus
average_speed = gps_data["Velocity (m/s)"].mean()

# Askelpituus (laskettu suodatetusta datasta)
step_length = total_distance / step_count

# Streamlit
st.title("Analyysi")

st.subheader("Tulokset")
st.write(f"Askelmäärä (suodatetusta datasta): {step_count}")
st.write(f"Askelmäärä (Fourier-analyysistä): {step_count_fourier:.0f}")
st.write(f"Keskinopeus: {average_speed:.2f} m/s")
st.write(f"Kuljettu matka: {total_distance:.2f} m")
st.write(f"Askelpituus (laskettu suodatetusta datasta): {step_length:.2f} m")

st.subheader("Suodatettu Kiihtyvyys (Y-akseli)")
fig, ax = plt.subplots()
ax.plot(accel_data["Time (s)"], accel_y_filt)
ax.set_xlabel("Aika (s)")
ax.set_ylabel("Suodatettu kiihtyvyys y-akselilla (m/s^2)")
st.pyplot(fig)

# Tehospektri, taajuudet 0.5-10 Hz
st.subheader("Tehospektri (Fourier) - Taajuudet 0.5-10 Hz")
fig, ax = plt.subplots()
valid_idx = (xf > 0.5) & (xf <= 10)
ax.plot(xf[valid_idx], power_spectrum[valid_idx])
ax.set_xlabel("Taajuus (Hz)")
ax.set_ylabel("Teho")
st.pyplot(fig)

st.subheader("Reitti kartalla")
start_coords = [gps_data["Latitude (°)"].iloc[0], gps_data["Longitude (°)"].iloc[0]]
mymap = folium.Map(location=start_coords, zoom_start=15)
coordinates = list(zip(gps_data["Latitude (°)"], gps_data["Longitude (°)"]))
folium.PolyLine(locations=coordinates, color="blue", weight=2.5, opacity=1).add_to(mymap)
st_data = st_folium(mymap, width=700, height=500)

