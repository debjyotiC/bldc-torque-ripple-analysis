import pandas as pd
import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt

path = "data/with_Buck.csv"

df = pd.read_csv(path)

torque = np.array(df["Torque"])
time_base = np.array(df["Time"])

N = torque.size
T = 1.006e-06

ft = abs(fft(torque) * T)
freq = abs(fftfreq(N, d=T))


fig, axs = plt.subplots(2, 1)
plt.suptitle(f"Torque spectra {path.split('.')[0]}", fontsize=14)

# plot Torque
axs[0].plot(time_base, torque, '-', label='Torque')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Torque')
axs[0].grid(True)
axs[0].legend(loc='best')

# plot Torque FFT
axs[1].semilogy(freq/1000, ft, '-', label='Torque FFT')
axs[1].set_xlabel('Frequency (kHz)')
axs[1].set_ylabel('FFT Mag')
axs[1].grid(True)
axs[1].legend(loc='best')
plt.savefig(f"{path.split('.')[0]}", dpi=600)
plt.show()
