from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np

wav_file = 'C:\\Users\\sree2\\Desktop\\cs_599_deep_learning\\CSCI-599-DL\\IRMAS-Sample\\Training\\sax\\sax_1.wav' # Filename of the wav file

# Load the data and calculate the time of each sample
samplerate, data = wavfile.read(wav_file)
times = np.arange(len(data))/float(samplerate)

# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(10, 10))
plt.fill_between(times, data[:,0], data[:,1], color='k') 
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.savefig('sample.png', dpi=100)
plt.show()