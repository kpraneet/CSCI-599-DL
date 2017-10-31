#working with only 1 wav file for now

import matplotlib.pyplot as plt
from scipy.io import wavfile

def graph_spectrogram(wav_file, opath):
    rate, data = get_wav_info(wav_file)
    nfft = 256  # Length of the windowing segments
    fs = 256    # Sampling frequency
    print data.shape, '**'
    pxx, freqs, bins, im = plt.specgram(data[:,0], nfft,fs)

    plt.axis('off')
    plt.savefig(opath,
                dpi=100, # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0) # Spectrogram saved as a .png

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

if __name__ == '__main__': # Main function
    wav_file = '/home/khyati/Documents/599 DL/IRMAS-Sample/Training/sax/118__[sax][nod][jaz_blu]1702__3.wav' # Filename of the wav file
    graph_spectrogram(wav_file)