import matplotlib.pyplot as plt ###
from scipy.io import wavfile

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
	
	print(rate)
	
    nfft = 256  # Length of the windowing segments
    fs = 256    # Sampling frequency
    print(data.shape, '**')
    pxx, freqs, bins, im = plt.specgram((data[:,0]+data[:,1])/2, nfft,fs)

    plt.axis('off')
    plt.savefig('test_spectro.png',
                dpi=100, # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0) # Spectrogram saved as a .png

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

if __name__ == '__main__': # Main function
    wav_file = 'C:\\Users\\sree2\\Desktop\\cs_599_deep_learning\\CSCI-599-DL\\sax_1_downsample.wav' # Filename of the wav file
    graph_spectrogram(wav_file)
	
	
