import matplotlib.pyplot as plt ###
from scipy import signal
from scipy.io import wavfile
import pylab
import numpy as np
import librosa
from librosa import display
import os


def graph_melspectrogram(wav_file, des):
	print '%%%%',wav_file, des
	data = librosa.load(wav_file)
	wav_data = data[0]
	#now convert to stft
	D = librosa.stft(data[0], 1024, 512)
	#convert to melspectrogram
	S = librosa.feature.melspectrogram(S=D)
	print(S.shape)
	plt.figure(figsize=(10, 4))
	librosa.display.specshow(librosa.power_to_db(S,ref=np.max),	y_axis = 'log', x_axis = 'time')
	#plt.show()
	plt.savefig(des)
'''
def graph_melspectrogram(wav_file, des):
	rate, data = get_wav_info(wav_file)
	nfft = 1024  # Length of the windowing segments
	fs = 256    # Sampling frequency
#print(data.shape, '**')

	#f, t, Z = signal.stft(data, fs, nperseg=1000)
	Zxx = librosa.core.stft(data, nfft, 512)
	print Zxx
	librosa.display.specshow(librosa.amplitude_to_db(Zxx,	ref = np.max),	y_axis = 'log', x_axis = 'time')
	plt.title('Power spectrogram')
	plt.colorbar(format='%+2.0f dB')
	plt.tight_layout()
	#plt.show()
	plt.savefig(des)
'''


def get_wav_info(wav_file):
	rate, data = wavfile.read(wav_file)
	return rate, data

#if __name__ == '__main__': # Main function
#   wav_file = './sax_1_downsample_new.wav' # Filename of the wav file
#  graph_melspectrogram(wav_file)

proc_wavs = '/home/khyati/Dropbox/CSCI-599-DL/IRMAS-training_proc_wavs'

if __name__ == '__main__': # Main function
	for root, subdirs, files in os.walk(proc_wavs):
		if 'IRMAS-training_proc_wavs' in root:  ###
			newDir = root.replace('IRMAS-training_proc_wavs','IRMAS-training_spectrograms')
			if not os.path.exists(newDir):
				os.mkdir(newDir)

			for i in files:
				if i.endswith('.wav'):

					input = os.path.join(root,i)
					outpath = root.replace('IRMAS-training_proc_wavs','IRMAS-training_spectrograms')
					#out_wav_path = root.replace('IRMAS-Sample','IRMAS-proc_wavs')
					outpath = outpath+'/'+i[:-4]+'.png'
					#print 'i###',input, 'o######',outpath
					if not os.path.exists(outpath):
						#convert_to_image.graph_spectrogram(input, outpath)
						print '@@@', i
						graph_melspectrogram(input, outpath)