import wave
import audioop
import sys
import os


def downsampleWav(src, dst, inrate=44100, outrate=22050, inchannels=1, outchannels=1):
	if not os.path.exists(src):
		print('Source not found!')
		return False

	if not os.path.exists(os.path.dirname(dst)):
		os.makedirs(os.path.dirname(dst))

	try:
		s_read = wave.open(src, 'r')
		s_write = wave.open(dst, 'w')
	except:
		print('Failed to open files!')
		return False

	n_frames = s_read.getnframes()
	
	data = s_read.readframes(n_frames)
	
	data = audioop.tomono(data,2,0.5,0.5)
	
	try:
		converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
		#if outchannels == 1 & inchannels != 1:
			#converted[0] = audioop.tomono(converted[0], 2, 1, 0)
	except:
		print('Failed to downsample wav')
		return False
		
	#divide by the max time domain values
	try:
		max_val = audioop.maxpp(converted[0],2)
		coverted = audioop.mul(converted[0],2,1/max_val)
	except:
		print('FAILURE IN TIME DOMAIN CONVERSION')
		return False

				
	try:
		s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
		s_write.writeframes(converted[0])
	except:
		print('Failed to write wav')
		return False

	try:
		s_read.close()
		s_write.close()
	except:
		print('Failed to close wav files')
		return False

	return True
'''
if __name__ == "__main__":
	path_in = 'C:\\Users\\sree2\\Desktop\\cs_599_deep_learning\\CSCI-599-DL\\IRMAS-Sample\\Training\\sax\\'
	path_out = 'C:\\Users\\sree2\\Dropbox\\CSCI-599-DL\\'

	for file in os.listdir(path_in):
		
		if file.endswith(".wav"):
			src = path_in + file 
			dst = path_out + file.split(".")[0] + "_downsample_new.wav"
			downsampleWav(src, dst)
'''
if __name__ == '__main__': # Main function
	import  os
	#ipath = '/home/khyati/Documents/599 DL/Project/IRMAS-Sample'
    #opath = '/home/khyati/Documents/599 DL/Project/IRMAS-spectrograms'
	ipath = '/home/khyati/Dropbox/CSCI-599-DL/IRMAS-TrainingData'
	proc_wavs = '/home/khyati/Dropbox/CSCI-599-DL/IRMAS-training_proc_wavs'
	opath = '/home/khyati/Dropbox/CSCI-599-DL/IRMAS-training_spectrograms'
	for root, subdirs, files in os.walk(ipath):
		if 'IRMAS-TrainingData' in root:  ###
			newDir = root.replace('IRMAS-TrainingData','IRMAS-training_proc_wavs')
		if not os.path.exists(newDir):
			os.mkdir(newDir)

		for i in files:
			if i.endswith('.wav'):
				input = os.path.join(root,i)
				outpath = root.replace('IRMAS-TrainingData','IRMAS-training_proc_wavs')
				out_wav_path = root.replace('IRMAS-TrainingData','IRMAS-training_proc_wavs')
				outpath = outpath+'/'+i[:-4]+'.wav'
				#print outpath
				#convert_to_image.graph_spectrogram(input, outpath)
				downsampleWav(input, outpath)
