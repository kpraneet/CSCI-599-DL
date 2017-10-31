#working with only 1 wav file for now
import convert_to_image

if __name__ == '__main__': # Main function
    import  os
    ipath = '/home/khyati/Documents/599 DL/Project/IRMAS-Sample'
    opath = '/home/khyati/Documents/599 DL/Project/IRMAS-spectrograms'
    for root, subdirs, files in os.walk(ipath):
        if 'IRMAS-Sample' in root:
            newDir = root.replace('IRMAS-Sample','IRMAS-spectrograms')
            if not os.path.exists(newDir):
                os.mkdir(newDir)

        for i in files:
            if i.endswith('.wav'):
                 input = os.path.join(root,i)
                 outpath = root.replace('IRMAS-Sample','IRMAS-spectrograms')
                 outpath = outpath+'/'+i[:-4]+'_0.png'
                 convert_to_image.graph_spectrogram(input, outpath)
