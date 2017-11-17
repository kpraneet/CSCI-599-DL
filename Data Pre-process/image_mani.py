import cv2
import os

#inpath = '/home/khyati/Documents/599_DL/Project/IRMAS-spectrograms'
inpath = '/home/khyati/Dropbox/CSCI-599-DL/IRMAS-training_spectrograms'

image_list = []
image_labels = []

def image_transform(input_img):
    image = cv2.imread(input_img, 0)
    new_dim = (43, 128)
    resized = cv2.resize(image, new_dim)
    return resized

for root, subdirs, files in os.walk(inpath):
    if 'IRMAS-training_spectrograms' in root:
        for i in files:
            if i.endswith('.png'):
                ind_ = []
                path = root+'/'+i

                if '__' in path:
                    print root
                    try:
                        label = root[-3:]
                    except:
                        print '**not found for',path
                    image_list.append(image_transform(path))
                    image_labels.append(label)

print len(image_labels)