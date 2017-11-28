import os
import cv2
import numpy as np


def main():
    filecount = 0
    train_data = []
    train_label = []
    eval_data = []
    eval_label = []
    test_data = []
    test_label = []
    label_dict = {"cel": 0, "cla": 1, "flu": 2, "gac": 3, "gel": 4, "org": 5, "pia": 6, "sax": 7, "tru": 8, "vio": 9,
                  "voi": 10}
    path = "/Users/praneet/Documents/USC - MS/Fall 2017/CS 599 - DL/CSCI-599-DL/CNN/IRMAS-training_spectrograms"
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith(".png"):
                filecount += 1
                file_path = os.path.abspath(os.path.join(root, file_name))
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fixed_size = (128, 43)
                img = cv2.resize(img, dsize=fixed_size)
                print(file_name)
                file_label = root.split('/')[-1]
                if filecount % 7 == 0:
                    eval_data.append(img)
                    tmp = [0] * 11
                    tmp[label_dict[file_label]] = 1
                    eval_label.append(tmp)
                elif filecount % 11 == 0:
                    test_data.append(img)
                    tmp = [0] * 11
                    tmp[label_dict[file_label]] = 1
                    test_label.append(tmp)
                else:
                    train_data.append(img)
                    tmp = [0] * 11
                    tmp[label_dict[file_label]] = 1
                    train_label.append(tmp)
    print(filecount)
    print(len(train_data), len(train_label), len(eval_data), len(eval_label), len(test_data), len(test_label))
    train_data = np.array(train_data, np.float32) / 255.
    train_label = np.array(train_label, np.int32)
    eval_data = np.array(eval_data, np.float32) / 255.
    eval_label = np.array(eval_label, np.int32)
    test_data = np.array(test_data, np.float32) / 255.
    test_label = np.array(test_label, np.int32)
    print(train_data.shape, train_label.shape)
    print(eval_data.shape, eval_label.shape)
    print(test_data.shape, test_label.shape)
    np.save('train_data', train_data)
    np.save('train_label', train_label)
    np.save('eval_data', eval_data)
    np.save('eval_label', eval_label)
    np.save('test_data', test_data)
    np.save('test_label', test_label)


if __name__ == '__main__':
    main()
