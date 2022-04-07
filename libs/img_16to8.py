import cv2
import os
import numpy as np


class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix='.jpg'):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        files_list = []

        for dirpath, dirnames, filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..').
            filenames is a list of the names of the non-directory files in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list


if __name__ == "__main__":
    dir_depth = r"/Users/yidawang/Documents/database/TableTop/depth"
    dir_target = r"/Users/yidawang/Documents/database/CAMPAR_depth"
    scan = ScanFile(dir_depth)
    files = scan.scan_files()

    for file in files:
        # read depth images as 16 bit array
        img16 = cv2.imread(file, -1)
        # convert 16 bit to 8 bit
        img8 = (img16 * 255.0 / 1480).astype('uint8')
        # convert gray scale to rgb
        img = cv2.merge((img8, img8, img8))
        label_start = file.rfind('/') + 10
        #import pdb; pdb.set_trace()
        cv2.imwrite(dir_target + '/depth/lab/01_' + file[label_start:], img)

    a = img

    dir_depth = r"/Users/yidawang/Documents/database/TableTop/rgb"
    scan = ScanFile(dir_depth)
    files = scan.scan_files()

    for file in files:
        # 1 for RGB and 0 for gray
        img = cv2.imread(file, 1)
        label_start = file.rfind('/') + 10
        cv2.imwrite(dir_target + '/rgb/lab/01_' + file[label_start:], img)
