__author__ = "Ivan Zhezhera"
__date__ = "19.09.2024"


import cv2 
#from itertools import izip as zip
import time
import warnings
import logging
import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pandas import cut 
from lightglue import LightGlue, SuperPoint, DISK, viz2d
from lightglue.utils import load_image, rbd
import random




#Parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-img", "--input_image", default="./img2/", required = False,
	help = "Path to the directory that contains the imags")
ap.add_argument("-p", "--persent", default=0.1, required = False,
    help = "Part of images to processing ")

args = vars(ap.parse_args())



warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S') 


class Hist_analysis(object):
    def __init__(self, logging = logging, verbose = True, 
        image_path = args["input_image"], random_part = args["persent"]):

        self.verbose = verbose
        self.logging = logging
        self.image_path = image_path
        self.max_num_keypoints = 256
        self.device_type  = 'cpu' # 'mps'  
        self.device = torch.device("cuda" if torch.cuda.is_available() else self.device_type)
        self.random_part = random_part
        self.bins = 10


    def __read_files(self):
        return os.listdir(self.image_path)


    def __getPairs(self, file_list):
        res = [(a, b) for idx, a in enumerate(file_list) for b in file_list[idx + 1:]]
        return res


    def variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()


    def __match_pairs(self, pairs):
        pass


    def blur_analysis(self, file_list):
        #print(file_list)
        blur_levels = []
        for i in tqdm(range(len(file_list))):
            image = cv2.imread(self.image_path + file_list[i])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = self.variance_of_laplacian(image = gray)
            blur_levels.append(fm)
        return blur_levels

    def matches_detection(self, image0, image1)-> tuple:
        ''' Method for the matches finding with LightGlue algorithm'''

        # setup SUperGlue detector
        extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)  
        matcher = LightGlue(features="superpoint").eval().to(self.device)

        # finding points in both images
        feats0 = extractor.extract(image0.to(self.device))
        feats1 = extractor.extract(image1.to(self.device))

        # matching pairs
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        if self.verbose:
            self.logging.info(f"Detected: {len(matches)} pairs")
        return m_kpts0, m_kpts1, matches01, kpts0, kpts1

    def overlay_front_analysis(self, file_list, fast_file_list = None):
        overlay_levels = []
        height, width, channels = cv2.imread(self.image_path + file_list[0]).shape
        if fast_file_list is None:
            if self.verbose:
                self.logging.info(f"[INFO] Normal (full) analysis")

            for i in tqdm(range(len(file_list))):
                if i > 0:
                    image0 = load_image(self.image_path + file_list[i-1], resize = 0.5)
                    image1 = load_image(self.image_path + file_list[i], resize = 0.5)
                    m_kpts0, m_kpts1, _, _, _ = self.matches_detection(image0 = image0, image1 = image1)
                    #print(m_kpts0, m_kpts1)
                    
                    if len(m_kpts0) > 25:
                        max_y = 0
                        for j in range(len(m_kpts0)):
                            y = m_kpts0[j][1].numpy().astype(int)
                            if y > max_y:
                                max_y = y
                        overlap_value = max_y / height * 100
                        if overlap_value > 0:
                            overlay_levels.append(overlap_value)
                        else:
                            print("max_y = 0")

        else:
            if self.verbose:
                self.logging.info(f"[INFO] Fast analysis")

            for i in tqdm(range(len(fast_file_list))):
                if i > 0:
                    image0 = load_image(self.image_path + fast_file_list[i], resize = 0.5)
                    image1 = load_image(self.image_path + file_list[file_list.index(fast_file_list[i])-1], resize = 0.5)
                    m_kpts0, m_kpts1, _, _, _ = self.matches_detection(image0 = image0, image1 = image1)

                    if len(m_kpts0) > 25:
                        max_y = 0
                        for j in range(len(m_kpts0)):
                            y = m_kpts0[j][1].numpy().astype(int)
                            if y > max_y:
                                max_y = y
                        overlap_value = max_y / height * 100
                        if overlap_value > 0:
                            overlay_levels.append(overlap_value)
                        else:
                            print("max_y = 0")


        return overlay_levels

    def overlay_side_analysis(self, file_list, fast_file_list = None):
        overlay_levels = []
        height, width, channels = cv2.imread(self.image_path + file_list[0]).shape
        if fast_file_list is not None:
            if self.verbose:
                self.logging.info(f"[INFO] Fast analysis") 
            for i in tqdm(range(len(fast_file_list))):
                if i > 0:
                    image0 = load_image(self.image_path + fast_file_list[i], resize = 0.5)
                    index = file_list.index(fast_file_list[i])
                    
                    max_x = 0

                    for j in range(len(file_list)):
                        if j > index:
                            image1 = load_image(self.image_path + file_list[file_list.index(fast_file_list[i])-1], resize = 0.5)
                            m_kpts0, m_kpts1, _, _, _ = self.matches_detection(image0 = image0, image1 = image1)
                            #print(image0)
                            #print(image1)
                            if len(m_kpts0) > 25:
                                for j in range(len(m_kpts0)):
                                    x = m_kpts0[j][0].numpy().astype(int)
                                    if x > max_x:
                                        max_x = x
                                overlap_value = max_x / width * 100
                                if overlap_value > 0:
                                    overlay_levels.append(overlap_value)
                                    break
                                else:
                                    print("max_x = 0")

        
        return overlay_levels


    def main_process(self):
        file_list = self.__read_files()
        if len(file_list) > 5:
            fast_file_list_number = int(self.random_part * len(file_list))
            fast_file_list = random.choices(population=file_list, k=fast_file_list_number)
            pairs = self.__getPairs(file_list = fast_file_list)
            
            
            # BLUR 
            if self.verbose:
                self.logging.info(f"[INFO] Blur analysis") 
            blur_levels = self.blur_analysis(file_list = file_list)
            
            # OVERLAY FRONT
            if self.verbose:
                self.logging.info(f"[INFO] Overlap front analysis") 

            overlay_levels_front = self.overlay_front_analysis(file_list = file_list, fast_file_list = fast_file_list)

            # OVERLAY SIDE
            if self.verbose:
                self.logging.info(f"[INFO] Overlap side analysis") 

            overlay_levels_side = self.overlay_side_analysis(file_list = file_list, fast_file_list = fast_file_list)



            plt.subplots_adjust(hspace=.4)
            plt.subplot(311)
            plt.title('Blur')
            blur_levels = np.array(blur_levels) 
            _, bins = cut(blur_levels, bins=self.bins, retbins=True)
            plt.hist(blur_levels, bins)
            
            plt.subplot(312)
            plt.title('Overlap front')
            overlay_levels_front = np.array(overlay_levels_front) 
            _, bins = cut(overlay_levels_front, bins=self.bins, retbins=True)
            plt.hist(overlay_levels_front, bins)
            
            plt.subplot(313)
            plt.title('Overlap side')
            overlay_levels_side = np.array(overlay_levels_side) 
            _, bins = cut(overlay_levels_side, bins=self.bins, retbins=True)
            plt.hist(overlay_levels_side, bins)

            plt.savefig('res.png')
            plt.show()

        else:
            self.logging.warning(f"[WARNING] Not enought images")


if __name__ == "__main__":
    ha = Hist_analysis()
    ha.main_process()
    
