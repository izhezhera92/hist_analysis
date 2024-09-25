import cv2
import torch
from tqdm import tqdm

from lightglue import LightGlue, SuperPoint, DISK, viz2d
from lightglue.utils import load_image, rbd
import warnings

import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S') 

class AnalysisProcess(object):
    
    def __init__(self, image_path, verbose):
        self.image_path = image_path
        self.verbose = verbose
        #self.logging = logging
        self.max_num_keypoints = 256
        self.device_type  = 'cpu' # 'mps'  
        self.device = torch.device("cuda" if torch.cuda.is_available() else self.device_type)


        if torch.cuda.is_available():
            logging.info(f"Cude is available")
        else:
            logging.warning(f"Cude is not available")

        self.minimal_edge = 25
        self.resize_coef = 0.5

        self.blur_levels = None
        self.overlay_levels_front = None
        self.overlay_levels_side = None


    def __variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()


    def blur_analysis(self, file_list): 
        blur_levels = []
        for i in range(len(file_list)): #
            image = cv2.imread(self.image_path + file_list[i])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = self.__variance_of_laplacian(image = gray)
            blur_levels.append(fm)
        return blur_levels
        #self.blur_levels = blur_levels


    def __matches_detection(self, image0, image1)-> tuple:
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
        #if self.verbose:
        #    logging.info(f"Detected: {len(matches)} pairs")
        return m_kpts0, m_kpts1, matches01, kpts0, kpts1


    def overlay_front_analysis(self, file_list, fast_file_list = None):

        overlay_levels = []
        height, width, channels = cv2.imread(self.image_path + file_list[0]).shape

        if fast_file_list is None:

            for i in range(len(file_list)): #tqdm(
                if i > 0:
                    image0 = load_image(self.image_path + file_list[i-1], resize = self.resize_coef)
                    image1 = load_image(self.image_path + file_list[i], resize = self.resize_coef)
                    m_kpts0, m_kpts1, _, _, _ = self.__matches_detection(image0 = image0, image1 = image1)
                    #print(m_kpts0, m_kpts1)
                    
                    if len(m_kpts0) > self.minimal_edge:
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

            for i in range(len(fast_file_list)):#tqdm(
                if i > 0:
                    image0 = load_image(self.image_path + fast_file_list[i], resize = self.resize_coef)
                    image1 = load_image(self.image_path + file_list[file_list.index(fast_file_list[i])-1], resize = self.resize_coef)
                    m_kpts0, m_kpts1, _, _, _ = self.__matches_detection(image0 = image0, image1 = image1)

                    if len(m_kpts0) > self.minimal_edge:
                        max_y = 0
                        for j in range(len(m_kpts0)):
                            y = m_kpts0[j][1].numpy().astype(int)
                            if y > max_y:
                                max_y = y

                        overlap_value = max_y / height * 100
                        if overlap_value > 0:
                            overlay_levels.append(overlap_value)
                        else:
                            #self.logging.warning(f"[WARNING] Uncorrect y value: {overlap_value}")
                            pass


        return overlay_levels
        #self.overlay_levels_front = overlay_levels


    def overlay_side_analysis(self, file_list, fast_file_list = None):
        overlay_levels = []
        height, width, channels = cv2.imread(self.image_path + file_list[0]).shape
        if fast_file_list is not None:

            for i in range(len(fast_file_list)): #tqdm(
                if i > 0:
                    image0 = load_image(self.image_path + fast_file_list[i], resize = self.resize_coef)
                    index = file_list.index(fast_file_list[i])
                    
                    max_x = 0

                    for j in range(len(file_list)):
                        if j > index:
                            image1 = load_image(self.image_path + file_list[file_list.index(fast_file_list[i])-1], resize = self.resize_coef)
                            m_kpts0, m_kpts1, _, _, _ = self.__matches_detection(image0 = image0, image1 = image1)

                            if len(m_kpts0) > self.minimal_edge:
                                for j in range(len(m_kpts0)):
                                    x = m_kpts0[j][0].numpy().astype(int)
                                    if x > max_x:
                                        max_x = x

                                overlap_value = max_x / width * 100
                                if overlap_value > 0:
                                    overlay_levels.append(overlap_value)
                                    break
                                else:
                                    #self.logging.warning(f"[WARNING] Uncorrect x value: {overlap_value}")
                                    pass

        
        return overlay_levels
        #self.overlay_levels_side = overlay_levels
