import cv2
import torch
from tqdm import tqdm

from lightglue import LightGlue, SuperPoint, DISK, viz2d
from lightglue.utils import load_image, rbd
import warnings

import logging
import pandas as pd
from pandas import cut 
import os

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S') 

class AnalysisProcess(object):
    
    def __init__(self, image_path, settings, verbose):
        self.image_path = image_path
        self.verbose = verbose
        self.settings = settings
        print(self.settings)
        #self.logging = logging
        self.max_num_keypoints = int(self.settings['settings']['max_num_keypoints'])
        self.device_type  = "cpu"#str(self.settings['settings']['cpu']) # 'mps'  
        self.device = torch.device("cuda" if torch.cuda.is_available() else self.device_type)
     

        if torch.cuda.is_available():
            logging.info(f"Cude is available")
        else:
            logging.warning(f"Cude is not available")

        self.minimal_edge = int(self.settings['settings']['minimal_edge'])
        self.resize_coef = float(self.settings['settings']['resize_coef'])

        self.blur_levels = None
        self.overlay_levels_front = None
        self.overlay_levels_side = None
        self.csv_path = str(self.settings['settings']['csv_path'])


    def __variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def __append_to_csv(self, data, csv_file, columns):
        file_exists = os.path.isfile(csv_file)
        #df = pd.DataFrame(data, columns=columns)  # Adjust columns as needed
        data.to_csv(csv_file, mode='a', header=not file_exists, index=False)
        return 0

    def files_concate(self, path):
        result = None
        csv_list = os.listdir(self.image_path)
        for i in range(len(csv_list)):
            df = pd.read_csv(csv_list[i])
            result = pd.concat([result, df], axis=1)
        return result


    def __draw_result(self, data = None):

        if data is not None:
            plt.subplots_adjust(hspace=.4)
            plt.subplot(211)
            plt.title('Blur')
            #_, bins = cut(self.blur_levels, bins=self.bins, retbins=True)
            #values, bins, _ = plt.hist(self.blur_levels, bins)
            
            data['blur'].hist(bins=10, edgecolor='black')
            # Add labels and title
            plt.title('Blur Level Distribution')
            plt.xlabel('Blur Level')
            plt.ylabel('Frequency')

            plt.subplot(212)
            plt.title('Overlap front')

            #categories, bins = cut(self.overlay_levels_front, bins=self.bins, retbins=True)
            #plt.hist(self.overlay_levels_front, bins)

            data['front'].hist(bins=10, edgecolor='black')
            # Add labels and title
            plt.title('Front Overlap Level')
            plt.xlabel('Overlap Level')
            plt.ylabel('Frequency')
                
            #ersentages = categories.value_counts().sort_index() 
            #result_persentages =  persentages/persentages.sum() * 100
            #logging.info(f"Persentages: {result_persentages}")
                    
            #plt.subplot(313)
            #plt.title('Overlap side')
            #_, bins = cut(self.overlay_levels_side, bins=self.bins, retbins=True)
            #plt.hist(self.overlay_levels_side, bins)

            plt.savefig(self.result_image, dpi=300, bbox_inches='tight')
            plt.close()
            return 0
        return 1


    def blur_analysis(self, fast_file_list):
        columns=["Iteration", "blur"]
        csv_file = self.csv_path + "blur.csv"
        blur_levels = []
        for i in range(len(fast_file_list)): #
            image = cv2.imread(self.image_path + fast_file_list[i])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = round(self.__variance_of_laplacian(image = gray),2)
            df = pd.DataFrame([[i, fm, fast_file_list[i]]], columns)
            self.__append_to_csv(data = df, csv_file = csv_file, columns = columns)

            blur_levels.append([i, fm, fast_file_list[i]])
            print(f"blur {i}: {fast_file_list[i]}")

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
        columns = ["Iteration", "front", "name"]
        csv_file = self.csv_path +  "front.csv"

        overlay_levels = []
        height, width, channels = cv2.imread(self.image_path + file_list[0]).shape

        if fast_file_list is None:

            for i in range(len(file_list)): #tqdm(
                if i > 0:
                    print(f"front {i}: {file_list[i]}")
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
                        overlap_value = round(overlap_value,2)

                        if overlap_value > 0:
                            
                            df = pd.DataFrame([[i, overlap_value, file_list[i]]], columns=columns)
                            self.__append_to_csv(data = df, csv_file = csv_file, columns = columns)

                            overlay_levels.append([i, overlap_value, file_list[i]])
                        else:
                            print("max_y = 0")


        else:

            for i in range(len(fast_file_list)):#tqdm(
                if i > 0:
                    print(f"front {i}: {fast_file_list[i]}")
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
                        overlap_value = round(overlap_value, 2)
                        if overlap_value > 0:
                            
                            df = pd.DataFrame([[i, overlap_value, fast_file_list[i]]], columns=columns)
                            self.__append_to_csv(data = df, csv_file = csv_file, columns = columns)

                            overlay_levels.append([i, overlap_value, fast_file_list[i]])
                        else:
                            #self.logging.warning(f"[WARNING] Uncorrect y value: {overlap_value}")
                            pass


        return overlay_levels
        #self.overlay_levels_front = overlay_levels


    def overlay_side_analysis(self, file_list, fast_file_list = None):
        overlay_levels = []
        height, width, channels = cv2.imread(self.image_path + file_list[0]).shape
        if fast_file_list is not None:

            for i in tqdm(range(len(fast_file_list))): #tqdm(
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
                                   # overlay_levels.append(overlap_value)
                                    overlay_levels.append([i, overlap_value])
                                    break
                                else:
                                    #self.logging.warning(f"[WARNING] Uncorrect x value: {overlap_value}")
                                    pass

        
        return overlay_levels
        #self.overlay_levels_side = overlay_levels
