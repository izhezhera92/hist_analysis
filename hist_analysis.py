__author__ = "Ivan Zhezhera"
__date__ = "19.09.2024"


import cv2 
import time
import warnings
import logging
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pandas import cut 
from scipy.stats import norm

import random
import multiprocessing
from multiprocessing import Process, Queue
import queue 

import json
from json import JSONEncoder
import numpy

import analysis_process as anp



ap = argparse.ArgumentParser()
ap.add_argument("-img", "--input_image", default="./img3/", required = False,
    help = "Path to the directory that contains the imags")
ap.add_argument("-p", "--persent", default=0.08, required = False,
    help = "Part of images to processing ")

args = vars(ap.parse_args())



warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S') 
  


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)



class Hist_analysis(object):
    def __init__(self, logging = logging, verbose = True, demo = True,
        result_image = './res.png', bins = 10, minimal_image_quantity = 5,
        image_path = args["input_image"], random_part = args["persent"]):

        self.verbose = verbose
        self.demo = demo
        #self.logging = logging
        self.image_path = image_path
        self.result_image = result_image
        self.random_part = random_part

        self.bins = bins
        self.minimal_image_quantity = minimal_image_quantity

        self.blur_levels = []
        self.overlay_levels_front = []
        self.overlay_levels_side = []

        self.analytic = anp.AnalysisProcess(image_path = self.image_path, 
            verbose = self.verbose) #logging = self.logging
        self.health_images_edge = 90
        self.health = "ok"

        self.minim_blur_val = 100
        self.minim_overlap_front_value = 45
        self.minim_overlap_side_value = 40


    def __read_files(self):
        return os.listdir(self.image_path)


    def __count_elements_less_than(self, lst, value):
        qantity = len([x for x in lst if x < value])
        relative = qantity / len(lst) * 100
        return relative


    def __draw_result(self):
        if self.demo:
            plt.subplots_adjust(hspace=.4)

            plt.subplot(311)
            plt.title('Blur')
            _, bins = cut(self.blur_levels, bins=self.bins, retbins=True)
            (mu, sigma) = norm.fit(self.blur_levels)
            x = np.linspace(min(self.blur_levels), max(self.blur_levels), 100)
            values, bins, _ = plt.hist(self.blur_levels, bins)

            area = sum(np.diff(bins) * values)
            plt.plot(x, mlab.normpdf(x, mu, sigma))
                
            plt.subplot(312)
            plt.title('Overlap front')

            _, bins = cut(self.overlay_levels_front, bins=self.bins, retbins=True)
            plt.hist(self.overlay_levels_front, bins)
                
            plt.subplot(313)
            plt.title('Overlap side')
            _, bins = cut(self.overlay_levels_side, bins=self.bins, retbins=True)
            plt.hist(self.overlay_levels_side, bins)

            plt.savefig(self.result_image)
            plt.show()


    def __resume(self):

        blurred_images_persent = self.__count_elements_less_than(lst = self.blur_levels, value = self.minim_blur_val)

        if (100 - self.health_images_edge) >= blurred_images_persent:
            logging.info(f"Blured images: {round(blurred_images_persent,1)} %. It's ok!")
        else:
            self.health = "not_ok"
            logging.warning(f"Attention! Blured images: {round(blurred_images_persent,1)} %. It's too much!")

        low_front_overlap_persent = self.__count_elements_less_than(lst = self.overlay_levels_front, value = self.minim_overlap_front_value)

        if (100 - self.health_images_edge) >= low_front_overlap_persent:
            logging.info(f"Low front overlap images: {round(low_front_overlap_persent,1)} %. It's ok!")
        else:
            self.health = "not_ok"
            logging.warning(f"Attention! Low front overlap images: {round(low_front_overlap_persent,1)} %. It's too much!")

        low_side_overlap_persent = self.__count_elements_less_than(lst = self.overlay_levels_side, value = self.minim_overlap_side_value)

        if (100 - self.health_images_edge) >= low_side_overlap_persent:
            logging.info(f"Low front overlap images: {round(low_side_overlap_persent,1)} %. It's ok!")
        else:
            self.health = "not_ok"
            logging.warning(f"Attention! Low front overlap images: {round(low_side_overlap_persent,1)} %. It's too much!")


    def __core_calculation(self, fast_file_list, queue):
        blur_process = Process(target=self.blur_analysis, args=(self.file_list, queue))
        front_overlap_process = Process(target=self.overlay_front_analysis, args=(self.file_list, fast_file_list, queue))
        side_overlap_process = Process(target=self.overlay_side_analysis, args=(self.file_list, fast_file_list, queue))

        blur_process.start()
        front_overlap_process.start()
        side_overlap_process.start()

        blur_process.join()
        front_overlap_process.join()
        side_overlap_process.join()

        return queue


    def blur_analysis(self, file_list, queue = None):
        if self.verbose:
            logging.info(f"Blur analysis starting ..")
        blur_levels = self.analytic.blur_analysis(file_list = file_list)
        queue.put({"blur_levels":blur_levels})
        if self.verbose:
            logging.info(f"Blur analysis complete.")


    def overlay_front_analysis(self, file_list, fast_file_list = None, queue = None):
        if self.verbose:
            logging.info(f"Overlap front analysis starting ..")
        overlay_levels_front = self.analytic.overlay_front_analysis(file_list = file_list, fast_file_list = fast_file_list)
        queue.put({"overlay_levels_front":overlay_levels_front})
        if self.verbose:
            logging.info(f"Overlap front analysis complete.")


    def overlay_side_analysis(self, file_list, fast_file_list = None, queue = None):
        if self.verbose:
            logging.info(f"Overlap side analysis starting ..")
        overlay_levels_side = self.analytic.overlay_side_analysis(file_list = file_list, fast_file_list = fast_file_list)
        queue.put({"overlay_levels_side":overlay_levels_side})
        if self.verbose:
            logging.info(f"Overlap side analysis complete.")



    def main_process(self):
        queue = Queue()
        self.file_list = self.__read_files()
        self.file_list = sorted(self.file_list)
        quantity = len(self.file_list)
        in_use = int(quantity * self.random_part)

        logging.info(f"Total quantity of images: {quantity}")
        logging.info(f"In use images: {in_use}")


        if quantity > self.minimal_image_quantity:
            fast_file_list_number = int(self.random_part * quantity)
            fast_file_list = random.choices(population=self.file_list, k=fast_file_list_number)

            queue = self.__core_calculation(fast_file_list = fast_file_list, queue = queue)

            analysis_data = {}
            while not queue.empty():
                analysis_data.update(queue.get())

            self.blur_levels = analysis_data['blur_levels']
            self.overlay_levels_front = analysis_data['overlay_levels_front']
            self.overlay_levels_side = analysis_data['overlay_levels_side']

            
            self.blur_levels = np.array(self.blur_levels)

            if len(self.overlay_levels_front)>=self.minimal_image_quantity:
                self.overlay_levels_front = np.array(self.overlay_levels_front)
            if len(self.overlay_levels_side)>=self.minimal_image_quantity:
                self.overlay_levels_side = np.array(self.overlay_levels_side) 

            self.__draw_result()

            self.__resume()

        else:
            logging.warning(f"Not enought images")


        numpyBlurData = {"blur": self.blur_levels}
        encodedNumpyBlurData = json.dumps(numpyBlurData, cls=NumpyArrayEncoder)
 
        numpyFrontData = {"front": self.overlay_levels_front}
        encodedNumpyFrontData = json.dumps(numpyFrontData, cls=NumpyArrayEncoder)

        numpySideData = {"side": self.overlay_levels_side}
        encodedNumpySideData = json.dumps(numpySideData, cls=NumpyArrayEncoder)

        return encodedNumpyBlurData, encodedNumpyFrontData, encodedNumpySideData, self.health    



if __name__ == "__main__":
    ha = Hist_analysis()
    ha.main_process()
    


