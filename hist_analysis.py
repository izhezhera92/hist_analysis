__author__ = "Ivan Zhezhera"
__date__ = "19.09.2024"


import warnings
import logging
import os
import yaml

from datetime import datetime

import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

import random
from multiprocessing import Process, Queue
import queue 

from json import JSONEncoder, dumps

import analysis_process as anp




ap = argparse.ArgumentParser()
ap.add_argument("-img", "--input_image", default="E:/farsightvision/67/ADTi_F003/", required = False, #
    help = "Path to the directory that contains the imags")
ap.add_argument("-p", "--persent", default=1.0, required = False,
    help = "Part of images to processing ")
ap.add_argument("-set", "--path_default_settings", default="./config.yaml", required = False,
    help = "Path to config file")

args = vars(ap.parse_args())


warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S') 
  


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)



class Hist_analysis(object):
    def __init__(self, logging = logging, 
        result_image = './res_test_05.png', 
        image_path = args["input_image"], random_part = args["persent"],
        settings = None):

        if settings is None:
            settings_path = args["path_default_settings"]
            with open(settings_path, "r") as file:
                self.settings = yaml.safe_load(file)
        else:
            self.settings = settings

        
        self.verbose = bool(self.settings['settings']['verbose'])
        self.demo = bool(self.settings['settings']['demo'])
        self.image_path = image_path
        self.result_image = result_image
        self.random_part = random_part

        self.minimal_image_quantity = int(self.settings['settings']['minimal_image_quantity'])

        self.blur_levels = []
        self.overlay_levels_front = []
        self.overlay_levels_side = []

        self.analytic = anp.AnalysisProcess(image_path = self.image_path, 
            verbose = self.verbose, settings = self.settings) #logging = self.logging
        
        self.health_images_edge = int(self.settings['settings']['health_images_edge'])
        self.health = "ok"

        self.minim_blur_val = int(self.settings['settings']['minim_blur_val'])
        self.minim_overlap_front_value = int(self.settings['settings']['minim_overlap_front_value'])
        self.minim_overlap_side_value = int(self.settings['settings']['minim_overlap_side_value'])
        self.bins = int(self.settings['settings']['bins'])
        self.mock_mode = bool(self.settings['settings']['mock_mode'])
        self.csv_path = str(self.settings['settings']['csv_path'])




    def __count_elements_less_than(self, lst, value):
        qantity = len([x for x in lst if x < value])
        return qantity / len(lst) * 100



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

        #low_side_overlap_persent = self.__count_elements_less_than(lst = self.overlay_levels_side, value = self.minim_overlap_side_value)

        #if (100 - self.health_images_edge) >= low_side_overlap_persent:
        #    logging.info(f"Low front overlap images: {round(low_side_overlap_persent,1)} %. It's ok!")
        #else:
        #    self.health = "not_ok"
        #    logging.warning(f"Attention! Low front overlap images: {round(low_side_overlap_persent,1)} %. It's too much!")


    def __core_calculation(self, fast_file_list, queue):
        blur_process = Process(target=self.blur_analysis, args=(fast_file_list, queue))
        front_overlap_process = Process(target=self.overlay_front_analysis, args=(self.file_list, fast_file_list, queue))
        #side_overlap_process = Process(target=self.overlay_side_analysis, args=(self.file_list, fast_file_list, queue))

        blur_process.start()
        front_overlap_process.start()
        #side_overlap_process.start()

        blur_process.join()
        front_overlap_process.join()
        #side_overlap_process.join()

        return queue


    def blur_analysis(self, fast_file_list, queue = None):
        if self.verbose:
            logging.info(f"Blur analysis starting ..")
        blur_levels = self.analytic.blur_analysis(fast_file_list = fast_file_list)
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


    '''def overlay_side_analysis(self, file_list, fast_file_list = None, queue = None):
        if self.verbose:
            logging.info(f"Overlap side analysis starting ..")
        overlay_levels_side = self.analytic.overlay_side_analysis(file_list = file_list, fast_file_list = fast_file_list)
        queue.put({"overlay_levels_side":overlay_levels_side})
        if self.verbose:
            logging.info(f"Overlap side analysis complete.")'''


    def main_process(self):

        
        if self.mock_mode:
            self.random_part = 1.0

        start_time = datetime.now()
        formatted_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"Starting time: {formatted_time}")
            
        queue = Queue()
        self.file_list = sorted(os.listdir(self.image_path))
        logging.info(f"File list: {self.file_list}")

        quantity = len(self.file_list)
        in_use = int(quantity * self.random_part)

        logging.info(f"Total quantity of images: {quantity}")
        logging.info(f"In use images: {in_use}")


        if quantity > self.minimal_image_quantity:
            #fast_file_list_number = int(self.random_part * quantity)
            #fast_file_list = random.choices(population=self.file_list, k=fast_file_list_number)

            #
            fast_file_list = self.file_list.copy()
            logging.info(f"File list: {fast_file_list}")


            queue = self.__core_calculation(fast_file_list = fast_file_list, queue = queue)

            analysis_data = {}
            while not queue.empty():
                analysis_data.update(queue.get())

            self.blur_levels = analysis_data['blur_levels']
            self.blur_levels = [item[0] for item in self.blur_levels]


            self.overlay_levels_front = analysis_data['overlay_levels_front']
            self.overlay_levels_front = [item[0] for item in self.overlay_levels_front]
            #self.overlay_levels_side = analysis_data['overlay_levels_side']

            
            self.blur_levels = np.array(self.blur_levels)

            if len(self.overlay_levels_front)>=self.minimal_image_quantity:
                self.overlay_levels_front = np.array(self.overlay_levels_front)
            #if len(self.overlay_levels_side)>=self.minimal_image_quantity:
            #    self.overlay_levels_side = np.array(self.overlay_levels_side) 
            
            
            
            concated_dataframe = self.analytic.files_concate(path = self.csv_path)
            #self.analytic.__draw_result(data = concated_dataframe)

            self.__resume()

        else:
            logging.warning(f"Not enought images")


        numpyBlurData = {"blur": self.blur_levels}
        encodedNumpyBlurData = dumps(numpyBlurData, cls=NumpyArrayEncoder)
 
        numpyFrontData = {"front": self.overlay_levels_front}
        encodedNumpyFrontData = dumps(numpyFrontData, cls=NumpyArrayEncoder)

        #numpySideData = {"side": self.overlay_levels_side}
        #encodedNumpySideData = dumps(numpySideData, cls=NumpyArrayEncoder)

        end_time = datetime.now()
        formatted_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"End time: {formatted_time}")

        return encodedNumpyBlurData, encodedNumpyFrontData, self.health #encodedNumpySideData,   
        #concated_dataframe = self.analytic.files_concate(path = self.csv_path)
        #self.analytic.__draw_result(data = concated_dataframe)



if __name__ == "__main__":
    ha = Hist_analysis()
    ha.main_process()
    


