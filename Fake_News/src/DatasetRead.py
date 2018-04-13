#import pandas as pd
import numpy as np
import csv
import os

class DatasetLoad():

    def __init__(self, name='train', path="fnc-1"):
        self.name = name
        self.path = path

    def set_name(self, name='train'):
        self.name = name

    def set_path(self, path='fnc-1'):
        self.path = path

    def get_stance(self):
        stances = []
        file_name = self.name+"_stances.csv"
        with open(os.path.join(self.path, file_name), 'r') as stance_file:
            read_stances = csv.reader(stance_file)
            for stance in read_stances:
                stances.append(stance)
        return stances

    def get_bodies(self):
        bodies = []
        file_name = self.name+"_bodies.csv"
        #bodies_csv = pd.read_csv(self.path+"/"+file_name)
        with open(os.path.join(self.path, file_name), 'r') as bodies_file:
            read_bodies = csv.reader(bodies_file)
            for body in read_bodies:
                bodies.append(body)
        return bodies

    def get_combined_data(self, stances, bodies, data_type="train"):
        headlines = list()
        bodies_list = list()
        stance_list = list()
        body_content = {}
        for body in bodies:
            if body[0] == 'Body ID':
                continue
            body_content[body[0]] = body[1]
        if data_type == "train":
            for data in stances:
                if data[0] == 'Headline':
                    continue
                body = body_content[data[1]]
                headlines.append(data[0])
                bodies_list.append(body)
                stance_list.append(data[2])
            return headlines, bodies_list, stance_list
        else:
            for data in stances:
                if data[0] == 'Headline':
                    continue
                body = body_content[data[1]]
                headlines.append(data[0])
                bodies_list.append(body)
            return headlines, bodies_list
