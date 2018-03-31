import pandas as pd
import numpy as np

class DatasetLoad():

    def __init__(self, name='train', path="fnc-1"):
        self.name = name
        self.path = path

    def set_name(self, name='train'):
        self.name = name

    def set_path(self, path='fnc-1'):
        self.path = path

    def get_stance(self):
        file_name = self.name+"_stances.csv"
        stance_csv = pd.read_csv(self.path+"/"+file_name)
        return stance_csv

    def get_bodies(self):
        file_name = self.name+"_bodies.csv"
        bodies_csv = pd.read_csv(self.path+"/"+file_name)
        return bodies_csv

    def get_combined_data(self, stances, bodies, data_type="train"):
        headlines = list()
        bodies_list = list()
        stance_list = list()
        body_content = {}
        for body, id2 in zip(bodies['articleBody'], bodies['Body ID']):
            body_content[id2] = body
        if data_type == "train":
            for headline, body_id, stance in zip(stances['Headline'], stances['Body ID'], stances['Stance']):
                body = body_content[body_id]
                headlines.append(headline)
                bodies_list.append(body)
                stance_list.append(stance)
            return headlines, bodies_list, stance_list
        else:
            for headline, body_id in zip(stances['Headline'], stances['Body ID']):
                body = body_content[body_id]
                headlines.append(headline)
                bodies_list.append(body)
            return headlines, bodies_list
