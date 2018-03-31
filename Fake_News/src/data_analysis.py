import matplotlib.pyplot as plt
import numpy as np

class DataAnalysis:

    def plotHistogram(self, data):
        n, bins, patches = plt.hist(data, 100)
        plt.show()

    def histogramOfStrLength(self, str_list):
        str_length = []
        for string in str_list:
            str_length.append(len(string))
        print(max(set(str_length), key=str_length.count))
        self.plotHistogram(str_length)