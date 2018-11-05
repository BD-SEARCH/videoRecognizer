import cv2 as cv
import numpy as np
import csv

class Video():
    def __init__(self):
        print('init')

    def call_video(self, names):
        data_set = []

        for name in names:
            path = "./dataset/~"
            print(path)

            data_set_ = []
            f = open(path, "r")
            reader = csv.reader(f)
            for line in reader:
                if int(line[0])<int(line[1]) and int(line[1])-int(line[0])<200:
                    sett = {"start":line[0], "end":line[1], "label":line[2]}
                    data_set_.append(sett)
            f.close()

            interval = 4
            for i in data_set_:
                for j in range(int(i["start"]), int(i["end"])+1, interval):
                    img = cv.resize(cv.imread(path+str[j]+".jpg"), (self.width, self.height))
                    data_set.append({"image":image, "label":int(i["label"])})



init = Video(./)

llist =
init.call_video(llist)
