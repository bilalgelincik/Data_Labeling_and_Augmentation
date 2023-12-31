import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image,ImageDraw
import os
import cv2
import random

list_point=os.listdir(r"C:\Users\bilal\Desktop\new_model_strategy\object_detection_train\label_ham")
for i in list_point:
    dosya=open(r"C:\Users\bilal\Desktop\new_model_strategy\object_detection_train\label_ham" + "\\" + i , "r" , encoding="utf-8")
    sayac_sınıf=0
    bbox=""
    for _ in range(19):
        dizi=dosya.readline().rstrip().split(",")

        y_label,x_label=int(dizi[1]),int(dizi[0])

        random_y1=random.randint(150,300)
        random_y2=random.randint(150,300)
        random_x1=random.randint(150,300)
        random_x2=random.randint(150,300)

        min_y=y_label-random_y1
        max_y=y_label+random_y2
        min_x=x_label-random_x1
        max_x=x_label+random_x2

        bbox=bbox + str(sayac_sınıf) + " " + str((min_x+max_x)/(2*1935)) + " " + str((max_y+min_y)/(2*2400)) + " " + str((max_x-min_x)/1935) + " " + str((max_y-min_y)/2400) + "\n"
        sayac_sınıf+=1

    dosya.close()
    dosya_2=open(r"C:\Users\bilal\Desktop\new_model_strategy\object_detection_train\labels" + "\\" + i ,"w" , encoding="utf-8")
    dosya_2.write(bbox)
    dosya_2.close()