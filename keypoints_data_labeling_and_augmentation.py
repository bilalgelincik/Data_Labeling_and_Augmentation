import cv2
import random 
import albumentations as A
import matplotlib.pyplot as plt
import os

keypoints_color = (0, 0, 255)

def visualize(image, keypoints, color=keypoints_color, diameter=15):
    image = image.copy()
    
    for x,y in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, color, -1)
        
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(image)
    plt.show()



list_point=os.listdir(r"C:\Users\bilal\Desktop\images\train")
sayac=902

for i in list_point:
    #Reading images
    read_image_path=r"C:\Users\bilal\Desktop\images\train" + "\\" + i
    img = cv2.imread(read_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Reading Labels
    text_path_i=i.split(".")[0]
    dosya=open(r"C:\Users\bilal\Desktop\data_ham_train" + "\\" + text_path_i + ".txt","r",encoding="utf-8")
        
    dizi_1=dosya.readline().rstrip().split(",")

    dizi_2=dosya.readline().rstrip().split(",")

    dizi_3=dosya.readline().rstrip().split(",")

    dizi_4=dosya.readline().rstrip().split(",")

    dizi_5=dosya.readline().rstrip().split(",")

    dizi_6=dosya.readline().rstrip().split(",")

    dizi_7=dosya.readline().rstrip().split(",")

    dizi_8=dosya.readline().rstrip().split(",")

    dizi_9=dosya.readline().rstrip().split(",")

    dizi_10=dosya.readline().rstrip().split(",")

    dizi_11=dosya.readline().rstrip().split(",")

    dizi_12=dosya.readline().rstrip().split(",")

    dizi_13=dosya.readline().rstrip().split(",")

    dizi_14=dosya.readline().rstrip().split(",")

    dizi_15=dosya.readline().rstrip().split(",")

    dizi_16=dosya.readline().rstrip().split(",")

    dizi_17=dosya.readline().rstrip().split(",")

    dizi_18=dosya.readline().rstrip().split(",")

    dizi_19=dosya.readline().rstrip().split(",")

    dosya.close()
    #Read keypoints


    keypoints = [
        (int(dizi_1[0]),int(dizi_1[1])),
        (int(dizi_2[0]),int(dizi_2[1])),
        (int(dizi_3[0]),int(dizi_3[1])),
        (int(dizi_4[0]),int(dizi_4[1])),
        (int(dizi_5[0]),int(dizi_5[1])),
        (int(dizi_6[0]),int(dizi_6[1])),
        (int(dizi_7[0]),int(dizi_7[1])),
        (int(dizi_8[0]),int(dizi_8[1])),
        (int(dizi_9[0]),int(dizi_9[1])),
        (int(dizi_10[0]),int(dizi_10[1])),
        (int(dizi_11[0]),int(dizi_11[1])),
        (int(dizi_12[0]),int(dizi_12[1])),
        (int(dizi_13[0]),int(dizi_13[1])),
        (int(dizi_14[0]),int(dizi_14[1])),
        (int(dizi_15[0]),int(dizi_15[1])),
        (int(dizi_16[0]),int(dizi_16[1])),  
        (int(dizi_17[0]),int(dizi_17[1])),
        (int(dizi_18[0]),int(dizi_18[1])),
        (int(dizi_19[0]),int(dizi_19[1]))
    ]


    #4. Augmentation Pipeline
    transform = A.Compose([A.ShiftScaleRotate(p=1)], #Random crop 1200x1400 yapıldı
                        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    #5. Pass the data to the augmentation pipeline
    transformed = transform(image=img, keypoints=keypoints)

    #6. Receive the data
    image = transformed["image"]
    keypoints = transformed["keypoints"]


    write_path=r"C:\Users\bilal\Desktop\image_rotation" + "\\" + str(sayac)+ ".bmp"
    cv2.imwrite(write_path,image)

    width,height=int(image.shape[1]),int(image.shape[0])

    empty_list_x=[]
    empty_list_y=[]
    for x,y in keypoints:
        if int(x)<width and int(y)<height and int(x)>0 and int(y)>0:
            empty_list_x.append(int(x))
            empty_list_y.append(int(y))

    min_x=min(empty_list_x)
    max_x=max(empty_list_x)
    min_y=min(empty_list_y)
    max_y=max(empty_list_y)
    
    bbox="0" + " " + str((min_x+max_x)/(2*width)) + " " + str((max_y+min_y)/(2*height)) + " " + str((max_x-min_x)/width) + " " + str((max_y-min_y)/height) + " "
    
    empty_string=""

    for x,y in keypoints:
        if int(x)>width or int(y)>height or int(x)<0 or int(y)<0:
            empty_string=empty_string + str(float(0))+ " " + str(float(0)) + " " + "0" + " "

        else:
            empty_string=empty_string + str(float(x)/width)+ " " + str(float(y)/height) + " " + "2" + " "

    dosya_2=open(r"C:\Users\bilal\Desktop\train_rotation_label" + "\\" + str(sayac) + ".txt" ,"w" , encoding="utf-8")
    dosya_2.write(bbox+empty_string)
    dosya_2.close()
    sayac+=1

