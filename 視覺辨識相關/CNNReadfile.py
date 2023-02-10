import numpy as np
import os
import cv2
import numpy as np


#img_array1=cv2.imread(r"test\a\8225343_254.jpg", cv2.IMREAD_GRAYSCALE)
#new_img_array=cv2.resize(img_array1, dsize=(300, 300))
#cv2.imshow("img",new_img_array)
#cv2.waitKey(0)

x=[]
y=[]
def create_test_data(path):
    for p in os.listdir(path):
        for file in os.listdir(os.path.join(path,p)):
            img_array = cv2.imread(os.path.join(os.path.join(path,p),file),cv2.IMREAD_GRAYSCALE)
            new_img_array = cv2.resize(img_array, dsize=(200, 200))
            x.append(new_img_array)
            y.append(p)

a=r"test"

create_test_data(a)
x=np.array(x).reshape(-1, 200, 200, 1)
print(x.shape)
nx=x/255
print(nx[0])

