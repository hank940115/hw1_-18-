import cv2
import numpy as np
import os
os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
class HDR:
    '''主要操作的物件'''
    #self.imgs : 圖片物件的陣列
    #self.ltimes : 每個圖片的曝光時間對2的對數

    def __init__(self):
        self.imgs = []
        self.ltimes = []
    
    def openImage(self, filename: str, ltime: int):
        '''加入一個圖檔和該圖的曝光時間對2的對數'''
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        self.imgs.append(img)
        self.ltimes.append(ltime)

if __name__ == "__main__":
    obj = HDR()
    obj.openImage(r"./data/PPT範例亮圖.png", 0)
    obj.openImage(r"./data/PPT範例暗圖.png", -1)
    cv2.imshow(str(obj.ltimes[0]), obj.imgs[0])
    cv2.waitKey(0)