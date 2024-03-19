import cv2
import numpy as np
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