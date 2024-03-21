import cv2
import numpy as np
import os
import itertools
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
    
    def alignment(self):
        '''圖片對齊'''
        #把所有圖片依曝光時間排序
        imgtuple = list(zip(self.ltimes, self.imgs))
        self.ltimes = []
        self.imgs = []
        for t, i in sorted(imgtuple, key=lambda obj:obj[0]):
            self.ltimes.append(t)
            self.imgs.append(i)
        
        imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for img in self.imgs]
        resize_time = 5
        dx = [0] * (len(imgs_gray)-1)
        dy = [0] * (len(imgs_gray)-1)
        for t in range(resize_time, -1, -1):
            imgs_resize = [cv2.resize(img, None,
                0.5 ** t, 0.5 ** t) for img in imgs_gray]
            thresholds = [np.percentile(data, [30, 70])
                for data in imgs_resize]
            dx = [x*2 for x in dx]
            dy = [y*2 for y in dy]
            for i in range(len(dx)):
                min_cnt = None
                p1 = imgs_resize[i]
                t1 = thresholds[i]
                p2 = imgs_resize[i+1]
                t2 = thresholds[i+1]
                for dx0, dy0 in itertools.product(
                    range(dx[i]-1, dx[i]+2), range(dy[i]-1, dy[i]+2)):
                    cnt = 0
                    for x, y in itertools.product(
                        range(dx[i]+1, min(p1.shape[0], p2.shape[0]-dx[i]-1)),
                        range(dy[i]+1, min(p1.shape[1], p2.shape[1]-dy[i]-1))):
                        if ((p1[x, y] <= t1[0] and p2[x+dx0, y+dy0] >= t2[1])
                            or (p1[x, y] >= t1[1] and p2[x+dx0, y+dy0] <= t2[0])):
                            cnt += 1
                    if min_cnt is None or cnt < min_cnt:
                        min_cnt = cnt
                        min_dx = dx0
                        min_dy = dy0
                dx[i] = min_dx
                dy[i] = min_dy
        dx = [0] + itertools.accumulate(dx)
        mn = min(dx)
        dx = [n - mn for n in dx]
        dy = [0] + itertools.accumulate(dy)
        mn = min(dy)
        dy = [n - mn for n in dy]
        szx = min(p.shape[0] - d for p,d in
            zip(imgs_gray, dx))
        szy = min(p.shape[1] - d for p,d in
            zip(imgs_gray, dy))

if __name__ == "__main__":
    obj = HDR()
    obj.openImage(r"./data/PPT範例亮圖.png", 0)
    obj.openImage(r"./data/PPT範例暗圖.png", -1)
    cv2.imshow(str(obj.ltimes[0]), obj.imgs[0])
    cv2.waitKey(0)