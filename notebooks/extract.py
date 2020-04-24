import cv2
import os

class Observation:
    def __init__(self, line):
        lst = line.split(',')
        lst = [int(x) for x in lst]
        self.id = lst[0]
        self.frame = lst[1]
        self.UpperPointShort = [lst[2], lst[3]]
        self.UpperPointCorner = [lst[4], lst[5]]
        self.UpperPointLong = [lst[6], lst[7]]
        self.CrossCorner = [lst[8], lst[9]]
        self.ShortSide = [lst[10], lst[11]]
        self.Corner = [lst[12], lst[13]]
        self.LongSide = [lst[14], lst[15]]
        self.LowerCrossCorner = [lst[16], lst[17]]

    def topLeftCorner(self):
        return min([self.Corner[0], self.CrossCorner[0], self.LongSide[0], self.LowerCrossCorner[0],
                    self.ShortSide[0], self.UpperPointCorner[0], self.UpperPointShort[0], self.UpperPointLong[0]]),\
               min([self.Corner[1], self.CrossCorner[1], self.LongSide[1], self.LowerCrossCorner[1],
                    self.ShortSide[1], self.UpperPointCorner[1], self.UpperPointShort[1], self.UpperPointLong[1]])

    def lowerRightCorner(self):
        return max([self.Corner[0], self.CrossCorner[0], self.LongSide[0], self.LowerCrossCorner[0],
                     self.ShortSide[0], self.UpperPointCorner[0], self.UpperPointShort[0], self.UpperPointLong[0]]),\
               max([self.Corner[1], self.CrossCorner[1], self.LongSide[1], self.LowerCrossCorner[1],
                     self.ShortSide[1], self.UpperPointCorner[1], self.UpperPointShort[1], self.UpperPointLong[1]])

    def selectionSize(self):
        tlx, tly = self.topLeftCorner()
        lrx, lry = self.lowerRightCorner()
        return lrx-tlx, lry-tly

    def getSquaredCorners(self):
        w, h = self.selectionSize()
        tlx, tly = self.topLeftCorner()
        lrx, lry = self.lowerRightCorner()
        if w > h:
            half1 = int((w-h) / 2)
            half2 = w-h-half1
            tly -= half1
            lry += half2
        else:
            half1 = int((h - w) / 2)
            half2 = h - w - half1
            tlx -= half1
            lrx += half2
        if lrx-tlx != lry-tly:
            print('hupsz')
        return (tlx, tly), (lrx, lry)


# TODO !!!
# 1A:1, 1B:3, 2A:1, 2B:3, 3A:1, 3B:3
#kamera = '1A'
#frame_disp = 1
just_show = False
# !!!

for (kamera, frame_disp) in [('1A', 1), ('1B', 3), ('2A', 1), ('2B', 3), ('3A', 1), ('3B', 3), ('4A', 1), ('4B', 3), ('5A', 1), ('5B', 3)]:

    if not os.path.exists('frames/{}'.format(kamera)):
        os.makedirs('frames/{}'.format(kamera))

    vidcap = cv2.VideoCapture('e:\\{}.mov'.format(kamera))
    success, image = vidcap.read()

    lst = [line.rstrip('\n') for line in open('e:\\{}_annotations.txt'.format(kamera))]
    lst = lst[1:]   #remove first line
    lst = [Observation(line) for line in lst]
    lst.sort(key=lambda x: x.frame)


    frame_id = frame_disp
    while len(lst) > 0:
        o = lst[0]
        lst = lst[1:]
        while frame_id < o.frame:
            success, image = vidcap.read()
            frame_id += 1
            #if frame_id % 100 == 0:
            #    print('current frame: ', frame_id)
        if frame_id == o.frame:
            start, end = o.getSquaredCorners()
            crop = image[start[1]:end[1], start[0]:end[0]]
            if crop.shape[0] != crop.shape[1]:
                diff = abs(crop.shape[0] - crop.shape[1])
                half1 = int(diff / 2)
                half2 = diff - half1
                if crop.shape[0] > crop.shape[1]:
                    crop = crop[half1:crop.shape[0]-half2, :]
                else:
                    crop = crop[:, half1:crop.shape[1]-half2]
            if just_show:
                cv2.rectangle(image, start, end, (255, 255, 255))
                cv2.imshow("image", image)
                cv2.waitKey(0)
            else:
                cv2.imwrite("frames/{}/{}_{}_{}.jpg".format(kamera, o.id, o.frame, crop.shape[0]), crop)
            #print(crop.shape, ' saved as ', "frames/{}_{}_{}.jpg".format(o.id, o.frame, crop.shape[0]))
        #if len(lst) % 100 == 0:
        #    print(len(lst), ' observations remaining')
    print(kamera, 'done!')
print('done!')

'''
count = 0
while count < 1797:
    success, f = vidcap.read()
    count += 1
    if count % 100 == 0:
        print('current: ', count)
#f = cv2.imread('frames/frame1800.jpg')
#cv2.rectangle(f, (648, 240), (521, 243), (0,0,255))
#cv2.rectangle(f, (405, 183), (500, 181), (0,255,0))
#cv2.rectangle(f, (652, 415), (528, 423), (255,0,0))
#cv2.rectangle(f, (412, 316), (506, 312), (255,0,255))
#UpperPointLong[x y] -> ShortSide[x y]
#cv2.rectangle(f, (405, 183), (652, 415), (255,255,255))
o = Observation('1,1800,999,266,937,271,862,252,922,247,998,339,936,346,862,322,921,316')
tl, lr = o.getSquaredCorners()
cv2.rectangle(f, tl, lr, (255,0,0))
cv2.imshow("f", f)
cv2.waitKey(0)
'''