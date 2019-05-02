import cv2
import numpy as np
import time
import sys
from imutils import face_utils
from face_utilities import Face_utilities
from signal_processing import Signal_processing
from collections import OrderedDict


def draw_poly(bg, shape, pts, color):
    poly = np.array([[shape[pts[0]][0],shape[pts[1]][1]],[shape[pts[2]][0],shape[pts[3]][1]],
                    [shape[pts[4]][0],shape[pts[5]][1]],[shape[pts[6]][0],shape[pts[7]][1]]], dtype=np.int32)
    #bg = np.zeros(SIZE,dtype=np.uint8)
    cv2.fillPoly(bg, [poly], color)
    
    return bg
    
    # cv2.imshow("add",img1)
    # cv2.imshow("bg",bg)
    # cv2.waitKey(0)
    


FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("between_eyebrown", ((21,19,22,24,22,27,21,27),(216, 206, 17))),
    ("chin", ((7,4,9,12,9,10,7,6),(104, 153, 74))),
    ("right_cheek", ((4,2,41,29,39,30,5,48),(127, 80, 65))),
    ("left_cheek", ((12,14,46,29,42,30,12,54),(127, 80, 65))),
    ("nose", ((32,27,34,27,34,30,32,30),(0, 255, 255))),
    ("around_lip_area", ((48,33,54,33,54,52,48,50),(230, 68, 255))),
    ("under_left_eye", ((42,28,45,28,45,29,42,29),(35, 68, 132))),
    ("under_right_eye", ((36,28,39,28,39,29,36,29),(35, 68, 132))),
    ("forehead", ((68,68,69,69,69,24,68,19),(0, 0, 255)))
    
])


img = cv2.imread("1.jpg")

fu = Face_utilities()
sp = Signal_processing()

ret_process = fu.no_age_gender_face_process(img, "68")

rects, face, shape, aligned_face, aligned_shape = ret_process

(h,w,c) = aligned_face.shape
SIZE = [h,w,c]

(x, y, w, h) = face_utils.rect_to_bb(rects[0])
cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# add forehead pts
y1 = 2*aligned_shape[19][1] - aligned_shape[36][1]
y2 = 2*aligned_shape[24][1] - aligned_shape[45][1]

if y1 < 0:
    y1 = 0
if y2 < 0:
    y2 = 0

print("{},{},{},{}".format(aligned_shape[36][1],aligned_shape[19],y1,y2))

pt1 = np.array([aligned_shape[17][0],y1])
pt2 = np.array([aligned_shape[26][0],y2])

aligned_shape = np.concatenate((aligned_shape, [pt1]))
aligned_shape = np.concatenate((aligned_shape, [pt2]))


for (x, y) in aligned_shape: 
    cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)            

bg = np.zeros(SIZE,dtype=np.uint8)

for k, v in FACIAL_LANDMARKS_68_IDXS.items():
    pts = v[0]
    print(pts)
    color = v[1]
    print(color)
    bg = draw_poly(bg, aligned_shape, pts, color)
    
aligned_face = cv2.addWeighted(aligned_face,1,bg,0.3,0)
    
cv2.imshow("img",img)
cv2.imshow("aligned_face",aligned_face)
cv2.waitKey(0)
