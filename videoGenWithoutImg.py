import glob

import cv2

img_array = []
filenames = glob.glob('img/*.png')

for x in range(500):
    img = cv2.imread('img/img' + str(x) + '.png')
    img_array.append(img)

out = cv2.VideoWriter('vid/vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60,
                      (1000, 1000))

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
