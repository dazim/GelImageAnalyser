import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from scipy import signal

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop

def irf(expression):

    return int(round(float(expression)))

class ShapeDetector:

	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"

		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"

		# return the name of the shape
		return shape

def cropToGel(image):

    borderpixels = []
    borderpixels.append(image[0])
    borderpixels.append(image[-1])

    res_x = len(image[0]) #768
    res_y = len(image)    #576

    bg_thresh = irf(np.amin(borderpixels) * 0.95)

    ret,threshed_img = cv2.threshold(image,bg_thresh,255,cv2.THRESH_BINARY)

    threshed_image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE)

    areas = [0]

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        if (w * h) < (0.99 * (res_x * res_y)):
            if (w * h) > max(areas):
                x_fin, y_fin, w_fin, h_fin = x, y, w, h
                rect = cv2.minAreaRect(c)
        # draw a green rectangle to visualize the bounding rect
            areas.append(w*h)
    cv2.rectangle(image, (x_fin, y_fin), (x_fin+w_fin, y_fin+h_fin), (0, 255, 0), 2)
    image = crop_minAreaRect(image, rect)

    return image

def detectPockets(img):

    sobel = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    rows = []

    for i in range(0, len(img -2)):

        rows.append(np.mean(sobel[i]))

    rows_median = np.mean(rows)

    for i in range(0, len(img -2)):

       if not (0.5 * rows_median) < rows[i] < (1.5 * rows_median):

            rows[i] = rows_median

    plt.plot(rows)
    plt.show()

def xy_sobel(img):

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

    return abs(sobelx + sobely)

def scan_cols(image):

    ylen = len(image)

    cols = []

    for i in range(0, ylen - 2):

        cols.append(np.mean(image[i]))

    plt.plot(cols)
    plt.show()


def scan_rows(image):

    xlen = len(image[0])
    ylen = len(image)

    rows = []

    for i in range(0, xlen - 2):

        tmp = []

        for j in range(0, ylen - 2):

            tmp.append(image[j][i])

        rows.append(np.mean(tmp))

    plt.plot(rows)
    plt.show()

def cropToROI(img):

    row = []

    for i in range(0, gel_y - 2):

        row.append(np.std(img[i]))

    print(np.mean(img[55]))
    plt.plot(row)
    plt.show()

def detectLadders(image):

    lanes = []

    for i in range(1, gel_x - 2):

        column = []

        for j in range(1, gel_y - 2):

            column.append(gel[j][i])

        lanes.append(np.mean(column))
    lanes_median = np.median(lanes)
    for i in range(len(lanes)):

        if not (0.5 * lanes_median) < lanes[i] < (1.5 * lanes_median):

            lanes[i] = lanes_median

    for i in range(len(lanes)):

        lanes[i] = max(lanes) - lanes[i]


    moving_window = irf(0.025* gel_x)
    lanes_moving = np.convolve(lanes, np.ones((moving_window,))/moving_window, mode='valid')



    peaks = signal.find_peaks(lanes_moving, prominence=1, width=(0.01 * gel_x))


    lanes = []

    for peak in peaks[0]:

        lane = []

        for i in range(0, len(image) - 2):

            lane.append(image[i][peak])

        lanes.append((peak, (np.mean(lane))))
        plt.axvline(x=peak)

    lanes =  sorted(lanes, key=lambda x: int(x[1]))[:2]

    tmp = []

    for i in range(0, len(image) - 2):

        tmp.append(image[i][83])

    tmp_max = max(tmp)
    tmp_min = min(tmp)

    for i in range(0, len(tmp)):

        if not (tmp[i] < irf((tmp_max - tmp_min)*0.02)):

            tmp[i] = tmp_max

        else:

            tmp[i] = tmp_min

    groups = []

    group_start = 0
    group_end = 0

    for i in range(0, len(tmp)-1):

        if tmp[i] == tmp[i+1]:

            group_end += 1

        else:

            groups.append((group_start, group_end))
            group_start = group_end + 1
            #group_start = group_end = 0

    new_list = []

    for group in groups:

        if tmp[group[0]] == tmp_min:

            new_list.append(group)


    print(new_list)


    plt.plot(tmp)

    #plt.plot(band)

    plt.show()



image = cv2.imread('gelbilder/20171120.JPG', cv2.IMREAD_GRAYSCALE)
#image = cv2.imread('gelbilder/20180220.JPG', cv2.IMREAD_GRAYSCALE)
#image = cv2.imread('gelbilder/20180125_8.JPG', cv2.IMREAD_GRAYSCALE)
image = cv2.medianBlur(image,5)

gel = cropToGel(image)
gel_x = len(gel[0]) #653
gel_y = len(gel)    #449

sobel_gel = xy_sobel(gel)

#scan_rows(sobel_gel)
print(np.median(gel))
detectLadders(gel)
plt.imshow(gel)
plt.show()

