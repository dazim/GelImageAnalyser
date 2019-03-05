import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils
from scipy import signal

def exploPlot(img):


    img_x = len(img[0]) #653
    img_y = len(img)    #449

    # deconstruct image

    pos = []
    int = []

    for i in range(img_x):

        for j in range(img_y):

            pos.append(np.log2(abs(i-(img_x/2)) * abs(j-(img_y/2))))
            int.append(img[j][i])

            if img[j][i] > 100:

                img[j][i] = 255

    plt.plot(pos, int, ".")
    #plt.imshow(img)
    cv2.imwrite("filtered.bmp", img)
    plt.show()
 #   df = pd.DataFrame()
  #  df.columns = ["X", "Y", "Dist", "Int"]
    print(img[0][0])

#    np.append(px, [1])

    #plt.imshow(img)
    #plt.show()

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
    #cv2.rectangle(image, (x_fin, y_fin), (x_fin+w_fin, y_fin+h_fin), (0, 255, 0), 2)
    image = crop_minAreaRect(image, rect)

    return image

def detectLanes(image):

    rows = scan_rows(image)

    # Find first and last heavy slope (ladder left and right)

    ladder_thresh = 1.1 * np.min(rows)

    ladder_pos = []

    i = 0

    for row in rows:

        if row < ladder_thresh:

            ladder_pos.append(i)

        i += 1

    # Binarize peaks

    gradient_std = np.std((np.gradient(rows[ladder_pos[0]:ladder_pos[-1]])))

    peaks_up = np.gradient(rows[ladder_pos[0]:ladder_pos[-1]]) > gradient_std

    peaks_down = -1*(np.gradient(rows[ladder_pos[0]:ladder_pos[-1]]) < -gradient_std )

    # Sharpen peaks

    i = 0
    peak_start = 0
    peak_end = 0

    while i < (len(peaks_up)-1):

        if peaks_up[i] == 0 and peaks_up[i+1] == 1:

            peak_start = i

        elif peaks_up[i] == 1 and peaks_up[i+1] == 0:

            peak_end = i

        i += 1

        if i == len(peaks_up)-1:

            peak_end = i

        if peak_start and peak_end:

            peak_center = int(peak_start+(peak_end-peak_start)/2)
            peaks_up[(peak_start-1):(peak_end+1)] = 0
            peaks_up[peak_center] = 1
            peak_start = 0
            peak_end = 0

    i = 0
    peak_start = 0
    peak_end = 0

    if peaks_down[0] == -1:

        peak_start = True

    while i < (len(peaks_down)-1):

        if peaks_down[i] == 0 and peaks_down[i+1] == -1:

            peak_start = i

        elif peaks_down[i] == -1 and peaks_down[i+1] == 0:

            peak_end = i

        i += 1

        if i == len(peaks_down)-1:

            peak_end = i

        if peak_start and peak_end:

            peak_center = int(peak_start+(peak_end-peak_start)/2)
            peaks_down[(peak_start-1):(peak_end+1)] = 0
            peaks_down[peak_center] = -1
            peak_start = 0
            peak_end = 0

    # Check peak distances and fill missing

    i = 0
    peak_pos = []

    while i < (len(peaks_up)-1):

        if peaks_up[i] == 1:

            peak_pos.append(i)

        i += 1

    i = 0
    peak_pos_dists = []
    peak_pos_tmp = []

    while i < (len(peak_pos)-1):

        peak_pos_dists.append([peak_pos[i], peak_pos[i+1],
            peak_pos[i+1]-peak_pos[i]])
        peak_pos_tmp.append(peak_pos[i+1]-peak_pos[i])

        i += 1

    peak_pos_filled = []
    dist_med = np.median(peak_pos_tmp)
    i = 1

    for pos in peak_pos_dists:

        if not 0.9 * dist_med < pos[2] < 1.1 * dist_med:

            missing_lanes = int(round(pos[2]/dist_med))

            for j in range(1, missing_lanes):

                expected_pos = pos[0]+(int(pos[2]/int(round(pos[2]/dist_med))))
                peaks_up[expected_pos] = 1

        i += 1

    i = 0
    peak_pos = []

    while i < (len(peaks_down)-1):

        if peaks_down[i] == -1:

            peak_pos.append(i)

        i += 1

    i = 0
    peak_pos_dists = []
    peak_pos_tmp = []

    while i < (len(peak_pos)-1):

        peak_pos_dists.append([peak_pos[i], peak_pos[i+1],
            peak_pos[i+1]-peak_pos[i]])
        peak_pos_tmp.append(peak_pos[i+1]-peak_pos[i])

        i += 1

    dist_med = np.median(peak_pos_tmp)
    i = 1

    for pos in peak_pos_dists:

        if not 0.9 * dist_med < pos[2] < 1.1 * dist_med:

            missing_lanes = int(round(pos[2]/dist_med))

            for j in range(1, missing_lanes):

                expected_pos = pos[0]+(int(pos[2]/int(round(pos[2]/dist_med))))
                peaks_down[expected_pos] = -1
                print(pos[0]+(int(pos[2]/int(round(pos[2]/dist_med)))))

        i += 1

    i = 0
    all_peaks = abs(peaks_up+peaks_down)
    peak_pos_filled = []

    while i < (len(all_peaks)-1):

        if all_peaks[i]:

            peak_pos_filled.append(i)

        i += 1

    i = 0
    lane_pos = peaks_up * 0
    lane_pos_in_gel = []

    while i < (len(peak_pos_filled)-1):

        lane_center = int(peak_pos_filled[i]+(peak_pos_filled[i+1]-peak_pos_filled[i])/2)
        lane_pos[lane_center] = 1
        lane_pos_in_gel.append(ladder_pos[0]+lane_center)
        i += 2


    print(lane_pos_in_gel)

    #plt.plot(np.gradient(rows[ladder_pos[0]:ladder_pos[-1]]))

    #plt.plot(peaks_up)
    #plt.plot(peaks_down)
    #plt.plot(all_peaks)
    plt.imshow(gel)
    plt.vlines(lane_pos_in_gel, 0, len(gel))

    #plt.imshow(image[ladder_pos[0]:ladder_pos[-1]])
    #plt.plot(rows[ladder_pos[0]:ladder_pos[-1]])
    plt.show()

    return lane_pos_in_gel

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

    #plt.plot(np.gradient(rows)**2)
    #plt.plot(rows)
    #plt.show()

    return rows

def cropToROI(img):

    row = []

    for i in range(0, gel_y - 2):

        row.append(np.std(img[i]))

    print(np.mean(img[55]))
    plt.plot(row)
    plt.show()

def detectPeaks(data):

    peaks = signal.find_peaks(data, prominence=1, width=2)

    for peak in peaks[0]:

       print(peak)


def detectLadders(image):

    gel_x = len(image[0]) #653
    gel_y = len(image)    #449

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


    plt.plot(tmp)

    #plt.plot(band)

    plt.show()

def plotCol(image, col):

    tmp = []


    for row in image:

        tmp.append(row[col])

    plt.plot(tmp)
    plt.show()



img = cv2.imread('gelbilder/20171120.JPG', cv2.IMREAD_GRAYSCALE)
#exploPlot(img)
image = img
gel = cropToGel(image)
#image = cv2.imread('gelbilder/20180220.JPG', cv2.IMREAD_GRAYSCALE)
#image = cv2.imread('gelbilder/20180125_8.JPG', cv2.IMREAD_GRAYSCALE)
#gray = cv2.medianBlur(img,5)


#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#flag, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

#cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#	cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)

# Find contours
#img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#contours = sorted(contours, key=cv2.contourArea, reverse=True)



# Select long perimeters only
#perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
#listindex=[i for i in range(15) if perimeters[i]>perimeters[0]/2]
#numcards=len(listindex)

#card_number = -1 #just so happened that this is the worst case
#stencil = np.zeros(img.shape).astype(img.dtype)
#cv2.drawContours(stencil, [contours[listindex[card_number]]], 0, (255, 255, 255), cv2.FILLED)
#res = cv2.bitwise_and(img, stencil)


#cv2.imwrite("out.bmp", res)

#kernel = np.ones((5,5), np.uint8)

#img_dilation = cv2.dilate(res, kernel, iterations=1)
#img_erosion = cv2.erode(img_dilation, kernel, iterations=1)

#cv2.imwrite('Erosion.bmp', img_dilation)


#canny = cv2.Canny(img_dilation, 100, 200)

#cv2.imwrite("contours.bmp", canny)

#for c in contours:
#    # compute the center of the contour
#    M = cv2.moments(c)
#    cX = int(M["m10"] / M["m00"])
#    cY = int(M["m01"] / M["m00"])

    # draw the contour and center of the shape on the image
#    cv2.drawContours(canny, [c], -1, (0, 255, 0), 2)
#    cv2.circle(canny, (cX, cY), 7, (255, 255, 255), -1)
#    cv2.putText(canny, "center", (cX - 20, cY - 20),
#		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # show the image

#cv2.imwrite("canny.bmp", canny)

#img = cv2.imread('gelbilder/20171120.JPG', cv2.IMREAD_GRAYSCALE)
#exploPlot(img)

#gel_x = len(gel[0]) #653
#gel_y = len(gel)    #449

#sobel_gel = xy_sobel(gel)


lanes = detectLanes(np.sqrt(gel))

plotCol(np.sqrt(gel), 61)

#scan_rows(np.sqrt(gel))
#cv2.imwrite("gel.bmp", gel)#
#scan_rows(np.gradient(gel))
#print(np.median(gel))
#detectLadders(gel)
#plt.imshow(gel)
#plt.show()

