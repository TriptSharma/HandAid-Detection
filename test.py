import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np 
from sklearn.cluster import DBSCAN

def smoothEdge(img, toBlur=True, canny=False):
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	kernel1 = np.ones((5,5),np.float32)/25
	kernel2 = np.ones((3,3),np.float32)/9
	img = cv2.erode(img, kernel2)
	img = cv2.dilate(img, kernel2)
	# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
	if toBlur:
		img = cv2.medianBlur(img,5)
	
	if canny:
		img = cv2.Canny(img, 50, 30)
		img = cv2.dilate(img, kernel1)
		img = cv2.erode(img, kernel2)
	# cv2.imshow('sdgds', img)
	# cv2.waitKey(0)
	return img

# img1 = cv2.imread('with3.png')
# img2 = cv2.imread("without3.png")

img1 = cv2.imread('with2.jpeg')
img2 = cv2.imread("without2.jpeg")

img1 = cv2.resize(img1, (480,320))
img2 = cv2.resize(img2, (480,320))


s1 = smoothEdge(img1.copy(), toBlur=False)
s2 = smoothEdge(img2.copy())

def kmeans(img, K=6):		
	Z = img.reshape((-1,3))

	# convert to np.float32
	Z = np.float32(Z)
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	return cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

ret,label,center = kmeans(s2)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((s2.shape))

cv2.imshow('res2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(center)

# hsvColor = []
# for i in center:
# 	i = np.reshape(i, (1,1,3))
# 	hsvc = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
# 	hsvc = np.reshape(hsvc, (3))
# 	# print(hsvc)
# 	upper =  np.array([min(hsvc[0]+10,179), min(hsvc[1]+10,255), min(hsvc[2]+40,255)])
# 	lower =  np.array([max(hsvc[0]-10, 0), max(hsvc[1]-10,0), max(hsvc[2]-40,0)])
	
# 	hsvColor.append([upper, lower])

# # print(hsvColor)
# hsv_img = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
# # cv2.imshow('thresh', hsv_img)
# # cv2.waitKey(0)

contours=[]
for color in center:
# for color in hsvColor:
	# print(color)
	# frame_threshed = cv2.inRange(hsv_img, color[1], color[0])
	frame_threshed = cv2.inRange(res2, color, color)
	
	cnts, pyr = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	#found re	ct contours store 'em
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.015 * peri, True)
		
		# print(approx)
		if cv2.contourArea(approx)>20 and cv2.contourArea(approx)<7500 and cv2.isContourConvex(approx):
			contours.append(approx)

print(len(contours))
cv2.drawContours(s2, contours, -1, (0, 255, 0), 3)

methods = ['cv2.TM_CCOEFF_NORMED']#, 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
	img = s1.copy()
	method = eval(meth)
		
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		template = s2[y:y+h, x:x+w]
		
		# Apply template Matching
		# cv2.imshow('fdgd',template)
		# cv2.waitKey(0)
		res = cv2.matchTemplate(img, template, method)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

		# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
		if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
			top_left = min_loc
		else:
			top_left = max_loc
		bottom_right = (top_left[0] + w, top_left[1] + h)

		# diff = cv2.absdiff(cv2.Canny(template, 50, 30),
		# 	cv2.Canny(s1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], 50, 30))
		diff = cv2.absdiff(cv2.cvtColor(img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], cv2.COLOR_BGR2GRAY), 
			cv2.cvtColor(template, cv2.COLOR_BGR2GRAY))
		
		# ret, diff = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
		# kernel = np.ones((3,3),np.float32)/25
		# diff = cv2.dilate(diff, kernel)
		
		circles = cv2.HoughCircles(diff, cv2.HOUGH_GRADIENT, 1.2, 50, param1=20, param2=15, minRadius=2, maxRadius=10)
		print(circles)
		if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers 2, 5, param1=15, param2=12
			circles = np.round(circles[0, :]).astype("int")
		
			# loop over the (x, y) coordinates and radius of the circles
			for (x, y, r) in circles:
				# draw the circle in the output image, then draw a rectangle
				# corresponding to the center of the circle
				cv2.circle(diff, (x, y), r, 255, 4)
				# cv2.rectangle(diff, (x - 5, y - 5), (x + 5, y + 5), 255, -1)

		cv2.imshow('diff', diff)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
		cv2.rectangle(img,top_left, bottom_right, 255, 2)

	cv2.imshow('thresh', img)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	