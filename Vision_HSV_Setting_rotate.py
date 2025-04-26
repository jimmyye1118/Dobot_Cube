import cv2 
import numpy as np

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def do_nothing(x):
    pass

Video_num = 1
kernel = np.ones((7, 7), np.uint8) 
capture = cv2.VideoCapture(Video_num)
img_mask = cv2.imread("mask.png")
cX = 0
cY = 0

cv2.namedWindow('Track')
cv2.createTrackbar('hmin' ,'Track', 0 , 255 , do_nothing)
cv2.createTrackbar('hmax' ,'Track', 255 , 255 , do_nothing)
cv2.createTrackbar('smin' ,'Track', 0 , 255 , do_nothing)
cv2.createTrackbar('smax' ,'Track', 255 , 255 , do_nothing)
cv2.createTrackbar('vmin' ,'Track', 0 , 255 , do_nothing)
cv2.createTrackbar('vmax' ,'Track', 255 , 255 , do_nothing)

while True:

    ret, cap = capture.read()
    if ret==False:
        break
    cap_copy = cap.copy()
    #cap_copy = rotate_image(cap_copy, 20)
    blur_image = cv2.GaussianBlur(cap_copy , (5,5), 0)

    hmin = cv2.getTrackbarPos('hmin','Track')
    hmax = cv2.getTrackbarPos('hmax','Track')
    smin = cv2.getTrackbarPos('smin','Track')
    smax = cv2.getTrackbarPos('smax','Track')
    vmin = cv2.getTrackbarPos('vmin','Track')
    vmax = cv2.getTrackbarPos('vmax','Track')
    color_lower = np.array([hmin,smin,vmin])
    color_upper = np.array([hmax,smax,vmax])
       
    hsv_image = cv2.cvtColor(blur_image,cv2.COLOR_BGR2HSV)     
    mask_image = cv2.inRange(hsv_image, color_lower, color_upper)    
    
    #mask = cv.erode(mask,kernel,iterations=1)
    #mask = cv.dilate(mask,kernel,iterations=1)    
    """
    contours, hierarchy = cv2.findContours(mask_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) > 0):
        c = max(contours, key = cv2.contourArea)
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        print("cX = " + str(cX))
        print("cY = " + str(cY))
        cv2.circle(mask_image,(cX,cY),10,(1,227,254),-1)
    """
    cv2.imshow("camera_input" ,cap_copy)
    cv2.imshow("camera_mask",mask_image)

    keypress = cv2.waitKey(1)
    if keypress == ord('q'):     
        break

cv2.destroyAllWindows()

