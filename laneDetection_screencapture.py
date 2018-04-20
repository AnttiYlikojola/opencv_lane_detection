import cv2
import numpy as np
import math
import win32gui
import win32ui
import win32con
import win32api
import matplotlib.pyplot as plt
from calibration import calibrate, undistort


mtx, dist = calibrate()

# demovideo
# cap = cv2.VideoCapture('solidWhiteRight.mp4')
cap = cv2.VideoCapture('project_video.mp4')

# https://www.programcreek.com/python/example/14102/win32gui.GetDesktopWindow
# mahdollisia videoita tai pelejeä varten ruudunkaappaus -- ei käytössä
def grab_screen(region=(0, 100, 920 , 560)):

    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB), cv2.COLOR_BGR2RGB)

# rajataan kuvaa
def region_of_interest(canny_image, original_image):

    if 'width' not in locals() or 'height' not in locals():
        width = canny_image.shape[1]
        height = canny_image.shape[0]
        vertices = [
            (width * 0.10, height), 
            (width / 2.2, height / 1.8), 
            (width - (width / 2.2) , height / 1.8), 
            (width * 0.90, height)
            ]
        # vertices1 = [(0.0, height),(width* 0.20, height), (0.0, 0.0),(width* 0.20, 0.0)]
        # vertices2 = [(width, height),(width* 0.80, height), (width, 0.0),(width* 0.80, 0.0)]

    mask = np.zeros_like(canny_image)    


    masked_image = cv2.bitwise_and(canny_image, mask)
    mask_for_regular = np.zeros_like(original_image)
    img = cv2.fillConvexPoly(mask_for_regular, np.array([vertices], np.int32), (255,170,255))
    regular_masked_image = cv2.addWeighted(original_image, 1, img, 0.1, 0.0)

    


    return masked_image, regular_masked_image

def process_img(original_image):

    undistorted_image = undistort(original_image, mtx, dist)

    gray_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    warped_image = change_perspective(canny_image)
    cropped_image, regular_masked_image = region_of_interest(warped_image, original_image)

    # palautetaan myös kaistaviivat

    return cropped_image, regular_masked_image

def change_perspective(original_image):
    image_size = (original_image.shape[1], original_image.shape[0])

    bot_width = .76
    mid_width = .08
    height_pct = .62
    bottom_trim = .935
    offset = image_size[0]*.25
    src = np.float32([[original_image.shape[1]*(.5 - mid_width/2), original_image.shape[0]*height_pct], [original_image.shape[1]*(.5 + mid_width/2), original_image.shape[0]*height_pct],\
    [original_image.shape[1]*(.5 + bot_width/2), original_image.shape[0]*bottom_trim], [original_image.shape[1]*(.5 - bot_width/2), original_image.shape[0]*bottom_trim]])
    dst = np.float32([[offset, 0], [image_size[0] - offset, 0], [image_size[0] - offset, image_size[1]], [offset, image_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)

    #transform the image to birds eye view given the transform matrix
    warped = cv2.warpPerspective(original_image, M, (image_size[0], image_size[1]))
    return warped

def lanes():
    return 0


while (cap.isOpened()): #True:#
    ret, frame = cap.read()

    # original_image = grab_screen()
    cropped_image, original_image_masked = process_img(frame)
    warped_image = change_perspective(cropped_image)

    # histogram = np.sum(warped_image[600:,:], axis=0)
    # plt.plot(histogram)
    # plt.title("histogram")
    # plt.show()
    # hsv = cv2.cvtColor(warped_image,cv2.COLOR_BGR2HSV)
    # hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    # plt.imshow(hist,interpolation = 'nearest')
    # plt.show()

    


    cv2.imshow("cropped image", cropped_image)
    # cv2.imshow("warped image", warped_image)       
    cv2.imshow("original image masked", original_image_masked)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
