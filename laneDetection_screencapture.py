import math
import time

import matplotlib.pyplot as plt
import numpy as np

import cv2
from calibration import calibrate, undistort

mtx, dist = calibrate()

# video stream from pi
imageUrl = 'http://192.168.0.12/html/cam_pic_new.php'
cap = cv2.VideoCapture(imageUrl)

def region_of_interest(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    width = img.shape[1]
    height = img.shape[0]
    vertices = [(0, height), (width / 2, height / 2), (width, height)]
    # maaritellaan tyhja matriisi joka tasmaa kuvan korkeuteen/leveyteen
    mask = np.zeros_like(img)

    # haetaan kuvan varikanavat
    channel_count = 3

    # luodaan tasmaava vari samoilla varikanavilla
    match_mask_color = (255,) * channel_count

    # taytetaan polygoni
    cv2.fillPoly(mask, np.int32([vertices]), match_mask_color)

    # palautetaan kuva jossa vain maskatut pikselit tasmaavat
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def process_img(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    undistorted_image = undistort(gray_image, mtx, dist)
    inputImage = cv2.blur(gray_image, (2,2))


    canny_image = cv2.Canny(inputImage, 50, 150)
    rio = region_of_interest(canny_image)
    cv2.imshow("canny", gray_image)

    #thresholded_image = threshold(original_image)
    warped_image = to_birdseye(canny_image)

    # palautetaan myös kaistaviivat

    return warped_image, rio


def lr_curvature(warped_image):

    histogram = np.sum(warped_image[600:,:], axis=0)
    # plt.plot(histogram)
    # plt.show()
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped_image, warped_image, warped_image))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines



    # plt.imshow(out_img)
    # plt.title('before windows')
    # plt.show()

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 36
    # Set height of windows
    window_height = np.int(warped_image.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    print(nonzerox, nonzeroy)
    # Set the width of the windows +/- margin
    margin = 120
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    ## https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_image.shape[0] - (window+1)*window_height
        win_y_high = warped_image.shape[0] - window*window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        print(win_xright_low, win_xright_high)
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,140,0), 2)
        # print('rectangle 1', (win_xleft_low,win_y_low),(win_xleft_high,win_y_high))
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,140,0), 2)
        # print('rectangle 2', (win_xright_low,win_y_low), (win_xright_high,win_y_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        print(win_y_low, nonzeroy)
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    print(righty, rightx)
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # At this point, you're done! But here is how you can visualize the result as well:
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [30, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 30]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    # plt.show()

    #convert from pixel space to meter space
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    #calculate radisu of curvature
    left_eval = np.max(lefty)
    right_eval = np.max(righty)
    left_curverad = ((1 + (2*left_fit_cr[0]*left_eval + left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*right_eval + right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])

    # calculate left_min by finding minimum value in first index of array
    left_min = np.amin(leftx, axis=0)
    # print('left_min', left_min)
    right_max = np.amax(rightx, axis=0)
    # print('right max', right_max)
    actual_center = (right_max + left_min)/2
    dist_from_center =  actual_center - (1280/2)
    # print('pix dist from center', dist_from_center)

    meters_from_center = xm_per_pix * dist_from_center
    string_meters = str(round(meters_from_center, 2))

    return left_fitx, lefty, right_fitx, righty, ploty

def to_birdseye(thresholded_image):
    cv2.imshow("12", thresholded_image)
    height = thresholded_image.shape[0]
    width = thresholded_image.shape[1]

    # RIO
    top_height = 0.8
    top_left_corner_width = 0.2
    top_right_corner_width = 0.75
    offset = width*0.10
    src = np.float32([
        [width*top_left_corner_width, height*top_height],
        [width*top_right_corner_width, height*top_height],
        [width*0.76, height],
        [width*0.12, height]
    ])
    print(src)
    # plt.imshow(thresholded_image)
    # plt.show()
    dst = np.float32([
        [offset, 0],
        [width - offset, 0],
        [width - offset, height],
        [offset, height]
    ])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(thresholded_image, M, (width, height))
    return warped

def from_birdseye(warped_image):

    height = warped_image.shape[0]
    width = warped_image.shape[1]


    top_height = 0.8
    top_left_corner_width = 0.2
    top_right_corner_width = 0.8
    offset = width*0.10
    dst = np.float32([
        [width*top_left_corner_width, height*top_height],
        [width*top_right_corner_width, height*top_height],
        [width*0.76, height],
        [width*0.12, height]
    ])

    src = np.float32([
        [offset, 0],
        [width - offset, 0],
        [width - offset, height],
        [offset, height]
    ])


    Minv = cv2.getPerspectiveTransform(src, dst)
    # warp the blank back oto the original image using inverse perspective matrix
    unwarped = cv2.warpPerspective(warped_image, Minv, (width, height))

    return unwarped
def threshold(original_image):

    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    white_low = [100,100,100]
    white_high = [150,150,150]


    yellow_low = [0, 135, 168]
    yellow_high = [0, 204, 255]

    lower_white = np.array(white_low, dtype=np.uint8)
    higher_white = np.array(white_high, dtype=np.uint8)

    mask_white = cv2.inRange(original_image, lower_white, higher_white)

    res_white = cv2.bitwise_and(original_image,original_image, mask= mask_white)


    return res_white

def draw_lanes(left_fitx, lefty, right_fitx, righty, ploty):
    # create img to draw the lines on
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # recast x and y into usable format for cv2.fillPoly
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # print('pts left', pts_left.shape, 'pts right', pts_right.shape)
    pts = np.hstack((pts_left, pts_right))

    # draw the lane onto the warped blank img
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    unwarped = from_birdseye(color_warp)
    # combine the result with the original
    result = cv2.addWeighted(frame, 1, unwarped, 0.3, 0)
    return result


while cap.isOpened():  # True:#

    # try:

    ret, frame = cap.read()

    if ret:
        warped_image, rio = process_img(frame)

      


        left_fitx, lefty, right_fitx, righty, ploty = lr_curvature(warped_image)
        result = draw_lanes(left_fitx, lefty, right_fitx, righty, ploty)
        cv2.imshow("th", warped_image)
         # result = )
        cv2.imshow("th1", result)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


    # except Exception as e:
    #     print(str(e))
    #     pass

cv2.destroyAllWindows()
