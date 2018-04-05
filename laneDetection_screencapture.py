from time import sleep
import cv2
import numpy as np
import math
import win32gui
import win32ui
import win32con
import win32api

# https://www.programcreek.com/python/example/14102/win32gui.GetDesktopWindow


def grab_screen(region=(0, 53, 960, 590)):

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
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def process_img(original_image):

    if 'width' not in locals() or 'height' not in locals():
        width = original_image.shape[1]
        height = original_image.shape[0]
        region_of_interest_vertices = [
            (0, height), (width / 2, height / 2), (width, height)]

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(
        canny_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi/60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) 
            if math.fabs(slope) < 0.5:  
                continue
            if slope <= 0: 
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: 
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
 
    min_y = int(original_image.shape[0] * (3 / 5))
    max_y = int(original_image.shape[0])

    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
        
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    processed_lines = [[
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y],
    ]]

    return cropped_image, processed_lines


def region_of_interest(img, vertices):

    # maaritellaan tyhja matriisi joka tasmaa kuvan korkeuteen/leveyteen
    mask = np.zeros_like(img)

    # haetaan kuvan varikanavat
    channel_count = 1

    # luodaan tasmaava vari samoilla varikanavilla
    match_mask_color = (255,) * channel_count

    # taytetaan polygoni
    cv2.fillPoly(mask, vertices, match_mask_color)

    # palautetaan kuva jossa vain maskatut pikselit tasmaavat
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, vertices):
    # luodaan kaistan piirrolle musta kuva
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)


    # piirretaan kaistaviivat
    for line in processed_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), [250, 250, 0], 5)

    # lisataan alkuperaiseen kuvaan kaistaviivat
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    return img


while True:

    original_image = grab_screen()
    cropped_image, processed_lines = process_img(original_image)

    if processed_lines is not None:
                                                # piirra viivat
        image_with_lines = draw_lines(original_image, processed_lines)
        cv2.imshow("frame", image_with_lines)
    cv2.imshow("asas", cropped_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
