"""
Created by Anurag at 14-10-2021
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def manual_draw(input_img):
    count = [0.0] * 256
    w, h = input_img.shape
    for i in range(w):
        for j in range(h):
            count[input_img[i, j]] += 1
    return np.array(count)


def cumulative_sum(hist_aray):
    # count = 0
    # cum_array = []
    # for x in pixel_value:
    #     count += x
    #     cum_array.append(count)
    # return cum_array
    return [sum(hist_aray[:x+1]) for x in range(len(hist_aray))]


def manual_equalize(img):
    m, n = img.shape
    man_array = manual_draw(img)
    normalize_array = man_array/(m*n)
    cummulative_array = cumulative_sum(normalize_array)
    last_array = np.array(cummulative_array) * 255
    new_img = np.ones_like(img)
    for i in range(m):
        for j in range(n):
            new_img[i, j] = last_array[img[i, j]]
    return new_img, manual_draw(new_img)
        

if __name__ == '__main__':
    img = cv2.imread('03.jpg', 0)

    # manual histogram plot
    hist_manual = manual_draw(img)

    # manual equalized image and equalized array
    manual_equalize_img, equalize_array = manual_equalize(img)

    # opencv histogram equalization of image
    opencv_equalize_image = cv2.equalizeHist(img)


    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original image')

    plt.subplot(2, 3, 2)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.title('Opencv histogram plot')

    plt.subplot(2, 3, 3)
    plt.plot(hist_manual)
    plt.title('manual histogram plot')
    plt.xlim(0, 255)

    plt.subplot(2, 3, 4)
    plt.plot(equalize_array)
    plt.title('manual histogram equalized plot')
    plt.xlim(0, 255)

    plt.subplot(2, 3, 5)
    plt.imshow(opencv_equalize_image, cmap='gray')
    plt.title('opencv_equalized_image')

    plt.subplot(2, 3, 6)
    plt.imshow(manual_equalize_img, cmap='gray')
    plt.title('manual_equalized_image')
    plt.show()


