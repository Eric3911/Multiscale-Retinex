import cv2
import numpy as np


# 单尺度Retinex
def singleScaleRetinex(img, sigma):
    # 按照公式计算
    _temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(_temp == 0, 0.001, _temp)
    img_ssr = np.log10(img + 0.01) - np.log10(gaussian)
    # 量化到0--255
    for i in range(img_ssr.shape[2]):
        img_ssr[:, :, i] = (img_ssr[:, :, i] - np.min(img_ssr[:, :, i])) / \
                           (np.max(img_ssr[:, :, i]) - np.min(img_ssr[:, :, i])) * 255
    img_ssr = np.uint8(np.minimum(np.maximum(img_ssr, 0), 255))
    return img_ssr


def singleScaleRetinexTemp(img, sigma):
    # 按照公式计算
    _temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(_temp == 0, 0.001, _temp)
    retinex = np.log10(img + 0.01) - np.log10(gaussian)

    return retinex


# 多尺度Retinex, sigma_list[15,80,250]
def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img * 1.0)
    for sigma in sigma_list:
        print("sigma:", sigma)
        retinex += singleScaleRetinexTemp(img, sigma)
    img_msr = retinex / len(sigma_list)
    for i in range(img_msr.shape[2]):
        img_msr[:, :, i] = (img_msr[:, :, i] - np.min(img_msr[:, :, i])) / \
                           (np.max(img_msr[:, :, i]) - np.min(img_msr[:, :, i])) * 255
    img_msr = np.uint8(np.minimum(np.maximum(img_msr, 0), 255))
    return img_msr


if __name__ == '__main__':
    imageSrc = cv2.imread("D:\\opencvpy\\003\\0000.jpg")
    cv2.imshow('src', imageSrc)

    dstsrc = singleScaleRetinex(imageSrc, 300)
    cv2.imshow('ssr', dstsrc)

    sigma_list = [15, 80, 250]
    dstmsr = multiScaleRetinex(imageSrc, sigma_list)
    cv2.imshow('msr', dstmsr)
    cv2.waitKey(0)