import cv2

def rotate(image, angle=20, scale=1.0, grayscale=False):
    if grayscale:
        height, width = image.shape
    else:
        height, width, _ = image.shape
    #rotate matrix
    translation = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,translation,(width,height))

    return image
