import cv2

def resize_frame(frame, scale_percent):
    """
    Resizes an image by a given scale percentage.

    Args:
        frame (numpy.ndarray): The original image/frame.
        scale_percent (int): The scaling percentage.

    Returns:
        numpy.ndarray: The resized image/frame.
    """
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized