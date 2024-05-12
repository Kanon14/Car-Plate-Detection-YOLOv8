import hydra
import torch
import easyocr
import cv2
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.plotting import Annotator, colors, save_one_box

def getOCR(img, coords):
    """
    Extracts text from a specified region of an image using EasyOCR.

    Args:
        img (numpy.ndarray): The input image from which text is to be extracted.
        coords (tuple): A tuple containing four coordinates (x, y, w, h) that define
                        a rectangular region in the image.

    Returns:
        str: The extracted text from the specified region of the image.
    """
    # Extract coordinates from the input and convert them to integers
    x, y, w, h = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
    
    # Crop the image to the region defined by the coordinates
    img = img[y:h, x:w]
    
    # Set the confidence threshold for OCR results
    conf = 0.2
    
    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    ocr = ""
    
    # Iterate over the OCR results
    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) > 1 and len(results[1]) > 6 and results[2] > conf:
            ocr = results[1]
    
    return str(ocr)
    
    
if __name__ == "__main__":
    reader = easyocr.Reader(['en'])