import cv2
import numpy as np
import pytesseract

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image, convert to grayscale, sharpen, apply Otsu's threshold
image = cv2.imread('img.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened_image = cv2.filter2D(gray_image, -1, sharpen_kernel)
thresholded_image = cv2.threshold(sharpened_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Perform OCR
extracted_text = pytesseract.image_to_string(thresholded_image, lang='eng', config='--psm 6')

# Extract numbers from the text
extracted_numbers = ''.join(filter(str.isdigit, extracted_text))

# Print extracted numbers
print("Extracted Numbers:", extracted_numbers)
