import cv2
import pytesseract
import matplotlib.pyplot as plt

# Read the image using OpenCV
image = cv2.imread('image.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment the text from the background
_, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Perform some morphological operations to enhance text extraction
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)

# Use Tesseract to perform OCR (Optical Character Recognition)
text = pytesseract.image_to_string(morph_image)

# Display the image using Matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

# Print the extracted text
print("Extracted Text:")
print(text)
