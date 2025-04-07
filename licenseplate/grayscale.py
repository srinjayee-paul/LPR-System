import cv2

# Load the input image
image_path = r"C:\Users\Sneha\Pictures\Screenshots\Screenshot_23.jpg"  # Corrected file path
image = cv2.imread(image_path)

# Check if the image is loaded properly
if image is None:
    print("Error: Could not read the image. Check the file path.")
else:
    cv2.imshow('Original', image)
    cv2.waitKey(0)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Grayscale', gray_image)
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
