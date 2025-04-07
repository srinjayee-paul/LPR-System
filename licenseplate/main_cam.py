#REAL TIME WITHOUT TESSARACT
import cv2
import numpy as np
import os
import DetectChars
import DetectPlates
import PossiblePlate

SCALAR_BLACK = (0, 0, 0)
SCALAR_WHITE = (255, 255, 255)
SCALAR_YELLOW = (0, 255, 255)
SCALAR_GREEN = (0, 255, 0)
SCALAR_RED = (0, 0, 255)

showSteps = False

def capture_image():
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.imshow("Camera Feed - Press SPACE to Capture", frame)

        key = cv2.waitKey(1) & 0xFF 

        if cv2.getWindowProperty("Camera Feed - Press SPACE to Capture", cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == 32: 
            image_path = "captured_image.png"
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved as '{image_path}'")
            cap.release()
            cv2.destroyAllWindows()
            return image_path

        elif key == 27: 
            break

    cap.release()  
    cv2.destroyAllWindows() 
    return None

def process_image(image_path):
    if not DetectChars.loadKNNDataAndTrainKNN():
        print("\nError: KNN training was not successful\n")
        return
    
    imgOriginalScene = cv2.imread(image_path)
    if imgOriginalScene is None:
        print("\nError: Image not read from file\n")
        os.system("pause")
        return
    
    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    cv2.imshow("imgOriginalScene", imgOriginalScene)

    if len(listOfPossiblePlates) == 0:
        print("\nNo license plates were detected\n")
    else:
        listOfPossiblePlates.sort(key=lambda plate: len(plate.strChars), reverse=True)

        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:
            print("\nNo characters were detected\n")
            return

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

        print("\nLicense plate read from image = " + licPlate.strChars + "\n")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)

        cv2.imshow("imgOriginalScene", imgOriginalScene)
        cv2.imwrite("processed_image.png", imgOriginalScene)

    cv2.waitKey(0)

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    try:
        p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
        p2fRectPoints = np.array(p2fRectPoints, dtype=np.int32)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
    
    except Exception as e:
        print(f"Error drawing rectangle: {e}")

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    sceneHeight, sceneWidth, _ = imgOriginalScene.shape
    plateHeight, plateWidth, _ = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = plateHeight / 30.0
    intFontThickness = int(round(fltFontScale * 1.5))

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)

    ((intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), _) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)
    intPlateCenterY = int(intPlateCenterY)

    if intPlateCenterY < (sceneHeight * 0.75):
        ptCenterOfTextAreaY = int(round(intPlateCenterY + plateHeight * 1.6))
    else:
        ptCenterOfTextAreaY = int(round(intPlateCenterY - plateHeight * 1.6))

    textSizeWidth, textSizeHeight = textSize
    ptLowerLeftTextOriginX = int(intPlateCenterX - textSizeWidth / 2)
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + textSizeHeight / 2)

    cv2.putText(imgOriginalScene,
                licPlate.strChars,
                (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY),
                intFontFace,
                fltFontScale,
                SCALAR_YELLOW,
                intFontThickness)

if __name__ == "__main__":
    image_path = capture_image()
    if image_path:
        process_image(image_path)
