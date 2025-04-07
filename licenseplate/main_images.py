#IMAGES WITHOUT TESSARACT
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

showSteps = True

def main():
    if not DetectChars.loadKNNDataAndTrainKNN():
        print("\nError: KNN training was not successful\n")
        return
    
    imgOriginalScene = cv2.imread("12.png")
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
        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)

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
    main()
