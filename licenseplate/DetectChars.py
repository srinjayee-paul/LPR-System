import os
import cv2
import numpy as np
import math
import random

import Main
import Preprocess
import PossibleChar

kNearest = cv2.ml.KNearest_create()

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

def loadKNNDataAndTrainKNN():
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return False

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return False

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))

    kNearest.setDefaultK(1)
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    return True

def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:
        return listOfPossiblePlates

    for possiblePlate in listOfPossiblePlates:

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)

        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx=1.6, fy=1.6)

        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if len(listOfListsOfMatchingCharsInPlate) == 0:
            possiblePlate.strChars = ""
            continue

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            listOfListsOfMatchingCharsInPlate[i].sort(key=lambda matchingChar: matchingChar.intCenterX)
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])

        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i

        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

    return listOfPossiblePlates

def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []
    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        possibleChar = PossibleChar.PossibleChar(contour)
        if checkIfPossibleChar(possibleChar):
            listOfPossibleChars.append(possibleChar)

    return listOfPossibleChars

def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and 
        possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False

def findListOfListsOfMatchingChars(listOfPossibleChars):
    listOfListsOfMatchingChars = []

    for possibleChar in listOfPossibleChars:
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)
        listOfMatchingChars.append(possibleChar)

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue
        
        listOfListsOfMatchingChars.append(listOfMatchingChars)

        listWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveMatches = findListOfListsOfMatchingChars(listWithCurrentMatchesRemoved)
        
        for recursiveMatch in recursiveMatches:
            listOfListsOfMatchingChars.append(recursiveMatch)
        
        break
    
    return listOfListsOfMatchingChars

def findListOfMatchingChars(possibleChar, listofchars):
    matchingchars=[]
    
    for char in listofchars:
        
        if char==possibleChar:
            continue
        
        fltDistanceBetween=distanceBetweenChars(possibleChar,char)
        
        fltAngleBetween=angleBetweenChars(possibleChar,char)
        
        fltChangeArea=float(abs(char.intBoundingRectArea-possibleChar.intBoundingRectArea))/float(possibleChar.intBoundingRectArea)
        
        fltChangeWidth=float(abs(char.intBoundingRectWidth-possibleChar.intBoundingRectWidth))/float(possibleChar.intBoundingRectWidth)
        
        fltChangeHeight=float(abs(char.intBoundingRectHeight-possibleChar.intBoundingRectHeight))/float(possibleChar.intBoundingRectHeight)
        
        if(fltDistanceBetween<(possibleChar.fltDiagonalSize*MAX_DIAG_SIZE_MULTIPLE_AWAY) and fltAngleBetween<MAX_ANGLE_BETWEEN_CHARS and fltChangeArea<MAX_CHANGE_IN_AREA and fltChangeWidth<MAX_CHANGE_IN_WIDTH and fltChangeHeight<MAX_CHANGE_IN_HEIGHT):
            
            matchingchars.append(char)
    
    return matchingchars

def distanceBetweenChars(firstchar,secondchar):
    
    intX=abs(firstchar.intCenterX-secondchar.intCenterX)
    
    intY=abs(firstchar.intCenterY-secondchar.intCenterY)
    
    return math.sqrt((intX**2)+(intY**2))

def angleBetweenChars(firstchar,secondchar):
    
    fltAdj=float(abs(firstchar.intCenterX-secondchar.intCenterX))
    
    fltOpp=float(abs(firstchar.intCenterY-secondchar.intCenterY))
    
    if fltAdj!=0.0:
        
       angleRad=math.atan(fltOpp/fltAdj)
       
    else:
        
       angleRad=1.5708
    
    angleDeg=angleRad*(180.0/math.pi)
    
    return angleDeg

def removeInnerOverlappingChars(matchingchars):
    
   charsNoOverlap=list(matchingchars)
   
   for current in matchingchars:
       
       for other in matchingchars:
           
           if current!=other and distanceBetweenChars(current,other)<(current.fltDiagonalSize*MIN_DIAG_SIZE_MULTIPLE_AWAY):
               
               if current.intBoundingRectArea<other.intBoundingRectArea and current in charsNoOverlap: 
                   charsNoOverlap.remove(current)
               elif other in charsNoOverlap: 
                   charsNoOverlap.remove(other)
   
   return charsNoOverlap
   
def recognizeCharsInPlate(imgThresh,listofmatchingchars):
    
   strchars=""
   
   height,width=imgThresh.shape
   
   imgThreshColor=np.zeros((height,width,3),np.uint8)
   
   listofmatchingchars.sort(key=lambda match:match.intCenterX)

   cv2.cvtColor(imgThresh,cv2.COLOR_GRAY2BGR,imgThreshColor)

   for char in listofmatchingchars:

       imgROI=imgThresh[char.intBoundingRectY:char.intBoundingRectY+char.intBoundingRectHeight,char.intBoundingRectX:char.intBoundingRectX+char.intBoundingRectWidth]

       imgROIResized=cv2.resize(imgROI,(RESIZED_CHAR_IMAGE_WIDTH,RESIZED_CHAR_IMAGE_HEIGHT))

       npaROIResized=np.float32(imgROIResized.reshape((1,-1)))

       retval,npaResults,nr,dists=kNearest.findNearest(npaROIResized,k=1)

       strCurrent=str(chr(int(npaResults[0][0])))

       strchars+=strCurrent
   
   return strchars
