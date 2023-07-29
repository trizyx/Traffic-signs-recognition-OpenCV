import cv2 as cv
cap=cv.VideoCapture(0)

kirpich=cv.imread('C:/Users/vadim/Documents/GitHub/Traffic-signs-recognition-OpenCV/kirpich.jpg')
mainRoad=cv.imread('C:/Users/vadim/Documents/GitHub/Traffic-signs-recognition-OpenCV/main road.jpg')
forward=cv.imread('C:/Users/vadim/Documents/GitHub/Traffic-signs-recognition-OpenCV/forward.jpg')
kirpich=cv.resize(kirpich, (256, 256))
mainRoad=cv.resize(mainRoad, (256,256))
forward=cv.resize(forward, (256,256))
cv.imshow('kirpich', kirpich)
cv.imshow('mainRoad', mainRoad)
cv.imshow('forward', forward)
while True:
    ret, frame=cap.read()
    cv.imshow('frame', frame)

    clear=frame.copy()
    
    hsv=cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    cv.imshow('hsv', hsv)

    hsv=cv.blur(hsv, (5,5))
    cv.imshow('blur', hsv)

    mask=cv.inRange(hsv, (0, 162, 67), (255, 255, 255))
    cv.imshow('mask', mask)

    maskEr=cv.erode(mask, None, iterations=2)
    cv.imshow('erode', maskEr)
    maskDi=cv.dilate(maskEr, None, iterations=4)
    cv.imshow('dilate', maskDi)

    contours=cv.findContours(maskDi, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours=contours[0]

    if contours:
        contours=sorted(contours, key=cv.contourArea, reverse=True)
        cv.drawContours(frame, contours, -1, (0, 234, 135), 2)
        (x,y,w,h)=cv.boundingRect(contours[0])
        cv.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), 2)
        cv.imshow('contours', frame)

        obj=clear[y:y+h,x:x+w]
        cv.imshow('object', obj)

    

    if cv.waitKey(1)==ord('q'):
        break
cap.release()
cv.destroyAllWindows()