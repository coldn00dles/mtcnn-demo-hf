import cv2
from mtcnn import MTCNN

detector = MTCNN()

cap = cv2.VideoCapture(0)

while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            x,y,w,h = bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]
            roi = frame[y:y+h, x:x+w]
            roi = cv2.blur(roi, (50,50))
            # impose this blurred image on original image to get final image
            frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()