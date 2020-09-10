import cv2
import argparse

# define list of ages, gender for detection
AGES = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
        "(38-43)", "(48-53)", "(60-100)"]
GENDER = ['Male', 'Female']

MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# parsing argument
arg = argparse.ArgumentParser()
arg.add_argument('--image')
args = arg.parse_args()

# read face detector model
faceProt = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
faceNet = cv2.dnn.readNet(faceModel, faceProt)

# read age detector model
ageProt = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
ageNet = cv2.dnn.readNet(ageModel, ageProt)

# read gender detector model
genderProt = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'
genderNet = cv2.dnn.readNet(genderModel, genderProt)

# detection face and drawing frame
def selectFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    (h, w) = frameOpencvDnn.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detect = net.forward()
    faceBox = []

    for i in range(detect.shape[2]):
        conf = detect[0, 0, i, 2]
        if conf > conf_threshold:
            xStart = int(detect[0, 0, i, 3] * w)
            yStart = int(detect[0, 0, i, 4] * h)
            xEnd = int(detect[0, 0, i, 5] * w)
            yEnd = int(detect[0, 0, i, 6] * h)
            faceBox.append([xStart, yStart, xEnd, yEnd])
            cv2.rectangle(frameOpencvDnn, (xStart, yStart), (xEnd, yEnd), (255, 0, 255), int(round((h / 150), 8)))
    return frameOpencvDnn, faceBox

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

while cv2.waitKey(1) < 0:
    isFrame, frame = video.read()

    if not isFrame:
        cv2.waitKey()
        break

    finalImg, faceBoxes = selectFace(faceNet, frame)

    if not faceBoxes:
        print("Face not found")
        cv2.waitKey()
        break

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding): \
                                                                    min(faceBox[2] + padding, frame.shape[1] - 1)]

# pass the blob through the network and obtain the gender detections and predict
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MEAN_VALUES, False)
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = GENDER[genderPred[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = AGES[agePred[0].argmax()]
        print(f'Age: {age[1:-1]} years')

# showing result and save
        cv2.putText(finalImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", finalImg)
        cv2.imwrite(f'{gender}, {age}.jpg', finalImg)
