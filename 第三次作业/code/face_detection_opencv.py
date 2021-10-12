import cv2


def face(img):
    face_cascade = cv2.CascadeClassifier('D:/anaconda/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 绘制人脸矩形框
    for id, (x, y, w, h) in enumerate(faces):
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # 写上文本
    for id, (x, y, w, h) in enumerate(faces):
        cv2.putText(img, "face", (x + w // 2 - 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img


if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("output0.avi", fourcc, 30, (720, 480))
    cap = cv2.VideoCapture("input.avi")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = face(frame)
            videoWriter.write(frame)
            cv2.imshow("frame", frame)
        else:
            break
        key = cv2.waitKey(10)
        if key == 27:
            break
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

