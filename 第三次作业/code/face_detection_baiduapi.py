import cv2
from aip import AipFace
import base64
import math


APP_ID = '24967392'
API_KEY = "Qu71W4CQHT0ocj3xWCD90aEV"
SECRET_KEY = "xeRQG52AwwWHcZLbqvo3G9DLiWDSyfS3"
aipFace = AipFace(APP_ID, API_KEY, SECRET_KEY)


def face(img):
    image = img
    cv2.imwrite('temp.png', img)
    with open('temp.png', 'rb') as fp:
        content = base64.b64encode(fp.read())
    img = str(content, 'utf-8')
    options = {
        "face_field": "age",
        "max_face_num": 3,
        "face_type": "LIVE"
    }
    result = aipFace.detect(img, image_type="BASE64", options=options)

    if 'result' not in result or result['result'] is None:
        return image

    face_num = result['result']['face_num']
    # 绘制人脸矩形框
    for num in range(0, int(face_num)):
        location = result['result']['face_list'][num - 1]['location']

        Theta = location['rotation'] / 60
        A = (int(location['left']), int(location['top']))
        B = (int(location['left']) + int(location['width'] * math.cos(Theta)),
             int(location['top']) + int(location['width'] * math.sin(Theta)))
        AC_Len = math.sqrt(location['width'] ** 2 + location['height'] ** 2)
        AC_Theta = math.atan(location['height'] / location['width']) + location['rotation'] / 60
        C = (int(location['left']) + int(AC_Len * math.cos(AC_Theta)),
             int(location['top']) + int(AC_Len * math.sin(AC_Theta)))
        D = (int(location['left']) - int(location['height'] * math.sin(Theta)),
             int(location['top']) + int(location['height'] * math.cos(Theta)))
        cv2.line(image, A, B, (0, 0, 255), 2)
        cv2.line(image, B, C, (0, 0, 255), 2)
        cv2.line(image, C, D, (0, 0, 255), 2)
        cv2.line(image, D, A, (0, 0, 255), 2)
        cv2.putText(image, "face", (int(location['left']) + int(location['width'] * math.cos(Theta)) // 2, int(location['top']) + int(location['width'] * math.sin(Theta))//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image


if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("output1.avi", fourcc, 30, (720, 480))
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
