import cv2
import os

image_path = "C:/develop/vision-test/file/image1.jpg"
# image_path = "C:/develop/vision-test/file/bg_image.webp"
# image_path = "C:/develop/vision-test/file/old_image.webp"
# image_path = "C:/develop/vision-test/file/blur_image.webp"
if not os.path.exists(image_path):
    print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    exit()

img = cv2.imread(image_path)
if img is None:
    print("이미지를 읽을 수 없습니다.")
    exit()

# OpenCV의 내장 Haar Cascade를 사용
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

if len(faces) > 0:
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    print(f"{len(faces)}개의 얼굴을 감지했습니다.")
else:
    print("얼굴을 감지하지 못했습니다.")

cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
