import torch
import cv2
import face_recognition
import pygame
import threading

#对比图片
picture =face_recognition.load_image_file("ypx.jpg")
known_encoding = face_recognition.face_encodings(picture)[0]
img =face_recognition.load_image_file("me.jpg")#没事用，只是为了初始化

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
cap=cv2.VideoCapture(0)
cv2.namedWindow("test", cv2.WINDOW_NORMAL)
lock=threading.Lock()
def process_dection(dections,conf=0.3):
    #conf置信度
    person_dect=[]
    for det in dections:
        if det[5]==0:#person
            if det[4]>=conf:
                person_dect.append(det[:4].cpu().numpy())
    return person_dect
def playdayi():
    pygame.mixer.init()
    pygame.mixer.music.load("dayi.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    pygame.quit()


def draw_dection(image,dections,color=(0,255,0),thickness=2):
    perimg=[]
    for det in dections:
        x1,y1,x2,y2=det
        h=image.shape[0]
        w=image.shape[1]
        x1=int(x1*w)
        x2=int(x2*w)
        y1=int(y1*h)
        y2=int(y2*h)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image
def imgprocess(encoding):
    global img
    while cv2.getWindowProperty("test", 0) >= 0:
        unknow_encoding = face_recognition.face_encodings(img)
        if len(unknow_encoding) > 0:
            sim = face_recognition.compare_faces([encoding], unknow_encoding[0], tolerance=0.5)
            if sim[0]:
                # 播放音频
                playdayi()
def camcap():
    global img
    while cv2.getWindowProperty("test", 0) >= 0:
        ret, img = cap.read()  # # or file, Path, PIL, OpenCV, numpy, list
        if not ret:
            break
        result = model(img).xyxyn
        result = result[0]
        dections = process_dection(result)
        img = draw_dection(img, dections)
        cv2.imshow("test", img)
        cv2.waitKey(1)
    cap.realse()
    cv2.destroyAllWindows()

threadpro=threading.Thread(target=imgprocess,args=(known_encoding,))
threadpro.start()
camcap()
threadpro.join()
