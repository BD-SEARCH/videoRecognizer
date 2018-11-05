# STEP1. 영상 찍기

import cv2 as cv
import os

opt = input("(0)train? or (1)test?: ")
while opt!='0' and opt!='1':
    print("error")
    opt = input("(0)for train? or (1)test?: ")
if opt=='0': opt='training_set'
elif opt=='1': opt='test_set'

filename = input("write what/who it is : ")
# cv2로부터 VideoCapture 객체 생성
# cap = cv.VideoCapture('input.mp4')
cap = cv.VideoCapture(0)

if not os.path.exists('./dataset_img/training_set/'+filename):
    os.mkdir('./dataset_img/training_set/'+filename)
    os.mkdir('./dataset_img/test_set/'+filename)
    print("Directory " , filename ,  " Created ")
else:
    print("Directory " , filename ,  " already exists")

# fourcc = cv.VideoWriter_fourcc(*'XVID')
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('./output.mp4',fourcc,30.0, (640,480))

if cap.isOpened() == False:
    print("error01")

while(cap.isOpened()):
    # capture the image from camera
    ret, frame = cap.read()
    # if not captured / error happened
    if ret==0: break

    # 상하 뒤집기
    frame = cv.resize(frame,(640,480))
    # 이미지를 파일로 저장. VideoWriter 개체에 연속적 저장 : 동영상으로
    out.write(frame)

    # 화면에 이미지 출력, 연속적으로 화면에 출력하면 동영상이 된다
    cv.imshow('frame',frame)

    # ESC누르면 종료
    if cv.waitKey(1) & 0xFF==27:
        print("fin")
        break

# VideoCapture 객체의 메모리 해제하고 모든 윈도 창 종료
cap.release()
out.release()
cv.destroyAllWindows()

# STEP2. 영상 to 이미지

import cv2
import os
import time

def video_to_frames(video, path_output_dir):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, str(round(time.time()))+'%d.jpg') % count, image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()

video_to_frames('./output.mp4', './dataset_img/'+opt+'/'+filename)
print('save '+filename)
print('dir : ./dataset_img/'+opt+'/'+filename)
os.remove('./output.mp4')
