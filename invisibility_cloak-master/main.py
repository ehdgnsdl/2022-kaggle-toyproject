# 이미지 처리 패키지
import cv2
# 행렬 연산 패키지
import numpy as np
import time, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', help='Input video path')
args = parser.parse_args()

# cv2.VideoCapture() 비디오 파일을 읽는다. 0으로 설정하면 웹캠을 읽는다.
cap = cv2.VideoCapture(args.video if args.video else 0)

# 카메라가 켜지는데에 3초가 걸리기 때문에 3초간 sleep
time.sleep(3)

# Grap background image from first part of the video
# 사람이 나오지 않는 이미지가 필요함. (2초 정도, background)
# background에 저장해줌.
for i in range(60):
  ret, background = cap.read()

# 결과 값을 기록하기 위헤서
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('videos/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (background.shape[1], background.shape[0]))
out2 = cv2.VideoWriter('videos/original.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (background.shape[1], background.shape[0]))

while(cap.isOpened()):
      # cap.read() 한프레임씩 읽어온다.
  ret, img = cap.read()
  if not ret:
    break
  
  # Convert the color space from BGR to HSV
  # cv2.cvtColor() 컬러시스템을 변경한다.
  # cv2.COLOR_BGR2HSV: BGR을 HSV로 바꾼다.
  # 그 이유는 사람이 인식하는 색깔의 수치와 HSV 컬러시스템이 표현하는 방식이 가장 비슷하다고함.
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Generate mask to detect red color
  # H: Hue 색조
  # S: Saturation 채도
  # V: Value 명도
  
  # 빨간색 범위는 0-10, 170-180이라 두 개로 나누어 마스크 생성 후 더함!
  lower_red = np.array([0, 120, 70])
  upper_red = np.array([10, 255, 255])
  # cv2.inRange() 범위 안에 해당하는 값들로 마스크를 생성
  mask1 = cv2.inRange(hsv, lower_red, upper_red)

  lower_red = np.array([170, 120, 70])
  upper_red = np.array([180, 255, 255])
  mask2 = cv2.inRange(hsv, lower_red, upper_red)

  mask1 = mask1 + mask2

  # lower_black = np.array([0, 0, 0])
  # upper_black = np.array([255, 255, 80])
  # mask1 = cv2.inRange(hsv, lower_black, upper_black)

  '''
  # Refining the mask corresponding to the detected red color
  https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
  '''
  # Remove noise
  # cv2.morphologyEx()는 noise를 삭제
  # cv2.dilate()는 픽셀을 늘려주는 함수(약간 넓게)  
  mask_cloak = cv2.morphologyEx(mask1, op=cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)
  mask_cloak = cv2.dilate(mask_cloak, kernel=np.ones((3, 3), np.uint8), iterations=1)
  mask_bg = cv2.bitwise_not(mask_cloak)

  cv2.imshow('mask_cloak', mask_cloak)

  # Generate the final output
  # cv2.bitwise_and()는 두 개의 행렬이 0이 아닌 것만 통과됨. 즉 마스크 영역만 남음(And 연산)
  # mask_cloak만큼만 추출. (즉, 사물을 제외한 배경이 지워짐 대신 보여지는 것은 사물이 있었던 곳을 background로 보여줌)
  res1 = cv2.bitwise_and(background, background, mask=mask_cloak)
  # mask_bg만큼만 추출 (즉, 사물만 지워짐)
  res2 = cv2.bitwise_and(img, img, mask=mask_bg)
  # cv2.addWeighted()로 두 개의 이미지를 합친다.
  result = cv2.addWeighted(src1=res1, alpha=1, src2=res2, beta=1, gamma=0)

  cv2.imshow('res1', res1)
  cv2.imshow('res2', res2)

  # cv2.imshow('ori', img)
  # cv2.imshow(): 이미지를 윈도우에 띄운다.
  cv2.imshow('result', result)
  out.write(result)
  out2.write(img)

  if cv2.waitKey(1) == ord('q'):
    break

out.release()
out2.release()
cap.release()
