import cv2 # 이미지 합성
import dlib # 얼굴인식 카메라 만들기 얼굴 영역 탐지, 랜드마크 탐지
from imutils import face_utils, resize # resize하는 것
import numpy as np # 형 변환하는 것

orange_img = cv2.imread('orange.jpg') # imread()를 이용해서 오렌지 이미지를 불러옴.
orange_img = cv2.resize(orange_img, dsize=(512, 512))

detector = dlib.get_frontal_face_detector() # 얼굴영역 탐지
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 랜드마크 탐지

cap = cv2.VideoCapture('video.mp4') # 0으로 설정하면 웹캠을 사용 ('01.mp4'로 설정하면 video를 사용.)

while cap.isOpened():
    ret, img = cap.read() # cap.read()를 통해서 image를 읽어준 다음에

    if not ret: # 프레임이 더 없으면 반복문을 빠져 나온다.
        break

    faces = detector(img) # 얼굴 영역을 인식 

    # result는 orange_img를 copy한 이미지
    result = orange_img.copy()

    if len(faces) > 0: # 얼굴이 1개 이상이면
        face = faces[0] # 얼굴은 하나만 이용할 것이기 때문에 0번 인덱스만
        
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_img = img[y1:y2, x1:x2].copy() # 크랍해서 face_img에 저장

        shape = predictor(img, face) # 랜드마크 68점을 구하는 것 (코드 30~31번)
        shape = face_utils.shape_to_np(shape) # shape에 68개의 점이 담기게 됨. 
        # shape_to_np는 numpy로 바꾸는 것임.

        for p in shape: # 이미지(얼굴)에 68개의 점이 찍힘.
            cv2.circle(face_img, center=(p[0] - x1, p[1] - y1), radius=2, color=255, thickness=-1)

        # eyes
        # 왼쪽 눈 짜를 때, x축은 36번, 39번 y축은 37번, 41번
        le_x1 = shape[36, 0]
        le_y1 = shape[37, 1]
        le_x2 = shape[39, 0]
        le_y2 = shape[41, 1]
        le_margin = int((le_x2 - le_x1) * 0.18) # 너무 타이트하게 짜르면 안되니까 margin을 줌.

        # 오른쪽 눈도 왼쪽눈과 같이 마찬가지로.
        re_x1 = shape[42, 0]
        re_y1 = shape[43, 1]
        re_x2 = shape[45, 0]
        re_y2 = shape[47, 1]
        re_margin = int((re_x2 - re_x1) * 0.18) # 너무 타이트하게 짜르면 안되니까 margin을 줌.

        # 왼쪽 눈, 오른쪽 눈에다가 margin을 줘서 크랍을 한다.
        left_eye_img = img[le_y1-le_margin:le_y2+le_margin, le_x1-le_margin:le_x2+le_margin].copy()
        right_eye_img = img[re_y1-re_margin:re_y2+re_margin, re_x1-re_margin:re_x2+re_margin].copy()

        # 가로를 100으로 resize 해줌.
        left_eye_img = resize(left_eye_img, width=100)
        right_eye_img = resize(right_eye_img, width=100)

        # seamlessClone()이란 seamless하게 티가 안나게 합성을 해주는 것이다.
        result = cv2.seamlessClone(
            left_eye_img, #왼쪽 눈 합성하고
            result, # result애 합성을 한다. (orange 이미지를 copy한 이미지)
            np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
            (100, 200),
            # 옵션으로 MIXED_CLONE을 주면 알아서 합성을 해줌.
            cv2.MIXED_CLONE
            # cv2.NORMAL_CLONE 대신 하면 더 잘보인다.
        )

        result = cv2.seamlessClone(
            right_eye_img, # 오른쪽 눈 합성하는데,
            result, # result에 합성을 한다. (orange 이미지를 copy한 이미지)
            np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
            (250, 200),
            # 옵션으로 MIXED_CLONE을 주면 알아서 합성을 해줌.
            cv2.MIXED_CLONE
            # cv2.NORMAL_CLONE 대신 하면 더 잘보인다.
        )

        # mouth
        # 입을 짜를 때, x축은 48번, 54번 y축은 50번, 57번
        mouth_x1 = shape[48, 0]
        mouth_y1 = shape[50, 1]
        mouth_x2 = shape[54, 0]
        mouth_y2 = shape[57, 1]
        mouth_margin = int((mouth_x2 - mouth_x1) * 0.1) # margin은 0.1정도로 줌.

        # 크랍을 해서 mouth_img에 저장
        mouth_img = img[mouth_y1-mouth_margin:mouth_y2+mouth_margin, mouth_x1-mouth_margin:mouth_x2+mouth_margin].copy()

        # resize하고
        mouth_img = resize(mouth_img, width=250)

        # seamlessClone을 한다.
        result = cv2.seamlessClone(
            mouth_img,
            result,
            np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
            (180, 320),
            cv2.MIXED_CLONE
            # cv2.NORMAL_CLONE 대신 하면 더 잘보인다.
        )

        
        cv2.imshow('left', left_eye_img)
        cv2.imshow('right', right_eye_img)
        cv2.imshow('mouth', mouth_img)
        cv2.imshow('face', face_img)

        cv2.imshow('result', result)

    # cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break