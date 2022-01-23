# 손가락 마디의 각도를 계산하는 부분만 남겨둠.
import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 1
# 제스처의 클래스를 정의하는 부분에서 11번 fy 클래스를 만들어준다.
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok', 11:'fy'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition data
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
print(file.shape)

# 0으로하면 웹캠
cap = cv2.VideoCapture(0)

# 클릭했을 때 현재 각도 data를 원본 file에 추가하도록 만들 것이다.
def click(event, x, y, flags, param):
    global data, file
    # 클릭했을 때의 핸들러
    if event == cv2.EVENT_LBUTTONDOWN:
        # numpy의 vstack을 사용해서 이어 붙여 준다.
        file = np.vstack((file, data))
        print(file.shape)

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', click)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) # Convert radian to degree

            data = np.array([angle], dtype=np.float32)
            data = np.append(data, 11)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Dataset', img)
    # q를 눌러서 while문을 빠져나온다.
    if cv2.waitKey(1) == ord('q'):
        break

# click할 때마다 모은 데이터를 여기에다가 저장
np.savetxt('data/gesture_train_fy.csv', file, delimiter=',')
