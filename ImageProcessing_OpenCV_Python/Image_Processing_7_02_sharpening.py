import numpy as np, cv2
from Common.filters import filter
# 앞에서 만들었던 filter()함수를 'Common.filters.py' 파일에 추가하고 import

image = cv2.imread("images/filter_sharpen.jpg", cv2.IMREAD_GRAYSCALE) # 영상 읽기
if image is None: raise Exception("영상파일 읽기 오류")

#샤프닝 마스크 원소 지정
datal = [0, -1, 0, #1차원 리스트
         -1, 5, -1,
         0, -1, 0]

data2 = [[-1, -1, -1], # 2차원 리스트
         [-1, 9, -1],
         [-1, -1, -1]]

# 1차원 리스트로 ndarray 객체를 만들어 reshape()함수로 2차원 행렬로 변경
mask1 = np.array(data1, np.float32).reshape(3, 3)

# 2차원 리스트로 ndarray 객체를 만든다, 형태 변경 필요없음
mask2 = np.array(data2, np.float32)

# 구현 filter()함수 호출해서 회선 수행
sharpen1 = filter(image, mask1) 
sharpen2 = filter(image, mask2)

sharpen1 = cv2.convertscaleAbs(sharpen1)
sharpen2 = cv2.convertscaleAbs(sharpen2)
# filter함수의 반환 행렬의 자료형이 실수(float32)형이기 때문에
# cv2.convertscaleabs()함수를 이용해서 행럴 원소의 절댓값 계산 및 스케일 조정을 하고 정수형(unit8)으로 번환

cv2.imshow("image", image)
cv2.imshow("sharpen1", cv2.convertscaleAbs(sharpen1)) # 원도우 표시 위한 형변환
cv2.imshow("sharpen2", cv2.convertscaleAbs(sharpen2))
# cv2.imshow 함수로 윈도우에 ndarray 객체를 영상으로 표시할 때, 주로 행렬의 자료형이 정수(unit8)형을 사용
# 실수형일 경우에는 원소값이 0~1 사이로 구성되어야 함
cv2.waitkey( )
