import numpy as np, cv2, time

# 회선 수행 함수 - 행렬 처리 방식(속도 면에서 유리)
def filter(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)               # 회선 결과 저장 행렬 
                           #계산 과정에서 소수 부분을 보존하기 위해 실수형으로 생성
    ycenter, xcenter = mask.shape[1]//2, mask.shape[0]//2  # 마스크 중심 좌표

    for i in range(ycenter, rows - ycenter):               # 입력 행렬 반복 순회
        for j in range(xcenter, cols - xcenter): 
            # 입력 영상의 상하좌우 끝부분에서 마스크 절반 크기는 계산 대상에서 제외
            # 회선 과정에서 입력 영상 범위를 벗어나는 좌표를 제외하기 위함
            y1, y2 = i - ycenter, i + ycenter + 1          # 관심영역 높이 범위
            x1, x2 = j - xcenter, j + xcenter + 1          # 관심영역 너비 범위
            roi = image[y1:y2, x1:x2].astype("float32")    # 관심영역 형변환 
                        # 계산 과정에서 소수 부분을 보존하기 위해 실수형으로 형변환
            
            tmp = cv2.multiply(roi, mask)                  # 회선 적용 - OpenCV 곱셈 
            # cv2.multiply()함수 이용:마스크 행렬과 입력영상의 해당 변위의 원소간 곱셈으로 회선 수행

            dst[i, j] = cv2.sumElems(tmp)[0]               # 출력화소 저장 
            # 곱셈이 수행된 관심 영역 행렬의 원소를 모두 합하기 위해 cv2.sumElems()함수 사용
            # 입력행렬이 단일채널이기 때문에 0번째 원소만 가져오면 됨
    return dst                                             # 자료형 변환하여 반환

# 회선 수행 함수 - 화소 직접 접근 (수행 속도가 상당히 느림)
def filter2(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)               # 회선 결과 저장 행렬
    ycenter, xcenter = mask.shape[1]//2, mask.shape[0]//2  # 마스크 중심 좌표

    for i in range(ycenter, rows - ycenter):               # 입력 행렬 반복 순회
        for j in range(xcenter, cols - xcenter):
            sum = 0.0
            for u in range(mask.shape[0]):  
                # 마스크 원소 순회하며 마스크 원소와 마스크 입력 범위 화소를 조회
                for v in range(mask.shape[1]):
                    y, x = i - ycenter + u, i - xcenter + v 
                    # 마스크 범위의 입력 화소의 좌표(y,x)를 계산

                    sum += image[y, x] * mask[u, v]        
                    #회선 수식 #마스크 원소와 마스크 범위 입력 화소값을 곱하여 sum변수에 누적함
            dst[i, j] = sum
    return dst

image = cv2.imread("images/filter_blur.jpg", cv2.IMREAD_GRAYSCALE) # 영상 읽기
if image is None: raise Exception("영상파일 읽기 오류")

# 블러링 마스크 원소 지정 # 25개의 원소를 갖는 1차원 리스트 생성
data = [1/25, 1/25, 1/25,1/25, 1/25,
    1/25, 1/25, 1/25,1/25, 1/25,
    1/25, 1/25, 1/25,1/25, 1/25,
    1/25, 1/25, 1/25,1/25, 1/25,
    1/25, 1/25, 1/25,1/25, 1/25 ]

mask = np.array(data, np.float32).reshape(5, 5) 
# data 리스트로 ndarray 객체를 생성하고, 2차원(5x5)로 변경하여 마스크로 구성
blur1 = filter(image, mask)   # 회선 수행 - 행렬 처리 방식
blur2 = filter2(image, mask)  # 회선 수행 - 화소 직접 접근

cv2.imshow("image", image)
cv2.imshow("blur1", blur1.astype("uint8")) 
# 회선 결과 행렬이 실수형이기 때문에 윈도우에 영상으로 표시하려면 정수형으로 변환해야 함

cv2.imshow("blur2", cv2.convertScaleAbs(blur1)) 
# OpenCV의 cv2.convertScaleAbs() 함수 사용하는 방법 예시

cv2.waitKey(0)
