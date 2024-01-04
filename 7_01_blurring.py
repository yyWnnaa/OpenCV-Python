import numpy as np, cv2, time

# 회선 수행 함수 - 행렬 처리 방식(속도 면에서 유리)
def filter(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)               # 회선 결과 저장 행렬 
                           #계산 과정에서 소수 부분을 보존하기 위해 실수형으로 생성
    ycenter, xcenter = mask.shape[1]//2, mask.shape[0]//2  # 마스크 중심 좌표
    # mask.shape[1]은 열의 개수 , mask.shape[0]은 행의 개수
    # mask.shape[1]//2: 열의 중심 좌표를 계산
    # 마스크의 열 개수가 홀수이면 중앙에 정확한 중심이 존재하고, 
    # 짝수이면 중간의 두 열 중 왼쪽에 위치한 열을 중심으로 취함
    # mask.shape[0]//2: 행의 중심 좌표를 계산
    # 마스크의 행 개수가 홀수이면 중앙에 정확한 중심이 존재하고, 
    # 짝수이면 중간의 두 행 중 위쪽에 위치한 행을 중심으로 취함
    # 따라서 ycenter는 마스크의 행 중심 좌표, xcenter는 마스크의 열 중심 좌표

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

# ndarray는 자료형을 하나로 고정시키기 때문에 값에 접근하는 속도가 더 빠름

# 회선 수행 함수 - 화소 직접 접근 (수행 속도가 상당히 느림)
def filter2(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)               # 회선 결과 저장 행렬
    ycenter, xcenter = mask.shape[1]//2, mask.shape[0]//2  # 마스크 중심 좌표

    for i in range(ycenter, rows - ycenter):               # 입력 행렬 반복 순회
        for j in range(xcenter, cols - xcenter):
            # i, j: 현재 픽셀의 행, 열 좌표
            sum = 0.0
            for u in range(mask.shape[0]):  
                # 마스크 원소 순회하며 마스크 원소와 마스크 입력 범위 화소를 조회
                for v in range(mask.shape[1]):
                    # u, v: 마스크의 상대적인 행, 열 위치

                    y, x = i - ycenter + u, i - xcenter + v 
                    # 마스크 범위의 입력 화소의 좌표(y,x)를 계산
                    # 회선 연산을 수행하기 위해 
                    # 현재 픽셀 위치 (i, j)를 중심으로 하는 마스크의 상대적인 위치 (u, v)로 변환

                    # i - ycenter + u : 현재 픽셀의 행 좌표 i에서 마스크의 행 중심 좌표 ycenter를 빼면
                    # 현재 픽셀을 중심으로 하는 좌표계에서 마스크의 중심 위치가 원점(0,0)이 됨
                    # 마스크의 상대적인 행 위치(마스크의 중심을 기준으로 하는 행 위치) u를 더하면 
                    # 마스크를 적용할 입력 화소의 행 좌표 y가 계산됨

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

# data 리스트로 ndarray 객체를 생성하고, 2차원(5x5)로 변경하여 마스크로 구성
mask = np.array(data, np.float32).reshape(5, 5) 

blur1 = filter(image, mask)   # 회선 수행 - 행렬 처리 방식
blur2 = filter2(image, mask)  # 회선 수행 - 화소 직접 접근

cv2.imshow("image", image)

# 회선 결과 행렬이 실수형이기 때문에 윈도우에 영상으로 표시하려면 정수형으로 변환해야 함
cv2.imshow("blur1", blur1.astype("uint8")) 

# OpenCV의 cv2.convertScaleAbs() 함수 사용하는 방법 예시
cv2.imshow("blur2", cv2.convertScaleAbs(blur1)) 

cv2.waitKey(0)
