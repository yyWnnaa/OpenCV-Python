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