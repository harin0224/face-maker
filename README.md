# face-maker

참고 소스 : https://github.com/kairess/edges2portrait_gan

졸업 작품으로 사용하기 위해 수정해 본 모델입니다.

# 변경점

- DataGenerator.py 파일과 main.py 파일은 주석을 달아 코드를 해석하여 사용하였습니다.
- grab_images.py
  - 경로 수정
  - 웹사이트에서 학습 데이터 이미지를 가져오는 부분을 내장메모리에서 가져오도록 변경
  - 이미지 저장 시 이름 자동 저장 코드 수정 및 추가
  - landmark 범위 표시 기능 추가
  - landmark 범위 수정
  - 일부 기능 제거
  - 결과 이미지 출력 기능 추가
