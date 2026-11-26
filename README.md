# multimodal-dialect-emotion

방언 화자에 대한 음성 감정 인식 편향과, 텍스트(STT)를 결합한 멀티모달 보정 효과를 검증하는 개인 프로젝트.

## 개요

- **질문**: 일반 화자 데이터로 학습한 음성 감정 인식 모델이 경상도 화자의 *감정 없는* 발화를 분노·부정으로 오인식하는가? 텍스트(STT)를 결합하면 이 오인식이 줄어드는가?
- **학습**: AI Hub「감정 분류를 위한 대화 음성 데이터셋」5차년도_2차 (7감정: happiness, angry, disgust, fear, neutral, sadness, surprise / 48kHz)
- **평가**: AI Hub「중·노년층 한국어 방언 데이터」2인발화
  - 경상도: 강원·경상 세트
  - 충청도(비교군): 충청·전라·제주 세트
  - 감정 태그(긍정/부정/양립/무관) 중 **무관** 발화를 "감정 없는 발화"로 보고 오인식률을 비교
- **환경**: 학습·실험은 Google Colab(T4), 서빙은 로컬(FastAPI + Docker + k8s)

## 폴더 구조

```
notebooks/        Colab 노트북
  00_download.ipynb   AI Hub → Google Drive 다운로드·압축 해제 (CPU 런타임)
  01_explore.ipynb    감정 csv / 방언 json 탐색, 그래프 저장
src/              노트북 공용 파이썬 모듈 (data.py: csv·json 로딩)
serving/          로컬 서빙 (FastAPI app.py, Dockerfile, k8s.yaml)
reports/figures/  탐색·실험 그래프
```

## Colab 실행 순서

1. 이 저장소를 Google Drive에 clone
   ```
   /content/drive/MyDrive/mmdl/multimodal-dialect-emotion   ← 저장소
   /content/drive/MyDrive/mmdl/data                         ← 데이터 (저장소 밖)
   ```
2. AI Hub에서 세 데이터셋 이용 신청 후 승인, 마이페이지에서 API 키 발급
3. `notebooks/00_download.ipynb` — **CPU 런타임**. 설정 셀에 datasetkey/filekey 입력 → API 키는 실행 시 `getpass`로 입력
4. `notebooks/01_explore.ipynb` — CPU 런타임. 첫 실행 후 출력된 컬럼 목록을 보고 설정 셀의 `EMO_LABEL_COL`, `EMO_TEXT_COL`, json glob을 맞춤

경로는 각 노트북 맨 위 **설정 셀**에서 한 번에 바꿀 수 있습니다.

## 로컬 서빙

```bash
pip install -r requirements.txt
uvicorn app:app --app-dir serving --reload        # http://localhost:8000/health

docker build -t mmde-serving:dev serving/
kubectl apply -f serving/k8s.yaml                 # http://localhost:30800/health
```

아직 `/health`만 있습니다. 예측 엔드포인트는 모델 학습 후 추가합니다.

## 데이터 출처 및 주의

- 데이터 출처: [AI Hub](https://www.aihub.or.kr)
  - 감정 분류를 위한 대화 음성 데이터셋
  - 중·노년층 한국어 방언 데이터 (강원도·경상도 / 충청도·전라도·제주도)
- **데이터는 이 저장소에 포함하지 않습니다.** AI Hub 이용약관상 재배포가 금지되어 있으며, 각자 AI Hub에서 신청·다운로드해야 합니다.
- `.gitignore`로 `data/`, `*.wav`, `*.zip` 등을 제외하지만, **노트북 출력에도 전사 텍스트 원문이 남습니다.** 커밋 전에 Colab에서 *수정 → 모든 출력 지우기* 후 저장하세요.
