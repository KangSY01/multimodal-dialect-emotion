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
  00_download.ipynb   Drive의 원본 zip → Colab 로컬 압축 해제 (CPU 런타임)
  01_explore.ipynb    감정 csv / 방언 json 탐색, 그래프 저장
  02_preprocess.ipynb 방언 문장 표·평가셋·감정 데이터 분할 (CPU 런타임)
src/              노트북 공용 파이썬 모듈 (data.py: csv·json 로딩, 전처리 함수)
serving/          로컬 서빙 (FastAPI app.py, Dockerfile, k8s.yaml)
reports/figures/  탐색·실험 그래프
```

## Colab 실행 순서

1. 이 저장소를 Google Drive에 clone
   ```
   /content/drive/MyDrive/mmdl/multimodal-dialect-emotion   ← 저장소
   /content/drive/MyDrive/mmdl/data/raw                     ← 원본 zip (저장소 밖)
   /content/drive/MyDrive/mmdl/processed                    ← 전처리 결과 (저장소 밖)
   ```
2. AI Hub 웹사이트에서 세 데이터셋 이용 신청·승인 후 파일 다운로드
3. 압축을 풀지 말고 zip 그대로 Drive raw 폴더에 업로드
   ```
   raw/dialect_gs/  gs_wav.zip, gs_label.zip
   raw/dialect_cc/  cc_wav.zip, cc_label.zip
   raw/emotion/     emotion.zip, emotion.csv
   ```
4. `notebooks/00_download.ipynb` — **CPU 런타임**. Colab 로컬(`/content/data`)에 압축 해제. 로컬은 세션이 끝나면 지워지므로 세션마다 다시 실행
5. `notebooks/02_preprocess.ipynb` — **CPU 런타임**. `dialect_all.csv`, `eval_set.csv`, `emotion_meta.csv`, 평가셋 문장 wav(`segments/`)를 processed 폴더에 저장. 첫 실행 후 `emotion_tag` 출력을 보고 설정 셀의 `NEUTRAL_TAG`를 맞춤

(선택 사항) 웹 대신 AI Hub API(aihubshell)로 받으려면 00 노트북 맨 아래 섹션을 사용하세요.

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
