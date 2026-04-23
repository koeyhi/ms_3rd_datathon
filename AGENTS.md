# AGENTS.md

이 문서는 본 저장소에서 작업하는 AI 에이전트(및 기여자)를 위한 안내서입니다. 프로젝트의 구조, 개발 환경, 실행 방법, 코드 작성 규칙 등을 정리합니다.

## 프로젝트 개요

- **주제**: e스포츠(LoL) 팀/선수 지표 기반 경기 분석 및 결과 예측
- **데이터 출처**: [Oracle's Elixir](http://oracleselixir.com/) LoL 경기 데이터 (2022 ~ 2024, 약 3년치)
- **결과물**: 경기 결과 예측 모델 + Streamlit 기반 대시보드
- 상세한 프로젝트 목적/대상/기대효과는 `README.md` 참고.

## 저장소 구조

```
.
├── README.md
├── requirements.txt
├── .devcontainer/                 # VSCode / Codespaces dev container 설정
├── LoLesports_data/               # 원본 및 전처리된 CSV 데이터
├── output/                        # 학습된 모델(.pkl, .cbm), 예측 결과, 메타 JSON
├── catboost_info/                 # CatBoost 학습 중 생성되는 로그 (커밋 대상 아님)
├── streamlit/                     # Streamlit 웹 앱
│   ├── main.py                    # 앱 엔트리포인트 (네비게이션)
│   ├── pg/                        # 각 페이지 (home/dashboard/inference/result/test)
│   └── fonts/                     # 한글 폰트(NanumGothic)
├── common_data_preprocess.ipynb   # 공통 데이터 전처리
├── analysis_data_preprocess*.ipynb
├── analysis_data_split.ipynb
├── predict_data_preprocess.ipynb
├── feature_engineering*.ipynb
├── modeling*.ipynb
└── ensemble.ipynb                 # 스태킹/보팅 등 앙상블
```

### 파일 명명 규칙

- 노트북 파일명에 담당자 이름이 괄호로 포함된 경우가 있음: `feature_engineering(현지).ipynb`.
  경로 처리 시 괄호와 한글 때문에 인용(`"..."`)이 필요함.
- 모델/결과 파일명에는 날짜(MMDD) 또는 담당자 이니셜이 포함됨: `cat_0107.cbm`, `jh_pred_test.csv`.

## 개발 환경

### Python 버전
- **Python 3.11** (dev container 기준: `mcr.microsoft.com/devcontainers/python:1-3.11-bullseye`)

### 필수 패키지 (`requirements.txt`)
- `joblib`, `catboost`, `lightgbm`, `scikit-learn==1.5.2`, `xgboost`, `seaborn`
- Streamlit 앱을 돌리려면 `streamlit`도 추가 설치 필요
  (dev container의 `updateContentCommand`에서 자동 설치됨)

### 설치

```bash
pip install -r requirements.txt
pip install streamlit  # dev container 밖에서 앱 돌릴 때
```

노트북 실행이 필요한 경우 `jupyter`도 별도로 설치.

## 주요 실행 방법

### Streamlit 앱

저장소 루트에서 실행(상대경로로 폰트/데이터를 읽으므로 **반드시 루트에서 실행**).

```bash
streamlit run streamlit/main.py
```

- 기본 포트: `8501`
- 페이지 구성은 `streamlit/main.py`의 `pg_list` 참고.

### 노트북

- 루트 기준 상대경로(`LoLesports_data/...`, `output/...`)를 사용하므로 Jupyter를 **저장소 루트**에서 기동할 것.
- 데이터 전처리 → 피처 엔지니어링 → 모델링 → 앙상블 순서로 실행하는 것이 기본 흐름.

## 데이터 및 모델 규약

- 원본/전처리 CSV는 `LoLesports_data/`에만 둔다.
- 학습된 모델, 예측 결과, 참조용 JSON은 `output/`에 저장한다.
- 대용량 산출물(모델, 대형 CSV)은 가능하면 LFS 또는 별도 스토리지를 고려하고, 무분별하게 커밋하지 않는다.
- `catboost_info/`는 학습 시 자동 생성되는 로그 폴더이므로 변경사항을 커밋하지 않는다.

## 코드 규칙

- 한국어 주석/문서는 허용하며 기존 스타일과 일관성을 유지한다.
- Python 코드는 PEP8을 기본으로 하되, 노트북에서는 실험성 코드를 허용한다.
- 전처리 로직에 변경이 생기면 가능한 한 해당 노트북 상단 마크다운 셀에 간단한 변경 이유를 남긴다.
- 새로운 피처나 모델을 추가할 때는 `feature_engineering*.ipynb` / `modeling*.ipynb`의 네이밍 컨벤션(담당자/날짜 suffix)을 따른다.
- 한글 시각화가 필요한 경우 `streamlit/fonts/NanumGothic-Regular.ttf`를 사용한다(`streamlit/main.py` 참고).

## 테스트 / 검증

- 별도의 자동화 테스트 스위트는 없다.
- 변경사항 검증은 다음 중 적절한 것으로 수행한다:
  1. 영향을 받는 노트북 셀을 재실행해 에러가 없고 지표가 합리적인지 확인.
  2. 예측 파이프라인(`ensemble.ipynb` 등)의 최종 정확도/로그로스 등 메트릭을 이전 결과와 비교.
  3. Streamlit 앱을 로컬에서 기동해 변경된 페이지의 UI/그래프가 정상 렌더링되는지 수동 확인.
- 에이전트가 UI에 영향을 주는 변경을 할 경우, Streamlit 앱을 실제로 실행하고 스크린샷 또는 영상을 첨부한다.

## Git 워크플로우

- 기본 브랜치: `main`
- 기능 브랜치는 `cursor/<설명>` 형식을 권장(에이전트 작업 시 필수 prefix).
- 커밋 메시지는 한글 또는 영문 모두 허용하나, 무엇을/왜 바꿨는지 명확히 기술한다.
- PR을 열기 전에 노트북의 불필요한 대용량 출력은 제거하거나 축소하는 것을 권장.

## 팀원

- 권지혁 (조장)
- 강나현 (회의 주도)
- 위건 (GitHub 업데이트 독려)
- 이현지 (서기 / 기획서 및 노션 정리)
