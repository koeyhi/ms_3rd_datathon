## Cursor Cloud specific instructions

- 이 저장소의 핵심 서비스는 Streamlit 앱(`streamlit/main.py`)이며, 상대 경로(`LoLesports_data/`, `output/`, `streamlit/fonts/`)를 사용하므로 항상 저장소 루트(`/workspace`)에서 실행한다.
- CSV/PKL 모델 아티팩트는 Git LFS로 추적된다(`.gitattributes` 참고). 브랜치 전환/새 세션 시작 후 앱 실행 전 `git lfs pull`과 `git lfs checkout`이 필요하다.
- 앱 실행 기본 명령은 `.devcontainer/devcontainer.json`의 `postAttachCommand`를 기준으로 한다(`streamlit run streamlit/main.py ...`).
- 헬로월드 검증은 `경기 전 사전예측` 페이지에서 기본값으로 `예측` 버튼을 눌러 승리 확률 2줄이 출력되는지 확인하는 플로우를 사용한다.
- `LCK 팀 지표 확인` 페이지는 GitHub Raw CSV를 직접 읽으므로 네트워크 제한이 있으면 해당 페이지 데이터 렌더링이 실패할 수 있다.
- 이 저장소에는 전용 lint/test 설정 파일이 없다. 빠른 검증은 `python -m py_compile streamlit/main.py streamlit/pg/*.py`(구문 점검)와 `python -m unittest discover -v`(테스트 스위트 존재 여부 점검)로 수행한다.
