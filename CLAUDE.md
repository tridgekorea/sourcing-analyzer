# CLAUDE.md

## 프로젝트 개요

- 트릿지 TDS raw 수입 데이터를 분석하는 Streamlit 앱. **`analysis_tool.py` 단일 파일 구조**다.
- 고객사 데이터 분석가가 직접 로그인해서 쓰는 도구다.
- `main` 브랜치에 푸시하면 Streamlit Community Cloud가 자동 재배포한다. → `main` 푸시 = 운영 배포.

## analysis_tool.py 구조

위에서 아래 순서로:

1. `TEXTS` 딕셔너리(ko/en) + `T(key, **kwargs)` — 다국어 문구
2. 공용 분석 함수 — `weighted_avg`, `weighted_avg_groupby`, `remove_outliers_iqr` 등
3. 페이지별 `reset_*_states()` — `reset_analysis_states`, `reset_market_analysis_states`, `reset_flow_states`, `reset_risk_states`, `reset_season_states`, `reset_churn_states`, `reset_pivot_states`, `reset_scorer_states`
4. 파일/컬럼 헬퍼 — `read_uploaded_table`, `load_uploaded_df`, `detect_standard_columns`, `detect_extra_dimension_columns`, `build_axis_map`
5. 스코어러(페이지 8) 로직 — `_p8_*`, `compute_scorer`, `build_scorer_report_pdf`
6. PDF/차트 출력 — `fig_to_png_bytes`, `_find_korean_font_path`, `build_pdf_report`, `GUIDE_CONTENT`, `build_user_guide_pdf`
7. 비밀번호 — `_hash_pw`, `_get_configured_password_hash`, `_save_new_password`, `check_password`
8. 세션 상태 초기화 블록 (`if '..._raw_df' not in st.session_state:` 등)
9. 사이드바 — 언어 토글, 로고 SVG, `option_menu` 네비게이션
10. 페이지 1~8 본문 (`# ===` 구분선으로 나뉨): 고객사 효율 / 시장 경쟁력 / 공급망 흐름(Sankey) / 집중도 리스크 / 가격 추세·계절성 / 신규·이탈 거래처 / 신규사업 스코어러 / 자유 피벗

## 코드 규칙

- **다국어**: 모든 화면 문구는 `TEXTS`의 `ko`와 `en` 양쪽에 동시에 추가하고 `T('key')`로 불러온다. 한쪽만 추가 금지. 하드코딩 문자열 금지.
- **평균 단가**: 반드시 `weighted_avg` / `weighted_avg_groupby`(물량가중 VWAP)를 쓴다. 단가에 단순 `mean()` 금지.
- **세션 상태**: 새 페이지를 만들면 세션 상태 초기화 블록과 `reset_*_states()` 함수를 짝으로 만든다.
- **재사용**: 파일 업로드는 `load_uploaded_df`, 컬럼 인식은 `detect_standard_columns`를 재사용한다. 새로 만들지 않는다.
- **한글 폰트**: `assets/fonts/NanumGothic-Regular.ttf` 번들 방식을 유지한다. `packages.txt`는 비워둔 상태로 유지한다 (apt 패키지 추가 금지).
- **plotly<6 고정**: `kaleido==0.2.1`(자체 Chromium 내장, apt 불필요)과 호환되는 마지막 계열이라 고정한다. plotly 6+는 kaleido>=1(시스템 Chrome 필요)을 요구해 PDF 차트 내보내기가 깨진다.
- **건드리지 않는 영역** (요청이 있을 때만 수정):
  - 비밀번호 로직 (`check_password`, `_hash_pw`, `_get_configured_password_hash`, `_save_new_password`)
  - 사이드바 브랜딩 — 네이비 `#26224F`, 인디고 `#4F46E5`, 로고 SVG

## 작업 방식

- 큰 변경 전에는 계획을 먼저 보여주고 승인받은 뒤 구현한다.
- 메뉴를 추가·삭제·이름 변경하거나 기능이 바뀌면 사용자 가이드(`build_user_guide_pdf`, 가이드 `TEXTS` ko/en)도 같은 작업에서 함께 수정한다.
- 수정 후에는 `python -m py_compile analysis_tool.py`로 문법 확인까지 한다.
- 커밋과 푸시는 사용자가 요청할 때만 한다.
- 설명은 짧고 직접적으로. 끝나면 어떤 파일이 바뀌었는지만 명확히 알려준다.
