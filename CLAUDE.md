# CLAUDE.md

## 프로젝트 개요

- 트릿지 TDS raw 수입 데이터를 분석하는 Streamlit 앱. 앱 본체는 **`analysis_tool.py` 단일 파일**이고, 새 PDF 양식만 `report_theme.py`로 분리되어 있다.
- 고객사 데이터 분석가가 직접 로그인해서 쓰는 도구다.
- `main` 브랜치에 푸시하면 Streamlit Community Cloud가 자동 재배포한다. → `main` 푸시 = 운영 배포.

## 파일 구성

- `analysis_tool.py` — 앱 전체 (문구, 계산, 화면, 기존 PDF, 사용법 가이드)
- `report_theme.py` — 새 PDF 리포트 공통 테마. 폰트·색 토큰·로고·페이지 템플릿(표지/본문/클로징)·KPI 카드·브랜드 표·배지·증빙 박스·클로징 블록, PDF 차트(matplotlib) 스타일. 진입점 `make_report(meta, sections)`. 문구는 넣지 않고 `analysis_tool.py`의 `TEXTS`에서 넘겨받는다
- `build_report.py` — PDF 디자인 목업(가상 데이터). 새 PDF 디자인의 기준이며 앱에서는 쓰지 않는다
- `assets/fonts/` — `Pretendard-*.ttf` 5종(새 PDF·PDF 차트), `NanumGothic-Regular.ttf`(기존 PDF·kaleido 차트·사용법 가이드)
- `assets/logo/` — `tridge_logo_black.png`(밝은 배경), `tridge_logo_white.png`(어두운 배경)
- `assets/guide/` — 사용법 가이드 PDF에 들어가는 메뉴 스크린샷
- `tests/` — 고정 테스트. `test_normalize_product_name.py`(①②⑥ 품목명 정제·품목 묶기·포장 단위 구성 변화). 실행: `python tests/test_normalize_product_name.py`

## 메뉴 (11개, 사이드바 순서)

① 도입 효과 분석 · ② 포트폴리오 헬스체크 · ③ 시장 경쟁력 분석 · ④ 공급망 흐름도 · ⑤ 집중도 리스크 진단 · ⑥ 가격 추세 & 계절성 · ⑦ 신규·이탈 거래처 추적 · ⑧ 신규사업 스코어러 · ⑨ 범용상품 스크리너 · ⑩ 입찰 대응 분석 · ⑪ 자유 피벗 빌더

- 메뉴 목록은 사이드바 `option_menu`의 `_menu_keys`(TEXTS 키 `menu_opt_*`)로 관리한다. 메뉴 key는 `main_menu`로 고정되어 있고, 다른 메뉴로 코드에서 이동할 때는 `st.session_state['_menu_jump'] = 인덱스`를 쓴다(⑨ → ⑩ 넘기기).
- 새 PDF 양식(`report_theme`) 적용 메뉴: ①(기존 양식은 보조 버튼으로 유지), ⑨, ⑩. 나머지는 기존 `build_pdf_report`(NanumGothic) 양식이며 순서대로 전환 중이다. 사용법 가이드 PDF는 D 단계 마지막에 전환한다.

## analysis_tool.py 구조

위에서 아래 순서로:

1. `TEXTS` 딕셔너리(ko/en) + `T(key, **kwargs)` — 화면·PDF 문구 전부 (접두사: `p1_`~`p8_` 기존 메뉴, `ph_` 헬스체크, `sc_` 스크리너, `bd_` 입찰 대응, `rp_`·`scr_rp_`·`bd_rp_` 새 PDF 양식)
2. 공용 분석 함수 — `weighted_avg`, `weighted_avg_groupby`, `remove_outliers_iqr` 등
3. ① 헬퍼 — 품목명 정제 `normalize_product_name`(정제 규칙은 이 함수에서만 관리, 포장 단위·끝 제품 코드만 떼고 같은 이름끼리 묶음), ①②⑥ 공통 품목 묶기 `product_group_labels`(일반 명칭 품목은 HS 10단위로 한 번 더 나눔), 포장 단위 구성 변화 `pack_mix_change`/`pack_change_text`, 절감 그룹 `split_savings_groups`/`render_savings_cards` + 페이지별 `reset_*_states()` (`reset_analysis_states` … `reset_health_states`, `reset_screener_states`, `reset_bid_states`)
4. 파일/컬럼 헬퍼 — `read_uploaded_table`(HS 등 코드 컬럼은 문자열로 읽고 앞자리 0 복원), `detect_standard_columns`, `detect_extra_dimension_columns`, `build_axis_map`
5. ② 헬스체크 계산 — `prepare_health_df`, `compute_health_check`, `health_display_table` 등
6. ⑨⑩ 공통 — 블라인드(`blind_importer_map`, `apply_blind`), 거래 전처리(`detect_trade_columns`, `prepare_trade_df`), ⑨ `compute_screener`, ⑩ `bid_filter`·`bid_supplier_table`·`bid_stability`
7. 새 PDF 리포트 — `_rp_meta`, `build_screener_report`, `build_bid_report` (+ matplotlib 차트 함수)
8. 스코어러(⑧) 로직 — `detect_scorer_columns`, `_p8_*`, `compute_scorer`
9. 기존 PDF/차트 출력 — `fig_to_png_bytes`, `_find_korean_font_path`, `build_pdf_report`, `build_scorer_report_pdf`
10. 사용법 가이드 — `GUIDE_CONTENT`(ko/en 가이드 문구, `TEXTS`가 아님) + `build_user_guide_pdf`
11. 비밀번호 — `_hash_pw`, `_get_configured_password_hash`, `_save_new_password`, `check_password`
12. 세션 상태 초기화 블록 (`if '..._raw_df' not in st.session_state:` 등)
13. 사이드바 — 언어 토글, 로고 SVG, `option_menu` 네비게이션, 가이드 PDF 다운로드
14. 페이지 본문 (`if selected == T('menu_opt_...')`, `# ===` 구분선으로 나뉨). 파일 안 순서는 메뉴 순서와 다르다: ① → ② → ③ → ④ → ⑤ → ⑥ → ⑦ → ⑪ 피벗 → ⑧ → ⑨ → ⑩

## 코드 규칙

- **다국어**: 모든 화면 문구는 `TEXTS`의 `ko`와 `en` 양쪽에 동시에 추가하고 `T('key')`로 불러온다. 한쪽만 추가 금지. 하드코딩 문자열 금지.
- **평균 단가**: 반드시 `weighted_avg` / `weighted_avg_groupby`(물량가중 VWAP)를 쓴다. 단가에 단순 `mean()` 금지.
- **세션 상태**: 새 페이지를 만들면 세션 상태 초기화 블록과 `reset_*_states()` 함수를 짝으로 만든다.
- **재사용**: 파일 업로드는 `load_uploaded_df`, 컬럼 인식은 `detect_standard_columns`를 재사용한다. 새로 만들지 않는다.
- **한글 폰트**: 폰트는 `assets/fonts`에 번들한다. 새 PDF는 Pretendard(ReportLab에서 한글이 깨지는 Inter 금지), 기존 PDF는 NanumGothic. `packages.txt`는 비워둔 상태로 유지한다 (apt 패키지 추가 금지).
- **새 PDF 양식**: `report_theme`의 블록과 색 토큰만 쓴다(목업 `build_report.py` 기준). PDF 차트는 matplotlib(`report_theme.new_chart`/`apply_pdf_chart_style`/`chart_image`), 화면 차트는 plotly.
- **블라인드**: 한국 수입사명을 화면·PDF·CSV에 내보낼 때는 반드시 `apply_blind`를 거친다. 해외 공급사명은 그대로 둔다.
- **plotly<6 고정**: `kaleido==0.2.1`(자체 Chromium 내장, apt 불필요)과 호환되는 마지막 계열이라 고정한다. plotly 6+는 kaleido>=1(시스템 Chrome 필요)을 요구해 PDF 차트 내보내기가 깨진다.
- **건드리지 않는 영역** (요청이 있을 때만 수정):
  - 비밀번호 로직 (`check_password`, `_hash_pw`, `_get_configured_password_hash`, `_save_new_password`)
  - 사이드바 브랜딩 — 네이비 `#26224F`, 인디고 `#4F46E5`, 로고 SVG

## 작업 방식

- 새 기능이나 화면 변경만 계획을 먼저 보여주고 승인받은 뒤 구현한다.
- 메뉴를 추가·삭제·이름 변경하거나 기능이 바뀌면 사용자 가이드(`build_user_guide_pdf`, 가이드 문구 `GUIDE_CONTENT` ko/en)도 같은 작업에서 함께 수정한다.
- 수정 후에는 `python -m py_compile analysis_tool.py report_theme.py`로 문법 확인까지 한다. ① 품목명 정제를 건드렸으면 `python tests/test_normalize_product_name.py`도 돌린다.
- **단가 계산**: 단가는 `derive_unit_price`(Unit Price 우선, 없으면 금액 ÷ 중량)로 구한다. ①⑨⑩ 공통.
- 버그 수정처럼 테스트를 통과한 작은 변경은 확인 없이 커밋·푸시까지 진행한다. 그 외 커밋·푸시는 사용자가 요청할 때만 한다.
- 작업 중 발견한 다른 문제는 바로 고치지 말고, 작업이 끝난 뒤 목록으로만 보고한다.
- 설명은 짧고 직접적으로. 끝나면 어떤 파일이 바뀌었는지만 명확히 알려준다.
