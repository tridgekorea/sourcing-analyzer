import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import datetime
import os   # <--- [비밀번호 기능] 환경변수(Codespaces secret) 읽기용
import hmac # <--- [비밀번호 기능] 타이밍 공격에 안전한 문자열 비교용
import hashlib # <--- [비밀번호 기능] 최초 실행 시 설정한 비밀번호를 해시로 저장하기 위함
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from streamlit_option_menu import option_menu
import statsmodels.api as sm
import codecs # <--- [추가 1] 인코딩 처리를 위해 추가

# --------------------------------#
# 다국어 지원 (i18n) / Multilingual support #
# --------------------------------#
TEXTS = {
    'ko': {
        'app_menu_title': '메뉴',
        'menu_opt_customer': '고객사 효율 분석',
        'menu_opt_market': '시장 경쟁력 분석',
        'menu_opt_flow': '공급망 흐름도',
        'menu_opt_risk': '집중도 리스크 진단',
        'menu_opt_season': '가격 추세 & 계절성',
        'menu_opt_churn': '신규·이탈 거래처 추적',
        'menu_opt_pivot': '자유 피벗 빌더',
        'summary_table_no_data': '요약할 데이터가 없습니다.',
        'summary_table_header_price': '수입 단가(USD/KG)',
        'summary_table_max': '최대',
        'summary_table_mean': '평균',
        'summary_table_min': '최소',
        'outlier_warning_capped': '이상치가 전체의 {cap_percent:.0%}를 초과하여, 가장 극단적인 {removed_rows}건(상한 적용)만 제거했습니다.',
        'outlier_warning_removed': '분석의 정확도를 위해 시장 데이터의 단가(Unit Price) 이상치 {n}건을 제거했습니다.',

        'p1_title': '💲 고객사 효율 분석 (Overview)',
        'p1_reset_btn': '새로운 분석 시작 (다시하기)',
        'p1_settings_header': '⚙️ 분석 설정',
        'p1_upload_label': '고객사 데이터 파일을 업로드하세요',
        'p1_upload_caption': '※ 하나의 회사 정보를 가지고 있는 TDS raw file을 업로드해주세요.',
        'p1_missing_importer_col_warning': "업로드된 파일에 'Raw Importer Name' 컬럼이 없습니다. 아래에 직접 입력해주세요.",
        'p1_customer_name_input': '분석할 수입 업체 이름을 입력해주세요.',
        'p1_file_read_fail_error': '파일을 읽는 데 실패했습니다. 지원되는 인코딩(utf-8, euc-kr, cp949)이 아니거나 파일이 손상되었을 수 있습니다.',
        'p1_file_read_error_generic': '파일을 읽는 중 오류가 발생했습니다: {e}. 파일 형식이나 컬럼명을 확인해주세요.',
        'p1_contract_date_input': '계약 시작일 (Contract Date)을 선택하세요',
        'p1_run_btn': '분석 실행',
        'p1_spinner': '고객사 데이터를 분석 중입니다...',
        'p1_missing_required_cols_error': '필수 컬럼이 부족합니다. 파일 내용을 확인해주세요.',
        'p1_analysis_complete_success': "'{customer_name}' 고객사 분석 완료!",
        'p1_exp1_title': '1. 계약 전후 예상 절감액 분석',
        'p1_total_savings_subheader': '총 예상 절감액',
        'p1_total_savings_caption': '※ 계약일({date}) 이후, 고객사의 자체 구매 단가 변화에 따른 총 예상 절감액입니다.',
        'p1_savings_detail_subheader': '품목군별 상세 절감 내역',
        'p1_exp2_title': '2. 수입 품목군 정제 및 군집화 (DBSCAN & PCA)',
        'p1_too_many_clusters_info': '클러스터가 너무 많아, 수입량 기준 상위 {n}개 품목군만 그리드에 시각화합니다.',
        'p1_scatter_title': '[{customer}] 품목 유사도 기반 군집화 (상위 품목군 Grid)',
        'p1_scatter_subtitle': '수입 중량 기준 상위 {n}개 품목군',
        'p1_cluster_list_subheader': '클러스터 리스트 (수입 중량순)',
        'p1_exp3_title': '3. 주요 수입 품목군 분석 (월별 수입량)',
        'p1_monthly_chart_title': '[{customer}] 주요 수입 품목군 월별 수입량(KG)',
        'p1_recent3m_subheader': '최근 3개월 수입 품목 비중',
        'p1_recent3m_pie_title': '[{customer}] 최근 3개월 수입 비중',
        'p1_recent3m_pie_subtitle': '{start} ~ {end} 기준',
        'p1_no_recent_data_info': '최근 3개월간의 수입 데이터가 없습니다.',
        'p1_no_data_warning': '분석할 데이터가 없습니다.',
        'p1_exp4_title': '4. 계약 이후 소싱 변화 분석',
        'p1_new_clusters_subheader': '신규 거래 품목군',
        'p1_new_clusters_text': '계약 이후 새로 수입하기 시작한 품목군은 총 **{n}**개 입니다.',
        'p1_no_new_clusters_info': '계약 이후 새로 추가된 품목군은 없습니다.',
        'p1_new_origins_subheader': '신규 거래 원산지',
        'p1_new_origins_text': '계약 이후 새로 거래를 시작한 원산지는 총 **{n}**곳 입니다.',
        'p1_no_new_origins_info': '계약 이후 새로 추가된 원산지는 없습니다.',
        'p1_new_exporters_subheader': '신규 거래 공급사',
        'p1_new_exporters_text': '계약 이후 새로 거래를 시작한 공급사는 총 **{n}**곳 입니다.',
        'p1_no_new_exporters_info': '계약 이후 새로 추가된 공급사는 없습니다.',

        'col_new_cluster': '신규 품목군',
        'col_rep_product': '대표 품목명',
        'col_new_origin': '신규 원산지',
        'col_related_products': '관련 품목명',
        'col_new_exporter': '신규 공급사',

        'axis_yearmonth': '연-월',
        'axis_volume_kg': '수입량(KG)',
        'legend_cluster': '품목 클러스터',
        'axis_pca1': 'PCA Component 1',
        'axis_pca2': 'PCA Component 2',

        'p2_title': '🏆 시장 경쟁력 상세 분석 (Drill-down)',
        'p2_reset_btn': '새로운 시장 분석 시작 (다시하기)',
        'p2_intro_text': '특정 품목에 대한 전체 시장 데이터를 업로드하여, 고객사의 시장 내 경쟁력을 심층 분석합니다.',
        'p2_upload_label': '분석할 품목의 전체 시장 데이터를 업로드하세요.',
        'p2_upload_caption': '※ 하나의 품목에 대한 여러 회사의 정보가 포함된 TDS raw file을 업로드해주세요.',
        'p2_select_customer_label': '분석할 고객사를 선택해주세요.',
        'p2_missing_importer_col_warning': "업로드된 파일에 'Raw Importer Name' 컬럼이 없습니다. 아래에 직접 입력해주세요.",
        'p2_customer_name_input': '분석할 수입 업체 이름을 입력해주세요.',
        'p2_file_read_error': '파일을 읽는 중 오류가 발생했습니다: {e}. 컬럼명을 확인해주세요.',
        'p2_product_name_input': '분석할 품목명을 입력하세요 (예: 건면)',
        'p2_contract_date_input': '분석 기준이 될 계약 시작일을 선택하세요.',
        'p2_iqr_slider_label': '이상치 제거 민감도 (IQR 배수)',
        'p2_iqr_slider_help': "값이 클수록 '정상' 데이터 범위를 넓게 봅니다. (예: 3.0은 더 적은 이상치를 제거)",
        'p2_run_btn': '시장 경쟁력 분석 시작',
        'p2_processing_spinner': '시장 데이터를 분석 중입니다. 파일 크기에 따라 시간이 걸릴 수 있습니다...',
        'p2_missing_cols_error': "필수 컬럼이 누락되었습니다: {cols}. 'Export Country' 등의 컬럼이 파일에 포함되어 있는지 확인해주세요.",
        'p2_result_subheader': "'{product}' 품목 시장 분석 결과 (기준 고객사: {customer})",
        'p2_country_filter_label': '원산지(수출국) 필터:',
        'p2_country_filter_all': '전체',
        'p2_country_status_line': '원산지: {countries}',
        'p2_no_data_for_filter_warning': "'{product}'에 대해 선택하신 원산지에 해당하는 데이터가 없습니다.",
        'p2_exp1_title': '1. [{product}] 구매 경쟁력 분석',
        'p2_scatter_subheader': 'Volume 대비 Unit Price 분포 및 시장 추세',
        'p2_scatter_title': '시장 내 거래 분포 및 평균 가격 추세선',
        'p2_scatter_subtitle': 'LOWESS 회귀분석 기반',
        'axis_unit_price': '단가(USD/KG)',
        'p2_top10_subheader': '구매 경쟁력 상위 10개사',
        'p2_rank_info': '참고: **{customer}**의 구매 경쟁력 순위는 (필터링된 결과) {total}개사 중 **{rank}위**입니다.',
        'p2_rank_not_in_filter_warning': '**{customer}**의 데이터가 현재 필터에 포함되지 않습니다.',
        'p2_exp2_title': '2. [{product}] 단가 추세 및 경쟁 우위 그룹 벤치마킹',
        'p2_monthly_comp_subheader': '구매 경쟁력 지수 월별 추이',
        'legend_market_avg_index': '시장 전체 평균 지수',
        'legend_customer_index': '{customer} 경쟁력 지수',
        'legend_top_group_avg_index': '경쟁 우위 그룹 평균 지수',
        'p2_comp_trend_chart_title': '[{product}] 구매 경쟁력 지수 월별 추이',
        'axis_comp_index': '구매 경쟁력 지수',
        'p2_comp_trend_caption': '※ 이 그래프는 시장의 기대 단가 대비 실제 구매 단가의 차이(경쟁력 지수)가 시간에 따라 어떻게 변하는지를 보여줍니다.',
        'p2_price_trend_subheader': '월별 평균 단가 추세',
        'legend_market_avg_price': '시장 전체 평균 단가',
        'legend_customer_avg_price': '{customer} 평균 단가',
        'legend_top_group_avg_price': '경쟁 우위 그룹 평균',
        'p2_benchmark_info': '**벤치마크: 경쟁 우위 그룹 평균**',
        'p2_benchmark_caption': "※ '경쟁 우위 그룹'은 '구매 경쟁력 분석'의 순위에서 현재 선택된 고객사보다 높은 순위를 기록한 모든 기업들의 평균입니다.",
        'p2_benchmark_success': '**벤치마크 분석:** `{customer}`님이 (현재 필터에서) 가장 우수한 구매 경쟁력을 보이고 있습니다!',
        'p2_price_trend_chart_title': '[{product}] 단가 추세',
        'p2_price_compare_subheader': '전체 기간 평균 단가 비교',
        'metric_market_avg': '시장 전체 평균',
        'metric_customer_avg': '{customer} 평균',
        'metric_top_group_avg': '경쟁 우위 그룹 평균',
        'p2_sim_exp_title': '경쟁 우위 그룹 벤치마킹 시뮬레이션',
        'p2_sim_start_date': '시뮬레이션 시작일',
        'p2_sim_end_date': '시뮬레이션 종료일',
        'p2_sim_run_btn': '예상 절감액 계산',
        'p2_sim_success': '해당 기간 동안 **경쟁 우위 그룹**의 평균 단가를 따랐다면 **${amount}**를 추가로 절감할 수 있었습니다.',
        'p2_sim_caption': '※ 이 금액은 고객사의 월평균 단가가 경쟁 우위 그룹보다 높았던 달의 절감 가능액만을 합산한 값입니다.',
        'p2_sim_no_data_warning': '해당 기간에 비교할 데이터가 없습니다.',
        'p2_exp3_title': '3. [{product}] 시장 점유율 및 경쟁사 비교',
        'p2_ms_year_select': '시장 점유율 분석 연도 선택',
        'p2_others_label': '기타',
        'p2_ms_pie_title': '[{product}] {year}년 시장 점유율',
        'p2_ms_pie_subtitle': '수입 중량 기준',
        'p2_price_year_select': '수입 상위 5개사 단가 비교 연도',
        'p2_price_bar_title': '{year}년 고객사와 수입 상위 5개사 단가 비교',
        'p2_price_bar_subtitle': '수입 중량 기준 상위 5개사',
        'axis_importer': '수입사',
        'axis_avg_unit_price': '평균 단가(USD/KG)',
        'p2_exp4_title': '4. [{product}] 공급망(공급사/원산지) 분석',
        'p2_exporter_year_select': '공급망 분석 연도 선택',
        'p2_exporter_quarterly_subheader': '{year}년 분기별 공급사 단가 분포',
        'p2_exporter_box_title': '{year}년 분기별 공급사 단가 분포',
        'p2_exporter_box_subtitle': '수입 중량 기준 상위 10개 공급사',
        'axis_quarter': '분기',
        'p2_detail_data_expander': '상세 데이터 보기',
        'col_supplier': '공급사',
        'col_max_price': '최대 단가(USD/KG)',
        'col_avg_price': '평균 단가(USD/KG)',
        'col_min_price': '최소 단가(USD/KG)',
        'p2_customer_exporters_info': '**{customer}**가 {year}년에 거래한 공급사: **{exporters}**',
        'p2_exporter_detail_expander': "공급사 '{exporter}' 상세 비교 분석",
        'p2_volume_price_compare_subheader': 'Volume 및 평균 단가 비교',
        'legend_total_volume': '총 수입량(KG)',
        'legend_avg_import_price': '평균 수입단가(USD/KG)',
        'p2_exporter_bar_title': "'{exporter}' 거래 업체별 Volume 및 평균 단가",
        'axis_total_volume': '총 수입량(KG)',
        'axis_avg_import_price': '평균 수입단가(USD/KG)',
        'p2_price_dist_subheader': '단가 분포 비교',
        'p2_exporter_box2_title': "'{exporter}' 거래 업체별 단가 분포",
        'p2_exporter_box2_subtitle': '수입 중량 기준 상위 10개 수입사',
        'col_importer': '수입사',
        'p2_alt_sourcing_subheader': '{year}년 분기별 대안 소싱 옵션',
        'p2_q_expander': '**{q}분기** 대안 소싱 옵션',
        'p2_no_quarter_data': '- 해당 분기에 거래 데이터가 없습니다.',
        'p2_current_sourcing_md': '**현재 소싱 옵션**',
        'col_avg_price2': '평균 단가(USD/KG)',
        'p2_no_exporter_deals': '- 공급사 거래 없음',
        'col_origin': '원산지(수출국)',
        'p2_no_origin_deals': '- 원산지 거래 없음',
        'p2_alt_recommend_md': '**대안 추천 옵션 (더 저렴한)**',
        'col_recommend_supplier': '추천 공급사',
        'col_rep_item': '대표 품목',
        'p2_no_cheaper_exporter': '- 더 저렴한 공급사 없음',
        'col_recommend_origin': '추천 원산지(수출국)',
        'p2_no_cheaper_origin': '- 더 저렴한 원산지(수출국) 없음',
        'p2_no_supply_chain_cols_warning': "'Exporter' 또는 'Export Country' / 'Origin Country' 컬럼이 없어 공급망 분석을 수행할 수 없습니다.",

        'password_gate_title': '🔒 접근 제한',
        'password_label': '비밀번호를 입력하세요',
        'password_wrong': '비밀번호가 올바르지 않습니다.',
        'password_not_configured': '관리자가 아직 앱 비밀번호(APP_PASSWORD)를 설정하지 않았습니다. secrets.toml 또는 환경변수를 확인해주세요.',
        'password_setup_title': '🔒 최초 실행 - 비밀번호 설정',
        'password_setup_caption': '이 앱을 처음 실행하셨네요! 앞으로 사용할 비밀번호를 설정해주세요. 이후 접속할 때부터는 이 비밀번호를 입력해야 합니다.',
        'password_setup_new': '새 비밀번호',
        'password_setup_confirm': '비밀번호 확인',
        'password_setup_button': '비밀번호 설정하고 시작하기',
        'password_setup_empty': '비밀번호를 입력해주세요.',
        'password_setup_mismatch': '두 비밀번호가 일치하지 않습니다.',
        'password_setup_success': '비밀번호가 설정되었습니다!',
        'guide_download_btn': '📖 사용법 가이드 (PDF)',
        'menu_opt_scorer': '신규사업 스코어러',
        'p8_title': '🔬 원료 수입 신규사업 스코어러',
        'p8_intro': '전체 시장 거래 데이터에서 "최근 급상승한 기존 품목"과 "완전히 새로 등장한 품목"을 찾아, 우리 회사 기존 사업과의 적합도까지 반영해 진입 우선순위를 매깁니다.',
        'p8_upload_label': '전체 시장 거래 데이터를 업로드하세요',
        'p8_upload_caption': '※ 여러 수입사·공급사·원산지 정보가 담긴 TDS raw file을 업로드해주세요. HS코드명, Detailed HS-CODE, 카테고리, 한글품목명 컬럼이 있으면 자동으로 인식됩니다.',
        'p8_settings_header': '⚙️ 분석 설정',
        'p8_months_label': '분석 기간 (최근 N개월)',
        'p8_floor_label': '최소 물량 기준 (KG)',
        'p8_dim_label': '분석 기준',
        'p8_minship_label': '최소 선적 횟수 (신규진입 판정)',
        'p8_newkey_label': '신규 판정 기준 축',
        'p8_preset_label': '점수 가중치 프리셋',
        'p8_preset_growth': '성장 중심',
        'p8_preset_size': '규모 중심',
        'p8_preset_easy': '진입 용이성 중심',
        'p8_fitw_label': '적합도 반영 비중 (%)',
        'p8_fit_context_header': '🎯 우리 회사 적합도 기준 (선택 — 입력하면 4분면 분류까지 나옵니다)',
        'p8_fit_context_caption': '입력하지 않아도 분석은 되지만, 아래를 채우면 "우리 사업과 얼마나 잘 맞는지"까지 점수화됩니다.',
        'p8_fit_suppliers_label': '기존 공급사 목록 (쉼표 또는 줄바꿈으로 구분)',
        'p8_fit_origins_label': '기존 원산지 목록',
        'p8_fit_keywords_label': '기존 취급 품목 키워드',
        'p8_fit_moat_label': '가점 키워드 (선택, 있으면 추가 점수)',
        'p8_run_btn': '분석 실행',
        'p8_reset_btn': '새로운 분석 시작 (다시하기)',
        'p8_missing_cols_error': "필수 컬럼이 부족합니다: {cols}. 파일 내용을 확인해주세요.",
        'p8_no_data_warning': '설정 조건에 해당하는 데이터가 없습니다. 기간·최소물량 기준을 조정해보세요.',
        'p8_kpi_window': '분석 윈도우',
        'p8_kpi_total_yoy': '전체 물량 YoY',
        'p8_kpi_growing': '급상승 기존',
        'p8_kpi_new': '신규 진입',
        'p8_kpi_price': '가격 데이터',
        'p8_kpi_price_linked': '연동됨 (단가 YoY 반영)',
        'p8_kpi_price_none': '미연동 (물량 기준만)',
        'p8_section_s': '🏆 통합 추천 순위 (TOP 10)',
        'p8_section_a': '📈 기존 급상승 품목',
        'p8_section_b': '🆕 신규 진입 품목',
        'p8_col_rank': '#',
        'p8_col_type': '구분',
        'p8_col_item': '품목',
        'p8_col_score': '점수',
        'p8_col_fit': '적합도',
        'p8_col_quadrant': '4분면',
        'p8_col_reason': '근거',
        'p8_col_flags': '주의',
        'p8_col_yoy': 'YoY',
        'p8_col_rec_vol': '최근 물량',
        'p8_col_ly_vol': '전년 물량',
        'p8_col_delta': '증가량',
        'p8_col_price_yoy': '단가 YoY',
        'p8_col_concentration': '집중도',
        'p8_col_vol': '최근 물량',
        'p8_col_ship': '선적횟수',
        'p8_col_top_importer': '주수입사',
        'p8_type_existing': '기존·급상승',
        'p8_type_new': '신규진입',
        'p8_flag_low_base': '⚠ 저베이스 반등',
        'p8_flag_single_ship': '⚠ 단일선적 편중',
        'p8_flag_suspect': '⚠ 표기변형 의심',
        'p8_quad_priority': '우선 타깃',
        'p8_quad_growth': '성장 주도',
        'p8_quad_safe': '안전 인접',
        'p8_quad_low': '후순위',
        'p8_signal_premium': '프리미엄',
        'p8_signal_commodity': '코모디티화',
        'p8_signal_shock': '공급쇼크',
        'p8_label_growth': '성장',
        'p8_label_cagr': 'CAGR',
        'p8_label_share': '점유율',
        'p8_label_importer_count': '수입사',
        'p8_label_origin_count': '원산지',
        'p8_label_origin_unit': '국',
        'p8_label_fit': '적합',
        'p8_label_top_importer': '주수입사',
        'p8_label_recent': '최근',
        'p8_label_shipment': '선적',
        'p8_label_shipment_unit': '회',
        'p8_fitwhy_supplier': '공급사',
        'p8_fitwhy_origin': '원산지',
        'p8_fitwhy_keyword': '취급인접',
        'p8_fitwhy_moat': '가점KW',
        'p8_fitwhy_none': '매칭없음',
        'p8_chart_title': '통합 추천 순위 TOP 10 — 점수',
        'p8_dim_hs_name': 'HS Code Name',
        'p8_dim_detailed_hs': 'Detailed HS-CODE',
        'p8_dim_product': 'Reported Product Name',
        'p8_dim_importer': 'Importer',
        'p8_dim_origin': 'Origin Country',
        'p8_report_title': '수입 신규사업 스코어링 리포트',
        'p8_report_subtitle': 'SOURCING OPPORTUNITY SCORER',
        'p8_report_meta': '분석기간 {start} ~ {end} · 전년 동기({ly_start}~{ly_end}) 대비 · 윈도우 {months} · 발행 {today}',
        'p8_report_insights_header': '◆ 핵심 인사이트',
        'p8_report_data_summary': '분석 데이터: 거래 {n_tx}건 · 기간 {span} · 수입사 {n_imp}곳 · 품목 {n_prod}종 · 원산지 {n_origin}개국',
        'p8_report_top10_header': '★ 추천 TOP 10 (진입매력도 점수 통합순위)',
        'p8_report_top10_sub': '기존 급상승 품목과 신규 진입 품목을 하나의 점수(성장률·규모·가격추세·집중도 종합)로 통합 정렬했습니다.',
        'p8_report_section_a_header': 'A. 급상승 기존 품목 상세 (상위 {n})',
        'p8_report_section_a_sub': '작년에도 수입되던 품목 중 물량이 늘어난 항목, 점수순.',
        'p8_report_section_b_header': 'B. 신규 진입 품목 상세 (상위 {n})',
        'p8_report_section_b_sub': '과거에는 없다가 최근 처음 등장한 항목(최소 선적·물량 하한 적용), 점수순.',
        'p8_report_footer': '데이터: 업로드된 거래내역. 최근 기간 합계를 전년 동기 같은 기간과 비교(YoY)했습니다. 신규는 과거 전체 기간에 없다가 최근 등장한 항목 기준입니다. 진입매력도 점수는 성장률·물량규모·(가능시)단가추세·수입사 집중도를 정규화해 합산한 자체 지표이며, 저베이스 반등·단일선적 편중·표기변형 의심 항목은 감점됩니다. 관세율·수입규제·물류비·계약조건 등은 반영하지 않으므로 1차 스크리닝 용도로만 사용하고, 최종 판단 전 사람의 검수를 거치시기 바랍니다.',
        'p8_ins_total': '최근 {month_label} 총 수입은 전년 동기 대비 {sign}{pct}% ({ly} → {rec}).',
        'p8_ins_top_candidate': '진입매력도 최고 후보: {name} ({type}, 점수 {score}).',
        'p8_ins_top_growth': '성장률 최고: {name} — {sign}{pct}%.',
        'p8_ins_top_new': '신규 진입 주도: {name} ({vol}{origin_part}).',
        'p8_ins_top_new_origin_part': ', 주력 수입국 {origin}',
        'p8_ins_price_up': '단가도 함께 오른 품목: {name} — 단가 {sign}{pct}%.',
        'p8_ins_fit': '적합도 반영 결과 우선 타깃(성장·적합 모두 상위) {n}건 도출{rep_part}.',
        'p8_ins_fit_rep_part': ' — 대표: {name} (적합도 {fit})',
        'p8_detail_a_narrative': '전년 대비 {sign}{pct}% ({ly}→{rec}), 점유율 {share}%.{tags}',
        'p8_detail_big_tag': ' 대형 물량',
        'p8_detail_lowbase_tag': ' 저베이스',
        'p8_detail_b_narrative': '{tag} 최근 {vol} · 선적 {ship}회 · 주수입사 {imp} {share}%.{suspect}',
        'p8_detail_new_tag': '신규',
        'p8_detail_suspect_tag': ' 표기변형 의심',
        'p8_generate_report_btn': '📊 리포트 PDF 생성 (상세)',
        'p8_download_report_btn': '📥 리포트 PDF 다운로드',
        'p8_cat_all': '전체 품목',
        'p8_report_excluded_note_a': '※ 저베이스(작년 물량 거의 0)로 수치가 왜곡된 {n}건은 상세 목록에서 제외했습니다.',
        'p8_report_excluded_note_b': '※ 표기변형 의심(기존 품목이 이름만 바뀐 것으로 추정되는) {n}건은 상세 목록에서 제외했습니다.',
        'p8_report_note_col': '비고',

        'p3_title': '🔀 공급망 흐름도 (Sankey)',
        'p3_upload_label': '전체 시장/공급망 데이터를 업로드하세요',
        'p3_upload_caption': '※ 여러 공급사·수입사·원산지 정보가 포함된 TDS raw file을 업로드해주세요.',
        'p3_left_axis_label': '왼쪽 축 — 구분',
        'p3_left_entity_label': '왼쪽 축 — 대상 선택',
        'p3_right_axis_label': '오른쪽 축 — 비교 기준',
        'p3_axis_exporter': '공급사',
        'p3_axis_importer': '수입사',
        'p3_axis_origin': '원산지',
        'p3_axis_country': '수출대상국',
        'p3_axis_product': '품목',
        'p3_axis_hint': '※ 원산지 = 물건이 생산된 나라, 수출대상국 = 물건이 팔려나가는 나라 (서로 다른 컬럼입니다)',
        'p3_date_start': '시작일',
        'p3_date_end': '종료일',
        'p3_run_btn': '흐름도 그리기',
        'p3_reset_btn': '새로운 흐름도 분석 시작 (다시하기)',
        'p3_missing_cols_error': "필수 컬럼이 부족합니다: {cols}. 파일 내용을 확인해주세요.",
        'p3_no_data_warning': '선택하신 조건에 해당하는 데이터가 없습니다. 대상/기간을 다시 확인해주세요.',
        'p3_result_subheader': "'{left}' ({left_axis}) 기준 {right_axis}별 흐름 ({start} ~ {end})",
        'p3_sankey_title': "'{left}' → {right_axis}별 물량 흐름",
        'p3_table_subheader': '정렬된 상세 표',
        'p3_col_category': '{right_axis}',
        'p3_col_volume': '물량(KG)',
        'p3_col_share': '비중',
        'p3_col_avg_price': '평균 단가(USD/KG)',
        'p3_others_note': '※ 상위 {n}개 외 나머지는 "기타"로 묶었습니다.',

        'p4_title': '⚠️ 집중도 리스크 진단',
        'p4_upload_label': '전체 시장/공급망 데이터를 업로드하세요',
        'p4_upload_caption': '※ 여러 공급사·수입사·원산지 정보가 포함된 TDS raw file을 업로드해주세요.',
        'p4_axis_label': '기준',
        'p4_axis_exporter': '공급사별',
        'p4_axis_origin': '원산지별',
        'p4_axis_product': '품목별',
        'p4_scope_label': '범위',
        'p4_scope_all': '전체',
        'p4_scope_importer': '특정 수입사만',
        'p4_scope_entity_label': '수입사 선택',
        'p4_date_start': '시작일',
        'p4_date_end': '종료일',
        'p4_threshold_label': '위험 기준선',
        'p4_run_btn': '진단 실행',
        'p4_reset_btn': '새로운 진단 시작 (다시하기)',
        'p4_missing_cols_error': "필수 컬럼이 부족합니다: {cols}. 파일 내용을 확인해주세요.",
        'p4_no_data_warning': '선택하신 조건에 해당하는 데이터가 없습니다.',
        'p4_kpi_top1': '1위 비중',
        'p4_kpi_top3': '상위 3개 합산',
        'p4_kpi_count': '총 거래처 수',
        'p4_kpi_risk': '위험도',
        'p4_risk_danger': '위험',
        'p4_risk_caution': '주의',
        'p4_risk_safe': '안전',
        'p4_risk_reason_top1': '1위 비중 {v}%가 기준({t}%) 초과',
        'p4_risk_reason_top3': '상위3개 합산 {v}%가 {t}% 초과',
        'p4_risk_reason_ok': '기준선 이내',
        'p4_bar_chart_title': '{axis} 비중 (점선 = 위험 기준선 {t}%)',
        'p4_trend_chart_title': '1위 비중 월별 추이',
        'p4_axis_share': '비중(%)',
        'p4_col_name': '이름',
        'p4_col_volume': '물량(KG)',
        'p4_col_share': '비중',

        'p5_title': '📈 가격 추세 & 계절성',
        'p5_upload_label': '전체 시장 데이터를 업로드하세요',
        'p5_upload_caption': '※ 하나 이상의 품목에 대한 시계열 데이터가 포함된 TDS raw file을 업로드해주세요.',
        'p5_product_label': '품목 검색',
        'p5_prevyear_label': '전년 동기 겹쳐보기',
        'p5_breakdown_label': '비교 축',
        'p5_breakdown_none': '전체 평균만',
        'p5_breakdown_origin': '원산지별로 나눠보기',
        'p5_breakdown_exporter': '공급사별로 나눠보기',
        'p5_run_btn': '추세 그리기',
        'p5_reset_btn': '새로운 가격 추세 분석 시작 (다시하기)',
        'p5_missing_cols_error': "필수 컬럼이 부족합니다: {cols}. 파일 내용을 확인해주세요.",
        'p5_no_data_warning': '선택하신 품목에 해당하는 데이터가 없습니다.',
        'p5_kpi_current': '최근월 평균단가',
        'p5_kpi_yoy': '전년 동월 대비',
        'p5_kpi_frompeak': '연중 최고 대비 현재',
        'p5_season_badge': '계절 고점: {months}',
        'p5_no_season': '뚜렷한 계절 패턴 없음',
        'p5_chart_title': "'{product}' 월별 평균 단가 추이",
        'p5_axis_month': '연-월',
        'p5_axis_price': '단가(USD/KG)',
        'p5_legend_this_year': '선택 기간',
        'p5_legend_prev_year': '전년 동기',

        'p6_title': '🔀 신규·이탈 거래처 추적',
        'p6_upload_label': '전체 시장/공급망 데이터를 업로드하세요',
        'p6_upload_caption': '※ 여러 공급사·수입사·원산지 정보가 포함된 TDS raw file을 업로드해주세요.',
        'p6_axis_label': '추적 기준',
        'p6_axis_exporter': '공급사',
        'p6_axis_origin': '원산지',
        'p6_axis_product': '품목',
        'p6_axis_importer': '수입사',
        'p6_period_a': '기간 A (비교 대상)',
        'p6_period_b': '기간 B (기준)',
        'p6_run_btn': '비교 실행',
        'p6_reset_btn': '새로운 추적 시작 (다시하기)',
        'p6_missing_cols_error': "필수 컬럼이 부족합니다: {cols}. 파일 내용을 확인해주세요.",
        'p6_no_data_warning': '두 기간 중 하나 이상에 해당하는 데이터가 없습니다.',
        'p6_kpi_new': '신규',
        'p6_kpi_kept': '유지',
        'p6_kpi_lost': '이탈',
        'p6_new_header': '🟢 신규 거래처',
        'p6_lost_header': '🔴 이탈 거래처',
        'p6_no_new': '신규 항목 없음',
        'p6_no_lost': '이탈 항목 없음',
        'p6_col_name': '이름',
        'p6_col_volume': '물량(KG)',
        'p6_price_compare_subheader': '💰 가격으로 보면',
        'p6_kpi_new_price': '신규 거래처 평균단가',
        'p6_kpi_kept_price': '유지 거래처 평균단가',
        'p6_kpi_lost_price': '이탈 거래처 평균단가',
        'p6_price_insight_cheaper': '신규 거래처가 유지 거래처보다 평균 {pct}% {direction}.',
        'p6_price_insight_none': '가격 비교에 필요한 단가 데이터가 부족합니다.',
        'p6_concentration_subheader': '⚠️ 집중도 변화',
        'p6_concentration_warning': '기간 A→B 사이 1위 거래처 의존도가 {a}% → {b}%로 상승했습니다 (현재 1위: {name}). 거래처가 줄면서 남은 곳에 더 쏠리고 있는 신호일 수 있어요.',
        'p6_concentration_stable': '1위 거래처 의존도: {a}% → {b}% (큰 변화 없음)',
        'p6_concentration_improved': '1위 거래처 의존도가 {a}% → {b}%로 오히려 낮아졌습니다 (공급처 분산 개선).',

        'p7_title': '🧩 자유 피벗 빌더',
        'p7_upload_label': '분석할 데이터를 업로드하세요',
        'p7_upload_caption': '※ TDS raw file을 업로드해주세요. (고객사/시장 데이터 모두 가능)',
        'p7_row_label': '행 (기준)',
        'p7_row_month': '월별',
        'p7_row_exporter': '공급사별',
        'p7_row_origin': '원산지별',
        'p7_row_product': '품목별',
        'p7_row_importer': '수입사별',
        'p7_col_label': '열 (나눠보기)',
        'p7_col_none': '없음',
        'p7_col_origin': '원산지별',
        'p7_col_exporter': '공급사별',
        'p7_metric_label': '지표',
        'p7_metric_volume': '물량 합계',
        'p7_metric_price': '평균 단가',
        'p7_metric_count': '거래 건수',
        'p7_view_label': '보기',
        'p7_view_bar': '막대',
        'p7_view_line': '선',
        'p7_view_table': '표만',
        'p7_run_btn': '피벗 생성',
        'p7_reset_btn': '새로운 피벗 시작 (다시하기)',
        'p7_missing_cols_error': "필수 컬럼이 부족합니다: {cols}. 파일 내용을 확인해주세요.",
        'p7_no_data_warning': '집계할 데이터가 없습니다.',
        'p7_table_subheader': '데이터 표',

        'pdf_generate_btn': '📄 PDF 보고서 생성',
        'pdf_download_btn': '📥 PDF 다운로드',
        'pdf_generating_msg': 'PDF를 생성하고 있습니다...',
        'pdf_error_msg': 'PDF 생성 중 오류가 발생했습니다: {msg}',
        'multi_product_label': '품목 검색 (여러 개 선택 가능)',
        'multi_product_help': '실제 수입신고 명칭은 같은 상품이라도 표기가 조금씩 다를 수 있어요. 관련된 품목명을 모두 선택하면 하나로 합쳐서 분석합니다.',
        'insight_box_title': '💡 이번 분석에서 확인할 수 있는 것',
        'insight_cheapest_month': '과거 데이터 기준, {month}에 구매하면 평균 대비 가장 저렴했습니다 ({pct}% 낮음).',
        'insight_current_vs_seasonal_avg': '최근월 단가는 계절 평균 대비 {sign}{pct}% {direction}.',
        'insight_low_confidence': '⚠️ 이 품목은 데이터가 {years}년치뿐이라, 지금 나온 "계절 패턴"이 우연일 수도 있습니다. 2년 이상 데이터가 쌓이면 신뢰도가 높아집니다.',
        'direction_higher': '높습니다',
        'direction_lower': '낮습니다',
        'p6_scope_label': '범위',
        'p6_scope_all': '전체 시장 (모든 수입사 포함)',
        'p6_scope_importer': '특정 수입사만 (우리 회사 관점)',
        'p6_scope_entity_label': '수입사 선택',
        'p6_scope_all_caption': '※ "전체" 선택 시, 시장 전체에서 사라지거나 새로 생긴 거래관계를 보여줍니다 (특정 회사 관점이 아닙니다).',
        'p3_compare_years_label': '두 기간 비교하기',
        'p3_period_a': '기간 A',
        'p3_period_b': '기간 B',
        'p3_compare_col_a': '기간 A 물량',
        'p3_compare_col_b': '기간 B 물량',
        'p3_compare_col_diff': '증감',
        'p3_compare_col_diff_pct': '증감률',
        'p3_compare_subheader': "'{left}' 기준, {right_axis}별 기간 A→B 비교",
        'p7_filter_label': '필터 (선택)',
        'p7_filter_col': '필터할 컬럼',
        'p7_filter_none': '없음',
        'p7_filter_values': '포함할 값',
        'p7_view_pie': '파이',
        'p7_view_stacked': '누적 막대',
        'p7_view_heatmap': '히트맵',
        'p7_row_label_multi': '행 (기준, 여러 개 선택 가능 — 선택 순서대로 중첩됩니다)',
        'p7_col_label_multi': '열 (나눠보기, 여러 개 선택 가능)',
        'p7_values_label': '값 (지표, 여러 개 선택 가능)',
        'p7_metric_volume_sum': '물량 합계',
        'p7_metric_volume_mean': '물량 평균',
        'p7_metric_price_mean': '단가 평균',
        'p7_metric_price_max': '단가 최대',
        'p7_metric_price_min': '단가 최소',
        'p7_metric_count': '거래건수',
        'p7_filter_cols_label': '필터링할 컬럼 (여러 개 가능)',
        'p7_no_rows_warning': '행 기준을 최소 1개는 선택해주세요.',
        'p7_no_values_warning': '값(지표)을 최소 1개는 선택해주세요.',
        'p7_multi_metric_chart_note': '※ 지표를 여러 개 고르시면 표/히트맵만 지원됩니다. 차트는 첫 번째로 고른 지표만 그립니다.',
        'p7_heatmap_needs_dims_note': '※ 히트맵은 행과 열이 각각 1개일 때 가장 보기 좋습니다.',
        'p7_metric_label_multi': '지표 (복수 선택 가능)',
    },
    'en': {
        'app_menu_title': 'Menu',
        'menu_opt_customer': 'Customer Efficiency Analysis',
        'menu_opt_market': 'Market Competitiveness Analysis',
        'menu_opt_flow': 'Supply Chain Flow',
        'menu_opt_risk': 'Concentration Risk',
        'menu_opt_season': 'Price Trend & Seasonality',
        'menu_opt_churn': 'New/Lost Trading Partners',
        'menu_opt_pivot': 'Free Pivot Builder',
        'summary_table_no_data': 'No data to summarize.',
        'summary_table_header_price': 'Import Unit Price (USD/KG)',
        'summary_table_max': 'Max',
        'summary_table_mean': 'Avg',
        'summary_table_min': 'Min',
        'outlier_warning_capped': "Outliers exceeded {cap_percent:.0%} of the total, so only the most extreme {removed_rows} records were removed (cap applied).",
        'outlier_warning_removed': 'Removed {n} unit price outliers from the market data for analysis accuracy.',

        'p1_title': '💲 Customer Efficiency Analysis (Overview)',
        'p1_reset_btn': 'Start New Analysis (Reset)',
        'p1_settings_header': '⚙️ Analysis Settings',
        'p1_upload_label': 'Upload the customer data file',
        'p1_upload_caption': '※ Please upload a TDS raw file containing data for a single company.',
        'p1_missing_importer_col_warning': "The uploaded file has no 'Raw Importer Name' column. Please enter it manually below.",
        'p1_customer_name_input': 'Enter the name of the importer to analyze.',
        'p1_file_read_fail_error': 'Failed to read the file. It may not be in a supported encoding (utf-8, euc-kr, cp949), or the file may be corrupted.',
        'p1_file_read_error_generic': 'An error occurred while reading the file: {e}. Please check the file format or column names.',
        'p1_contract_date_input': 'Select the contract start date',
        'p1_run_btn': 'Run Analysis',
        'p1_spinner': 'Analyzing customer data...',
        'p1_missing_required_cols_error': 'Required columns are missing. Please check the file contents.',
        'p1_analysis_complete_success': "Analysis for customer '{customer_name}' complete!",
        'p1_exp1_title': '1. Estimated Savings Before/After Contract',
        'p1_total_savings_subheader': 'Total Estimated Savings',
        'p1_total_savings_caption': "※ Total estimated savings from the customer's own purchase price changes after the contract date ({date}).",
        'p1_savings_detail_subheader': 'Detailed Savings by Product Group',
        'p1_exp2_title': '2. Import Product Group Refinement & Clustering (DBSCAN & PCA)',
        'p1_too_many_clusters_info': 'Too many clusters — visualizing only the top {n} product groups by import volume.',
        'p1_scatter_title': '[{customer}] Product Similarity-Based Clustering (Top Product Groups Grid)',
        'p1_scatter_subtitle': 'Top {n} product groups by import volume',
        'p1_cluster_list_subheader': 'Cluster List (by import volume)',
        'p1_exp3_title': '3. Major Import Product Group Analysis (Monthly Volume)',
        'p1_monthly_chart_title': '[{customer}] Monthly Import Volume by Product Group (KG)',
        'p1_recent3m_subheader': 'Import Product Share (Last 3 Months)',
        'p1_recent3m_pie_title': '[{customer}] Import Share (Last 3 Months)',
        'p1_recent3m_pie_subtitle': '{start} to {end}',
        'p1_no_recent_data_info': 'No import data for the last 3 months.',
        'p1_no_data_warning': 'No data to analyze.',
        'p1_exp4_title': '4. Sourcing Changes After Contract',
        'p1_new_clusters_subheader': 'New Product Groups',
        'p1_new_clusters_text': 'A total of **{n}** new product groups started being imported after the contract.',
        'p1_no_new_clusters_info': 'No new product groups were added after the contract.',
        'p1_new_origins_subheader': 'New Origin Countries',
        'p1_new_origins_text': 'A total of **{n}** new origin countries started trading after the contract.',
        'p1_no_new_origins_info': 'No new origin countries were added after the contract.',
        'p1_new_exporters_subheader': 'New Suppliers',
        'p1_new_exporters_text': 'A total of **{n}** new suppliers started trading after the contract.',
        'p1_no_new_exporters_info': 'No new suppliers were added after the contract.',

        'col_new_cluster': 'New Product Group',
        'col_rep_product': 'Representative Product',
        'col_new_origin': 'New Origin',
        'col_related_products': 'Related Products',
        'col_new_exporter': 'New Supplier',

        'axis_yearmonth': 'Year-Month',
        'axis_volume_kg': 'Import Volume (KG)',
        'legend_cluster': 'Product Cluster',
        'axis_pca1': 'PCA Component 1',
        'axis_pca2': 'PCA Component 2',

        'p2_title': '🏆 Market Competitiveness Detailed Analysis (Drill-down)',
        'p2_reset_btn': 'Start New Market Analysis (Reset)',
        'p2_intro_text': "Upload full market data for a specific product to deeply analyze the customer's competitiveness in the market.",
        'p2_upload_label': 'Upload the full market data for the product to analyze.',
        'p2_upload_caption': "※ Please upload a TDS raw file containing multiple companies' data for a single product.",
        'p2_select_customer_label': 'Select the customer to analyze.',
        'p2_missing_importer_col_warning': "The uploaded file has no 'Raw Importer Name' column. Please enter it manually below.",
        'p2_customer_name_input': 'Enter the name of the importer to analyze.',
        'p2_file_read_error': 'An error occurred while reading the file: {e}. Please check the column names.',
        'p2_product_name_input': 'Enter the product name to analyze (e.g. Dried Noodles)',
        'p2_contract_date_input': 'Select the contract start date to use as the analysis baseline.',
        'p2_iqr_slider_label': 'Outlier Removal Sensitivity (IQR Multiplier)',
        'p2_iqr_slider_help': "A higher value treats a wider range of data as 'normal' (e.g. 3.0 removes fewer outliers).",
        'p2_run_btn': 'Start Market Competitiveness Analysis',
        'p2_processing_spinner': 'Analyzing market data. This may take a while depending on file size...',
        'p2_missing_cols_error': "Required columns are missing: {cols}. Please check that columns such as 'Export Country' are included in the file.",
        'p2_result_subheader': "Market Analysis Results for '{product}' (Reference Customer: {customer})",
        'p2_country_filter_label': 'Origin (Export Country) Filter:',
        'p2_country_filter_all': 'All',
        'p2_country_status_line': 'Origin: {countries}',
        'p2_no_data_for_filter_warning': "No data found for '{product}' with the selected origins.",
        'p2_exp1_title': '1. [{product}] Purchasing Competitiveness Analysis',
        'p2_scatter_subheader': 'Unit Price vs. Volume Distribution & Market Trend',
        'p2_scatter_title': 'Market Transaction Distribution & Average Price Trendline',
        'p2_scatter_subtitle': 'Based on LOWESS regression',
        'axis_unit_price': 'Unit Price (USD/KG)',
        'p2_top10_subheader': 'Top 10 Companies by Purchasing Competitiveness',
        'p2_rank_info': "Note: **{customer}**'s purchasing competitiveness rank is **#{rank}** out of {total} companies (filtered result).",
        'p2_rank_not_in_filter_warning': "**{customer}**'s data is not included in the current filter.",
        'p2_exp2_title': '2. [{product}] Price Trend & Competitive Advantage Group Benchmarking',
        'p2_monthly_comp_subheader': 'Monthly Trend of Purchasing Competitiveness Index',
        'legend_market_avg_index': 'Market Average Index',
        'legend_customer_index': '{customer} Competitiveness Index',
        'legend_top_group_avg_index': 'Competitive Advantage Group Average Index',
        'p2_comp_trend_chart_title': '[{product}] Monthly Trend of Purchasing Competitiveness Index',
        'axis_comp_index': 'Purchasing Competitiveness Index',
        'p2_comp_trend_caption': "※ This chart shows how the gap (competitiveness index) between the market's expected price and the actual purchase price changes over time.",
        'p2_price_trend_subheader': 'Monthly Average Price Trend',
        'legend_market_avg_price': 'Market Average Price',
        'legend_customer_avg_price': '{customer} Average Price',
        'legend_top_group_avg_price': 'Competitive Advantage Group Average',
        'p2_benchmark_info': '**Benchmark: Competitive Advantage Group Average**',
        'p2_benchmark_caption': "※ The 'Competitive Advantage Group' is the average of all companies that ranked higher than the currently selected customer in the 'Purchasing Competitiveness Analysis'.",
        'p2_benchmark_success': '**Benchmark Analysis:** `{customer}` currently shows the best purchasing competitiveness (within the current filter)!',
        'p2_price_trend_chart_title': '[{product}] Price Trend',
        'p2_price_compare_subheader': 'Average Price Comparison (Full Period)',
        'metric_market_avg': 'Market Average',
        'metric_customer_avg': '{customer} Average',
        'metric_top_group_avg': 'Competitive Advantage Group Average',
        'p2_sim_exp_title': 'Competitive Advantage Group Benchmarking Simulation',
        'p2_sim_start_date': 'Simulation Start Date',
        'p2_sim_end_date': 'Simulation End Date',
        'p2_sim_run_btn': 'Calculate Estimated Savings',
        'p2_sim_success': "If you had followed the **Competitive Advantage Group**'s average price during this period, you could have saved an additional **${amount}**.",
        'p2_sim_caption': "※ This amount only sums the potential savings for months where the customer's average monthly price was higher than the competitive advantage group's.",
        'p2_sim_no_data_warning': 'No data to compare for this period.',
        'p2_exp3_title': '3. [{product}] Market Share & Competitor Comparison',
        'p2_ms_year_select': 'Select Year for Market Share Analysis',
        'p2_others_label': 'Others',
        'p2_ms_pie_title': '[{product}] {year} Market Share',
        'p2_ms_pie_subtitle': 'By import volume',
        'p2_price_year_select': 'Select Year for Top 5 Importer Price Comparison',
        'p2_price_bar_title': '{year} Price Comparison: Customer vs. Top 5 Importers',
        'p2_price_bar_subtitle': 'Top 5 companies by import volume',
        'axis_importer': 'Importer',
        'axis_avg_unit_price': 'Average Unit Price (USD/KG)',
        'p2_exp4_title': '4. [{product}] Supply Chain (Supplier/Origin) Analysis',
        'p2_exporter_year_select': 'Select Year for Supply Chain Analysis',
        'p2_exporter_quarterly_subheader': '{year} Quarterly Price Distribution by Supplier',
        'p2_exporter_box_title': '{year} Quarterly Price Distribution by Supplier',
        'p2_exporter_box_subtitle': 'Top 10 suppliers by import volume',
        'axis_quarter': 'Quarter',
        'p2_detail_data_expander': 'View Detailed Data',
        'col_supplier': 'Supplier',
        'col_max_price': 'Max Price (USD/KG)',
        'col_avg_price': 'Avg Price (USD/KG)',
        'col_min_price': 'Min Price (USD/KG)',
        'p2_customer_exporters_info': 'Suppliers **{customer}** traded with in {year}: **{exporters}**',
        'p2_exporter_detail_expander': "Detailed Comparison for Supplier '{exporter}'",
        'p2_volume_price_compare_subheader': 'Volume & Average Price Comparison',
        'legend_total_volume': 'Total Import Volume (KG)',
        'legend_avg_import_price': 'Average Import Price (USD/KG)',
        'p2_exporter_bar_title': "'{exporter}': Volume & Average Price by Trading Company",
        'axis_total_volume': 'Total Import Volume (KG)',
        'axis_avg_import_price': 'Average Import Price (USD/KG)',
        'p2_price_dist_subheader': 'Price Distribution Comparison',
        'p2_exporter_box2_title': "'{exporter}': Price Distribution by Trading Company",
        'p2_exporter_box2_subtitle': 'Top 10 importers by import volume',
        'col_importer': 'Importer',
        'p2_alt_sourcing_subheader': '{year} Quarterly Alternative Sourcing Options',
        'p2_q_expander': '**Q{q}** Alternative Sourcing Options',
        'p2_no_quarter_data': '- No transaction data for this quarter.',
        'p2_current_sourcing_md': '**Current Sourcing Options**',
        'col_avg_price2': 'Average Price (USD/KG)',
        'p2_no_exporter_deals': '- No supplier transactions',
        'col_origin': 'Origin (Export Country)',
        'p2_no_origin_deals': '- No origin transactions',
        'p2_alt_recommend_md': '**Recommended Alternatives (Cheaper)**',
        'col_recommend_supplier': 'Recommended Supplier',
        'col_rep_item': 'Representative Item',
        'p2_no_cheaper_exporter': '- No cheaper supplier available',
        'col_recommend_origin': 'Recommended Origin (Export Country)',
        'p2_no_cheaper_origin': '- No cheaper origin (export country) available',
        'p2_no_supply_chain_cols_warning': "Cannot perform supply chain analysis because the 'Exporter', 'Export Country', or 'Origin Country' columns are missing.",

        'password_gate_title': '🔒 Restricted Access',
        'password_label': 'Enter password',
        'password_wrong': 'Incorrect password.',
        'password_not_configured': 'The app password (APP_PASSWORD) has not been configured yet. Please check secrets.toml or environment variables.',
        'password_setup_title': '🔒 First Run - Set a Password',
        'password_setup_caption': "This is the first time running this app! Please set a password you'll use from now on. You'll need to enter this password every time you access the app going forward.",
        'password_setup_new': 'New password',
        'password_setup_confirm': 'Confirm password',
        'password_setup_button': 'Set password and start',
        'password_setup_empty': 'Please enter a password.',
        'password_setup_mismatch': 'Passwords do not match.',
        'password_setup_success': 'Password has been set!',
        'guide_download_btn': '📖 User Guide (PDF)',
        'menu_opt_scorer': 'New Business Scorer',
        'p8_title': '🔬 Raw Material New-Business Scorer',
        'p8_intro': 'Finds "recently surging existing items" and "brand-new entrants" from full market trade data, ranked by opportunity — with an optional fit score against your own existing business.',
        'p8_upload_label': 'Upload the full market trade data',
        'p8_upload_caption': '※ Please upload a TDS raw file with multiple importers, suppliers, and origins. HS Code Name, Detailed HS-CODE, Category, and Korean product name columns are auto-detected if present.',
        'p8_settings_header': '⚙️ Analysis Settings',
        'p8_months_label': 'Analysis window (last N months)',
        'p8_floor_label': 'Minimum volume threshold (KG)',
        'p8_dim_label': 'Analysis dimension',
        'p8_minship_label': 'Minimum shipment count (for new entrants)',
        'p8_newkey_label': 'Dimension used to detect "new"',
        'p8_preset_label': 'Scoring weight preset',
        'p8_preset_growth': 'Growth-focused',
        'p8_preset_size': 'Size-focused',
        'p8_preset_easy': 'Ease-of-entry-focused',
        'p8_fitw_label': 'Fit score weight (%)',
        'p8_fit_context_header': '🎯 Your Company Fit Criteria (optional — enables quadrant classification)',
        'p8_fit_context_caption': "Analysis works without this, but filling it in scores how well each candidate fits your existing business.",
        'p8_fit_suppliers_label': 'Existing suppliers (comma or newline separated)',
        'p8_fit_origins_label': 'Existing origin countries',
        'p8_fit_keywords_label': 'Keywords for products you already handle',
        'p8_fit_moat_label': 'Bonus keywords (optional, adds extra score)',
        'p8_run_btn': 'Run Analysis',
        'p8_reset_btn': 'Start New Analysis (Reset)',
        'p8_missing_cols_error': "Required columns are missing: {cols}. Please check the file contents.",
        'p8_no_data_warning': 'No data matches the settings. Try adjusting the window or minimum volume.',
        'p8_kpi_window': 'Analysis Window',
        'p8_kpi_total_yoy': 'Total Volume YoY',
        'p8_kpi_growing': 'Surging Existing',
        'p8_kpi_new': 'New Entrants',
        'p8_kpi_price': 'Price Data',
        'p8_kpi_price_linked': 'Linked (unit price YoY applied)',
        'p8_kpi_price_none': 'Not linked (volume only)',
        'p8_section_s': '🏆 Combined Recommendation Ranking (Top 10)',
        'p8_section_a': '📈 Surging Existing Items',
        'p8_section_b': '🆕 New Entrant Items',
        'p8_col_rank': '#',
        'p8_col_type': 'Type',
        'p8_col_item': 'Item',
        'p8_col_score': 'Score',
        'p8_col_fit': 'Fit',
        'p8_col_quadrant': 'Quadrant',
        'p8_col_reason': 'Reason',
        'p8_col_flags': 'Flags',
        'p8_col_yoy': 'YoY',
        'p8_col_rec_vol': 'Recent Volume',
        'p8_col_ly_vol': 'Last Year Volume',
        'p8_col_delta': 'Change',
        'p8_col_price_yoy': 'Price YoY',
        'p8_col_concentration': 'Concentration',
        'p8_col_vol': 'Recent Volume',
        'p8_col_ship': 'Shipments',
        'p8_col_top_importer': 'Top Importer',
        'p8_type_existing': 'Existing·Surging',
        'p8_type_new': 'New Entrant',
        'p8_flag_low_base': '⚠ Low-base rebound',
        'p8_flag_single_ship': '⚠ Single-shipment skew',
        'p8_flag_suspect': '⚠ Possible re-labeling',
        'p8_quad_priority': 'Priority Target',
        'p8_quad_growth': 'Growth-led',
        'p8_quad_safe': 'Safe Adjacent',
        'p8_quad_low': 'Lower Priority',
        'p8_signal_premium': 'Premium',
        'p8_signal_commodity': 'Commoditizing',
        'p8_signal_shock': 'Supply Shock',
        'p8_label_growth': 'Growth',
        'p8_label_cagr': 'CAGR',
        'p8_label_share': 'Share',
        'p8_label_importer_count': 'Importers',
        'p8_label_origin_count': 'Origins',
        'p8_label_origin_unit': '',
        'p8_label_fit': 'Fit',
        'p8_label_top_importer': 'Top importer',
        'p8_label_recent': 'Recent',
        'p8_label_shipment': 'Shipments',
        'p8_label_shipment_unit': '',
        'p8_fitwhy_supplier': 'supplier',
        'p8_fitwhy_origin': 'origin',
        'p8_fitwhy_keyword': 'adjacent product',
        'p8_fitwhy_moat': 'bonus keyword',
        'p8_fitwhy_none': 'no match',
        'p8_chart_title': 'Top 10 Combined Ranking — Score',
        'p8_dim_hs_name': 'HS Code Name',
        'p8_dim_detailed_hs': 'Detailed HS-CODE',
        'p8_dim_product': 'Reported Product Name',
        'p8_dim_importer': 'Importer',
        'p8_dim_origin': 'Origin Country',
        'p8_report_title': 'Import New-Business Scoring Report',
        'p8_report_subtitle': 'SOURCING OPPORTUNITY SCORER',
        'p8_report_meta': 'Period {start} ~ {end} · vs. same period last year ({ly_start}~{ly_end}) · Window {months} · Published {today}',
        'p8_report_insights_header': '◆ Key Insights',
        'p8_report_data_summary': 'Data analyzed: {n_tx} transactions · {span} · {n_imp} importers · {n_prod} products · {n_origin} origin countries',
        'p8_report_top10_header': '★ Top 10 Recommendations (Combined Opportunity Score)',
        'p8_report_top10_sub': 'Surging existing items and new entrants ranked together on one score (growth, size, price trend, concentration).',
        'p8_report_section_a_header': 'A. Surging Existing Items — Detail (Top {n})',
        'p8_report_section_a_sub': 'Items imported last year too, with growing volume, ranked by score.',
        'p8_report_section_b_header': 'B. New Entrant Items — Detail (Top {n})',
        'p8_report_section_b_sub': 'Items absent in the entire prior history but recently appearing (min. shipment/volume applied), ranked by score.',
        'p8_report_footer': 'Data: uploaded transaction records. Recent-period totals compared to the same period last year (YoY). "New" means absent from the entire prior history but appearing recently. The opportunity score is a proprietary normalized combination of growth rate, volume, (where available) price trend, and importer concentration; low-base rebounds, single-shipment skew, and possible re-labeling are penalized. Tariffs, import regulations, logistics costs, and contract terms are not reflected — use this for initial screening only, and have a person review before final decisions.',
        'p8_ins_total': 'Total imports over the last {month_label} were {sign}{pct}% vs. the same period last year ({ly} → {rec}).',
        'p8_ins_top_candidate': 'Top opportunity candidate: {name} ({type}, score {score}).',
        'p8_ins_top_growth': 'Highest growth: {name} — {sign}{pct}%.',
        'p8_ins_top_new': 'Leading new entrant: {name} ({vol}{origin_part}).',
        'p8_ins_top_new_origin_part': ', primarily from {origin}',
        'p8_ins_price_up': 'Price also rose for: {name} — price {sign}{pct}%.',
        'p8_ins_fit': 'With fit criteria applied, {n} "Priority Target" (both high growth and high fit) items were found{rep_part}.',
        'p8_ins_fit_rep_part': ' — top example: {name} (fit {fit})',
        'p8_detail_a_narrative': '{sign}{pct}% vs. last year ({ly}→{rec}), share {share}%.{tags}',
        'p8_detail_big_tag': ' Large volume',
        'p8_detail_lowbase_tag': ' Low base',
        'p8_detail_b_narrative': '{tag} Recent {vol} · {ship} shipments · top importer {imp} {share}%.{suspect}',
        'p8_detail_new_tag': 'New',
        'p8_detail_suspect_tag': ' Possible re-labeling',
        'p8_generate_report_btn': '📊 Generate Detailed Report PDF',
        'p8_download_report_btn': '📥 Download Report PDF',
        'p8_cat_all': 'All items',
        'p8_report_excluded_note_a': '※ {n} items with a near-zero prior-year base (distorted figures) were excluded from the detail list.',
        'p8_report_excluded_note_b': '※ {n} items suspected to be re-labeled existing products were excluded from the detail list.',
        'p8_report_note_col': 'Note',

        'p3_title': '🔀 Supply Chain Flow (Sankey)',
        'p3_upload_label': 'Upload the full market/supply chain data',
        'p3_upload_caption': '※ Please upload a TDS raw file containing multiple suppliers, importers, and origins.',
        'p3_left_axis_label': 'Left axis — Type',
        'p3_left_entity_label': 'Left axis — Select entity',
        'p3_right_axis_label': 'Right axis — Compare by',
        'p3_axis_exporter': 'Supplier',
        'p3_axis_importer': 'Importer',
        'p3_axis_origin': 'Origin',
        'p3_axis_country': 'Export destination',
        'p3_axis_product': 'Product',
        'p3_axis_hint': '※ Origin = country where the product was produced. Export destination = country it was sold to (different columns).',
        'p3_date_start': 'Start date',
        'p3_date_end': 'End date',
        'p3_run_btn': 'Draw flow diagram',
        'p3_reset_btn': 'Start New Flow Analysis (Reset)',
        'p3_missing_cols_error': "Required columns are missing: {cols}. Please check the file contents.",
        'p3_no_data_warning': 'No data matches the selected conditions. Please check the entity/date range.',
        'p3_result_subheader': "Flow for '{left}' ({left_axis}) by {right_axis} ({start} to {end})",
        'p3_sankey_title': "'{left}' → volume flow by {right_axis}",
        'p3_table_subheader': 'Sorted detail table',
        'p3_col_category': '{right_axis}',
        'p3_col_volume': 'Volume (KG)',
        'p3_col_share': 'Share',
        'p3_col_avg_price': 'Avg Price (USD/KG)',
        'p3_others_note': '※ Everything outside the top {n} is grouped as "Others".',

        'p4_title': '⚠️ Concentration Risk Diagnosis',
        'p4_upload_label': 'Upload the full market/supply chain data',
        'p4_upload_caption': '※ Please upload a TDS raw file containing multiple suppliers, importers, and origins.',
        'p4_axis_label': 'By',
        'p4_axis_exporter': 'Supplier',
        'p4_axis_origin': 'Origin',
        'p4_axis_product': 'Product',
        'p4_scope_label': 'Scope',
        'p4_scope_all': 'All',
        'p4_scope_importer': 'Specific importer only',
        'p4_scope_entity_label': 'Select importer',
        'p4_date_start': 'Start date',
        'p4_date_end': 'End date',
        'p4_threshold_label': 'Risk threshold',
        'p4_run_btn': 'Run diagnosis',
        'p4_reset_btn': 'Start New Diagnosis (Reset)',
        'p4_missing_cols_error': "Required columns are missing: {cols}. Please check the file contents.",
        'p4_no_data_warning': 'No data matches the selected conditions.',
        'p4_kpi_top1': 'Top-1 share',
        'p4_kpi_top3': 'Top-3 combined',
        'p4_kpi_count': 'Total trading partners',
        'p4_kpi_risk': 'Risk level',
        'p4_risk_danger': 'High',
        'p4_risk_caution': 'Caution',
        'p4_risk_safe': 'Safe',
        'p4_risk_reason_top1': 'Top-1 share {v}% exceeds threshold ({t}%)',
        'p4_risk_reason_top3': 'Top-3 combined {v}% exceeds {t}%',
        'p4_risk_reason_ok': 'Within threshold',
        'p4_bar_chart_title': 'Share by {axis} (dashed = risk threshold {t}%)',
        'p4_trend_chart_title': 'Monthly Trend of Top-1 Share',
        'p4_axis_share': 'Share (%)',
        'p4_col_name': 'Name',
        'p4_col_volume': 'Volume (KG)',
        'p4_col_share': 'Share',

        'p5_title': '📈 Price Trend & Seasonality',
        'p5_upload_label': 'Upload the full market data',
        'p5_upload_caption': '※ Please upload a TDS raw file with time-series data for one or more products.',
        'p5_product_label': 'Search product',
        'p5_prevyear_label': 'Overlay previous year',
        'p5_breakdown_label': 'Compare by',
        'p5_breakdown_none': 'Overall average only',
        'p5_breakdown_origin': 'Split by origin',
        'p5_breakdown_exporter': 'Split by supplier',
        'p5_run_btn': 'Draw trend',
        'p5_reset_btn': 'Start New Price Trend Analysis (Reset)',
        'p5_missing_cols_error': "Required columns are missing: {cols}. Please check the file contents.",
        'p5_no_data_warning': 'No data found for the selected product.',
        'p5_kpi_current': 'Latest month avg price',
        'p5_kpi_yoy': 'YoY (same month)',
        'p5_kpi_frompeak': 'Current vs. yearly peak',
        'p5_season_badge': 'Seasonal peak: {months}',
        'p5_no_season': 'No clear seasonal pattern',
        'p5_chart_title': "'{product}' Monthly Average Price Trend",
        'p5_axis_month': 'Year-Month',
        'p5_axis_price': 'Unit Price (USD/KG)',
        'p5_legend_this_year': 'Selected period',
        'p5_legend_prev_year': 'Previous year',

        'p6_title': '🔀 New/Lost Trading Partners',
        'p6_upload_label': 'Upload the full market/supply chain data',
        'p6_upload_caption': '※ Please upload a TDS raw file containing multiple suppliers, importers, and origins.',
        'p6_axis_label': 'Track by',
        'p6_axis_exporter': 'Supplier',
        'p6_axis_origin': 'Origin',
        'p6_axis_product': 'Product',
        'p6_axis_importer': 'Importer',
        'p6_period_a': 'Period A (comparison)',
        'p6_period_b': 'Period B (baseline)',
        'p6_run_btn': 'Compare',
        'p6_reset_btn': 'Start New Tracking (Reset)',
        'p6_missing_cols_error': "Required columns are missing: {cols}. Please check the file contents.",
        'p6_no_data_warning': 'No data found for one or both periods.',
        'p6_kpi_new': 'New',
        'p6_kpi_kept': 'Kept',
        'p6_kpi_lost': 'Lost',
        'p6_new_header': '🟢 New Partners',
        'p6_lost_header': '🔴 Lost Partners',
        'p6_no_new': 'No new items',
        'p6_no_lost': 'No lost items',
        'p6_col_name': 'Name',
        'p6_col_volume': 'Volume (KG)',
        'p6_price_compare_subheader': '💰 Looking at price',
        'p6_kpi_new_price': 'New Partners Avg Price',
        'p6_kpi_kept_price': 'Kept Partners Avg Price',
        'p6_kpi_lost_price': 'Lost Partners Avg Price',
        'p6_price_insight_cheaper': 'New partners are on average {pct}% {direction} than kept partners.',
        'p6_price_insight_none': 'Not enough price data available for comparison.',
        'p6_concentration_subheader': '⚠️ Concentration Change',
        'p6_concentration_warning': "Top-1 partner dependency rose from {a}% to {b}% between Period A and B (current #1: {name}). This may signal increased reliance on fewer partners.",
        'p6_concentration_stable': 'Top-1 partner dependency: {a}% → {b}% (little change)',
        'p6_concentration_improved': 'Top-1 partner dependency actually fell from {a}% to {b}% (supplier base more diversified).',

        'p7_title': '🧩 Free Pivot Builder',
        'p7_upload_label': 'Upload the data to analyze',
        'p7_upload_caption': '※ Please upload a TDS raw file (customer or market data both work).',
        'p7_row_label': 'Rows (group by)',
        'p7_row_month': 'By month',
        'p7_row_exporter': 'By supplier',
        'p7_row_origin': 'By origin',
        'p7_row_product': 'By product',
        'p7_row_importer': 'By importer',
        'p7_col_label': 'Columns (split by)',
        'p7_col_none': 'None',
        'p7_col_origin': 'By origin',
        'p7_col_exporter': 'By supplier',
        'p7_metric_label': 'Metric',
        'p7_metric_volume': 'Total volume',
        'p7_metric_price': 'Average price',
        'p7_metric_count': 'Transaction count',
        'p7_view_label': 'View',
        'p7_view_bar': 'Bar',
        'p7_view_line': 'Line',
        'p7_view_table': 'Table only',
        'p7_run_btn': 'Generate pivot',
        'p7_reset_btn': 'Start New Pivot (Reset)',
        'p7_missing_cols_error': "Required columns are missing: {cols}. Please check the file contents.",
        'p7_no_data_warning': 'No data to aggregate.',
        'p7_table_subheader': 'Data table',

        'pdf_generate_btn': '📄 Generate PDF Report',
        'pdf_download_btn': '📥 Download PDF',
        'pdf_generating_msg': 'Generating PDF...',
        'pdf_error_msg': 'An error occurred while generating the PDF: {msg}',
        'multi_product_label': 'Search products (multi-select)',
        'multi_product_help': "Import declarations often spell the same product slightly differently. Select all related product names to combine them into one analysis.",
        'insight_box_title': '💡 What this analysis shows',
        'insight_cheapest_month': 'Historically, buying in {month} was cheapest on average ({pct}% below average).',
        'insight_current_vs_seasonal_avg': "The latest month's price is {sign}{pct}% {direction} the seasonal average.",
        'insight_low_confidence': '⚠️ This product only has {years} year(s) of data, so the "seasonal pattern" shown may be coincidental. Confidence improves with 2+ years of history.',
        'direction_higher': 'higher',
        'direction_lower': 'lower',
        'p6_scope_label': 'Scope',
        'p6_scope_all': 'Entire market (all importers)',
        'p6_scope_importer': 'Specific importer only (our company)',
        'p6_scope_entity_label': 'Select importer',
        'p6_scope_all_caption': '※ With "Entire market" selected, this shows relationships that appeared or disappeared market-wide (not from one company\'s perspective).',
        'p3_compare_years_label': 'Compare two periods',
        'p3_period_a': 'Period A',
        'p3_period_b': 'Period B',
        'p3_compare_col_a': 'Period A Volume',
        'p3_compare_col_b': 'Period B Volume',
        'p3_compare_col_diff': 'Change',
        'p3_compare_col_diff_pct': 'Change %',
        'p3_compare_subheader': "'{left}' by {right_axis}: Period A → B comparison",
        'p7_filter_label': 'Filter (optional)',
        'p7_filter_col': 'Filter column',
        'p7_filter_none': 'None',
        'p7_filter_values': 'Values to include',
        'p7_view_pie': 'Pie',
        'p7_view_stacked': 'Stacked bar',
        'p7_view_heatmap': 'Heatmap',
        'p7_row_label_multi': 'Rows (group by, multi-select — nested in the order chosen)',
        'p7_col_label_multi': 'Columns (split by, multi-select)',
        'p7_values_label': 'Values (metrics, multi-select)',
        'p7_metric_volume_sum': 'Total volume',
        'p7_metric_volume_mean': 'Average volume',
        'p7_metric_price_mean': 'Average price',
        'p7_metric_price_max': 'Max price',
        'p7_metric_price_min': 'Min price',
        'p7_metric_count': 'Transaction count',
        'p7_filter_cols_label': 'Columns to filter by (multi-select)',
        'p7_no_rows_warning': 'Please select at least one row field.',
        'p7_no_values_warning': 'Please select at least one value (metric).',
        'p7_multi_metric_chart_note': '※ With multiple metrics selected, only Table/Heatmap views are supported. Charts use only the first selected metric.',
        'p7_heatmap_needs_dims_note': '※ Heatmap looks best with exactly one row field and one column field.',
        'p7_metric_label_multi': 'Metrics (multi-select)',
    },
}

def T(key, **kwargs):
    """현재 언어에 맞는 텍스트를 반환하는 헬퍼 함수"""
    lang = st.session_state.get('lang', 'ko')
    text = TEXTS.get(lang, TEXTS['ko']).get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text

# --------------------------------#
# 데이터 전처리 및 분석 함수 #
# --------------------------------#

def preprocess_product_name(name):
    """'REPORTED PRODUCT NAME'을 정제하는 함수"""
    if not isinstance(name, str): return ''
    name = re.sub(r'\[.*?\]', '', name)
    name = name.split('_')[0]
    name = re.sub(r'(\(?\s*\d+\.?\d*\s*(kg|g|l|ml)\s*\)?)', '', name, flags=re.I)
    name = re.sub(r'[^A-Za-z0-9가-힣]', '', name)
    return name.strip().upper()  # 대소문자만 다른 표기(예: "Frozen"↔"FROZEN")도 같은 품목으로 묶이도록 대문자로 통일

def get_cluster_name(cluster_labels, preprocessed_names):
    """각 클러스터의 이름을 생성하는 함수"""
    cluster_name_map = {}
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label != -1:
            names_in_cluster = preprocessed_names[cluster_labels == label]
            if len(names_in_cluster) > 0:
                most_common_name = Counter(names_in_cluster).most_common(1)[0][0]
                cluster_name_map[label] = most_common_name
            else:
                cluster_name_map[label] = f'Cluster {label}'
    final_cluster_names = {}
    name_counts = Counter(cluster_name_map.values())
    used_names = {}
    for label, name in cluster_name_map.items():
        if name_counts[name] > 1:
            if name not in used_names: used_names[name] = 1
            final_cluster_names[label] = f"{name}_{used_names[name]}"
            used_names[name] += 1
        else:
            final_cluster_names[label] = name
    final_cluster_names[-1] = 'Noise'
    return final_cluster_names

# --- [수정 2] iqr_multiplier 파라미터 추가 (슬라이더 연동) ---
def remove_outliers_iqr(df, column_name, cap_percent=0.07, iqr_multiplier=1.5):
    """IQR 방식을 사용하되, 제거 비율을 최대 7%로 제한하여 이상치를 제거하는 함수"""
    if df.empty:
        return df

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    # --- [수정 3] 1.5 대신 iqr_multiplier 사용 ---
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    # 잠재적 이상치 식별
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    
    # 제거 비율이 상한선(7%)을 초과하는지 확인
    if not outliers.empty and (len(outliers) / len(df)) > cap_percent:
        num_to_remove = int(len(df) * cap_percent)
        
        # 중앙값에서 가장 멀리 떨어진 극단적인 값부터 제거하기 위해 거리 계산
        median = df[column_name].median()
        outliers = outliers.copy() # SettingWithCopyWarning 방지
        outliers['distance'] = (outliers[column_name] - median).abs()
        
        # 제거할 인덱스 선택
        indices_to_remove = outliers.nlargest(num_to_remove, 'distance').index
        
        df_filtered = df.drop(indices_to_remove)
        removed_rows = num_to_remove
        
        st.warning(T('outlier_warning_capped', cap_percent=cap_percent, removed_rows=removed_rows))
    else:
        # 상한선을 초과하지 않으면, 식별된 모든 이상치 제거
        df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
        removed_rows = len(df) - len(outliers)
        if len(outliers) > 0:
             st.warning(T('outlier_warning_removed', n=len(outliers)))

    return df_filtered

def generate_summary_table_html(df, group_by_col, header_name, value_col='unit_price'):
    """박스플롯에 대한 요약 테이블 HTML을 생성하는 함수"""
    if df.empty:
        return f"<p>{T('summary_table_no_data')}</p>"
    summary_df = df.groupby(group_by_col)[value_col].agg(['max', 'mean', 'min']).reset_index()
    
    html = f"""
    <style>
        .summary-table {{
            width: 100%; border-collapse: collapse;
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #e6e6e6; padding: 8px;
            text-align: left;
        }}
        .summary-table th {{
            background-color: #f2f2f2;
        }}
    </style>
    <table class="summary-table">
      <thead>
        <tr>
          <th rowspan="2" style="text-align: center; vertical-align: middle;">{header_name}</th>
          <th colspan="3" style="text-align: center;">{T('summary_table_header_price')}</th>
        </tr>
        <tr>
          <th style="text-align: center;">{T('summary_table_max')}</th>
          <th style="text-align: center;">{T('summary_table_mean')}</th>
          <th style="text-align: center;">{T('summary_table_min')}</th>
        </tr>
      </thead>
      <tbody>
    """

    for index, row in summary_df.iterrows():
        html += f"""
        <tr>
            <td>{row[group_by_col]}</td>
            <td style="text-align: right;">${row['max']:.2f}</td>
            <td style="text-align: right;">${row['mean']:.2f}</td>
            <td style="text-align: right;">${row['min']:.2f}</td>
        </tr>
        """
    
    html += "</tbody></table>"
    return html

def reset_analysis_states():
    """모든 분석 상태를 초기화하는 함수"""
    st.session_state.analysis_done = False
    st.session_state.market_analysis_done = False
    keys_to_reset = ['customer_name', 'plot_df', 'customer_df', 'contract_date', 
                     'tfidf_matrix', 'savings_df', 'total_savings', 'market_df',
                     'analyzed_product_name']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # [추가] 원산지 필터 세션도 리셋
    if 'analysis_countries' in st.session_state:
        del st.session_state['analysis_countries']


def reset_market_analysis_states():
    """목표 2 분석 상태만 초기화하는 함수"""
    st.session_state.market_analysis_done = False
    keys_to_reset = ['market_df', 'analyzed_product_name', 'selected_customer', 
                     'market_contract_date', 'top_competitors_list',
                     'all_competitors_ranked']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # [추가] 원산지 필터 세션도 리셋
    if 'analysis_countries' in st.session_state:
        del st.session_state['analysis_countries']


def reset_flow_states():
    """공급망 흐름도(페이지 3) 상태만 초기화하는 함수"""
    st.session_state.flow_raw_df = None
    st.session_state.flow_headers = None
    st.session_state.flow_result = None


def reset_risk_states():
    """집중도 리스크 진단(페이지 4) 상태만 초기화하는 함수"""
    st.session_state.risk_raw_df = None
    st.session_state.risk_headers = None
    st.session_state.risk_result = None


def reset_season_states():
    """가격 추세 & 계절성(페이지 5) 상태만 초기화하는 함수"""
    st.session_state.season_raw_df = None
    st.session_state.season_headers = None
    st.session_state.season_result = None


def reset_churn_states():
    """신규·이탈 거래처 추적(페이지 6) 상태만 초기화하는 함수"""
    st.session_state.churn_raw_df = None
    st.session_state.churn_headers = None
    st.session_state.churn_result = None


def reset_pivot_states():
    """자유 피벗 빌더(페이지 7) 상태만 초기화하는 함수"""
    st.session_state.pivot_raw_df = None
    st.session_state.pivot_headers = None


def reset_scorer_states():
    """신규사업 스코어러(페이지 8) 상태만 초기화하는 함수"""
    st.session_state.scorer_raw_df = None
    st.session_state.scorer_headers = None
    st.session_state.scorer_result = None


def read_uploaded_table(uploaded_file):
    """CSV(인코딩 자동 판별)/XLSX 파일을 읽어 DataFrame으로 반환하는 공통 헬퍼.
    실패 시 None을 반환한다 (호출부에서 오류 메시지 처리)."""
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith('.csv'):
        for enc in ('utf-8', 'euc-kr', 'cp949'):
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except UnicodeDecodeError:
                continue
        return None
    elif uploaded_file.name.endswith('.xlsx'):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)
    return None


def detect_standard_columns(headers):
    """TDS raw file에서 공통적으로 쓰는 8개 컬럼을 표준 이름으로 찾아 dict로 반환."""
    return {
        'date': find_column(headers, ['Date', 'date']),
        'importer': find_column(headers, ['Raw Importer Name', 'importer_name']),
        'exporter': find_column(headers, ['Exporter', 'exporter']),
        'origin': find_column(headers, ['Origin Country', 'origin_country']),
        'export_country': find_column(headers, ['Export Country', 'export_country', '수출국']),
        'product': find_column(headers, ['Reported Product Name', 'product_name']),
        'volume': find_column(headers, ['Volume', 'volume']),
        'price': find_column(headers, ['Unit Price', 'unit_price']),
    }


def detect_extra_dimension_columns(df, cols, max_unique_ratio=0.5):
    """표준 8개 컬럼 외에, 업로드된 파일에 실제로 존재하는 다른 '축으로 쓸 수 있는' 컬럼을
    동적으로 찾는다 (예: HS Code, Incoterm, Port of Discharge 등). 날짜/수치형이 아니고,
    사실상 자유서술 텍스트가 아닌(고유값 비율이 너무 높지 않은) 컬럼만 후보로 삼는다."""
    used = {cols.get(k) for k in ('date', 'importer', 'exporter', 'origin', 'export_country', 'product', 'volume', 'price')}
    extras = []
    n = max(len(df), 1)
    for c in df.columns:
        if c in used or c is None:
            continue
        if df[c].dtype == object:
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= max(50, n * max_unique_ratio):
                extras.append(c)
    return extras


def build_axis_map(standard_pairs, df, cols):
    """(라벨, 컬럼명) 표준 축 목록 + 파일에서 동적으로 찾은 추가 컬럼들을 합쳐
    선택창(selectbox)에 바로 쓸 수 있는 {라벨: 컬럼명} dict를 만든다."""
    axis_map = {label: col for label, col in standard_pairs if col}
    for extra_col in detect_extra_dimension_columns(df, cols):
        if extra_col not in axis_map.values():
            axis_map[extra_col] = extra_col
    return axis_map


# ============================================================
# 신규사업 스코어러 (페이지 8) — 원본 HTML 도구의 스코어링 엔진을 그대로 이식
# ============================================================
def detect_scorer_columns(headers):
    """표준 8개 컬럼 + 스코어러 전용 추가 컬럼(HS코드명, Detailed HS-CODE, 카테고리, 한글품목명)을 찾는다."""
    base = detect_standard_columns(headers)
    base['hs_name'] = find_column(headers, ['HS Code Name', 'hs_name', 'HSCodeName', 'HS코드명', '품목명'])
    base['detailed_hs'] = find_column(headers, ['Detailed HS-CODE', 'Detailed HS Code', 'detailed_hs', 'HS Code', 'HS코드'])
    base['category'] = find_column(headers, ['Category', 'category', '카테고리', '분류'])
    base['label_kr'] = find_column(headers, ['한글품목명', '국문품명', '품명국문', '한글명', 'label_kr', 'Korean Name'])
    return base


def _p8_norm_name(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ''
    return re.sub(r'\s+', ' ', str(s).strip().upper())


def _p8_split_list(s):
    if not s:
        return set()
    parts = re.split(r'[,\n;]+', s)
    return set(_p8_norm_name(p) for p in parts if p and p.strip())


def _p8_minmax_norm(value, arr):
    if not arr:
        return 50.0
    mn, mx = min(arr), max(arr)
    if mx <= mn:
        return 50.0
    return (value - mn) / (mx - mn) * 100.0


def _p8_mode_or_none(items):
    items = [x for x in items if x is not None and not (isinstance(x, float) and pd.isna(x)) and str(x).strip() != '']
    if not items:
        return None
    counts = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _p8_fit_of(supplier_set, origin_set, name_text, fit_ctx):
    """적합도(fit) 점수 계산: 공급사/원산지/키워드/가점키워드 매칭 여부로 0~100점."""
    if not fit_ctx['has_ctx']:
        return None, ''
    sup = ori = kw = moat = 0
    if fit_ctx['suppliers'] and supplier_set:
        for s in supplier_set:
            sn = _p8_norm_name(s)
            if any(sn in m or m in sn for m in fit_ctx['suppliers'] if m):
                sup = 1
                break
    if fit_ctx['origins'] and origin_set:
        for o in origin_set:
            if o and _p8_norm_name(o) in fit_ctx['origins']:
                ori = 1
                break
    name_norm = _p8_norm_name(name_text)
    if fit_ctx['keywords']:
        kw = 1 if any(w and w in name_norm for w in fit_ctx['keywords']) else 0
    if fit_ctx['moat']:
        moat = 1 if any(w and w in name_norm for w in fit_ctx['moat']) else 0
    score = max(0, min(100, round(sup * 45 + ori * 25 + kw * 25 + moat * 15)))
    why_parts = []
    if sup: why_parts.append(T('p8_fitwhy_supplier'))
    if ori: why_parts.append(T('p8_fitwhy_origin'))
    if kw: why_parts.append(T('p8_fitwhy_keyword'))
    if moat: why_parts.append(T('p8_fitwhy_moat'))
    why = '·'.join(why_parts) if why_parts else T('p8_fitwhy_none')
    return score, why


def compute_scorer(df, dim_col, months, floor, minship, new_key_col, preset, fit_ctx, fitw,
                    has_price, importer_col, exporter_col, origin_col, detailed_hs_col, label_kr_col, product_col):
    """원본 JS run() 함수를 그대로 이식한 핵심 계산. A(기존급상승)/B(신규진입)/S(통합순위) 를 반환."""
    latest = df['_date'].max()
    earliest = df['_date'].min()
    rec_start = (latest - pd.DateOffset(months=months - 1)).replace(day=1)
    ly_start = pd.Timestamp(year=rec_start.year - 1, month=rec_start.month, day=1)
    ly_end = pd.Timestamp(year=latest.year - 1, month=latest.month, day=latest.day)

    recent = df[(df['_date'] >= rec_start) & (df['_date'] <= latest)]
    lastyr = df[(df['_date'] >= ly_start) & (df['_date'] <= ly_end)]
    hist = df[df['_date'] < rec_start]

    tot_rec = recent['_volume'].sum()
    tot_ly = lastyr['_volume'].sum()
    span_months = max(1, round((latest - earliest).days / 30.44))

    # ---------- Section A: 기존 급상승 품목 ----------
    A_rows = []
    for key, g in recent.groupby(dim_col):
        if key is None or (isinstance(key, float) and pd.isna(key)):
            continue
        rec_vol = g['_volume'].sum()
        max_row = g['_volume'].max()
        top1_imp, top1_share = None, 0.0
        if importer_col and importer_col in g.columns:
            imp_vol = g.groupby(importer_col)['_volume'].sum().sort_values(ascending=False)
            if len(imp_vol):
                top1_imp, top1_share = imp_vol.index[0], (imp_vol.iloc[0] / rec_vol * 100 if rec_vol > 0 else 0)
        price_rec = None
        if has_price:
            valid = g.dropna(subset=['_value'])
            if valid['_volume'].sum() > 0:
                price_rec = valid['_value'].sum() / valid['_volume'].sum()
        dhs_set = set(g[detailed_hs_col].dropna().unique()) if detailed_hs_col else set()
        kr_name = _p8_mode_or_none(g[label_kr_col].tolist()) if label_kr_col else None
        supplier_set = set(g[exporter_col].dropna().unique()) if exporter_col else set()
        origin_set = set(g[origin_col].dropna().unique()) if origin_col else set()

        ly_g = lastyr[lastyr[dim_col] == key]
        ly_vol = ly_g['_volume'].sum()
        price_ly = None
        if has_price and len(ly_g):
            valid_ly = ly_g.dropna(subset=['_value'])
            if valid_ly['_volume'].sum() > 0:
                price_ly = valid_ly['_value'].sum() / valid_ly['_volume'].sum()

        en_name = preprocess_product_name(str(key)) if dim_col == 'product' else str(key)
        A_rows.append({
            'k': key, 'rec_vol': rec_vol, 'ly_vol': ly_vol, 'max_row': max_row, 'rows_n': len(g),
            'top1_imp': top1_imp, 'top1_share': top1_share, 'price_rec': price_rec, 'price_ly': price_ly,
            'dhs_set': dhs_set, 'kr_name': kr_name, 'en_name': en_name,
            'display_name': kr_name or str(key), 'supplier_set': supplier_set, 'origin_set': origin_set,
            'dim_is_importer': (dim_col == importer_col),
        })

    A = pd.DataFrame(A_rows)
    if not A.empty:
        A = A[(A['ly_vol'] > 0) & (A['rec_vol'] >= floor)].copy()
    if not A.empty:
        A['yoy'] = (A['rec_vol'] - A['ly_vol']) / A['ly_vol'] * 100
        A['delta'] = A['rec_vol'] - A['ly_vol']
        A['share'] = A['rec_vol'] / tot_rec * 100 if tot_rec > 0 else 0
        A['low_base'] = (A['ly_vol'] > 0) & (A['ly_vol'] < floor * 0.5)
        A['single_ship'] = A['max_row'] >= A['rec_vol'] * 0.6
        A['price_yoy'] = A.apply(lambda r: (r['price_rec'] - r['price_ly']) / r['price_ly'] * 100
                                  if (r['price_rec'] is not None and r['price_ly']) else None, axis=1)
        A = A.sort_values('yoy', ascending=False)

        pos_a = A[A['yoy'] > 0].copy()
        YOY_SCORE_CAP = 300  # 저베이스 극단치(YoY 수만 %)가 정규화 척도 전체를 왜곡하는 걸 방지하기 위한 상한선.
        # (표시용 '+999%+' 캡과는 별개 — 이건 점수 계산에만 쓰이는 내부 상한)
        yoy_arr = pos_a['yoy'].clip(upper=YOY_SCORE_CAP).tolist()
        share_arr = pos_a['share'].tolist()
        ease_arr = [50.0 if r['dim_is_importer'] else 100 - r['top1_share'] for _, r in pos_a.iterrows()]
        price_arr = [0 if pd.isna(v) else v for v in pos_a['price_yoy'].tolist()] if has_price else []

        W = {
            'growth': {'growth': .5, 'size': .2, 'ease': .2, 'price': .1},
            'size': {'growth': .2, 'size': .45, 'ease': .15, 'price': .2},
            'easy': {'growth': .25, 'size': .15, 'ease': .45, 'price': .15},
        }.get(preset, {'growth': .5, 'size': .2, 'ease': .2, 'price': .1})

        def _score_a(row):
            if row['yoy'] <= 0:
                return 0
            g = _p8_minmax_norm(min(row['yoy'], YOY_SCORE_CAP), yoy_arr)
            s = _p8_minmax_norm(row['share'], share_arr)
            e = 50.0 if row['dim_is_importer'] else _p8_minmax_norm(100 - row['top1_share'], ease_arr)
            w = dict(W)
            p_comp = None
            if has_price:
                pv = 0 if pd.isna(row['price_yoy']) else row['price_yoy']
                p_comp = _p8_minmax_norm(pv, price_arr)
            else:
                s_sum = w['growth'] + w['size'] + w['ease']
                w = {'growth': w['growth'] / s_sum, 'size': w['size'] / s_sum, 'ease': w['ease'] / s_sum, 'price': 0}
            sc = g * w['growth'] + s * w['size'] + e * w['ease'] + (p_comp * w['price'] if p_comp is not None else 0)
            if row['low_base']:
                sc -= 10
            if row['single_ship']:
                sc -= 10
            return max(0, min(100, round(sc)))

        A['score'] = A.apply(_score_a, axis=1)
    else:
        A['yoy'] = []; A['score'] = []

    # ---------- Section B: 신규 진입 품목 ----------
    hist_norm_set = set(_p8_norm_name(v) for v in hist[new_key_col].dropna().unique()) if new_key_col in hist.columns else set()
    hist_dhs_set = set(_p8_norm_name(v) for v in hist[detailed_hs_col].dropna().unique()) if detailed_hs_col and detailed_hs_col in hist.columns else set()

    B_rows = []
    if new_key_col in recent.columns:
        recent_new = recent[~recent[new_key_col].apply(lambda v: _p8_norm_name(v) in hist_norm_set if pd.notna(v) else True)]
        recent_new = recent_new.dropna(subset=[new_key_col])
        for nk, g in recent_new.groupby(recent_new[new_key_col].apply(_p8_norm_name)):
            if not nk:
                continue
            vol = g['_volume'].sum()
            ship = len(g)
            max_row = g['_volume'].max()
            top1_imp, top1_share = None, 0.0
            if importer_col and importer_col in g.columns:
                imp_vol = g.groupby(importer_col)['_volume'].sum().sort_values(ascending=False)
                if len(imp_vol):
                    top1_imp, top1_share = imp_vol.index[0], (imp_vol.iloc[0] / vol * 100 if vol > 0 else 0)
            disp_raw = _p8_mode_or_none(g[new_key_col].tolist())
            dom_dhs = _p8_norm_name(_p8_mode_or_none(g[detailed_hs_col].tolist())) if detailed_hs_col else None
            suspect = (new_key_col == product_col) and bool(dom_dhs) and (dom_dhs in hist_dhs_set)
            single_ship = max_row >= vol * 0.6
            kr_name = _p8_mode_or_none(g[label_kr_col].tolist()) if label_kr_col else None
            origin_val = _p8_mode_or_none(g[origin_col].tolist()) if origin_col else None
            en_name = preprocess_product_name(str(disp_raw)) if disp_raw else nk
            B_rows.append({
                'k': disp_raw, 'vol': vol, 'ship': ship, 'origin': origin_val, 'top1_imp': top1_imp,
                'top1_share': top1_share, 'suspect': suspect, 'single_ship': single_ship,
                'kr_name': kr_name, 'en_name': en_name, 'display_name': kr_name or str(disp_raw),
                'dhs_set': set(g[detailed_hs_col].dropna().unique()) if detailed_hs_col else set(),
                'origin_set': set(g[origin_col].dropna().unique()) if origin_col else set(),
            })

    B = pd.DataFrame(B_rows)
    if not B.empty:
        B = B[(B['ship'] >= minship) & (B['vol'] >= floor)].copy()
    if not B.empty:
        B = B.sort_values('vol', ascending=False)
        vol_arr = B['vol'].tolist()
        ship_arr = B['ship'].tolist()
        ease_arr_b = (100 - B['top1_share']).tolist()

        def _score_b(row):
            s = _p8_minmax_norm(row['vol'], vol_arr)
            sh = _p8_minmax_norm(row['ship'], ship_arr)
            e = _p8_minmax_norm(100 - row['top1_share'], ease_arr_b)
            sc = s * .35 + sh * .35 + e * .30
            if row['single_ship']:
                sc -= 10
            if row['suspect']:
                sc -= 15
            return max(0, min(100, round(sc)))

        B['score'] = B.apply(_score_b, axis=1)

    # ---------- 확장 지표: CAGR / 수입사수 증감 / 원산지수 / 가격신호 / 적합도 ----------
    def _cagr_of(key):
        rows_k = df[df[dim_col] == key]
        if rows_k.empty:
            return None
        yb = {}
        for _, r in rows_k.iterrows():
            mo = (latest.year - r['_date'].year) * 12 + (latest.month - r['_date'].month)
            if mo < 0:
                continue
            yi = mo // 12
            yb[yi] = yb.get(yi, 0) + r['_volume']
        if not yb:
            return None
        cur = yb.get(0, 0)
        old_i = None
        for i in sorted([k for k in yb.keys() if k >= 1], reverse=True):
            if yb.get(i, 0) > 0:
                old_i = i
                break
        if old_i is None or cur <= 0 or yb.get(old_i, 0) <= 0:
            return None
        return (pow(cur / yb[old_i], 1 / old_i) - 1) * 100

    def _price_signal(yoy, price_yoy):
        if price_yoy is None or pd.isna(price_yoy):
            return '-'
        if yoy > 0 and price_yoy >= 0:
            return T('p8_signal_premium')
        if yoy > 0 and price_yoy < 0:
            return T('p8_signal_commodity')
        if yoy <= 0 and price_yoy > 0:
            return T('p8_signal_shock')
        return '-'

    if not A.empty:
        A['cagr'] = A['k'].apply(_cagr_of)
        if importer_col:
            imp_rec_map = recent.groupby(dim_col)[importer_col].apply(lambda s: set(s.dropna().unique()))
            imp_ly_map = lastyr.groupby(dim_col)[importer_col].apply(lambda s: set(s.dropna().unique()))
            A['imp_delta'] = A['k'].apply(lambda k: len(imp_rec_map.get(k, set())) - len(imp_ly_map.get(k, set())))
        else:
            A['imp_delta'] = None
        A['origin_n'] = A['origin_set'].apply(len)
        A['price_signal'] = A.apply(lambda r: _price_signal(r['yoy'], r['price_yoy']) if has_price else '-', axis=1)

        fit_results = A.apply(lambda r: _p8_fit_of(r['supplier_set'], r['origin_set'], f"{r['display_name']} {r['en_name']}", fit_ctx), axis=1)
        A['fit_score'] = [f[0] for f in fit_results]
        A['fit_why'] = [f[1] for f in fit_results]

        def _final_a(r):
            fs = r['score'] if r['fit_score'] is None else round(r['score'] * (1 - fitw) + r['fit_score'] * fitw)
            if r['price_signal'] == T('p8_signal_premium'):
                fs = max(0, min(100, fs + 4))
            elif r['price_signal'] == T('p8_signal_commodity'):
                fs = max(0, min(100, fs - 6))
            return fs
        A['final_score'] = A.apply(_final_a, axis=1)

    if not B.empty:
        B['cagr'] = None
        B['imp_delta'] = None
        B['origin_n'] = B['origin_set'].apply(len)
        B['price_signal'] = '-'
        fit_results_b = B.apply(lambda r: _p8_fit_of(None, r['origin_set'], f"{r['display_name']} {r['en_name']}", fit_ctx), axis=1)
        B['fit_score'] = [f[0] for f in fit_results_b]
        B['fit_why'] = [f[1] for f in fit_results_b]
        B['final_score'] = B.apply(lambda r: r['score'] if r['fit_score'] is None else round(r['score'] * (1 - fitw) + r['fit_score'] * fitw), axis=1)

    # ---------- 4분면 분류 ----------
    has_ctx = fit_ctx['has_ctx']
    pos_a_final = A[A['yoy'] > 0] if not A.empty else A
    pooled_scores = (pos_a_final['score'].tolist() if not pos_a_final.empty else []) + (B['score'].tolist() if not B.empty else [])
    pooled_fits = ([0 if pd.isna(x) or x is None else x for x in pos_a_final['fit_score'].tolist()] if not pos_a_final.empty else []) + \
                  ([0 if pd.isna(x) or x is None else x for x in B['fit_score'].tolist()] if not B.empty else [])

    def _median(arr):
        if not arr:
            return 50
        a = sorted(arr)
        return a[len(a) // 2]

    if has_ctx and pooled_scores:
        g_med = _median(pooled_scores)
        f_med = _median(pooled_fits)

        def _quadrant(score, fit_score):
            fv = 0 if fit_score is None or pd.isna(fit_score) else fit_score
            g_ok = score >= g_med
            f_ok = fv > 0 and fv >= f_med
            if g_ok and f_ok:
                return T('p8_quad_priority')
            if g_ok and not f_ok:
                return T('p8_quad_growth')
            if (not g_ok) and f_ok:
                return T('p8_quad_safe')
            return T('p8_quad_low')

        if not A.empty:
            A['quadrant'] = A.apply(lambda r: _quadrant(r['score'], r['fit_score']) if r['yoy'] > 0 else '-', axis=1)
        if not B.empty:
            B['quadrant'] = B.apply(lambda r: _quadrant(r['score'], r['fit_score']), axis=1)
    else:
        if not A.empty:
            A['quadrant'] = '-'
        if not B.empty:
            B['quadrant'] = '-'

    # ---------- 통합 추천 S ----------
    S = []
    if not A.empty:
        for _, r in A[A['yoy'] > 0].iterrows():
            S.append({'type': 'existing', 'name': r['display_name'], 'score': r['final_score'], 'row': r})
    if not B.empty:
        for _, r in B.iterrows():
            S.append({'type': 'new', 'name': r['display_name'], 'score': r['final_score'], 'row': r})
    S.sort(key=lambda x: x['score'], reverse=True)

    meta = {
        'rec_start': rec_start, 'latest': latest, 'ly_start': ly_start, 'ly_end': ly_end,
        'tot_rec': tot_rec, 'tot_ly': tot_ly, 'span_months': span_months, 'has_price': has_price,
        'has_ctx': has_ctx, 'months': months,
    }
    return A, B, S, meta


def _p8_fmt_pct(v):
    """비정상적으로 극단적인 %(작년 물량이 0에 가까울 때 수학적으로 폭발하는 값)를
    보기 좋게 상한선을 씌워 표시한다. 실제 값이 아니라 표시용 캡이며, 이런 경우는
    '저베이스 반등' 플래그로 별도 표시되므로 근거 문장에서는 신뢰도를 과장하지 않는다."""
    if v is None or pd.isna(v):
        return 'N/A'
    if abs(v) >= 999:
        return f"{'+' if v >= 0 else '-'}999%+"
    sign = '+' if v >= 0 else ''
    return f"{sign}{v:.0f}%"


def _p8_reason_a(r, dim_is_importer):
    parts = []
    parts.append(f"{T('p8_label_growth')} {_p8_fmt_pct(r['yoy'])}")
    if r.get('cagr') is not None and not pd.isna(r.get('cagr')):
        parts.append(f"{T('p8_label_cagr')} {_p8_fmt_pct(r['cagr'])}")
    parts.append(f"{T('p8_label_share')} {r['share']:.1f}%")
    if r.get('imp_delta') is not None and r['imp_delta'] != 0:
        sign3 = '+' if r['imp_delta'] >= 0 else ''
        parts.append(f"{T('p8_label_importer_count')} {sign3}{int(r['imp_delta'])}")
    if r.get('origin_n', 0) > 1:
        parts.append(f"{T('p8_label_origin_count')} {int(r['origin_n'])}{T('p8_label_origin_unit')}")
    if r.get('price_signal') and r['price_signal'] != '-':
        parts.append(r['price_signal'])
    if r.get('fit_score') is not None:
        parts.append(f"{T('p8_label_fit')} {r['fit_score']}({r['fit_why']})")
    if not dim_is_importer:
        top_name = str(r['top1_imp'])[:14] if r['top1_imp'] else '-'
        parts.append(f"{T('p8_label_top_importer')} {top_name} {r['top1_share']:.0f}%")
    return ' · '.join(parts)


def _p8_reason_b(r):
    parts = [f"{T('p8_label_recent')} {r['vol']:,.0f}", f"{T('p8_label_shipment')} {r['ship']}{T('p8_label_shipment_unit')}"]
    top_name = str(r['top1_imp'])[:14] if r['top1_imp'] else '-'
    parts.append(f"{T('p8_label_top_importer')} {top_name} {r['top1_share']:.0f}%")
    return ' · '.join(parts)


def _p8_flags_a(r):
    f = []
    if r.get('low_base'):
        f.append(T('p8_flag_low_base'))
    if r.get('single_ship'):
        f.append(T('p8_flag_single_ship'))
    return ' '.join(f) if f else '-'


def _p8_flags_b(r):
    f = []
    if r.get('suspect'):
        f.append(T('p8_flag_suspect'))
    if r.get('single_ship'):
        f.append(T('p8_flag_single_ship'))
    return ' '.join(f) if f else '-'


def load_uploaded_df(uploaded_file, raw_df_key, headers_key, fileid_key):
    """파일이 '새로' 업로드된 경우에만 실제로 파싱하고, 그 외에는 세션에 캐시된 걸 그대로 재사용한다.
    (이게 없으면 위젯을 하나 바꿀 때마다 Streamlit이 스크립트를 처음부터 다시 실행하면서
    매번 파일 전체를 재파싱해서 체감 속도가 크게 느려진다.)"""
    if uploaded_file is None:
        return st.session_state.get(raw_df_key)
    file_id = getattr(uploaded_file, 'file_id', None) or (uploaded_file.name, uploaded_file.size)
    if st.session_state.get(fileid_key) != file_id:
        raw_df = read_uploaded_table(uploaded_file)
        if raw_df is None:
            st.error(T('p1_file_read_fail_error'))
            st.stop()
        st.session_state[raw_df_key] = raw_df
        st.session_state[headers_key] = list(raw_df.columns)
        st.session_state[fileid_key] = file_id
    return st.session_state.get(raw_df_key)


def cluster_product_names(raw_names):
    """제품명을 전처리 후 정확히 일치하는 것끼리 묶어 '품목군' 사전을 만든다
    (같은 상품이라도 실제 수입신고 명칭이 조금씩 다르게 들어오는 문제 대응).
    반환: {대표명(그룹 내 가장 짧은 원본명): [해당 그룹에 속한 원본 품목명 리스트]}"""
    groups = {}
    for name in raw_names:
        key = preprocess_product_name(name) or name
        groups.setdefault(key, []).append(name)
    result = {}
    for key, names in groups.items():
        uniq = sorted(set(names))
        rep = min(uniq, key=len)
        result[rep] = uniq
    return result


def fig_to_png_bytes(fig, width=900, height=500, scale=2):
    """Plotly figure를 PNG 바이트로 변환 (PDF에 삽입하기 위함). kaleido 필요."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        return None


def _find_korean_font_path():
    """시스템에서 한글을 지원하는 .ttf 폰트 파일 경로를 찾는다 (없으면 None).
    matplotlib.font_manager는 자체 캐시가 낡아있을 수 있어 신뢰하지 않고, 파일시스템에서 직접 찾는다.
    reportlab의 TTFont는 .ttc(트루타입 컬렉션)를 지원하지 않으므로 .ttf 파일만 대상으로 한다."""
    import glob
    search_patterns = [
        '/usr/share/fonts/**/NanumGothic.ttf',
        '/usr/share/fonts/**/Nanum*.ttf',
        '/usr/share/fonts/**/NotoSansKR*.ttf',
        '/usr/share/fonts/**/NotoSansCJK*.ttf',
        '/usr/share/fonts/**/malgun.ttf',
        '/System/Library/Fonts/**/*.ttf',
    ]
    for pattern in search_patterns:
        matches = [m for m in glob.glob(pattern, recursive=True) if m.lower().endswith('.ttf')]
        if matches:
            return matches[0]
    return None


def _pdf_table_col_widths(rows_as_str, total_width, min_ratio=0.05, max_ratio=0.32):
    """표 각 컬럼의 폭을, 그 컬럼에 실제로 들어가는 글자 수 비례로 배분한다.
    (min/max 클램프를 걸어서 어떤 컬럼은 안 보일 만큼 좁아지거나, 다른 컬럼을 다 밀어낼 만큼
    넓어지는 걸 방지 — 긴 '근거' 같은 텍스트 컬럼은 대신 뒤에서 Paragraph로 줄바꿈 처리한다.)"""
    ncols = len(rows_as_str[0])
    max_lens = []
    for c in range(ncols):
        col_vals = [row[c] for row in rows_as_str]
        max_lens.append(max((len(v) for v in col_vals), default=1) or 1)
    total_len = sum(max_lens) or 1
    raw = [total_width * (l / total_len) for l in max_lens]
    min_w, max_w = total_width * min_ratio, total_width * max_ratio
    clamped = [max(min_w, min(max_w, w)) for w in raw]
    scale = total_width / sum(clamped)
    return [w * scale for w in clamped]


def build_pdf_report(title, kpi_lines, figs, df_table=None, table_title=None):
    """제목 + KPI 텍스트 + Plotly 차트(이미지로 변환) + 표를 하나의 PDF로 조립해 바이트로 반환.
    reportlab + kaleido 필요."""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=15 * mm, bottomMargin=15 * mm, leftMargin=15 * mm, rightMargin=15 * mm)
    styles = getSampleStyleSheet()

    font_name = 'Helvetica'
    try:
        found_path = _find_korean_font_path()
        if found_path:
            pdfmetrics.registerFont(TTFont('KoreanFont', found_path))
            font_name = 'KoreanFont'
    except Exception:
        pass
    for s in styles.byName.values():
        s.fontName = font_name

    story = [Paragraph(title, styles['Title']), Spacer(1, 6)]
    for line in kpi_lines:
        story.append(Paragraph(line, styles['Normal']))
    story.append(Spacer(1, 10))

    for fig in figs:
        img_bytes = fig_to_png_bytes(fig)
        if img_bytes:
            story.append(RLImage(io.BytesIO(img_bytes), width=180 * mm, height=180 * mm * 500 / 900))
            story.append(Spacer(1, 8))

    if df_table is not None and not df_table.empty:
        if table_title:
            story.append(Paragraph(table_title, styles['Heading3']))

        display_table = df_table.copy()
        for c in display_table.columns:
            display_table[c] = display_table[c].astype(str)

        header_style = styles['Normal'].clone('PdfTableHeader')
        header_style.textColor = colors.white
        header_style.fontSize = 7.5
        header_style.leading = 9.5
        cell_style = styles['Normal'].clone('PdfTableCell')
        cell_style.fontSize = 7.5
        cell_style.leading = 9.5

        headers = [str(c) for c in display_table.columns]
        rows_as_str = [headers] + display_table.values.tolist()
        total_width = doc.width  # 페이지 여백을 뺀 실제 사용 가능 폭
        col_widths = _pdf_table_col_widths(rows_as_str, total_width)

        data = [[Paragraph(h, header_style) for h in headers]]
        for row in display_table.values.tolist():
            data.append([Paragraph(v, cell_style) for v in row])

        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d9488')),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f6f8')]),
        ]))
        story.append(t)

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# 신규사업 스코어러 전용 — "리포트" PDF (원본 buildReport() 100% 이식)
# 범용 build_pdf_report()보다 훨씬 상세한, 보고용 리포트 포맷
# ============================================================
def _p8_ton(v):
    if v is None or pd.isna(v):
        return '-'
    return f"{v / 1000:,.1f}t"


def _p8_split_signed_pct(v, cap=999):
    """부호와 숫자를 분리해서 반환 (극단치는 상한선 캡). 템플릿의 {sign}{pct}%에 그대로 끼워 넣는다."""
    if v is None or pd.isna(v):
        return '', 'N/A'
    if abs(v) >= cap:
        return ('+' if v >= 0 else '-'), f'{cap}+'
    return ('+' if v >= 0 else ''), f'{v:.0f}'


def _p8_build_insights(A, B, S, meta):
    lang = st.session_state.lang
    ins = []
    tot_yoy = ((meta['tot_rec'] - meta['tot_ly']) / meta['tot_ly'] * 100) if meta['tot_ly'] > 0 else 0
    months = meta['months']
    if months >= 12 and months % 12 == 0:
        yrs = months // 12
        month_label = f"{yrs}년" if lang == 'ko' else f"{yrs} year(s)"
    else:
        month_label = f"{months}개월" if lang == 'ko' else f"{months} months"
    sign, pct = _p8_split_signed_pct(tot_yoy)
    ins.append(T('p8_ins_total', month_label=month_label, sign=sign, pct=pct, ly=_p8_ton(meta['tot_ly']), rec=_p8_ton(meta['tot_rec'])))

    if S:
        top = S[0]
        type_label = T('p8_type_existing') if top['type'] == 'existing' else T('p8_type_new')
        ins.append(T('p8_ins_top_candidate', name=top['name'], type=type_label, score=top['score']))

    if not A.empty:
        pos_a = A[A['yoy'] > 0]
        if not pos_a.empty:
            top_a = pos_a.loc[pos_a['yoy'].idxmax()]
            sign2, pct2 = _p8_split_signed_pct(top_a['yoy'])
            ins.append(T('p8_ins_top_growth', name=top_a['display_name'], sign=sign2, pct=pct2))

    if not B.empty:
        top_b = B.iloc[0]
        origin_part = T('p8_ins_top_new_origin_part', origin=top_b['origin']) if top_b.get('origin') else ''
        ins.append(T('p8_ins_top_new', name=top_b['display_name'], vol=_p8_ton(top_b['vol']), origin_part=origin_part))

    if meta.get('has_price') and not A.empty and 'price_yoy' in A.columns:
        priced = A.dropna(subset=['price_yoy'])
        if not priced.empty:
            top_p = priced.loc[priced['price_yoy'].idxmax()]
            sign3, pct3 = _p8_split_signed_pct(top_p['price_yoy'])
            ins.append(T('p8_ins_price_up', name=top_p['display_name'], sign=sign3, pct=pct3))

    if meta.get('has_ctx'):
        prio_a = A[A.get('quadrant', pd.Series(dtype=object)) == T('p8_quad_priority')] if not A.empty and 'quadrant' in A.columns else pd.DataFrame()
        prio_b = B[B.get('quadrant', pd.Series(dtype=object)) == T('p8_quad_priority')] if not B.empty and 'quadrant' in B.columns else pd.DataFrame()
        prio_count = len(prio_a) + len(prio_b)
        rep_part = ''
        combined_prio = pd.concat([prio_a, prio_b]) if (not prio_a.empty or not prio_b.empty) else pd.DataFrame()
        if not combined_prio.empty:
            best = combined_prio.loc[combined_prio['fit_score'].fillna(0).idxmax()]
            rep_part = T('p8_ins_fit_rep_part', name=best['display_name'], fit=best['fit_score'])
        ins.append(T('p8_ins_fit', n=prio_count, rep_part=rep_part))

    return ins


def build_scorer_report_pdf(A, B, S, meta, dim_label):
    """신규사업 스코어러 전용 상세 리포트 PDF — 핵심 인사이트, TOP10, 품목별 서술형 상세까지 포함."""
    import io
    import datetime as _dt
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    lang = st.session_state.lang
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=15 * mm, bottomMargin=15 * mm, leftMargin=15 * mm, rightMargin=15 * mm)
    styles = getSampleStyleSheet()

    font_name = 'Helvetica'
    try:
        found_path = _find_korean_font_path()
        if found_path:
            pdfmetrics.registerFont(TTFont('KoreanFont', found_path))
            font_name = 'KoreanFont'
    except Exception:
        pass
    for s in styles.byName.values():
        s.fontName = font_name

    normal = styles['Normal']
    normal.fontSize = 9
    normal.leading = 13
    small = normal.clone('Small')
    small.fontSize = 7.5
    small.leading = 10
    small.textColor = colors.HexColor('#5c554a')
    header_cell = normal.clone('HeaderCell')
    header_cell.fontSize = 7.5
    header_cell.textColor = colors.white
    body_cell = normal.clone('BodyCell')
    body_cell.fontSize = 7.5
    body_cell.leading = 10
    narrative_cell = normal.clone('NarrativeCell')
    narrative_cell.fontSize = 7.5
    narrative_cell.leading = 10
    narrative_cell.textColor = colors.HexColor('#5c554a')

    story = [Paragraph(T('p8_report_subtitle'), small), Paragraph(T('p8_report_title'), styles['Title'])]
    today = _dt.date.today().isoformat()
    months = meta['months']
    month_label = f"{months}개월" if lang == 'ko' else f"{months} months"
    story.append(Paragraph(T('p8_report_meta',
        start=meta['rec_start'].strftime('%Y-%m'), end=meta['latest'].strftime('%Y-%m'),
        ly_start=meta['ly_start'].strftime('%Y-%m'), ly_end=meta['ly_end'].strftime('%Y-%m'),
        months=month_label, today=today), small))
    story.append(Spacer(1, 10))

    # KPI
    tot_yoy = ((meta['tot_rec'] - meta['tot_ly']) / meta['tot_ly'] * 100) if meta['tot_ly'] > 0 else 0
    a_growing = A[A['yoy'] > 0] if not A.empty else A
    kpi_data = [
        [Paragraph(T('p8_kpi_total_yoy'), body_cell), Paragraph(T('p8_kpi_growing'), body_cell), Paragraph(T('p8_kpi_new'), body_cell)],
        [Paragraph(f"{tot_yoy:+.0f}%", styles['Heading2']), Paragraph(str(len(a_growing)), styles['Heading2']), Paragraph(str(len(B)), styles['Heading2'])],
    ]
    kpi_t = Table(kpi_data, colWidths=[doc.width / 3] * 3)
    kpi_t.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#ddd4c4')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ddd4c4')),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#faf7f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_t)
    story.append(Spacer(1, 10))

    # 핵심 인사이트
    insights = _p8_build_insights(A, B, S, meta)
    ins_block = [Paragraph(T('p8_report_insights_header'), styles['Heading3'])]
    for line in insights:
        ins_block.append(Paragraph('• ' + line, normal))
    story.append(KeepTogether(ins_block))
    story.append(Spacer(1, 6))

    story.append(Paragraph(T('p8_report_data_summary',
        n_tx=f"{meta.get('n_tx', 0):,}", span=meta.get('span_str', ''),
        n_imp=f"{meta.get('n_imp', 0):,}", n_prod=f"{meta.get('n_prod', 0):,}", n_origin=f"{meta.get('n_origin', 0):,}"), small))
    story.append(Spacer(1, 10))

    # TOP 10
    story.append(Paragraph(T('p8_report_top10_header'), styles['Heading2']))
    story.append(Paragraph(T('p8_report_top10_sub'), small))
    S_top = S[:10]
    has_ctx = meta.get('has_ctx')
    if S_top:
        names = [s['name'] for s in S_top][::-1]
        scores = [s['score'] for s in S_top][::-1]
        bar_colors = ['#e11d48' if s['type'] == 'new' else '#0d9488' for s in S_top][::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=names, x=scores, orientation='h', marker_color=bar_colors))
        chart_h = max(220, 32 * len(S_top))
        fig.update_layout(margin=dict(l=160, r=30, t=10, b=30), height=chart_h, font=dict(size=13))
        img_bytes = fig_to_png_bytes(fig, width=800, height=chart_h)
        if img_bytes:
            story.append(RLImage(io.BytesIO(img_bytes), width=170 * mm, height=170 * mm * chart_h / 800))
            story.append(Spacer(1, 4))

        headers = ['#', T('p8_col_type'), T('p8_col_item'), T('p8_col_score')]
        if has_ctx:
            headers += [T('p8_col_fit'), T('p8_col_quadrant')]
        headers += [T('p8_col_reason')]
        rows_raw = [headers]
        for i, s in enumerate(S_top):
            r = s['row']
            reason = _p8_reason_a(r, r.get('dim_is_importer', False)) if s['type'] == 'existing' else _p8_reason_b(r)
            row = [str(i + 1), T('p8_type_existing') if s['type'] == 'existing' else T('p8_type_new'), s['name'], str(s['score'])]
            if has_ctx:
                row += [str(r['fit_score']) if r.get('fit_score') is not None else '-', r.get('quadrant', '-')]
            row += [reason]
            rows_raw.append(row)
        col_widths = _pdf_table_col_widths(rows_raw, doc.width)
        data = [[Paragraph(h, header_cell) for h in rows_raw[0]]] + [[Paragraph(v, body_cell) for v in row] for row in rows_raw[1:]]
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#17140f')),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f6f8')]),
        ]))
        story.append(t)
    story.append(Spacer(1, 14))

    # Section A 상세
    if not a_growing.empty:
        # 저베이스(작년 물량이 거의 0) 항목은 점유율과 무관하게 상세 목록에서 제외한다
        # (감점만으로는 리스트가 노이즈로 채워지는 걸 못 막았고, '상위 15'는 진짜 볼만한 기회만 보여줘야 함)
        meaningful_a = a_growing[~a_growing['low_base']]
        excluded_n = len(a_growing) - len(meaningful_a)
        a_top = meaningful_a.head(15)
        headers_a = ['#', T('p8_col_item'), T('p8_col_score'), T('p8_col_yoy'), T('p8_col_rec_vol'), T('p8_col_ly_vol'), T('p8_col_concentration'), T('p8_report_note_col')]
        rows_raw_a = [headers_a]
        for i, (_, r) in enumerate(a_top.iterrows()):
            big = bool(meta['tot_rec']) and r['rec_vol'] >= meta['tot_rec'] * 0.1
            note = T('p8_detail_big_tag').strip() if big else '-'
            rows_raw_a.append([str(i + 1), r['display_name'], str(r['final_score']), _p8_fmt_pct(r['yoy']),
                                _p8_ton(r['rec_vol']), _p8_ton(r['ly_vol']), f"{r['share']:.1f}%", note])
        col_widths_a = _pdf_table_col_widths(rows_raw_a, doc.width)
        data_a = [[Paragraph(h, header_cell) for h in rows_raw_a[0]]] + [[Paragraph(v, body_cell) for v in row] for row in rows_raw_a[1:]]
        t_a = Table(data_a, colWidths=col_widths_a, repeatRows=1)
        t_a.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d9488')),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f6f8')]),
        ]))
        excluded_note = [Paragraph(T('p8_report_excluded_note_a', n=excluded_n), small)] if excluded_n > 0 else []
        story.append(KeepTogether([
            Paragraph(T('p8_report_section_a_header', n=len(a_top)), styles['Heading2']),
            Paragraph(T('p8_report_section_a_sub'), small),
            Spacer(1, 4), t_a,
        ] + excluded_note))
        story.append(Spacer(1, 14))

    # Section B 상세
    if not B.empty:
        # 표기변형 의심(사실상 기존 품목이 이름만 바뀐 것) 항목은 무조건 제외
        meaningful_b = B[~B['suspect']]
        excluded_n_b = len(B) - len(meaningful_b)
        b_top = meaningful_b.head(15)
        headers_b = ['#', T('p8_col_item'), T('p8_col_score'), T('p8_col_vol'), T('p8_col_ship'), T('p8_col_top_importer')]
        rows_raw_b = [headers_b]
        for i, (_, r) in enumerate(b_top.iterrows()):
            top_imp_disp = f"{r['top1_imp']} {r['top1_share']:.0f}%" if r['top1_imp'] else '-'
            rows_raw_b.append([str(i + 1), r['display_name'], str(r['final_score']), _p8_ton(r['vol']), f"{r['ship']}회" if lang == 'ko' else str(r['ship']), top_imp_disp])
        col_widths_b = _pdf_table_col_widths(rows_raw_b, doc.width)
        data_b = [[Paragraph(h, header_cell) for h in rows_raw_b[0]]] + [[Paragraph(v, body_cell) for v in row] for row in rows_raw_b[1:]]
        t_b = Table(data_b, colWidths=col_widths_b, repeatRows=1)
        t_b.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e11d48')),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f6f8')]),
        ]))
        excluded_note_b = [Paragraph(T('p8_report_excluded_note_b', n=excluded_n_b), small)] if excluded_n_b > 0 else []
        story.append(KeepTogether([
            Paragraph(T('p8_report_section_b_header', n=len(b_top)), styles['Heading2']),
            Paragraph(T('p8_report_section_b_sub'), small),
            Spacer(1, 4), t_b,
        ] + excluded_note_b))
        story.append(Spacer(1, 14))

    story.append(Paragraph(T('p8_report_footer'), small))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# --------------------------------#
# 사용법 가이드 PDF (사이드바에서 다운로드)  #
# --------------------------------#
GUIDE_CONTENT = {
    'ko': {
        'doc_title': '소싱 분석기 사용 가이드',
        'intro': '이 문서는 앱의 각 기능을 처음 쓰시는 분도 화면을 보면서 그대로 따라 할 수 있도록 정리한 가이드입니다. 왼쪽 사이드바의 "📖 사용법 가이드" 버튼을 누르면 언제든 최신 버전을 다시 받으실 수 있습니다.',
        'label_steps': '따라하기',
        'label_results': '결과 해석',
        'sections': [
            {
                'heading': '🔒 로그인',
                'intro': '이 앱을 처음 실행하면 비밀번호를 직접 설정하는 화면이 나옵니다. 이후 접속할 때부터는 방금 설정한 비밀번호로 로그인하면 됩니다.',
                'steps': [
                    '새 비밀번호 입력란과 확인란에 동일한 비밀번호를 입력합니다.',
                    '"비밀번호 설정하고 시작하기" 버튼을 누르면 바로 앱으로 들어갑니다.',
                    '다음 접속부터는 "🔒 접근 제한" 화면에서 방금 정한 비밀번호를 입력하면 됩니다.',
                ],
                'tips': ['비밀번호는 별도 서버가 아니라 이 앱이 실행되는 환경(Codespace 등)에 저장되므로, 완전히 새로운 환경에서 처음 실행하면 다시 설정 화면이 나옵니다.'],
            },
            {
                'heading': '🌐 언어 전환 & 공통 파일 업로드 규칙',
                'intro': '왼쪽 사이드바 상단의 "한국어 / English" 버튼으로 화면 전체(제목, 버튼, 차트, 표 컬럼명까지) 언어가 즉시 전환됩니다.',
                'results': [
                    '모든 메뉴는 CSV 또는 XLSX 파일을 업로드하면서 시작합니다.',
                    '컬럼명은 정확히 같지 않아도 자동으로 인식됩니다 (예: "Date", "date", "거래일자" 모두 인식). 다만 날짜·물량·단가에 해당하는 컬럼은 반드시 있어야 합니다.',
                    'HS코드, Incoterm처럼 표준 항목 외의 컬럼도 파일에 있으면 자동으로 인식되어, 일부 메뉴(공급망 흐름도·집중도 진단·신규 이탈 거래처·피벗 빌더·신규사업 스코어러)의 축/필터 선택지에 그대로 추가됩니다.',
                ],
            },
            {
                'heading': '💲 ① 고객사 효율 분석',
                'image': 'p1_customer.png',
                'intro': '계약을 시작한 뒤로 우리(또는 특정 수입사)의 구매 단가가 실제로 얼마나 절감됐는지 숫자로 확인하고 싶을 때 씁니다. 계약 갱신이나 성과 보고 자료를 만들 때 가장 먼저 여는 화면이에요.',
                'steps': [
                    '거래 내역 파일을 업로드합니다 (여러 회사 데이터가 섞여 있어도 자동으로 가장 많이 등장하는 회사를 기준으로 분석합니다).',
                    '계약 시작일을 선택합니다.',
                    '"분석 실행" 버튼을 클릭합니다.',
                ],
                'results': [
                    '총 예상 절감액 — 계약일 전후로 실제 구매 단가 변화에 따른 절감 규모.',
                    '품목군별 카드 — 어떤 품목에서 얼마나 절감(또는 손해)됐는지 개별 확인.',
                    '수입 품목 클러스터 — 표기가 조금씩 다른 품목명을 자동으로 묶어서 정리.',
                    '월별 수입 추이, 최근 3개월 비중 — 최근 구매 패턴 파악.',
                    '계약 이후 새로 생기거나 사라진 품목·원산지·공급사 목록.',
                ],
            },
            {
                'heading': '🏆 ② 시장 경쟁력 분석',
                'image': 'p2_market.png',
                'intro': '특정 품목을 두고 여러 수입사가 같은 시장에서 경쟁하고 있을 때, 우리 회사가 그중 얼마나 잘 사고 있는지(=구매 경쟁력)를 확인하고 싶을 때 씁니다.',
                'steps': [
                    '한 품목에 대해 여러 회사의 거래 데이터가 섞인 시장 데이터를 업로드합니다.',
                    '분석할 고객사, 품목명, 계약(기준)일을 입력합니다.',
                    '필요하면 이상치 제거 민감도를 조절합니다.',
                    '"시장 경쟁력 분석 시작"을 클릭합니다.',
                ],
                'results': [
                    '구매 경쟁력 순위 — 시장 내 몇 위인지.',
                    '경쟁 우위 그룹(나보다 잘 사는 상위 그룹) 대비 단가 추이.',
                    '벤치마킹 시뮬레이션 — 그 그룹만큼만 샀어도 얼마를 아꼈을지.',
                    '시장 점유율, 원산지(수출국) 필터.',
                    '공급망(공급사·원산지)별 더 저렴한 대안 소싱 옵션.',
                ],
            },
            {
                'heading': '🔀 ③ 공급망 흐름도 (Sankey)',
                'image': 'p3_flow.png',
                'intro': '특정 공급사·수입사·원산지 "하나"를 골랐을 때, 그 대상이 정확히 어디서 와서 어디로 흘러가는지 시각적으로 보고 싶을 때 씁니다. 예: "이 공급사는 어느 나라로 얼마나 수출하고 있지?"',
                'steps': [
                    '전체 시장 데이터를 업로드합니다.',
                    '왼쪽 축(기준 구분 + 대상)을 고릅니다 — 예: 공급사 → 특정 회사명.',
                    '오른쪽 축(비교 기준)을 고릅니다 — 예: 수출대상국.',
                    '기간을 설정하고 "흐름도 그리기"를 클릭합니다.',
                ],
                'results': [
                    'Sankey 다이어그램 — 굵기가 물량 비중을 의미합니다.',
                    '정렬된 상세 표 — 물량, 비중, 평균 단가.',
                    '"두 기간 비교하기"를 켜면 기간 A/B 흐름도를 나란히 놓고, 어느 흐름이 늘고 줄었는지 증감표까지 확인할 수 있습니다.',
                ],
                'tips': ['"원산지"는 물건이 생산된 나라, "수출대상국"은 물건이 팔려나가는 나라로 서로 다른 개념입니다 — 헷갈리기 쉬우니 화면의 안내 문구를 확인하세요.'],
            },
            {
                'heading': '⚠️ ④ 집중도 리스크 진단',
                'image': 'p4_risk.png',
                'intro': '특정 공급사·원산지·품목 "하나"에 거래가 너무 쏠려있어서 위험하지는 않은지 자동으로 점검하고 싶을 때 씁니다. 공급망 리스크 관리 관점에서 정기적으로 확인하면 좋습니다.',
                'steps': [
                    '전체 시장 데이터를 업로드합니다.',
                    '진단 기준(공급사/원산지/품목별)을 고릅니다.',
                    '범위(전체 또는 특정 수입사만)를 고릅니다.',
                    '위험 기준선(%)을 필요하면 조절하고 "진단 실행"을 클릭합니다.',
                ],
                'results': [
                    '1위 비중, 상위 3개 합산 비중.',
                    '위험도 배지(안전/주의/위험)와 그 이유 — 기준선을 왜 넘었는지 문장으로 설명.',
                    '월별 1위 비중 추이 — 쏠림이 점점 심해지는지 확인.',
                ],
            },
            {
                'heading': '📈 ⑤ 가격 추세 & 계절성',
                'image': 'p5_season.png',
                'intro': '특정 품목의 단가가 시기별로 어떻게 움직이는지, 언제 사는 게 유리한지 알고 싶을 때 씁니다. 구매 시점을 계획할 때 참고하세요.',
                'steps': [
                    '전체 시장 데이터를 업로드합니다.',
                    '품목을 검색해 하나 이상 선택합니다 (표기가 비슷한 품목명은 자동으로 묶여서 후보에 뜹니다).',
                    '전년 비교 여부와 비교 축(원산지별/공급사별)을 설정합니다.',
                    '"추세 그리기"를 클릭합니다.',
                ],
                'results': [
                    '최근월 평균단가, 전년 동월 대비, 연중 최고가 대비 현재 위치.',
                    '계절 고점 자동 감지 — 매년 특정 시기에 비싸지는 패턴이 있는지.',
                    '"○월에 사면 가장 저렴했다" 같은 인사이트 문장이 자동 생성됩니다.',
                ],
                'tips': ['데이터가 1년치뿐이면 "이 계절 패턴은 우연일 수 있다"는 경고가 함께 표시됩니다 — 2년 이상 데이터가 쌓이면 신뢰도가 높아집니다.'],
            },
            {
                'heading': '🔀 ⑥ 신규·이탈 거래처 추적',
                'image': 'p6_churn.png',
                'intro': '임의의 두 기간을 비교해서, 그 사이에 새로 생기거나 사라진 거래처(공급사/원산지/품목/수입사)를 찾고 싶을 때 씁니다. "우리가 최근에 거래처를 바꿨는데 그게 잘한 선택이었나?"에 답합니다.',
                'steps': [
                    '전체 시장 데이터를 업로드합니다.',
                    '추적 기준(공급사/원산지/품목/수입사)을 고릅니다.',
                    '범위(전체 시장 또는 특정 수입사)를 고릅니다.',
                    '기간 A(비교 대상)와 기간 B(기준)를 설정하고 "비교 실행"을 클릭합니다.',
                ],
                'results': [
                    '신규/유지/이탈 개수와 각각의 목록(물량 포함).',
                    '💰 가격으로 보면 — 신규·유지·이탈 거래처의 평균단가를 비교해서, 이번 변화가 비용 절감으로 이어졌는지 알려줍니다.',
                    '⚠️ 집중도 변화 — 거래처가 줄면서 특정 공급사 의존도가 위험하게 높아졌는지 자동으로 경고합니다.',
                ],
            },
            {
                'heading': '🔬 ⑦ 신규사업 스코어러',
                'image': 'p7_scorer.png',
                'intro': '시장 전체 데이터에서 "최근 급상승한 기존 품목"과 "완전히 새로 등장한 품목"을 찾아, 새로운 소싱 기회의 우선순위를 매기고 싶을 때 씁니다. 다른 7개 메뉴가 "우리가 이미 하는 거래"를 들여다본다면, 이 메뉴는 "아직 안 하고 있지만 해볼 만한 것"을 찾아줍니다.',
                'steps': [
                    'HS코드명/Detailed HS-CODE 등이 포함된 전체 시장 데이터를 업로드합니다.',
                    '분석 기간, 최소 물량 기준, 분석 기준(HS코드명/품목명/수입사 등)을 설정합니다.',
                    '(선택) 우리 회사의 기존 공급사·원산지·취급 키워드를 입력하면, "우리 사업과 얼마나 잘 맞는지"까지 점수화됩니다.',
                    '"분석 실행"을 클릭합니다.',
                ],
                'results': [
                    '통합 추천 순위(TOP 10) — 기존 급상승 품목과 신규 진입 품목을 하나의 점수로 비교한 최종 순위.',
                    '적합도 기준을 입력했다면, 4분면(우선 타깃/성장 주도/안전 인접/후순위)까지 자동 분류됩니다.',
                    '⚠ 저베이스 반등, ⚠ 단일선적 편중, ⚠ 표기변형 의심 같은 주의 플래그로 착시 데이터를 걸러줍니다.',
                ],
                'tips': ['적합도 기준을 안 넣어도 분석 자체는 되지만, 넣으면 "우리 사업 맥락에서" 우선순위가 훨씬 정교해집니다.'],
            },
            {
                'heading': '🧩 ⑧ 자유 피벗 빌더',
                'image': 'p8_pivot.png',
                'intro': '정해진 화면으로는 답이 안 나올 때, 엑셀 피벗테이블처럼 원하는 축과 지표를 직접 조합해서 나만의 표·차트를 만들고 싶을 때 씁니다.',
                'steps': [
                    '데이터를 업로드합니다.',
                    '행(기준)을 하나 이상 고릅니다 — 여러 개 고르면 고른 순서대로 중첩됩니다.',
                    '열(나눠보기)을 선택합니다 (선택 사항).',
                    '값(지표)을 하나 이상 고릅니다 — 물량 합계/평균, 단가 평균/최대/최소, 거래건수 중 복수 선택 가능.',
                    '보기(표/히트맵/막대/선/파이/누적막대)를 고르고, 필요하면 필터를 걸어서 "피벗 생성"을 클릭합니다.',
                ],
                'results': [
                    '고른 조합대로 즉시 표와 차트가 생성됩니다.',
                    '지표를 여러 개 고르면 표·히트맵에는 전부 표시되지만, 막대·선·파이 차트는 첫 번째로 고른 지표만 그려집니다 (단위가 다른 지표를 한 차트에 억지로 섞으면 오히려 헷갈리기 때문입니다).',
                ],
            },
            {
                'heading': '📄 공통 — PDF 다운로드',
                'intro': '각 분석 결과 화면 하단의 "📄 PDF 보고서 생성" 버튼을 누르면, 해당 화면의 차트와 표가 담긴 PDF가 만들어지고 "📥 PDF 다운로드" 버튼으로 저장할 수 있습니다.',
                'tips': ['지금 읽고 계신 이 가이드 자체도 사이드바의 "📖 사용법 가이드" 버튼으로 언제든 다시 받을 수 있습니다.'],
            },
        ],
    },
    'en': {
        'doc_title': 'Sourcing Analyzer User Guide',
        'intro': 'This guide walks first-time users through every feature step by step. You can re-download the latest version anytime via the "📖 User Guide" button in the sidebar.',
        'label_steps': 'Steps',
        'label_results': 'What the results mean',
        'sections': [
            {
                'heading': '🔒 Login',
                'intro': 'The first time you run this app, you will be asked to set a password yourself. From then on, log in with that same password.',
                'steps': [
                    'Enter the same password in both the "new password" and "confirm" fields.',
                    'Click "Set password and start" to enter the app immediately.',
                    'On future visits, enter that password on the "🔒 Restricted Access" screen.',
                ],
                'tips': ['The password is stored in the environment the app runs in (e.g. the Codespace), not on an external server — a brand-new environment will show the setup screen again.'],
            },
            {
                'heading': '🌐 Language Toggle & Common File Upload Rules',
                'intro': 'Use the "한국어 / English" buttons at the top of the sidebar to switch the entire interface — titles, buttons, charts, and even table column names — instantly.',
                'results': [
                    'Every menu starts by uploading a CSV or XLSX file.',
                    'Column names do not need to match exactly (e.g. "Date", "date" are both recognized), but date, volume, and unit price columns must be present.',
                    'Columns beyond the standard set (e.g. HS Code, Incoterm) are auto-detected and added as axis/filter options in several menus (Supply Chain Flow, Concentration Risk, New/Lost Partners, Pivot Builder, New Business Scorer).',
                ],
            },
            {
                'heading': '💲 ① Customer Efficiency Analysis',
                'image': 'p1_customer.png',
                'intro': "Use this when you want to prove, in numbers, how much a company's purchase price dropped after a contract started. It's usually the first screen opened for renewal or performance reporting.",
                'steps': [
                    'Upload transaction data (if multiple companies are mixed in, the most frequent one is used automatically).',
                    'Select the contract start date.',
                    'Click "Run Analysis".',
                ],
                'results': [
                    'Total estimated savings before vs. after the contract.',
                    'Savings by product group.',
                    'Import product clusters — similarly-worded product names grouped automatically.',
                    'Monthly import trend and last-3-month share.',
                    'New/lost products, origins, and suppliers since the contract.',
                ],
            },
            {
                'heading': '🏆 ② Market Competitiveness Analysis',
                'image': 'p2_market.png',
                'intro': 'Use this when several importers compete for the same product, and you want to see how competitively your company is buying.',
                'steps': [
                    "Upload market data with multiple companies' transactions for one product.",
                    'Enter the customer, product name, and reference (contract) date.',
                    'Adjust outlier sensitivity if needed.',
                    'Click "Start Market Competitiveness Analysis".',
                ],
                'results': [
                    'Purchasing competitiveness ranking.',
                    'Price trend vs. the competitive advantage group.',
                    'Benchmarking simulation — potential savings if you matched that group.',
                    'Market share and an origin (export country) filter.',
                    'Cheaper alternative sourcing options by supplier/origin.',
                ],
            },
            {
                'heading': '🔀 ③ Supply Chain Flow (Sankey)',
                'image': 'p3_flow.png',
                'intro': 'Use this to see exactly where a single supplier, importer, or origin flows from and to. E.g. "Where does this supplier export to, and how much?"',
                'steps': [
                    'Upload the full market data.',
                    'Choose the left axis (type + entity), e.g. Supplier → a specific company.',
                    'Choose the right axis (compare by), e.g. export destination.',
                    'Set the date range and click "Draw flow diagram".',
                ],
                'results': [
                    'A Sankey diagram — ribbon width represents volume share.',
                    'A sorted detail table with volume, share, and average price.',
                    'Enabling "Compare two periods" shows Period A/B side by side plus a change table.',
                ],
                'tips': ['"Origin" is where the product was produced; "Export destination" is where it was sold — easy to mix up, so check the on-screen hint.'],
            },
            {
                'heading': '⚠️ ④ Concentration Risk Diagnosis',
                'image': 'p4_risk.png',
                'intro': 'Use this to automatically check whether your trade is dangerously concentrated in a single supplier, origin, or product. Good to check periodically for supply chain risk management.',
                'steps': [
                    'Upload the full market data.',
                    'Choose the diagnosis axis (supplier/origin/product).',
                    'Choose the scope (entire market or one importer).',
                    'Adjust the risk threshold if needed and click "Run diagnosis".',
                ],
                'results': [
                    'Top-1 share and top-3 combined share.',
                    'A risk badge (safe/caution/high) with a plain-language reason.',
                    'Monthly trend of top-1 share, to see if concentration is worsening.',
                ],
            },
            {
                'heading': '📈 ⑤ Price Trend & Seasonality',
                'image': 'p5_season.png',
                'intro': "Use this to see how a product's price moves over time and when it is cheapest to buy — useful for planning purchase timing.",
                'steps': [
                    'Upload the full market data.',
                    'Search and select one or more products (similarly-worded variants are grouped automatically as candidates).',
                    'Set the previous-year overlay and comparison axis.',
                    'Click "Draw trend".',
                ],
                'results': [
                    "Latest month's average price, year-over-year change, and position vs. the yearly peak.",
                    'Automatic seasonal-peak detection.',
                    'Auto-generated insight sentences like "buying in March was historically cheapest."',
                ],
                'tips': ['With only one year of data, a warning notes the seasonal pattern could be coincidental — confidence improves with 2+ years of history.'],
            },
            {
                'heading': '🔀 ⑥ New/Lost Trading Partners',
                'image': 'p6_churn.png',
                'intro': 'Use this to compare any two periods and find trading partners (supplier/origin/product/importer) that appeared or disappeared — answering "did switching partners actually pay off?"',
                'steps': [
                    'Upload the full market data.',
                    'Choose what to track (supplier/origin/product/importer).',
                    'Choose the scope (entire market or one importer).',
                    'Set Period A (comparison) and Period B (baseline), then click "Compare".',
                ],
                'results': [
                    'Counts and lists of new/kept/lost partners, with volume.',
                    '💰 Looking at price — compares average prices across the three groups to show whether the change saved money.',
                    '⚠️ Concentration Change — automatically warns if losing partners pushed dependency on the remaining ones to a risky level.',
                ],
            },
            {
                'heading': '🔬 ⑦ New Business Scorer',
                'image': 'p7_scorer.png',
                'intro': 'Use this to find "recently surging existing items" and "brand-new entrants" across the whole market, ranked by opportunity. While the other 7 menus examine trade you already do, this one surfaces opportunities you are not yet pursuing.',
                'steps': [
                    'Upload full market data that includes HS Code Name / Detailed HS-CODE if possible.',
                    'Set the analysis window, minimum volume threshold, and analysis dimension.',
                    "(Optional) Enter your company's existing suppliers, origins, and product keywords to get a fit score against your own business.",
                    'Click "Run Analysis".',
                ],
                'results': [
                    'Combined Recommendation Ranking (Top 10) — surging existing items and new entrants ranked on one scale.',
                    'If fit criteria were entered, items are auto-classified into a quadrant (Priority Target / Growth-led / Safe Adjacent / Lower Priority).',
                    'Warning flags such as low-base rebound, single-shipment skew, and possible re-labeling filter out misleading data.',
                ],
                'tips': ['Analysis works without fit criteria, but filling them in makes the ranking far more relevant to your specific business.'],
            },
            {
                'heading': '🧩 ⑧ Free Pivot Builder',
                'image': 'p8_pivot.png',
                'intro': 'Use this when no fixed screen answers your question — build your own table or chart by freely combining axes and metrics, Excel-pivot-table style.',
                'steps': [
                    'Upload the data.',
                    'Pick one or more Row fields — multiple fields nest in the order chosen.',
                    'Optionally pick Column fields.',
                    'Pick one or more Value metrics — total/average volume, average/max/min price, transaction count (multi-select).',
                    'Pick a View (table/heatmap/bar/line/pie/stacked), optionally filter, then click "Generate pivot".',
                ],
                'results': [
                    'A table and chart are generated instantly from your chosen combination.',
                    'With multiple metrics selected, the table and heatmap show all of them, but bar/line/pie charts plot only the first selected metric (mixing metrics with different units in one chart would be confusing).',
                ],
            },
            {
                'heading': '📄 Common — PDF Download',
                'intro': 'Click "📄 Generate PDF Report" at the bottom of any results screen to build a PDF with that screen\'s charts and tables, then save it via "📥 Download PDF".',
                'tips': ['This very guide can be re-downloaded anytime from the "📖 User Guide" button in the sidebar.'],
            },
        ],
    },
}
def build_user_guide_pdf(lang):
    """앱 전체 사용법을 스크린샷과 함께 정리한 PDF 가이드를 생성해 바이트로 반환한다."""
    import io
    import os
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, Image as RLImage, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    content = GUIDE_CONTENT.get(lang, GUIDE_CONTENT['ko'])
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'guide')

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=15 * mm, bottomMargin=15 * mm, leftMargin=18 * mm, rightMargin=18 * mm)
    styles = getSampleStyleSheet()

    font_name = 'Helvetica'
    try:
        found_path = _find_korean_font_path()
        if found_path:
            pdfmetrics.registerFont(TTFont('KoreanFont', found_path))
            font_name = 'KoreanFont'
    except Exception:
        pass
    for s in styles.byName.values():
        s.fontName = font_name
    styles['Normal'].fontSize = 9.5
    styles['Normal'].leading = 13
    tip_style = styles['Normal'].clone('TipStyle')
    tip_style.textColor = colors.HexColor('#6b7280')
    tip_style.fontSize = 8.5

    story = [Paragraph(content['doc_title'], styles['Title'])]
    if content.get('intro'):
        story.append(Paragraph(content['intro'], styles['Normal']))
    story.append(Spacer(1, 10))

    for sec in content['sections']:
        block = [Paragraph(sec['heading'], styles['Heading2'])]
        if sec.get('intro'):
            block.append(Paragraph(sec['intro'], styles['Normal']))
            block.append(Spacer(1, 4))
        if sec.get('steps'):
            block.append(Paragraph(content['label_steps'], styles['Heading4']))
            items = [ListItem(Paragraph(b, styles['Normal'])) for b in sec['steps']]
            block.append(ListFlowable(items, bulletType='1', leftIndent=14))
            block.append(Spacer(1, 4))
        if sec.get('results'):
            block.append(Paragraph(content['label_results'], styles['Heading4']))
            items = [ListItem(Paragraph(b, styles['Normal']), bulletColor='#0d9488') for b in sec['results']]
            block.append(ListFlowable(items, bulletType='bullet', leftIndent=14))
            block.append(Spacer(1, 4))
        if sec.get('tips'):
            for tip in sec['tips']:
                block.append(Paragraph('💡 ' + tip, tip_style))
        story.append(KeepTogether(block))

        img_file = sec.get('image')
        if img_file:
            img_path = os.path.join(assets_dir, img_file)
            if os.path.exists(img_path):
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(img_path) as im:
                        iw, ih = im.size
                    disp_w = 110 * mm
                    disp_h = disp_w * ih / iw
                    story.append(Spacer(1, 4))
                    story.append(RLImage(img_path, width=disp_w, height=disp_h))
                except Exception:
                    pass
        story.append(Spacer(1, 12))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def find_column(columns, candidates):
    """가능한 컬럼명 후보 중에서 실제 데이터에 있는 컬럼명을 찾는 함수"""
    for col in candidates:
        if col in columns:
            return col
    # 대소문자 구분 없이 재시도
    for col in candidates:
        for actual_col in columns:
            if actual_col.lower() == col.lower():
                return actual_col
    return None
# --- [추가 4] 끝 ---


# --- [비밀번호 기능] 앱 접근 비밀번호 게이트 (최초 실행 시 직접 설정) ---
# 비밀번호는 앱과 같은 폴더의 숨김 파일(.local_app_password.hash)에 '해시'로만 저장됩니다.
# (평문 비밀번호 자체는 어디에도 저장되지 않습니다.) 이 파일은 깃허브에 커밋하지 않도록
# .gitignore에 등록해두는 걸 권장합니다 (사람이 직접 파일 업로드하는 방식이면 애초에
# 이 파일이 로컬/코드스페이스에서만 생성되므로 리포지토리에 올라갈 일이 없습니다).
_PASSWORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local_app_password.hash")


def _hash_pw(pw):
    return hashlib.sha256(str(pw).encode("utf-8")).hexdigest()


def _get_configured_password_hash():
    """우선순위: 1) secrets.toml / 환경변수(APP_PASSWORD) 2) 최초 실행 때 로컬에 저장해둔 해시 파일."""
    try:
        pw = st.secrets.get("app_password", None)
        if pw:
            return _hash_pw(pw)
    except Exception:
        pass
    env_pw = os.environ.get("APP_PASSWORD")
    if env_pw:
        return _hash_pw(env_pw)
    if os.path.exists(_PASSWORD_FILE):
        try:
            with open(_PASSWORD_FILE, "r", encoding="utf-8") as f:
                saved = f.read().strip()
                if saved:
                    return saved
        except Exception:
            pass
    return None


def _save_new_password(pw):
    with open(_PASSWORD_FILE, "w", encoding="utf-8") as f:
        f.write(_hash_pw(pw))


def _render_lang_toggle_small():
    col1, col2 = st.columns([4, 1])
    with col2:
        lang_choice = st.radio(
            "🌐", options=['한국어', 'English'],
            index=0 if st.session_state.get('lang', 'ko') == 'ko' else 1,
            horizontal=True, label_visibility="collapsed"
        )
        st.session_state.lang = 'ko' if lang_choice == '한국어' else 'en'


def check_password():
    """비밀번호가 맞으면 True. 아직 비밀번호가 설정된 적 없으면 '최초 설정' 화면을,
    이미 설정되어 있으면 '로그인' 화면을 보여주고 False를 반환한다."""
    if st.session_state.get("password_correct", False):
        return True

    _render_lang_toggle_small()

    configured_hash = _get_configured_password_hash()

    # --- 아직 아무 비밀번호도 설정된 적 없음: 최초 설정 화면 ---
    if configured_hash is None:
        st.title(T('password_setup_title'))
        st.caption(T('password_setup_caption'))
        # st.form으로 감싸서 입력값과 제출 버튼 클릭이 한 번에 서버로 전달되도록 함
        # (form 없이 개별 위젯 + 버튼 조합이면, 버튼 클릭 순간 입력창의 최신 값이
        #  아직 서버에 반영되기 전이라 '빈 값'으로 읽히는 경우가 있음)
        with st.form("_password_setup_form"):
            pw1 = st.text_input(T('password_setup_new'), type="password")
            pw2 = st.text_input(T('password_setup_confirm'), type="password")
            submitted = st.form_submit_button(T('password_setup_button'))
        if submitted:
            if not pw1:
                st.warning(T('password_setup_empty'))
            elif pw1 != pw2:
                st.error(T('password_setup_mismatch'))
            else:
                _save_new_password(pw1)
                st.session_state["password_correct"] = True
                st.success(T('password_setup_success'))
                st.rerun()
        return False

    # --- 이미 비밀번호가 설정되어 있음: 로그인 화면 ---
    st.title(T('password_gate_title'))

    def _on_password_entered():
        entered = st.session_state.get("_password_input_widget", "")
        if hmac.compare_digest(_hash_pw(entered), configured_hash):
            st.session_state["password_correct"] = True
            del st.session_state["_password_input_widget"]
        else:
            st.session_state["password_correct"] = False

    st.text_input(
        T('password_label'), type="password",
        on_change=_on_password_entered, key="_password_input_widget"
    )
    if st.session_state.get("password_correct") is False:
        st.error(T('password_wrong'))
    return False
# --- [비밀번호 기능] 끝 ---


# --------------------------#
# 메인 애플리케이션 UI 및 로직 #
# --------------------------#

st.set_page_config(layout="wide")

# --------------------------#
#  <<< 인쇄용 CSS 추가 >>>  #
# --------------------------#
print_css = """
<style>
@media print {

    /* 2. 불필요한 UI 요소 숨기기 */
    [data-testid="stSidebar"], /* 사이드바 */
    [data-testid="stActionButton"], /* GitHub 버튼 등 */
    .stButton, /* 모든 Streamlit 버튼 */
    .stFileUploader, /* 파일 업로더 */
    .stForm, /* 폼 전체 */
    [data-testid="stHeader"], /* Streamlit 헤더 */
    footer { /* Streamlit 푸터 */
        display: none !important;
    }

    /* 3. 메인 콘텐츠 영역을 인쇄 페이지에 맞게 조정 */
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    
    section[data-testid="stAppViewContainer"] > section {
        left: 0 !important;
        width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* 4. Expander (보고서 섹션) 처리 */
    .stExpander > details[open] {
        page-break-inside: avoid !important; 
    }

    .stExpander {
        /* 인쇄 시 테두리 제거 */
        border: none !important;
        box-shadow: none !important;
    }

    .stExpander > details[open] > summary {
        font-size: 1.5rem !important;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .stExpander > details[open] > summary svg {
        display: none !important; /* 토글 화살표 숨기기 */
    }
    
    /* 닫힌 Expander는 인쇄하지 않음 */
    .stExpander > details:not([open]) {
         display: none !important;
    }

    /* 5. 콘텐츠 잘림 방지 */
    .plotly-chart, /* Plotly 차트 */
    [data-testid="stDataFrame"], /* 데이터프레임 */
    .stMarkdown, /* 마크다운 텍스트 */
    table.summary-table { /* 요약 테이블 */
        page-break-inside: avoid; /* 요소 내부에서 페이지가 나뉘지 않도록 함 */
    }

    h1, h2, h3, h4, h5 {
        page-break-after: avoid; /* 제목 바로 뒤에서 페이지가 나뉘지 않도록 함 */
    }
    
    /* 6. 차트 크기 최적화 */
    .stPlotlyChart {
        width: 100% !important;
        max-width: 100%;
        overflow: hidden;
    }
    
    /* 7. 배경색 및 색상 강제 인쇄 (브라우저 설정 필요할 수 있음) */
    body {
        -webkit-print-color-adjust: exact !important;
        color-adjust: exact !important;
    }
}
</style>
"""
st.markdown(print_css, unsafe_allow_html=True)
# --- 인쇄용 CSS 끝 ---


# --- 세션 상태 초기화 ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'ko'

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.customer_name = None
    st.session_state.plot_df = None
    st.session_state.customer_df = None
    st.session_state.contract_date = None
    st.session_state.tfidf_matrix = None
    st.session_state.savings_df = None
    st.session_state.total_savings = None

if 'market_analysis_done' not in st.session_state:
    st.session_state.market_analysis_done = False
    st.session_state.market_df = None
    st.session_state.analyzed_product_name = None
    st.session_state.selected_customer = None
    st.session_state.market_contract_date = None
    st.session_state.top_competitors_list = []
    st.session_state.all_competitors_ranked = None

if 'flow_raw_df' not in st.session_state:
    st.session_state.flow_raw_df = None
    st.session_state.flow_headers = None
    st.session_state.flow_result = None

if 'risk_raw_df' not in st.session_state:
    st.session_state.risk_raw_df = None
    st.session_state.risk_headers = None
    st.session_state.risk_result = None

if 'season_raw_df' not in st.session_state:
    st.session_state.season_raw_df = None
    st.session_state.season_headers = None
    st.session_state.season_result = None

if 'churn_raw_df' not in st.session_state:
    st.session_state.churn_raw_df = None
    st.session_state.churn_headers = None
    st.session_state.churn_result = None

if 'pivot_raw_df' not in st.session_state:
    st.session_state.pivot_raw_df = None
    st.session_state.pivot_headers = None
    st.session_state.pivot_result = None

if 'scorer_raw_df' not in st.session_state:
    st.session_state.scorer_raw_df = None
    st.session_state.scorer_headers = None
    st.session_state.scorer_result = None

# --- [비밀번호 기능] 비밀번호가 맞을 때까지 이 아래 앱 본문을 그리지 않음 ---
if not check_password():
    st.stop()
# --- [비밀번호 기능] 끝 ---

# --- 사이드바 메뉴 (원본 UI + 언어 토글) ---
with st.sidebar:
    lang_choice = st.radio(
        "🌐 Language / 언어",
        options=['한국어', 'English'],
        index=0 if st.session_state.lang == 'ko' else 1,
        horizontal=True,
    )
    st.session_state.lang = 'ko' if lang_choice == '한국어' else 'en'
    st.markdown("---")
    selected = option_menu(
        menu_title=T('app_menu_title'),
        options=[T('menu_opt_customer'), T('menu_opt_market'), T('menu_opt_flow'), T('menu_opt_risk'), T('menu_opt_season'), T('menu_opt_churn'), T('menu_opt_scorer'), T('menu_opt_pivot')],
        icons=["person-bounding-box", "graph-up-arrow", "diagram-3", "exclamation-triangle", "calendar3", "arrow-left-right", "binoculars", "grid-3x3"],
        menu_icon="cast",
        default_index=0,
    )
    st.markdown("---")
    try:
        _guide_pdf_bytes = build_user_guide_pdf(st.session_state.lang)
        st.download_button(T('guide_download_btn'), data=_guide_pdf_bytes, file_name="user_guide.pdf", mime="application/pdf", use_container_width=True)
    except Exception:
        pass  # 가이드 PDF 생성에 실패해도 앱 나머지 기능에는 영향 없도록 조용히 넘어감
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: grey; font-size: 0.8rem;">
            © Made by Seungha Lee
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# 페이지 1: 고객사 효율 분석 (원본 코드 - 파일 읽기 로직만 수정)
# ==============================================================================
if selected == T('menu_opt_customer'):
    st.title(T('p1_title'))
    
    if st.session_state.analysis_done:
        st.button(T('p1_reset_btn'), on_click=reset_analysis_states)
    
    if not st.session_state.analysis_done:
        st.header(T('p1_settings_header'))
        uploaded_file = st.file_uploader(T('p1_upload_label'), type=['csv', 'xlsx']) 
        st.caption(T('p1_upload_caption'))
        
        if uploaded_file:
            with st.form(key='analysis_form'):
                df_for_check = None
                try:
                    # --- [수정 5] 파일 읽기 오류 방지 로직 (seek(0) 사용) ---
                    if uploaded_file.name.endswith('.csv'):
                        try:
                            uploaded_file.seek(0)
                            df_for_check = pd.read_csv(uploaded_file, encoding='utf-8')
                        except UnicodeDecodeError:
                            try:
                                uploaded_file.seek(0)
                                df_for_check = pd.read_csv(uploaded_file, encoding='euc-kr')
                            except UnicodeDecodeError:
                                uploaded_file.seek(0)
                                df_for_check = pd.read_csv(uploaded_file, encoding='cp949')
                    elif uploaded_file.name.endswith('.xlsx'):
                         uploaded_file.seek(0)
                         df_for_check = pd.read_excel(uploaded_file)

                    if df_for_check is None:
                        st.error(T('p1_file_read_fail_error'))
                        st.stop()
                    # --- [수정 5] 끝 ---

                    customer_name_input = None
                    if 'Raw Importer Name' not in df_for_check.columns:
                        st.warning(T('p1_missing_importer_col_warning'))
                        customer_name_input = st.text_input(T('p1_customer_name_input'))

                except Exception as e:
                    st.error(T('p1_file_read_error_generic', e=e))
                    st.stop() # 폼 실행 중지
                
                contract_date_input = st.date_input(T('p1_contract_date_input'))
                submitted = st.form_submit_button(T('p1_run_btn'))

            if submitted:
                with st.spinner(T('p1_spinner')):
                    df = df_for_check.copy()
                    
                    rename_dict = {
                        'Date': 'date', 
                        'Reported Product Name': 'product_name', 
                        'Volume': 'volume', 
                        'Unit Price': 'unit_price',
                        'Origin Country': 'origin_country', 
                        'Exporter': 'Exporter'
                    }
                    if 'Raw Importer Name' in df.columns:
                        rename_dict['Raw Importer Name'] = 'importer_name'
                    
                    df.rename(columns=rename_dict, inplace=True)
                    
                    if 'importer_name' not in df.columns and customer_name_input:
                        df['importer_name'] = customer_name_input
                    
                    required_columns = ['date', 'importer_name', 'product_name', 'volume', 'unit_price']
                    if not all(col in df.columns for col in required_columns):
                        st.error(T('p1_missing_required_cols_error'))
                        st.stop()
                    
                    df['date'] = pd.to_datetime(df['date'])
                    df['year_month'] = df['date'].dt.to_period('M')
                    df['year'] = df['date'].dt.year
                    df = df.dropna(subset=['importer_name', 'product_name', 'volume', 'unit_price'])
                    
                    customer_name = df['importer_name'].mode()[0]
                    customer_df = df[df['importer_name'] == customer_name].copy()
                    customer_df['product_preprocessed'] = customer_df['product_name'].apply(preprocess_product_name)
                    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
                    tfidf_matrix = vectorizer.fit_transform(customer_df['product_preprocessed'])
                    dbscan = DBSCAN(eps=0.9, min_samples=3, metric='cosine')
                    cluster_labels = dbscan.fit_predict(tfidf_matrix)
                    cluster_name_map = get_cluster_name(cluster_labels, customer_df['product_preprocessed'])
                    customer_df['cluster'] = cluster_labels
                    customer_df['cluster_name'] = customer_df['cluster'].map(cluster_name_map)
                    plot_df = customer_df[customer_df['cluster'] != -1].copy()

                    contract_date = pd.to_datetime(contract_date_input)
                    before_contract_df = plot_df[plot_df['date'] < contract_date]
                    after_contract_df = plot_df[plot_df['date'] >= contract_date]
                    avg_price_before = before_contract_df.groupby('cluster_name')['unit_price'].mean().rename('avg_price_before')
                    avg_price_after = after_contract_df.groupby('cluster_name')['unit_price'].mean().rename('avg_price_after')
                    volume_after = after_contract_df.groupby('cluster_name')['volume'].sum().rename('volume_after')
                    savings_df = pd.concat([avg_price_before, avg_price_after, volume_after], axis=1).dropna()
                    savings_df['savings'] = (savings_df['avg_price_before'] - savings_df['avg_price_after']) * savings_df['volume_after']
                    savings_df = savings_df.sort_values('savings', ascending=False)
                    total_savings = savings_df['savings'].sum()

                    st.session_state.customer_name = customer_name
                    st.session_state.plot_df = plot_df
                    st.session_state.customer_df = customer_df
                    st.session_state.contract_date = contract_date
                    st.session_state.tfidf_matrix = tfidf_matrix
                    st.session_state.savings_df = savings_df
                    st.session_state.total_savings = total_savings
                    st.session_state.analysis_done = True
                    
                st.success(T('p1_analysis_complete_success', customer_name=customer_name))
                st.rerun()

    if st.session_state.analysis_done:
        with st.expander(T('p1_exp1_title'), expanded=True):
            st.subheader(T('p1_total_savings_subheader'))
            total_savings = st.session_state.total_savings
            color = "blue" if total_savings >= 0 else "red"
            st.markdown(f"## <span style='color:{color};'>${total_savings:,.2f}</span>", unsafe_allow_html=True)
            st.caption(T('p1_total_savings_caption', date=st.session_state.contract_date.date()))
            st.subheader(T('p1_savings_detail_subheader'))
            cols = st.columns(4)
            for i, row in enumerate(st.session_state.savings_df.itertuples()):
                col = cols[i % 4]
                color, arrow, val = ("blue", "▼", row.savings) if row.savings >= 0 else ("red", "▲", -row.savings)
                col.markdown(f"""<div style="border: 1px solid #e6e6e6; border-radius: 0.5rem; padding: 1rem; text-align: center; height: 120px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 1rem;"><strong>{row.Index}</strong><p style="font-size: 1.5rem; font-weight: bold; color: {color}; margin-top: 8px; margin-bottom: 0;">{arrow} ${val:,.0f}</p></div>""", unsafe_allow_html=True)
            
            st.dataframe(st.session_state.savings_df.style.format({
                'avg_price_before': '${:,.2f}',
                'avg_price_after': '${:,.2f}',
                'volume_after': '{:,.0f} KG',
                'savings': '${:,.2f}'
            }))

        with st.expander(T('p1_exp2_title'), expanded=True): # 인쇄 시 이 섹션부터 새 페이지
            if st.session_state.tfidf_matrix is not None and st.session_state.tfidf_matrix.shape[0] > 0:
                pca = PCA(n_components=2, random_state=42)
                components = pca.fit_transform(st.session_state.tfidf_matrix.toarray())
                vis_df = pd.DataFrame(components, columns=['x', 'y'])
                vis_df['cluster_name'] = st.session_state.customer_df['cluster_name'].values
                vis_df['product_name'] = st.session_state.customer_df['product_name'].values
                cluster_volume_sorted = st.session_state.plot_df.groupby('cluster_name')['volume'].sum().sort_values(ascending=False)
                top_clusters_for_viz = cluster_volume_sorted.head(15).index.tolist()
                vis_df_filtered = vis_df[vis_df['cluster_name'].isin(top_clusters_for_viz)]
                st.info(T('p1_too_many_clusters_info', n=len(top_clusters_for_viz)))
                
                fig1 = px.scatter(vis_df_filtered[vis_df_filtered['cluster_name'] != 'Noise'], x='x', y='y', color='cluster_name', facet_col='cluster_name', facet_col_wrap=4, height=800, 
                                  title=f"<b>{T('p1_scatter_title', customer=st.session_state.customer_name)}</b><br><span style='font-size: 0.8em; color:grey;'>{T('p1_scatter_subtitle', n=len(top_clusters_for_viz))}</span>", 
                                  labels={'x': T('axis_pca1'), 'y': T('axis_pca2')}, hover_data=['product_name'])
                fig1.update_traces(marker=dict(size=8, opacity=0.8))
                st.plotly_chart(fig1, use_container_width=True)
                st.subheader(T('p1_cluster_list_subheader'))
                plot_df_sorted = st.session_state.plot_df.copy()
                plot_df_sorted['cluster_name'] = pd.Categorical(plot_df_sorted['cluster_name'], categories=cluster_volume_sorted.index.tolist(), ordered=True)
                st.dataframe(plot_df_sorted[['product_name', 'product_preprocessed', 'cluster_name']].drop_duplicates().sort_values('cluster_name'))

        with st.expander(T('p1_exp3_title'), expanded=True): # 새 페이지
            plot_df_chart = st.session_state.plot_df.copy()
            plot_df_chart['year_month_str'] = plot_df_chart['year_month'].astype(str)
            cluster_volume = plot_df_chart.groupby(['year_month_str', 'cluster_name'])['volume'].sum().reset_index()
            sorted_clusters = st.session_state.plot_df.groupby('cluster_name')['volume'].sum().sort_values(ascending=False).index.tolist()
            
            fig2 = px.bar(cluster_volume, x='year_month_str', y='volume', color='cluster_name', 
                          title=f"<b>{T('p1_monthly_chart_title', customer=st.session_state.customer_name)}</b>", 
                          labels={'year_month_str': T('axis_yearmonth'), 'volume': T('axis_volume_kg'), 'cluster_name': T('legend_cluster')}, 
                          category_orders={'cluster_name': sorted_clusters})
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("---")

            st.subheader(T('p1_recent3m_subheader'))
            customer_df_for_pie = st.session_state.customer_df.copy()
            if not customer_df_for_pie.empty:
                latest_date = customer_df_for_pie['date'].max()
                three_months_ago = latest_date - pd.DateOffset(months=3)
                recent_df = customer_df_for_pie[customer_df_for_pie['date'] > three_months_ago]
                
                if not recent_df.empty:
                    pie_data = recent_df.groupby('cluster_name')['volume'].sum().reset_index()
                    fig_pie = px.pie(pie_data, values='volume', names='cluster_name',
                                     title=f"<b>{T('p1_recent3m_pie_title', customer=st.session_state.customer_name)}</b><br><span style='font-size: 0.8em; color:grey;'>{T('p1_recent3m_pie_subtitle', start=three_months_ago.strftime('%Y-%m-%d'), end=latest_date.strftime('%Y-%m-%d'))}</span>")
                    fig_pie.update_traces(textposition='inside', textinfo='percent')
                    fig_pie.update_layout(showlegend=True)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info(T('p1_no_recent_data_info'))
            else:
                st.warning(T('p1_no_data_warning'))

    
        with st.expander(T('p1_exp4_title'), expanded=True): # 새 페이지
            customer_df = st.session_state.customer_df
            contract_date = st.session_state.contract_date
            
            before_df = customer_df[customer_df['date'] < contract_date]
            after_df = customer_df[customer_df['date'] >= contract_date]

            st.subheader(T('p1_new_clusters_subheader'))
            before_clusters = set(before_df['cluster_name'].unique())
            after_clusters = set(after_df['cluster_name'].unique())
            new_clusters = list(after_clusters - before_clusters)
            if new_clusters:
                st.write(T('p1_new_clusters_text', n=len(new_clusters)))
                new_cluster_details_df = after_df[after_df['cluster_name'].isin(new_clusters)][['cluster_name', 'product_name']].drop_duplicates().rename(columns={'cluster_name': T('col_new_cluster'), 'product_name': T('col_rep_product')})
                st.dataframe(new_cluster_details_df)
            else:
                st.info(T('p1_no_new_clusters_info'))

            st.markdown("---")
            
            if 'origin_country' in customer_df.columns:
                st.subheader(T('p1_new_origins_subheader'))
                before_origins = set(before_df['origin_country'].dropna().unique())
                after_origins = set(after_df['origin_country'].dropna().unique())
                new_origins = list(after_origins - before_origins)
                if new_origins:
                    st.write(T('p1_new_origins_text', n=len(new_origins)))
                    new_origin_details = []
                    for origin in new_origins:
                        related_clusters = after_df[after_df['origin_country'] == origin]['product_name'].unique()
                        new_origin_details.append({
                            T('col_new_origin'): origin,
                            T('col_related_products'): ','.join(related_clusters)
                        })
                    st.dataframe(pd.DataFrame(new_origin_details))
            else:
                st.info(T('p1_no_new_origins_info'))
            
            st.markdown("---")

            if 'Exporter' in customer_df.columns:
                st.subheader(T('p1_new_exporters_subheader'))
                before_exporters = set(before_df['Exporter'].dropna().unique())
                after_exporters = set(after_df['Exporter'].dropna().unique())
                new_exporters = list(after_exporters - before_exporters)
                if new_exporters:
                    st.write(T('p1_new_exporters_text', n=len(new_exporters)))
                    new_exporter_details = []
                    for exporter in new_exporters:
                        related_clusters = after_df[after_df['Exporter'] == exporter]['product_name'].unique()
                        new_exporter_details.append({
                            T('col_new_exporter'): exporter,
                            T('col_related_products'): ','.join(related_clusters)
                        })
                    st.dataframe(pd.DataFrame(new_exporter_details))
                else:
                    st.info(T('p1_no_new_exporters_info'))

# ==============================================================================
# 페이지 2: 시장 경쟁력 분석 (모든 수정 사항이 이 섹션에 적용됨)
# ==============================================================================
if selected == T('menu_opt_market'):
    st.title(T('p2_title'))
    
    if st.session_state.get('market_analysis_done', False):
        st.button(T('p2_reset_btn'), on_click=reset_market_analysis_states)

    if not st.session_state.get('market_analysis_done', False):
        st.write(T('p2_intro_text'))
        market_file = st.file_uploader(T('p2_upload_label'), type=['csv', 'xlsx'], key="market_uploader")
        st.caption(T('p2_upload_caption'))
        
        if market_file:
            with st.form("market_analysis_form"):
                market_df_for_importers = None
                try:
                    # --- [수정 6] 파일 읽기 오류 방지 로직 (seek(0) 사용) ---
                    if market_file.name.endswith('.csv'):
                        try:
                            market_file.seek(0)
                            market_df_for_importers = pd.read_csv(market_file, encoding='utf-8')
                        except UnicodeDecodeError:
                            try:
                                market_file.seek(0)
                                market_df_for_importers = pd.read_csv(market_file, encoding='euc-kr')
                            except UnicodeDecodeError:
                                market_file.seek(0)
                                market_df_for_importers = pd.read_csv(market_file, encoding='cp949')
                    elif market_file.name.endswith('.xlsx'):
                         market_file.seek(0)
                         market_df_for_importers = pd.read_excel(market_file)

                    if market_df_for_importers is None:
                        st.error(T('p1_file_read_fail_error'))
                        st.stop()
                    # --- [수정 6] 끝 ---
                    
                    if 'Raw Importer Name' in market_df_for_importers.columns:
                        importer_list = sorted(market_df_for_importers['Raw Importer Name'].unique())
                        customer_name_selection = st.selectbox(T('p2_select_customer_label'), options=importer_list)
                    else:
                        st.warning(T('p2_missing_importer_col_warning'))
                        customer_name_selection = st.text_input(T('p2_customer_name_input'))
                
                except Exception as e:
                    st.error(T('p2_file_read_error', e=e))
                    customer_name_selection = None
                    st.stop() # 폼 실행 중지
                
                analyzed_product_name_input = st.text_input(T('p2_product_name_input'))
                contract_date_input = st.date_input(T('p2_contract_date_input'))

                # --- [수정 7] IQR 배수 조절 슬라이더 및 툴팁 추가 ---
                iqr_multiplier_input = st.slider(
                    T('p2_iqr_slider_label'), 
                    min_value=1.5, 
                    max_value=3.0, 
                    value=1.5, 
                    step=0.1, 
                    help=T('p2_iqr_slider_help')
                )
                # --- [수정 7] 끝 ---

                market_submitted = st.form_submit_button(T('p2_run_btn'))

            if market_submitted and customer_name_selection and analyzed_product_name_input:
                with st.spinner(T('p2_processing_spinner')):
                    market_df = market_df_for_importers.copy()
                    
                    # --- [수정 8] 'Export Country' 컬럼을 동적으로 찾도록 수정 (KeyError 해결) ---
                    # 컬럼을 동적으로 찾기
                    date_col = find_column(market_df.columns, ['Date', 'date'])
                    product_col = find_column(market_df.columns, ['Reported Product Name', 'product_name'])
                    volume_col = find_column(market_df.columns, ['Volume', 'volume'])
                    price_col = find_column(market_df.columns, ['Unit Price', 'unit_price'])
                    origin_col = find_column(market_df.columns, ['Origin Country', 'origin_country'])
                    export_col = find_column(market_df.columns, ['Export Country', 'export_country', '수출국']) # <-- 요청 사항 반영
                    importer_col = find_column(market_df.columns, ['Raw Importer Name', 'importer_name'])
                    exporter_col = find_column(market_df.columns, ['Exporter', 'exporter'])

                    rename_dict = {}
                    if date_col: rename_dict[date_col] = 'date'
                    if product_col: rename_dict[product_col] = 'product_name'
                    if volume_col: rename_dict[volume_col] = 'volume'
                    if price_col: rename_dict[price_col] = 'unit_price'
                    if origin_col: rename_dict[origin_col] = 'origin_country'
                    if export_col: rename_dict[export_col] = 'export_country' # <-- 요청 사항 반영
                    if importer_col: rename_dict[importer_col] = 'importer_name'
                    if exporter_col: rename_dict[exporter_col] = 'Exporter'

                    if not importer_col: # Handle manual input
                        market_df['importer_name'] = customer_name_selection 
                    
                    market_df.rename(columns=rename_dict, inplace=True)
                    # --- [수정 8] 끝 ---

                    market_df['date'] = pd.to_datetime(market_df['date'])
                    market_df['year_month'] = market_df['date'].dt.to_period('M')
                    market_df['year'] = market_df['date'].dt.year
                    market_df['quarter'] = market_df['date'].dt.quarter
                    
                    # --- [수정 9] 필수 컬럼 검증 로직 수정 (KeyError 해결) ---
                    required_market_cols = ['date', 'importer_name', 'product_name', 'volume', 'unit_price', 'Exporter', 'origin_country', 'export_country']
                    
                    # 필수 컬럼 누락 시 중지
                    missing_cols = [col for col in required_market_cols if col not in market_df.columns]
                    if missing_cols:
                        st.error(T('p2_missing_cols_error', cols=', '.join(missing_cols)))
                        st.stop()
                    
                    # dropna도 수정된 필수 컬럼 리스트로 수행
                    market_df = market_df.dropna(subset=required_market_cols)
                    # --- [수정 9] 끝 ---

                    # --- [수정 10] 함수 호출 시 iqr_multiplier_input 값 전달 ---
                    market_df = remove_outliers_iqr(
                        market_df, 
                        'unit_price', 
                        cap_percent=0.07, # 기존 캡 로직은 유지
                        iqr_multiplier=iqr_multiplier_input # 사용자가 선택한 값 전달
                    )
                    # --- [수정 10] 끝 ---
                    
                    lowess_results = sm.nonparametric.lowess(market_df['unit_price'], market_df['volume'], frac=0.5)
                    market_df['expected_price'] = np.interp(market_df['volume'], lowess_results[:, 0], lowess_results[:, 1])
                    market_df['competitiveness_index'] = market_df['expected_price'] - market_df['unit_price']
                    
                    all_competitors_ranked = market_df.groupby('importer_name')['competitiveness_index'].mean().sort_values(ascending=False).reset_index()
                    
                    customer_rank_info = all_competitors_ranked[all_competitors_ranked['importer_name'] == customer_name_selection]
                    customer_rank = customer_rank_info.index[0] if not customer_rank_info.empty else len(all_competitors_ranked)
                    top_competitors_list = all_competitors_ranked.iloc[:customer_rank]['importer_name'].tolist()
                    if customer_name_selection in top_competitors_list:
                        top_competitors_list.remove(customer_name_selection)
                    
                    st.session_state.market_df = market_df
                    st.session_state.analyzed_product_name = analyzed_product_name_input
                    st.session_state.selected_customer = customer_name_selection
                    st.session_state.market_contract_date = pd.to_datetime(contract_date_input)
                    st.session_state.top_competitors_list = top_competitors_list
                    st.session_state.all_competitors_ranked = all_competitors_ranked
                    st.session_state.market_analysis_done = True
                    st.session_state.analysis_countries = [T('p2_country_filter_all')] # <-- [수정 11] 분석 실행 시 필터 기본값 '전체'로 설정
                st.rerun()

    if st.session_state.get('market_analysis_done', False):
        customer_name = st.session_state.selected_customer
        
        # [수정 12] market_df를 세션에서 바로 가져오되, 필터링을 위해 'full_market_df'로 임시 저장
        full_market_df = st.session_state.market_df 
        
        analyzed_product_name = st.session_state.analyzed_product_name
        contract_date = st.session_state.market_contract_date
        
        # --- [수정 14] 요청하신 원산지(수출국) 필터 UI 추가 ---
        ALL_LABEL = T('p2_country_filter_all')
        all_countries = sorted(full_market_df['export_country'].astype(str).unique())
        all_countries_options = [ALL_LABEL] + all_countries
        
        col1, col2 = st.columns([2, 1]) # Title on the left, filter on the right
        with col1:
            # 원본 헤더
            st.subheader(T('p2_result_subheader', product=analyzed_product_name, customer=customer_name)) 
        with col2:
            # 상호작용(Interactive) 필터
            selected_countries = st.multiselect(
                T('p2_country_filter_label'),
                options=all_countries_options,
                default=st.session_state.get('analysis_countries', [ALL_LABEL]),
                key='country_filter' # Streamlit이 상태를 기억하도록 key 지정
            )
        
        # 필터 선택 값을 세션 상태에 즉시 업데이트
        st.session_state.analysis_countries = selected_countries
        # --- [수정 14] 필터 UI 끝 ---

        # --- [수정 15] 필터 적용 로직 ---
        current_selection = st.session_state.analysis_countries
        
        if not current_selection: # 아무것도 선택 안하면 '전체'로 간주
            st.session_state.analysis_countries = [ALL_LABEL]
            current_selection = [ALL_LABEL]

        # '전체'를 제외한 실제 필터링할 국가 리스트
        countries_to_filter = [c for c in current_selection if c != ALL_LABEL]

        if ALL_LABEL in current_selection or not countries_to_filter:
            # '전체'가 선택되었거나, '전체'가 아닌 리스트가 비어있으면 -> 필터링 안함
            market_df = full_market_df.copy() # 원본 df 사용
            with col1: # 제목 아래에 현재 상태 표시
                st.markdown(f"<small>{T('p2_country_status_line', countries=ALL_LABEL)}</small>", unsafe_allow_html=True)
        else:
            # 특정 국가만 선택된 경우 -> 데이터 필터링
            market_df = full_market_df[full_market_df['export_country'].isin(countries_to_filter)]
            with col1: # 제목 아래에 현재 상태 표시
                st.markdown(f"<small>{T('p2_country_status_line', countries=', '.join(countries_to_filter))}</small>", unsafe_allow_html=True)
        
        # 필터링 결과 데이터가 없는지 확인
        if market_df.empty:
            st.warning(T('p2_no_data_for_filter_warning', product=analyzed_product_name))
            st.stop() # 데이터 없으면 이하 분석 중지
        # --- [수정 15] 필터 적용 로직 끝 ---
        
        # --- [수정 16] 필터링된 market_df를 기준으로 경쟁사 랭킹 *다시 계산* ---
        filtered_competitors_ranked = market_df.groupby('importer_name')['competitiveness_index'].mean().sort_values(ascending=False).reset_index()
        customer_rank_info_filtered = filtered_competitors_ranked[filtered_competitors_ranked['importer_name'] == customer_name]
        customer_rank_filtered = customer_rank_info_filtered.index[0] if not customer_rank_info_filtered.empty else len(filtered_competitors_ranked)
        top_competitors_list_filtered = filtered_competitors_ranked.iloc[:customer_rank_filtered]['importer_name'].tolist()
        if customer_name in top_competitors_list_filtered:
            top_competitors_list_filtered.remove(customer_name)
        # --- [수정 16] 끝 ---


        with st.expander(T('p2_exp1_title', product=analyzed_product_name), expanded=True): # 새 페이지
            st.markdown(f"##### {T('p2_scatter_subheader')}")
            fig_comp = px.scatter(market_df, x='volume', y='unit_price', trendline="lowess", trendline_color_override="red", hover_data=['importer_name', 'date'], 
                                  title=f"<b>{T('p2_scatter_title')}</b><br><span style='font-size: 0.8em; color:grey;'>{T('p2_scatter_subtitle')}</span>",
                                  labels={'volume': T('axis_volume_kg'), 'unit_price': T('axis_unit_price')})
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.markdown(f"##### {T('p2_top10_subheader')}")
            # --- [수정 17] 필터링된 랭킹 사용 ---
            top_10_competitors = filtered_competitors_ranked.head(10)
            
            def highlight_customer(row):
                color = 'background-color: lightblue' if row.importer_name == customer_name else ''
                return [color] * len(row)
            
            st.dataframe(top_10_competitors.style.apply(highlight_customer, axis=1).format({'competitiveness_index': '{:,.2f}'}))
            
            # --- [수정 18] 필터링된 랭킹 정보 사용 ---
            if not customer_rank_info_filtered.empty:
                customer_rank = customer_rank_info_filtered.index[0] + 1
                if customer_rank > 10:
                    st.info(T('p2_rank_info', customer=customer_name, total=len(filtered_competitors_ranked), rank=customer_rank))
            else:
                 st.warning(T('p2_rank_not_in_filter_warning', customer=customer_name))


        with st.expander(T('p2_exp2_title', product=analyzed_product_name), expanded=True): # 새 페이지
            st.markdown(f"##### {T('p2_monthly_comp_subheader')}")
            monthly_competitiveness = market_df.groupby(['year_month', 'importer_name'])['competitiveness_index'].mean().unstack()
            
            market_avg_monthly_comp = monthly_competitiveness.mean(axis=1)
            customer_monthly_comp = monthly_competitiveness.get(customer_name)
            
            fig_comp_trend = go.Figure()
            fig_comp_trend.add_trace(go.Scatter(x=market_avg_monthly_comp.index.to_timestamp(), y=market_avg_monthly_comp, mode='lines', name=T('legend_market_avg_index'), line=dict(color='blue', width=3)))
            if customer_monthly_comp is not None:
                fig_comp_trend.add_trace(go.Scatter(x=customer_monthly_comp.index.to_timestamp(), y=customer_monthly_comp, mode='lines+markers', name=T('legend_customer_index', customer=customer_name), line=dict(color='red')))
            
            # --- [수정 19] 필터링된 랭킹 사용 ---
            if top_competitors_list_filtered: 
                top_competitors_monthly_comp = monthly_competitiveness[top_competitors_list_filtered]
                top_competitors_avg_monthly_comp = top_competitors_monthly_comp.mean(axis=1)
                fig_comp_trend.add_trace(go.Scatter(x=top_competitors_avg_monthly_comp.index.to_timestamp(), y=top_competitors_avg_monthly_comp, mode='lines+markers', name=T('legend_top_group_avg_index'), line=dict(color='green', dash='dash')))

            fig_comp_trend.update_layout(title=f"<b>{T('p2_comp_trend_chart_title', product=analyzed_product_name)}</b>", xaxis_title=T('axis_yearmonth'), yaxis_title=T('axis_comp_index'))
            st.plotly_chart(fig_comp_trend, use_container_width=True)
            st.caption(T('p2_comp_trend_caption'))
            st.markdown("---")

            st.markdown(f"##### {T('p2_price_trend_subheader')}")
            market_avg_price = market_df.groupby('year_month')['unit_price'].mean().rename('market_avg_price')
            customer_market_df = market_df[market_df['importer_name'] == customer_name]
            customer_avg_price = customer_market_df.groupby('year_month')['unit_price'].mean().rename('customer_avg_price')
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=market_avg_price.index.to_timestamp(), y=market_avg_price, mode='lines+markers', name=T('legend_market_avg_price'), line=dict(width=3)))
            fig4.add_trace(go.Scatter(x=customer_avg_price.index.to_timestamp(), y=customer_avg_price, mode='lines+markers', name=T('legend_customer_avg_price', customer=customer_name), line=dict(color='red')))
            
            # --- [수정 20] 필터링된 랭킹 사용 ---
            if top_competitors_list_filtered: 
                st.info(T('p2_benchmark_info'))
                st.caption(T('p2_benchmark_caption'))
                top_competitors_df = market_df[market_df['importer_name'].isin(top_competitors_list_filtered)] 
                top_competitors_avg_price = top_competitors_df.groupby('year_month')['unit_price'].mean().rename('top_competitors_avg_price')
                fig4.add_trace(go.Scatter(x=top_competitors_avg_price.index.to_timestamp(), y=top_competitors_avg_price, mode='lines+markers', name=T('legend_top_group_avg_price'), line=dict(color='green', dash='dash')))
            else:
                st.success(T('p2_benchmark_success', customer=customer_name))

            fig4.update_layout(title=f"<b>{T('p2_price_trend_chart_title', product=analyzed_product_name)}</b>", xaxis_title=T('axis_yearmonth'), yaxis_title=T('axis_avg_unit_price'))
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown(f"##### {T('p2_price_compare_subheader')}")
            col1, col2, col3 = st.columns(3)
            col1.metric(T('metric_market_avg'), f"${market_df['unit_price'].mean():.2f}")
            col2.metric(T('metric_customer_avg', customer=customer_name), f"${customer_market_df['unit_price'].mean():.2f}")
            # --- [수정 21] 필터링된 랭킹 사용 ---
            if top_competitors_list_filtered: 
                col3.metric(T('metric_top_group_avg'), f"${top_competitors_df['unit_price'].mean():.2f}")

        # --- [수정 22] 필터링된 랭킹 사용 ---
        if top_competitors_list_filtered: 
            with st.expander(T('p2_sim_exp_title'), expanded=True): # 새 페이지
                with st.form("simulation_form"):
                    sim_start_date = st.date_input(T('p2_sim_start_date'), contract_date)
                    sim_end_date = st.date_input(T('p2_sim_end_date'))
                    run_simulation = st.form_submit_button(T('p2_sim_run_btn'))
                
                if run_simulation:
                    sim_df = pd.merge(customer_avg_price, top_competitors_avg_price, left_index=True, right_index=True, how='inner')
                    customer_volume_monthly = customer_market_df.groupby('year_month')['volume'].sum()
                    sim_df = pd.merge(sim_df, customer_volume_monthly, left_index=True, right_index=True, how='inner')
                    
                    sim_period_start = pd.to_datetime(sim_start_date).to_period('M')
                    sim_period_end = pd.to_datetime(sim_end_date).to_period('M')
                    sim_df = sim_df[(sim_df.index >= sim_period_start) & (sim_df.index <= sim_period_end)]
                    
                    if not sim_df.empty:
                        sim_df['potential_savings'] = (sim_df['customer_avg_price'] - sim_df['top_competitors_avg_price']) * sim_df['volume']
                        total_potential_savings = sim_df[sim_df['potential_savings'] > 0]['potential_savings'].sum()
                        st.success(T('p2_sim_success', amount=f"{total_potential_savings:,.2f}"))
                        st.caption(T('p2_sim_caption'))
                    else:
                        st.warning(T('p2_sim_no_data_warning'))

        with st.expander(T('p2_exp3_title', product=analyzed_product_name), expanded=True): # 새 페이지
            col1, col2 = st.columns(2)
            with col1:
                years_with_data = sorted(market_df['year'].unique(), reverse=True)
                if years_with_data:
                    selected_year_ms = st.selectbox(T('p2_ms_year_select'), options=years_with_data, key=f"ms_year_{analyzed_product_name}")
                    ms_df = market_df[market_df['year'] == selected_year_ms]
                    ms_data = ms_df.groupby('importer_name')['volume'].sum().sort_values(ascending=False).reset_index()
                    display_data = ms_data.head(5)
                    if customer_name not in display_data['importer_name'].tolist() and not ms_data[ms_data['importer_name']==customer_name].empty:
                        customer_data = ms_data[ms_data['importer_name']==customer_name]
                        display_data = pd.concat([customer_data, display_data.head(4)])
                    others_volume = ms_data[~ms_data['importer_name'].isin(display_data['importer_name'])]['volume'].sum()
                    if others_volume > 0: display_data.loc[len(display_data)] = {'importer_name': T('p2_others_label'), 'volume': others_volume}
                    
                    competitors = [imp for imp in display_data['importer_name'] if imp != customer_name]
                    blue_shades = px.colors.sequential.Blues_r[::(len(px.colors.sequential.Blues_r)//(len(competitors)+1)) if competitors else 1]
                    color_map_pie = {comp: blue_shades[i % len(blue_shades)] for i, comp in enumerate(competitors)}
                    color_map_pie[customer_name] = 'red'
                    
                    fig5 = px.pie(display_data, values='volume', names='importer_name', color='importer_name',
                                  title=f"<b>{T('p2_ms_pie_title', product=analyzed_product_name, year=selected_year_ms)}</b><br><span style='font-size: 0.8em; color:grey;'>{T('p2_ms_pie_subtitle')}</span>", 
                                  hole=0.3, color_discrete_map=color_map_pie)
                    fig5.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig5, use_container_width=True)
            with col2:
                years_with_data_price = sorted(market_df['year'].unique(), reverse=True)
                if years_with_data_price:
                    selected_year_price = st.selectbox(T('p2_price_year_select'), options=years_with_data_price, key=f"price_year_{analyzed_product_name}")
                    price_comp_df = market_df[market_df['year'] == selected_year_price]
                    top_importers_by_vol = price_comp_df.groupby('importer_name')['volume'].sum().nlargest(5).index.tolist()
                    if customer_name not in top_importers_by_vol: top_importers_by_vol.append(customer_name)
                    price_comp_data = price_comp_df[price_comp_df['importer_name'].isin(top_importers_by_vol)]
                    avg_price_by_importer = price_comp_data.groupby('importer_name')['unit_price'].mean().sort_values().reset_index()
                    
                    competitors = [imp for imp in avg_price_by_importer['importer_name'] if imp != customer_name]
                    blue_shades = px.colors.sequential.Blues_r[::(len(px.colors.sequential.Blues_r)//(len(competitors)+1)) if competitors else 1]
                    color_map_bar = {comp: blue_shades[i % len(blue_shades)] for i, comp in enumerate(competitors)}
                    color_map_bar[customer_name] = 'red'

                    fig6 = px.bar(avg_price_by_importer, x='importer_name', y='unit_price', title=f"<b>{T('p2_price_bar_title', year=selected_year_price)}</b><br><span style='font-size: 0.8em; color:grey;'>{T('p2_price_bar_subtitle')}</span>", labels={'importer_name': T('axis_importer'), 'unit_price': T('axis_avg_unit_price')}, color='importer_name', color_discrete_map=color_map_bar)
                    st.plotly_chart(fig6, use_container_width=True)
        
        # --- [수정 23] 'export_country' 컬럼도 확인하도록 if문 변경 ---
        if 'Exporter' in market_df.columns and 'origin_country' in market_df.columns and 'export_country' in market_df.columns:
            with st.expander(T('p2_exp4_title', product=analyzed_product_name), expanded=True): # 새 페이지
                years_with_data_exporter = sorted(market_df['year'].unique(), reverse=True)
                if years_with_data_exporter:
                    selected_year_exporter = st.selectbox(T('p2_exporter_year_select'), options=years_with_data_exporter, key=f"exporter_year_{analyzed_product_name}")
                    exporter_analysis_df = market_df[market_df['year'] == selected_year_exporter]
                    
                    top_10_exporters_by_vol = exporter_analysis_df.groupby('Exporter')['volume'].sum().nlargest(10).index
                    exporter_analysis_df_top10 = exporter_analysis_df[exporter_analysis_df['Exporter'].isin(top_10_exporters_by_vol)]

                    st.subheader(T('p2_exporter_quarterly_subheader', year=selected_year_exporter))
                    fig9 = px.box(exporter_analysis_df_top10, x='quarter', y='unit_price', color='Exporter', 
                                  title=f"<b>{T('p2_exporter_box_title', year=selected_year_exporter)}</b><br><span style='font-size: 0.8em; color:grey;'>{T('p2_exporter_box_subtitle')}</span>", 
                                  labels={'quarter': T('axis_quarter'), 'unit_price': T('axis_unit_price')})
                    st.plotly_chart(fig9, use_container_width=True)
                    with st.expander(T('p2_detail_data_expander')):
                        summary_df_exp = exporter_analysis_df_top10.groupby('Exporter')['unit_price'].agg(['max', 'mean', 'min']).reset_index()
                        summary_df_exp.columns = [T('col_supplier'), T('col_max_price'), T('col_avg_price'), T('col_min_price')]
                        st.dataframe(summary_df_exp.style.format({T('col_max_price'): '${:,.2f}', T('col_avg_price'): '${:,.2f}', T('col_min_price'): '${:,.2f}'}))
                    
                    customer_exporters_in_year = exporter_analysis_df[exporter_analysis_df['importer_name'] == customer_name]['Exporter'].unique()
                    st.info(T('p2_customer_exporters_info', customer=customer_name, year=selected_year_exporter, exporters=', '.join(customer_exporters_in_year)))
                    
                    # <<-- 공급사별 분석을 각각의 expander에 표시 -->>
                    for exporter in customer_exporters_in_year:
                        with st.expander(T('p2_exporter_detail_expander', exporter=exporter), expanded=True): # 이 expander는 인쇄 시 페이지가 나뉘지 않습니다.
                            single_exporter_df = exporter_analysis_df[exporter_analysis_df['Exporter'] == exporter]
                            
                            st.subheader(T('p2_volume_price_compare_subheader'))
                            importer_summary = single_exporter_df.groupby('importer_name').agg(
                                total_volume=('volume', 'sum'),
                                avg_unit_price=('unit_price', 'mean')
                            ).sort_values('total_volume', ascending=False).reset_index()

                            fig8 = go.Figure()
                            fig8.add_trace(go.Bar(
                                x=importer_summary['importer_name'],
                                y=importer_summary['total_volume'],
                                name=T('legend_total_volume'),
                                marker_color=['red' if imp == customer_name else 'lightskyblue' for imp in importer_summary['importer_name']]
                            ))
                            fig8.add_trace(go.Scatter(
                                x=importer_summary['importer_name'],
                                y=importer_summary['avg_unit_price'],
                                name=T('legend_avg_import_price'),
                                yaxis='y2',
                                mode='lines+markers',
                                line=dict(color='orange')
                            ))
                            fig8.update_layout(
                                title=f"<b>{T('p2_exporter_bar_title', exporter=exporter)}</b>",
                                xaxis_title=T('col_importer'),
                                yaxis=dict(title=T('axis_total_volume')),
                                yaxis2=dict(title=T('axis_avg_import_price'), overlaying='y', side='right'),
                                legend=dict(x=0, y=1.1, orientation='h')
                            )
                            st.plotly_chart(fig8, use_container_width=True)

                            st.subheader(T('p2_price_dist_subheader'))
                            top_10_importers_by_vol = single_exporter_df.groupby('importer_name')['volume'].sum().nlargest(10).index
                            single_exporter_df_top10 = single_exporter_df[single_exporter_df['importer_name'].isin(top_10_importers_by_vol)]
                            
                            importers_in_plot = single_exporter_df_top10['importer_name'].unique()
                            competitors = [imp for imp in importers_in_plot if imp != customer_name]
                            blue_shades = px.colors.sequential.Blues_r
                            color_map_box = {comp: blue_shades[i % len(blue_shades)] for i, comp in enumerate(competitors)}
                            color_map_box[customer_name] = 'red'

                            fig10 = px.box(single_exporter_df_top10, x='importer_name', y='unit_price', 
                                           title=f"<b>{T('p2_exporter_box2_title', exporter=exporter)}</b><br><span style='font-size: 0.8em; color:grey;'>{T('p2_exporter_box2_subtitle')}</span>", 
                                           labels={'importer_name': T('col_importer'), 'unit_price': T('axis_unit_price')}, color='importer_name', color_discrete_map=color_map_box)
                            st.plotly_chart(fig10, use_container_width=True)
                            with st.expander(T('p2_detail_data_expander')):
                                summary_df_imp = single_exporter_df_top10.groupby('importer_name')['unit_price'].agg(['max', 'mean', 'min']).reset_index()
                                summary_df_imp.columns = [T('col_importer'), T('col_max_price'), T('col_avg_price'), T('col_min_price')]
                                st.dataframe(summary_df_imp.style.format({T('col_max_price'): '${:,.2f}', T('col_avg_price'): '${:,.2f}', T('col_min_price'): '${:,.2f}'}))

                    st.subheader(T('p2_alt_sourcing_subheader', year=selected_year_exporter))
                    
                    # --- [수정 24] 'export_country'를 사용하도록 groupby 컬럼 변경 ---
                    avg_prices = exporter_analysis_df.groupby(['quarter', 'Exporter', 'export_country']).agg(avg_price=('unit_price', 'mean'), representative_product=('product_name', 'first')).reset_index()
                    
                    # --- [수정 25] 'export_country'를 기준으로 고객사 원산지 추출 ---
                    customer_origins = exporter_analysis_df[exporter_analysis_df['importer_name'] == customer_name]['export_country'].unique()


                    # <<-- 분기별 소싱 옵션을 각각의 expander에 표시 -->>
                    for q in range(1, 5):
                        with st.expander(T('p2_q_expander', q=q), expanded=True): # 이 expander도 페이지가 나뉘지 않습니다.
                            q_df = avg_prices[avg_prices['quarter'] == q]
                            if q_df.empty:
                                st.write(T('p2_no_quarter_data'))
                                continue
                            
                            st.markdown(T('p2_current_sourcing_md'))
                            customer_exporters_q_df = q_df[q_df['Exporter'].isin(customer_exporters_in_year)].sort_values('avg_price')
                            if not customer_exporters_q_df.empty:
                                st.dataframe(customer_exporters_q_df[['Exporter', 'avg_price']].rename(columns={'Exporter': T('col_supplier'), 'avg_price': T('col_avg_price2')}).style.format({T('col_avg_price2'): '${:,.2f}'}))
                            else:
                                st.write(T('p2_no_exporter_deals'))
                            
                            # --- [수정 26] 'export_country' 기준으로 현재 원산지 옵션 표시 ---
                            customer_origins_q_df = q_df[q_df['export_country'].isin(customer_origins)].groupby('export_country')['avg_price'].mean().reset_index().sort_values('avg_price')
                            if not customer_origins_q_df.empty:
                                st.dataframe(customer_origins_q_df.rename(columns={'export_country': T('col_origin'), 'avg_price': T('col_avg_price2')}).style.format({T('col_avg_price2'): '${:,.2f}'}))
                            else:
                                st.write(T('p2_no_origin_deals'))

                            st.markdown(T('p2_alt_recommend_md'))
                            customer_avg_price_q = q_df[q_df['Exporter'].isin(customer_exporters_in_year)]['avg_price'].mean()
                            if not pd.isna(customer_avg_price_q):
                                cheaper_exporters = q_df[(~q_df['Exporter'].isin(customer_exporters_in_year)) & (q_df['avg_price'] < customer_avg_price_q)].sort_values('avg_price')
                                if not cheaper_exporters.empty:
                                    st.dataframe(cheaper_exporters[['Exporter', 'representative_product', 'avg_price']].rename(columns={'Exporter': T('col_recommend_supplier'), 'representative_product': T('col_rep_item'), 'avg_price': T('col_avg_price2')}).style.format({T('col_avg_price2'): '${:,.2f}'}))
                                else:
                                    st.write(T('p2_no_cheaper_exporter'))
                            
                            # --- [수정 27] 'export_country' 기준으로 대안 원산지 탐색 ---
                            customer_origin_avg_price_q = q_df[q_df['export_country'].isin(customer_origins)].groupby('export_country')['avg_price'].mean().mean()
                            if not pd.isna(customer_origin_avg_price_q):
                                cheaper_origins = q_df.groupby('export_country')['avg_price'].mean().reset_index()
                                cheaper_origins = cheaper_origins[(~cheaper_origins['export_country'].isin(customer_origins)) & (cheaper_origins['avg_price'] < customer_origin_avg_price_q)].sort_values('avg_price')
                                if not cheaper_origins.empty:
                                    st.dataframe(cheaper_origins.rename(columns={'export_country': T('col_recommend_origin'), 'avg_price': T('col_avg_price2')}).style.format({T('col_avg_price2'): '${:,.2f}'}))
                                else:
                                    st.write(T('p2_no_cheaper_origin'))
        # --- [수정 28] 경고 메시지 변경 ---
        else:
            st.warning(T('p2_no_supply_chain_cols_warning'))

# ==============================================================================
# 페이지 3: 공급망 흐름도 (Sankey)
# ==============================================================================

# ==============================================================================
# 페이지 3: 공급망 흐름도 (Sankey) — 2개 기간 비교 지원
# ==============================================================================
if selected == T('menu_opt_flow'):
    st.title(T('p3_title'))

    if st.session_state.flow_result is not None:
        st.button(T('p3_reset_btn'), on_click=reset_flow_states)

    if st.session_state.flow_result is None:
        flow_file = st.file_uploader(T('p3_upload_label'), type=['csv', 'xlsx'], key="flow_uploader")
        st.caption(T('p3_upload_caption'))
        raw_df = load_uploaded_df(flow_file, 'flow_raw_df', 'flow_headers', 'flow_fileid')

        if raw_df is not None:
            headers = st.session_state.flow_headers
            cols = detect_standard_columns(headers)
            missing = [k for k in ['date', 'importer', 'exporter', 'origin', 'export_country', 'product', 'volume', 'price'] if not cols[k]]
            if missing:
                st.error(T('p3_missing_cols_error', cols=', '.join(missing)))
                st.stop()

            AXIS_COL_MAP = build_axis_map([
                (T('p3_axis_exporter'), cols['exporter']),
                (T('p3_axis_importer'), cols['importer']),
                (T('p3_axis_origin'), cols['origin']),
            ], raw_df, cols)
            RIGHT_AXIS_COL_MAP = build_axis_map([
                (T('p3_axis_country'), cols['export_country']),
                (T('p3_axis_exporter'), cols['exporter']),
                (T('p3_axis_importer'), cols['importer']),
                (T('p3_axis_origin'), cols['origin']),
                (T('p3_axis_product'), cols['product']),
            ], raw_df, cols)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{T('p3_left_axis_label')}**")
                left_axis_label = st.selectbox(T('p3_left_axis_label'), options=list(AXIS_COL_MAP.keys()), label_visibility="collapsed", key="flow_left_axis")
                left_col = AXIS_COL_MAP[left_axis_label]
                left_entity_options = sorted(raw_df[left_col].dropna().astype(str).unique())
                left_entity = st.selectbox(T('p3_left_entity_label'), options=left_entity_options, key="flow_left_entity")
            with col2:
                st.markdown(f"**{T('p3_right_axis_label')}**")
                right_axis_label = st.selectbox(T('p3_right_axis_label'), options=list(RIGHT_AXIS_COL_MAP.keys()), label_visibility="collapsed", key="flow_right_axis")
                right_col = RIGHT_AXIS_COL_MAP[right_axis_label]
            st.caption(T('p3_axis_hint'))

            _parsed_dates = pd.to_datetime(raw_df[cols['date']], errors='coerce').dropna()
            _min_date = _parsed_dates.min().date() if len(_parsed_dates) else datetime.date.today()
            _max_date = _parsed_dates.max().date() if len(_parsed_dates) else datetime.date.today()
            _mid_date = _min_date + (_max_date - _min_date) / 2

            compare_mode = st.checkbox(T('p3_compare_years_label'), key="flow_compare_mode")

            if not compare_mode:
                col3, col4 = st.columns(2)
                with col3:
                    start_date = st.date_input(T('p3_date_start'), value=_min_date, key="flow_start_date")
                with col4:
                    end_date = st.date_input(T('p3_date_end'), value=_max_date, key="flow_end_date")
            else:
                st.markdown(f"**{T('p3_period_a')}**")
                colA1, colA2 = st.columns(2)
                with colA1:
                    a_start = st.date_input(T('p4_date_start'), value=_min_date, key="flow_a_start")
                with colA2:
                    a_end = st.date_input(T('p4_date_end'), value=_mid_date, key="flow_a_end")
                st.markdown(f"**{T('p3_period_b')}**")
                colB1, colB2 = st.columns(2)
                with colB1:
                    b_start = st.date_input(T('p4_date_start'), value=_mid_date, key="flow_b_start")
                with colB2:
                    b_end = st.date_input(T('p4_date_end'), value=_max_date, key="flow_b_end")

            if st.button(T('p3_run_btn')):
                df = raw_df.copy()
                df['_date'] = pd.to_datetime(df[cols['date']], errors='coerce')
                df['_volume'] = pd.to_numeric(df[cols['volume']], errors='coerce')
                df['_price'] = pd.to_numeric(df[cols['price']], errors='coerce')
                df = df.dropna(subset=['_date', '_volume', left_col, right_col])

                def _flow_grouped_for_range(d_start, d_end):
                    m = (df['_date'] >= pd.to_datetime(d_start)) & (df['_date'] <= pd.to_datetime(d_end)) & (df[left_col].astype(str) == str(left_entity))
                    s = df[m]
                    if s.empty:
                        return None
                    g = s.groupby(right_col).agg(volume=('_volume', 'sum'), avg_price=('_price', 'mean')).reset_index()
                    return g.sort_values('volume', ascending=False)

                if not compare_mode:
                    grouped = _flow_grouped_for_range(start_date, end_date)
                    if grouped is None:
                        st.warning(T('p3_no_data_warning'))
                    else:
                        TOP_N = 8
                        truncated = len(grouped) > TOP_N
                        if truncated:
                            top = grouped.iloc[:TOP_N].copy()
                            rest = grouped.iloc[TOP_N:]
                            others_vol = rest['volume'].sum()
                            others_price = (rest['volume'] * rest['avg_price']).sum() / others_vol if others_vol > 0 else 0
                            others_row = pd.DataFrame([{right_col: T('p2_others_label'), 'volume': others_vol, 'avg_price': others_price}])
                            grouped = pd.concat([top, others_row], ignore_index=True)
                        st.session_state.flow_result = {
                            'mode': 'single', 'left_axis_label': left_axis_label, 'left_entity': left_entity,
                            'right_axis_label': right_axis_label, 'grouped': grouped, 'right_col': right_col,
                            'start_date': start_date, 'end_date': end_date, 'top_n': TOP_N, 'truncated': truncated,
                        }
                        st.rerun()
                else:
                    grouped_a = _flow_grouped_for_range(a_start, a_end)
                    grouped_b = _flow_grouped_for_range(b_start, b_end)
                    if grouped_a is None or grouped_b is None:
                        st.warning(T('p3_no_data_warning'))
                    else:
                        merged = pd.merge(
                            grouped_a[[right_col, 'volume']].rename(columns={'volume': 'volume_a'}),
                            grouped_b[[right_col, 'volume']].rename(columns={'volume': 'volume_b'}),
                            on=right_col, how='outer'
                        ).fillna(0)
                        merged['diff'] = merged['volume_b'] - merged['volume_a']
                        merged['diff_pct'] = merged.apply(lambda r: (r['diff'] / r['volume_a'] * 100) if r['volume_a'] > 0 else None, axis=1)
                        merged = merged.sort_values('volume_b', ascending=False)

                        st.session_state.flow_result = {
                            'mode': 'compare', 'left_axis_label': left_axis_label, 'left_entity': left_entity,
                            'right_axis_label': right_axis_label, 'right_col': right_col,
                            'grouped_a': grouped_a.head(8), 'grouped_b': grouped_b.head(8), 'merged': merged,
                            'a_start': a_start, 'a_end': a_end, 'b_start': b_start, 'b_end': b_end,
                        }
                        st.rerun()

    if st.session_state.flow_result is not None:
        R = st.session_state.flow_result
        pdf_figs = []
        pdf_table = None

        if R['mode'] == 'single':
            grouped = R['grouped']
            st.subheader(T('p3_result_subheader', left=R['left_entity'], left_axis=R['left_axis_label'], right_axis=R['right_axis_label'], start=R['start_date'], end=R['end_date']))

            labels = [R['left_entity']] + grouped[R['right_col']].astype(str).tolist()
            n_targets = len(grouped)
            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=20, thickness=20, line=dict(color='rgba(0,0,0,0.2)', width=0.5), label=labels, color=['#0d9488'] + ['#60a5fa'] * n_targets),
                link=dict(source=[0] * n_targets, target=list(range(1, n_targets + 1)), value=grouped['volume'].tolist(), color=['rgba(13,148,136,0.35)'] * n_targets)
            )])
            fig.update_layout(title=T('p3_sankey_title', left=R['left_entity'], right_axis=R['right_axis_label']), font_size=13, height=max(350, 60 * n_targets))
            st.plotly_chart(fig, use_container_width=True)
            pdf_figs.append(fig)

            if R.get('truncated'):
                st.caption(T('p3_others_note', n=R['top_n']))

            st.subheader(T('p3_table_subheader'))
            category_col_label = T('p3_col_category', right_axis=R['right_axis_label'])
            display_df = grouped.copy()
            display_df['share'] = display_df['volume'] / display_df['volume'].sum()
            display_df = display_df.rename(columns={R['right_col']: category_col_label, 'volume': T('p3_col_volume'), 'avg_price': T('p3_col_avg_price'), 'share': T('p3_col_share')})
            display_df = display_df[[category_col_label, T('p3_col_volume'), T('p3_col_share'), T('p3_col_avg_price')]]
            st.dataframe(display_df.style.format({T('p3_col_volume'): '{:,.0f}', T('p3_col_share'): '{:.1%}', T('p3_col_avg_price'): '${:,.2f}'}))
            pdf_table = display_df
        else:
            st.subheader(T('p3_compare_subheader', left=R['left_entity'], right_axis=R['right_axis_label']))
            colA, colB = st.columns(2)
            for col_ui, grouped, period_label, period_range in [
                (colA, R['grouped_a'], T('p3_period_a'), f"{R['a_start']} ~ {R['a_end']}"),
                (colB, R['grouped_b'], T('p3_period_b'), f"{R['b_start']} ~ {R['b_end']}"),
            ]:
                with col_ui:
                    st.caption(f"{period_label}: {period_range}")
                    labels = [R['left_entity']] + grouped[R['right_col']].astype(str).tolist()
                    n_targets = len(grouped)
                    fig = go.Figure(data=[go.Sankey(
                        node=dict(pad=15, thickness=16, line=dict(color='rgba(0,0,0,0.2)', width=0.5), label=labels, color=['#0d9488'] + ['#60a5fa'] * n_targets),
                        link=dict(source=[0] * n_targets, target=list(range(1, n_targets + 1)), value=grouped['volume'].tolist(), color=['rgba(13,148,136,0.35)'] * n_targets)
                    )])
                    fig.update_layout(font_size=11, height=max(300, 50 * n_targets), margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    pdf_figs.append(fig)

            st.subheader(T('p3_table_subheader'))
            category_col_label = T('p3_col_category', right_axis=R['right_axis_label'])
            merged = R['merged'].rename(columns={R['right_col']: category_col_label, 'volume_a': T('p3_compare_col_a'), 'volume_b': T('p3_compare_col_b'), 'diff': T('p3_compare_col_diff'), 'diff_pct': T('p3_compare_col_diff_pct')})
            st.dataframe(merged.style.format({T('p3_compare_col_a'): '{:,.0f}', T('p3_compare_col_b'): '{:,.0f}', T('p3_compare_col_diff'): '{:+,.0f}', T('p3_compare_col_diff_pct'): '{:+.1f}%'}))
            pdf_table = merged

        if st.button(T('pdf_generate_btn'), key="flow_pdf_btn"):
            with st.spinner(T('pdf_generating_msg')):
                try:
                    pdf_bytes = build_pdf_report(
                        title=T('p3_title'),
                        kpi_lines=[f"{T('p3_left_entity_label')}: {R['left_entity']} ({R['left_axis_label']})"],
                        figs=pdf_figs, df_table=pdf_table, table_title=T('p3_table_subheader'),
                    )
                    st.download_button(T('pdf_download_btn'), data=pdf_bytes, file_name="supply_chain_flow.pdf", mime="application/pdf", key="flow_pdf_dl")
                except Exception as e:
                    st.error(T('pdf_error_msg', msg=str(e)))

# ==============================================================================
# 페이지 4: 집중도 리스크 진단
# ==============================================================================
if selected == T('menu_opt_risk'):
    st.title(T('p4_title'))

    if st.session_state.risk_result is not None:
        st.button(T('p4_reset_btn'), on_click=reset_risk_states)

    if st.session_state.risk_result is None:
        risk_file = st.file_uploader(T('p4_upload_label'), type=['csv', 'xlsx'], key="risk_uploader")
        st.caption(T('p4_upload_caption'))
        raw_df = load_uploaded_df(risk_file, 'risk_raw_df', 'risk_headers', 'risk_fileid')

        if raw_df is not None:
            headers = st.session_state.risk_headers
            cols = detect_standard_columns(headers)
            missing = [k for k in ['date', 'exporter', 'origin', 'product', 'volume'] if not cols[k]]
            if missing:
                st.error(T('p4_missing_cols_error', cols=', '.join(missing)))
                st.stop()

            AXIS_MAP = build_axis_map([
                (T('p4_axis_exporter'), cols['exporter']),
                (T('p4_axis_origin'), cols['origin']),
                (T('p4_axis_product'), cols['product']),
            ], raw_df, cols)

            col1, col2 = st.columns(2)
            with col1:
                axis_label = st.selectbox(T('p4_axis_label'), options=list(AXIS_MAP.keys()), key="risk_axis")
                axis_col = AXIS_MAP[axis_label]
            with col2:
                scope_label = st.selectbox(T('p4_scope_label'), options=[T('p4_scope_all'), T('p4_scope_importer')], key="risk_scope")

            scope_entity = None
            if scope_label == T('p4_scope_importer') and cols['importer']:
                importer_options = sorted(raw_df[cols['importer']].dropna().astype(str).unique())
                scope_entity = st.selectbox(T('p4_scope_entity_label'), options=importer_options, key="risk_scope_entity")

            _parsed_dates = pd.to_datetime(raw_df[cols['date']], errors='coerce').dropna()
            _min_date = _parsed_dates.min().date() if len(_parsed_dates) else datetime.date.today()
            _max_date = _parsed_dates.max().date() if len(_parsed_dates) else datetime.date.today()

            col3, col4 = st.columns(2)
            with col3:
                start_date = st.date_input(T('p4_date_start'), value=_min_date, key="risk_start_date")
            with col4:
                end_date = st.date_input(T('p4_date_end'), value=_max_date, key="risk_end_date")

            threshold = st.slider(T('p4_threshold_label'), min_value=20, max_value=80, value=50, step=5, key="risk_threshold")

            if st.button(T('p4_run_btn')):
                df = raw_df.copy()
                df['_date'] = pd.to_datetime(df[cols['date']], errors='coerce')
                df['_volume'] = pd.to_numeric(df[cols['volume']], errors='coerce')
                df = df.dropna(subset=['_date', '_volume', axis_col])

                mask = (df['_date'] >= pd.to_datetime(start_date)) & (df['_date'] <= pd.to_datetime(end_date))
                if scope_entity and cols['importer']:
                    mask &= (df[cols['importer']].astype(str) == str(scope_entity))
                sub = df[mask]

                if sub.empty:
                    st.warning(T('p4_no_data_warning'))
                else:
                    grouped = sub.groupby(axis_col)['_volume'].sum().reset_index().rename(columns={axis_col: 'name', '_volume': 'volume'})
                    grouped = grouped.sort_values('volume', ascending=False)
                    total = grouped['volume'].sum()
                    grouped['share'] = grouped['volume'] / total * 100

                    sub_copy = sub.copy()
                    sub_copy['_ym'] = sub_copy['_date'].dt.to_period('M').astype(str)
                    monthly = sub_copy.groupby(['_ym', axis_col])['_volume'].sum().reset_index()
                    monthly_total = sub_copy.groupby('_ym')['_volume'].sum().rename('total')
                    monthly = monthly.merge(monthly_total, on='_ym')
                    monthly['share'] = monthly['_volume'] / monthly['total'] * 100
                    trend = monthly.sort_values(['_ym', 'share'], ascending=[True, False]).groupby('_ym').first().reset_index().sort_values('_ym')

                    st.session_state.risk_result = {'axis_label': axis_label, 'grouped': grouped, 'threshold': threshold, 'trend': trend}
                    st.rerun()

    if st.session_state.risk_result is not None:
        R = st.session_state.risk_result
        grouped = R['grouped']
        threshold = R['threshold']
        top1 = grouped.iloc[0]
        top3_share = grouped.iloc[:3]['share'].sum()

        danger_top3 = min(95, threshold + 30)
        caution_top1 = threshold * 0.6
        caution_top3 = threshold + 10

        if top1['share'] >= threshold or top3_share >= danger_top3:
            risk_label = T('p4_risk_danger')
            reason = T('p4_risk_reason_top1', v=round(top1['share'], 1), t=threshold) if top1['share'] >= threshold else T('p4_risk_reason_top3', v=round(top3_share, 1), t=round(danger_top3, 1))
            risk_color = 'red'
        elif top1['share'] >= caution_top1 or top3_share >= caution_top3:
            risk_label = T('p4_risk_caution')
            reason = T('p4_risk_reason_top1', v=round(top1['share'], 1), t=round(caution_top1, 1)) if top1['share'] >= caution_top1 else T('p4_risk_reason_top3', v=round(top3_share, 1), t=round(caution_top3, 1))
            risk_color = 'orange'
        else:
            risk_label = T('p4_risk_safe')
            reason = T('p4_risk_reason_ok')
            risk_color = 'green'

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T('p4_kpi_top1'), f"{top1['share']:.1f}%", help=str(top1['name']))
        c2.metric(T('p4_kpi_top3'), f"{top3_share:.1f}%")
        c3.metric(T('p4_kpi_count'), f"{len(grouped)}")
        with c4:
            st.markdown(f"**{T('p4_kpi_risk')}**")
            badge_bg = {'red': '#fdecec', 'orange': '#fff7e6', 'green': '#e9f9ee'}[risk_color]
            badge_fg = {'red': '#b3261e', 'orange': '#92620b', 'green': '#0f7a3c'}[risk_color]
            st.markdown(f"<span style='background:{badge_bg};color:{badge_fg};padding:4px 12px;border-radius:8px;font-weight:700;'>{risk_label}</span>", unsafe_allow_html=True)
        st.caption(reason)

        display_grouped = grouped.head(12)
        bar_colors = ['#e11d48' if s >= threshold else '#0d9488' for s in display_grouped['share']]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=display_grouped['name'], x=display_grouped['share'], orientation='h', marker_color=bar_colors))
        fig.add_vline(x=threshold, line_dash='dash', line_color='#e11d48')
        fig.update_layout(title=T('p4_bar_chart_title', axis=R['axis_label'], t=threshold), xaxis_title=T('p4_axis_share'), yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

        figs_for_pdf = [fig]
        if len(R['trend']) > 1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=R['trend']['_ym'], y=R['trend']['share'], mode='lines+markers', line=dict(color='#0d9488')))
            fig2.update_layout(title=T('p4_trend_chart_title'), xaxis_title=T('axis_yearmonth'), yaxis_title=T('p4_axis_share'))
            st.plotly_chart(fig2, use_container_width=True)
            figs_for_pdf.append(fig2)

        display_df = grouped.rename(columns={'name': T('p4_col_name'), 'volume': T('p4_col_volume'), 'share': T('p4_col_share')})
        st.dataframe(display_df.style.format({T('p4_col_volume'): '{:,.0f}', T('p4_col_share'): '{:.1f}%'}))

        if st.button(T('pdf_generate_btn'), key="risk_pdf_btn"):
            with st.spinner(T('pdf_generating_msg')):
                try:
                    pdf_bytes = build_pdf_report(
                        title=T('p4_title'),
                        kpi_lines=[f"{T('p4_kpi_top1')}: {top1['share']:.1f}%", f"{T('p4_kpi_top3')}: {top3_share:.1f}%", f"{T('p4_kpi_risk')}: {risk_label} ({reason})"],
                        figs=figs_for_pdf, df_table=display_df, table_title=T('p4_col_name'),
                    )
                    st.download_button(T('pdf_download_btn'), data=pdf_bytes, file_name="concentration_risk.pdf", mime="application/pdf", key="risk_pdf_dl")
                except Exception as e:
                    st.error(T('pdf_error_msg', msg=str(e)))

# ==============================================================================
# 페이지 5: 가격 추세 & 계절성
# ==============================================================================
if selected == T('menu_opt_season'):
    st.title(T('p5_title'))

    if st.session_state.season_result is not None:
        st.button(T('p5_reset_btn'), on_click=reset_season_states)

    if st.session_state.season_result is None:
        season_file = st.file_uploader(T('p5_upload_label'), type=['csv', 'xlsx'], key="season_uploader")
        st.caption(T('p5_upload_caption'))
        raw_df = load_uploaded_df(season_file, 'season_raw_df', 'season_headers', 'season_fileid')

        if raw_df is not None:
            headers = st.session_state.season_headers
            cols = detect_standard_columns(headers)
            missing = [k for k in ['date', 'product', 'price'] if not cols[k]]
            if missing:
                st.error(T('p5_missing_cols_error', cols=', '.join(missing)))
                st.stop()

            raw_products = raw_df[cols['product']].dropna().astype(str).unique()
            product_groups = cluster_product_names(raw_products)
            selected_reps = st.multiselect(T('multi_product_label'), options=sorted(product_groups.keys()), key="season_products")
            st.caption(T('multi_product_help'))

            col1, col2 = st.columns(2)
            with col1:
                show_prev_year = st.checkbox(T('p5_prevyear_label'), value=True, key="season_prevyear")
            with col2:
                breakdown_options = {T('p5_breakdown_none'): 'none'}
                if cols['origin']:
                    breakdown_options[T('p5_breakdown_origin')] = 'origin'
                if cols['exporter']:
                    breakdown_options[T('p5_breakdown_exporter')] = 'exporter'
                breakdown_label = st.selectbox(T('p5_breakdown_label'), options=list(breakdown_options.keys()), key="season_breakdown")
                breakdown = breakdown_options[breakdown_label]

            if st.button(T('p5_run_btn')):
                if not selected_reps:
                    st.warning(T('p5_no_data_warning'))
                else:
                    selected_raw_names = set()
                    for rep in selected_reps:
                        selected_raw_names.update(product_groups[rep])

                    df = raw_df.copy()
                    df['_date'] = pd.to_datetime(df[cols['date']], errors='coerce')
                    df['_price'] = pd.to_numeric(df[cols['price']], errors='coerce')
                    df = df.dropna(subset=['_date', '_price', cols['product']])
                    sub = df[df[cols['product']].astype(str).isin(selected_raw_names)]

                    if sub.empty:
                        st.warning(T('p5_no_data_warning'))
                    else:
                        sub = sub.copy()
                        sub['_ym'] = sub['_date'].dt.to_period('M')
                        sub['_month'] = sub['_date'].dt.month
                        sub['_year'] = sub['_date'].dt.year

                        overall = sub.groupby('_ym')['_price'].mean().reset_index().sort_values('_ym')
                        overall['_ym_str'] = overall['_ym'].astype(str)

                        month_avg = sub.groupby('_month')['_price'].mean()
                        years_count = sub['_year'].nunique()
                        overall_avg = sub['_price'].mean()
                        cheapest_month = int(month_avg.idxmin())
                        cheapest_month_pct = (overall_avg - month_avg.min()) / overall_avg * 100

                        latest_year = sub['_year'].max()
                        monthly_avg_latest_year = sub[sub['_year'] == latest_year].groupby('_month')['_price'].mean()
                        peak_months = []
                        if len(monthly_avg_latest_year) >= 3:
                            lyr_mean = monthly_avg_latest_year.mean()
                            peak_months = sorted(monthly_avg_latest_year[monthly_avg_latest_year > lyr_mean * 1.05].index.tolist())

                        breakdown_series = {}
                        if breakdown != 'none':
                            bcol = cols[breakdown]
                            for name, g in sub.groupby(bcol):
                                s = g.groupby('_ym')['_price'].mean().reset_index().sort_values('_ym')
                                s['_ym_str'] = s['_ym'].astype(str)
                                breakdown_series[str(name)] = s

                        st.session_state.season_result = {
                            'product_label': ', '.join(selected_reps), 'overall': overall,
                            'show_prev_year': show_prev_year, 'breakdown': breakdown,
                            'breakdown_series': breakdown_series, 'peak_months': peak_months,
                            'cheapest_month': cheapest_month, 'cheapest_month_pct': cheapest_month_pct,
                            'years_count': years_count, 'overall_avg': overall_avg,
                        }
                        st.rerun()

    if st.session_state.season_result is not None:
        R = st.session_state.season_result
        overall = R['overall']

        ym_to_price = dict(zip(overall['_ym'], overall['_price']))
        current_ym = overall['_ym'].iloc[-1]
        current_price = overall['_price'].iloc[-1]
        peak_price = overall['_price'].max()
        from_peak = (current_price - peak_price) / peak_price * 100

        prev_ym = current_ym - 12
        prev_price = ym_to_price.get(prev_ym)
        yoy = (current_price - prev_price) / prev_price * 100 if prev_price else None

        c1, c2, c3 = st.columns(3)
        c1.metric(T('p5_kpi_current'), f"${current_price:.2f}")
        c2.metric(T('p5_kpi_yoy'), f"{yoy:+.1f}%" if yoy is not None else "N/A")
        c3.metric(T('p5_kpi_frompeak'), f"{from_peak:+.1f}%")

        if R['peak_months']:
            months_str = ', '.join([f"{m}월" if st.session_state.lang == 'ko' else f"Month {m}" for m in R['peak_months']])
            st.info(T('p5_season_badge', months=months_str))
        else:
            st.caption(T('p5_no_season'))

        fig = go.Figure()
        if R['breakdown'] == 'none':
            fig.add_trace(go.Scatter(x=overall['_ym_str'], y=overall['_price'], mode='lines+markers', name=T('p5_legend_this_year'), line=dict(color='#0d9488')))
        else:
            colors = ['#0d9488', '#e11d48', '#2563eb', '#f59e0b', '#a855f7']
            for i, (name, s) in enumerate(R['breakdown_series'].items()):
                fig.add_trace(go.Scatter(x=s['_ym_str'], y=s['_price'], mode='lines+markers', name=name, line=dict(color=colors[i % len(colors)])))

        prev_year_added = False
        if R['show_prev_year']:
            prev_year_prices = [ym_to_price.get(ym - 12) for ym in overall['_ym']]
            if any(v is not None for v in prev_year_prices):
                fig.add_trace(go.Scatter(x=overall['_ym_str'], y=prev_year_prices, mode='lines', name=T('p5_legend_prev_year'), line=dict(color='#94a3b8', dash='dash'), connectgaps=True))
                prev_year_added = True

        fig.update_layout(title=T('p5_chart_title', product=R['product_label']), xaxis_title=T('p5_axis_month'), yaxis_title=T('p5_axis_price'))
        st.plotly_chart(fig, use_container_width=True)

        if R['show_prev_year'] and not prev_year_added:
            st.caption('※ ' + ('전년도에 해당하는 데이터가 없어 전년 비교선을 표시하지 못했습니다 (최소 13개월치 데이터가 필요합니다).' if st.session_state.lang == 'ko' else 'No prior-year data available to overlay (needs at least 13 months of history).'))

        st.markdown(f"### {T('insight_box_title')}")
        month_label = f"{R['cheapest_month']}월" if st.session_state.lang == 'ko' else f"Month {R['cheapest_month']}"
        st.write("- " + T('insight_cheapest_month', month=month_label, pct=f"{R['cheapest_month_pct']:.1f}"))
        cur_vs_avg_pct = (current_price - R['overall_avg']) / R['overall_avg'] * 100
        direction = T('direction_higher') if cur_vs_avg_pct >= 0 else T('direction_lower')
        sign = '+' if cur_vs_avg_pct >= 0 else ''
        st.write("- " + T('insight_current_vs_seasonal_avg', sign=sign, pct=f"{abs(cur_vs_avg_pct):.1f}", direction=direction))
        if R['years_count'] < 2:
            st.warning(T('insight_low_confidence', years=R['years_count']))

        if st.button(T('pdf_generate_btn'), key="season_pdf_btn"):
            with st.spinner(T('pdf_generating_msg')):
                try:
                    pdf_bytes = build_pdf_report(
                        title=T('p5_title'),
                        kpi_lines=[
                            f"{T('p5_kpi_current')}: ${current_price:.2f}",
                            f"{T('p5_kpi_yoy')}: {yoy:+.1f}%" if yoy is not None else f"{T('p5_kpi_yoy')}: N/A",
                            f"{T('p5_kpi_frompeak')}: {from_peak:+.1f}%",
                        ],
                        figs=[fig], df_table=overall[['_ym_str', '_price']].rename(columns={'_ym_str': T('p5_axis_month'), '_price': T('p5_axis_price')}),
                        table_title=T('p5_axis_month'),
                    )
                    st.download_button(T('pdf_download_btn'), data=pdf_bytes, file_name="price_seasonality.pdf", mime="application/pdf", key="season_pdf_dl")
                except Exception as e:
                    st.error(T('pdf_error_msg', msg=str(e)))

# ==============================================================================
# 페이지 6: 신규·이탈 거래처 추적 (범위 토글 추가)
# ==============================================================================
if selected == T('menu_opt_churn'):
    st.title(T('p6_title'))

    if st.session_state.churn_result is not None:
        st.button(T('p6_reset_btn'), on_click=reset_churn_states)

    if st.session_state.churn_result is None:
        churn_file = st.file_uploader(T('p6_upload_label'), type=['csv', 'xlsx'], key="churn_uploader")
        st.caption(T('p6_upload_caption'))
        raw_df = load_uploaded_df(churn_file, 'churn_raw_df', 'churn_headers', 'churn_fileid')

        if raw_df is not None:
            headers = st.session_state.churn_headers
            cols = detect_standard_columns(headers)
            missing = [k for k in ['date', 'volume'] if not cols[k]]
            if missing:
                st.error(T('p6_missing_cols_error', cols=', '.join(missing)))
                st.stop()

            AXIS_MAP = build_axis_map([
                (T('p6_axis_exporter'), cols['exporter']),
                (T('p6_axis_origin'), cols['origin']),
                (T('p6_axis_product'), cols['product']),
                (T('p6_axis_importer'), cols['importer']),
            ], raw_df, cols)
            if not AXIS_MAP:
                st.error(T('p6_missing_cols_error', cols='Exporter/origin_country/product_name/importer_name'))
                st.stop()

            col1, col2 = st.columns(2)
            with col1:
                axis_label = st.selectbox(T('p6_axis_label'), options=list(AXIS_MAP.keys()), key="churn_axis")
                axis_col = AXIS_MAP[axis_label]
            with col2:
                scope_label = st.selectbox(T('p6_scope_label'), options=[T('p6_scope_all'), T('p6_scope_importer')], key="churn_scope")

            scope_entity = None
            if scope_label == T('p6_scope_all'):
                st.caption(T('p6_scope_all_caption'))
            elif cols['importer']:
                importer_options = sorted(raw_df[cols['importer']].dropna().astype(str).unique())
                scope_entity = st.selectbox(T('p6_scope_entity_label'), options=importer_options, key="churn_scope_entity")

            _parsed_dates = pd.to_datetime(raw_df[cols['date']], errors='coerce').dropna()
            _min_date = _parsed_dates.min().date() if len(_parsed_dates) else datetime.date.today()
            _max_date = _parsed_dates.max().date() if len(_parsed_dates) else datetime.date.today()
            _mid_date = _min_date + (_max_date - _min_date) / 2

            st.markdown(f"**{T('p6_period_a')}**")
            colA1, colA2 = st.columns(2)
            with colA1:
                a_start = st.date_input(T('p4_date_start'), value=_min_date, key="churn_a_start")
            with colA2:
                a_end = st.date_input(T('p4_date_end'), value=_mid_date, key="churn_a_end")

            st.markdown(f"**{T('p6_period_b')}**")
            colB1, colB2 = st.columns(2)
            with colB1:
                b_start = st.date_input(T('p4_date_start'), value=_mid_date, key="churn_b_start")
            with colB2:
                b_end = st.date_input(T('p4_date_end'), value=_max_date, key="churn_b_end")

            if st.button(T('p6_run_btn')):
                df = raw_df.copy()
                df['_date'] = pd.to_datetime(df[cols['date']], errors='coerce')
                df['_volume'] = pd.to_numeric(df[cols['volume']], errors='coerce')
                if cols['price']:
                    df['_price'] = pd.to_numeric(df[cols['price']], errors='coerce')
                df = df.dropna(subset=['_date', '_volume', axis_col])
                if scope_entity and cols['importer']:
                    df = df[df[cols['importer']].astype(str) == str(scope_entity)]

                a_df = df[(df['_date'] >= pd.to_datetime(a_start)) & (df['_date'] <= pd.to_datetime(a_end))]
                b_df = df[(df['_date'] >= pd.to_datetime(b_start)) & (df['_date'] <= pd.to_datetime(b_end))]

                if a_df.empty or b_df.empty:
                    st.warning(T('p6_no_data_warning'))
                else:
                    a_set = set(a_df[axis_col].astype(str).unique())
                    b_set = set(b_df[axis_col].astype(str).unique())
                    new_items = b_set - a_set
                    lost_items = a_set - b_set
                    kept_items = a_set & b_set

                    b_vol = b_df.groupby(axis_col)['_volume'].sum()
                    a_vol = a_df.groupby(axis_col)['_volume'].sum()

                    new_list = sorted([(n, float(b_vol.get(n, 0))) for n in new_items], key=lambda x: -x[1])
                    lost_list = sorted([(n, float(a_vol.get(n, 0))) for n in lost_items], key=lambda x: -x[1])

                    # --- ① 가격 비교: 신규/유지/이탈 거래처의 평균단가 ---
                    new_price = lost_price = kept_price = None
                    if cols['price']:
                        if new_items:
                            v = b_df[b_df[axis_col].astype(str).isin(new_items)]['_price'].mean()
                            new_price = float(v) if pd.notna(v) else None
                        if lost_items:
                            v = a_df[a_df[axis_col].astype(str).isin(lost_items)]['_price'].mean()
                            lost_price = float(v) if pd.notna(v) else None
                        if kept_items:
                            v = b_df[b_df[axis_col].astype(str).isin(kept_items)]['_price'].mean()
                            kept_price = float(v) if pd.notna(v) else None

                    # --- ② 집중도 변화: 기간 A→B 사이 1위 거래처 의존도가 어떻게 바뀌었는지 ---
                    def _top1_share(sub_df):
                        if sub_df.empty:
                            return None, None
                        g = sub_df.groupby(axis_col)['_volume'].sum().sort_values(ascending=False)
                        total = g.sum()
                        if total <= 0:
                            return None, None
                        return g.index[0], float(g.iloc[0] / total * 100)

                    top1_name_a, top1_share_a = _top1_share(a_df)
                    top1_name_b, top1_share_b = _top1_share(b_df)

                    st.session_state.churn_result = {
                        'axis_label': axis_label, 'scope_label': scope_label, 'scope_entity': scope_entity,
                        'new_list': new_list, 'lost_list': lost_list, 'kept_count': len(kept_items),
                        'new_price': new_price, 'lost_price': lost_price, 'kept_price': kept_price,
                        'top1_name_a': top1_name_a, 'top1_share_a': top1_share_a,
                        'top1_name_b': top1_name_b, 'top1_share_b': top1_share_b,
                    }
                    st.rerun()

    if st.session_state.churn_result is not None:
        R = st.session_state.churn_result
        scope_desc = R['scope_entity'] if R['scope_entity'] else R['scope_label']
        st.caption(f"{T('p6_scope_label')}: {scope_desc}")

        c1, c2, c3 = st.columns(3)
        c1.metric(T('p6_kpi_new'), f"{len(R['new_list'])}")
        c2.metric(T('p6_kpi_kept'), f"{R['kept_count']}")
        c3.metric(T('p6_kpi_lost'), f"{len(R['lost_list'])}")

        col1, col2 = st.columns(2)
        new_df = None
        lost_df = None
        with col1:
            st.subheader(T('p6_new_header'))
            if R['new_list']:
                new_df = pd.DataFrame(R['new_list'], columns=[T('p6_col_name'), T('p6_col_volume')])
                st.dataframe(new_df.style.format({T('p6_col_volume'): '{:,.0f}'}))
            else:
                st.caption(T('p6_no_new'))
        with col2:
            st.subheader(T('p6_lost_header'))
            if R['lost_list']:
                lost_df = pd.DataFrame(R['lost_list'], columns=[T('p6_col_name'), T('p6_col_volume')])
                st.dataframe(lost_df.style.format({T('p6_col_volume'): '{:,.0f}'}))
            else:
                st.caption(T('p6_no_lost'))

        # --- ① 가격 비교 인사이트 ---
        if R.get('new_price') is not None or R.get('lost_price') is not None or R.get('kept_price') is not None:
            st.markdown(f"### {T('p6_price_compare_subheader')}")
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric(T('p6_kpi_new_price'), f"${R['new_price']:.2f}" if R['new_price'] is not None else "N/A")
            pc2.metric(T('p6_kpi_kept_price'), f"${R['kept_price']:.2f}" if R['kept_price'] is not None else "N/A")
            pc3.metric(T('p6_kpi_lost_price'), f"${R['lost_price']:.2f}" if R['lost_price'] is not None else "N/A")
            if R['new_price'] is not None and R['kept_price'] is not None and R['kept_price'] > 0:
                diff_pct = (R['new_price'] - R['kept_price']) / R['kept_price'] * 100
                direction = T('direction_lower') if diff_pct < 0 else T('direction_higher')
                st.info(T('p6_price_insight_cheaper', pct=f"{abs(diff_pct):.1f}", direction=direction))
        else:
            st.caption(T('p6_price_insight_none'))

        # --- ② 집중도 변화 인사이트 ---
        if R.get('top1_share_a') is not None and R.get('top1_share_b') is not None:
            st.markdown(f"### {T('p6_concentration_subheader')}")
            a_pct, b_pct = R['top1_share_a'], R['top1_share_b']
            delta = b_pct - a_pct
            if delta >= 5:
                st.warning(T('p6_concentration_warning', a=f"{a_pct:.1f}", b=f"{b_pct:.1f}", name=str(R['top1_name_b'])))
            elif delta <= -5:
                st.success(T('p6_concentration_improved', a=f"{a_pct:.1f}", b=f"{b_pct:.1f}"))
            else:
                st.caption(T('p6_concentration_stable', a=f"{a_pct:.1f}", b=f"{b_pct:.1f}"))

        if st.button(T('pdf_generate_btn'), key="churn_pdf_btn"):
            with st.spinner(T('pdf_generating_msg')):
                try:
                    combined = pd.concat([
                        (new_df if new_df is not None else pd.DataFrame(columns=[T('p6_col_name'), T('p6_col_volume')])).assign(**{T('p6_kpi_new'): 'O'}),
                    ], ignore_index=True) if new_df is not None else None
                    pdf_bytes = build_pdf_report(
                        title=T('p6_title'),
                        kpi_lines=[f"{T('p6_kpi_new')}: {len(R['new_list'])}", f"{T('p6_kpi_kept')}: {R['kept_count']}", f"{T('p6_kpi_lost')}: {len(R['lost_list'])}"],
                        figs=[], df_table=new_df, table_title=T('p6_new_header'),
                    )
                    st.download_button(T('pdf_download_btn'), data=pdf_bytes, file_name="new_lost_partners.pdf", mime="application/pdf", key="churn_pdf_dl")
                except Exception as e:
                    st.error(T('pdf_error_msg', msg=str(e)))

# ==============================================================================
# 페이지 7: 자유 피벗 빌더 (동적 컬럼 + 필터 + 보기 확장)
# ==============================================================================
if selected == T('menu_opt_pivot'):
    st.title(T('p7_title'))

    if st.session_state.pivot_raw_df is None:
        pivot_file = st.file_uploader(T('p7_upload_label'), type=['csv', 'xlsx'], key="pivot_uploader")
        st.caption(T('p7_upload_caption'))
        load_uploaded_df(pivot_file, 'pivot_raw_df', 'pivot_headers', 'pivot_fileid')
        if st.session_state.pivot_raw_df is not None:
            st.rerun()

    if st.session_state.pivot_raw_df is not None:
        st.button(T('p7_reset_btn'), on_click=reset_pivot_states)
        raw_df = st.session_state.pivot_raw_df

        headers = st.session_state.pivot_headers
        cols = detect_standard_columns(headers)
        missing = [k for k in ['date', 'volume', 'price'] if not cols[k]]
        if missing:
            st.error(T('p7_missing_cols_error', cols=', '.join(missing)))
            st.stop()

        DIM_MAP = build_axis_map([
            (T('p7_row_exporter'), cols['exporter']),
            (T('p7_row_origin'), cols['origin']),
            (T('p7_row_product'), cols['product']),
            (T('p7_row_importer'), cols['importer']),
        ], raw_df, cols)

        ROW_OPTIONS = [T('p7_row_month')] + list(DIM_MAP.keys())

        METRIC_OPTIONS = {
            T('p7_metric_volume_sum'): ('_volume', 'sum'),
            T('p7_metric_volume_mean'): ('_volume', 'mean'),
            T('p7_metric_price_mean'): ('_price', 'mean'),
            T('p7_metric_price_max'): ('_price', 'max'),
            T('p7_metric_price_min'): ('_price', 'min'),
            T('p7_metric_count'): (None, 'count'),
        }
        VIEW_MAP = {
            T('p7_view_table'): 'table',
            T('p7_view_heatmap'): 'heatmap',
            T('p7_view_bar'): 'bar',
            T('p7_view_line'): 'line',
            T('p7_view_pie'): 'pie',
            T('p7_view_stacked'): 'stacked',
        }

        col1, col2 = st.columns(2)
        with col1:
            row_labels = st.multiselect(T('p7_row_label_multi'), options=ROW_OPTIONS, default=[ROW_OPTIONS[0]], key="pivot_rows")
        with col2:
            col_labels = st.multiselect(T('p7_col_label_multi'), options=list(DIM_MAP.keys()), key="pivot_cols")

        value_labels = st.multiselect(T('p7_values_label'), options=list(METRIC_OPTIONS.keys()), default=[T('p7_metric_volume_sum')], key="pivot_values")
        view_label = st.selectbox(T('p7_view_label'), options=list(VIEW_MAP.keys()), key="pivot_view")

        st.markdown(f"**{T('p7_filter_label')}**")
        filter_col_labels = st.multiselect(T('p7_filter_cols_label'), options=list(DIM_MAP.keys()), key="pivot_filter_cols")
        filter_selections = {}
        for fc_label in filter_col_labels:
            fcol = DIM_MAP[fc_label]
            with st.expander(fc_label):
                opts = sorted(raw_df[fcol].dropna().astype(str).unique())
                filter_selections[fcol] = st.multiselect(T('p7_filter_values'), options=opts, key=f"pivot_filter_vals_{fcol}")

        if st.button(T('p7_run_btn')):
            if not row_labels:
                st.warning(T('p7_no_rows_warning'))
            elif not value_labels:
                st.warning(T('p7_no_values_warning'))
            else:
                df = raw_df.copy()
                df['_date'] = pd.to_datetime(df[cols['date']], errors='coerce')
                df['_volume'] = pd.to_numeric(df[cols['volume']], errors='coerce')
                df['_price'] = pd.to_numeric(df[cols['price']], errors='coerce')

                for fcol, vals in filter_selections.items():
                    if vals:
                        df = df[df[fcol].astype(str).isin(vals)]

                if T('p7_row_month') in row_labels:
                    df['_row_month'] = df['_date'].dt.to_period('M').astype(str)

                row_fields = []
                for rl in row_labels:
                    if rl == T('p7_row_month'):
                        row_fields.append('_row_month')
                    else:
                        actual_col = DIM_MAP[rl]
                        df[f'_row_{actual_col}'] = df[actual_col].astype(str)
                        row_fields.append(f'_row_{actual_col}')

                col_fields = []
                for cl in col_labels:
                    actual_col = DIM_MAP[cl]
                    df[f'_col_{actual_col}'] = df[actual_col].astype(str)
                    col_fields.append(f'_col_{actual_col}')

                needed = row_fields + col_fields + ['_volume']
                df = df.dropna(subset=[c for c in needed if c in df.columns])

                if df.empty:
                    st.warning(T('p7_no_data_warning'))
                    st.session_state.pivot_result = None
                else:
                    tables = {}
                    group_keys = row_fields + col_fields
                    for vl in value_labels:
                        source_col, aggfunc = METRIC_OPTIONS[vl]
                        if aggfunc == 'count':
                            s = df.groupby(group_keys).size()
                        else:
                            s = df.groupby(group_keys)[source_col].agg(aggfunc)
                        if col_fields:
                            tables[vl] = s.unstack(col_fields)
                        else:
                            tables[vl] = s.rename(vl)

                    combined = pd.concat(tables, axis=1)

                    if combined.empty:
                        st.warning(T('p7_no_data_warning'))
                        st.session_state.pivot_result = None
                    else:
                        st.session_state.pivot_result = {
                            'combined': combined, 'row_labels': row_labels, 'col_labels': col_labels,
                            'value_labels': value_labels, 'view': VIEW_MAP[view_label], 'has_col': bool(col_fields),
                        }

        if st.session_state.get('pivot_result'):
            R = st.session_state.pivot_result
            combined = R['combined']
            multi_metric = len(R['value_labels']) > 1

            fig = None
            first_metric = R['value_labels'][0]
            cdata = combined[first_metric]
            if isinstance(cdata, pd.Series):
                cdata = cdata.to_frame(first_metric)

            def _flat_label(idx_val):
                return ' | '.join(str(x) for x in idx_val) if isinstance(idx_val, tuple) else str(idx_val)

            if R['view'] == 'table':
                pass
            elif R['view'] == 'heatmap':
                if multi_metric:
                    st.caption(T('p7_multi_metric_chart_note'))
                x_labels = [_flat_label(c) for c in cdata.columns]
                y_labels = [_flat_label(i) for i in cdata.index]
                fig = go.Figure(data=go.Heatmap(z=cdata.values, x=x_labels, y=y_labels, colorscale='Teal'))
                fig.update_layout(title=first_metric, height=max(350, 32 * len(y_labels)))
                if len(R['row_labels']) > 1 or len(R['col_labels']) > 1:
                    st.caption(T('p7_heatmap_needs_dims_note'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                if multi_metric:
                    st.caption(T('p7_multi_metric_chart_note'))
                x_labels = [_flat_label(i) for i in cdata.index]
                fig = go.Figure()
                if R['view'] == 'pie':
                    series = cdata.iloc[:, 0].fillna(0)
                    fig = go.Figure(data=[go.Pie(labels=x_labels, values=series)])
                    fig.update_layout(title=first_metric)
                elif R['has_col']:
                    for c in cdata.columns:
                        y = cdata[c]
                        if R['view'] in ('bar', 'stacked'):
                            fig.add_trace(go.Bar(x=x_labels, y=y, name=_flat_label(c)))
                        else:
                            fig.add_trace(go.Scatter(x=x_labels, y=y, mode='lines+markers', name=_flat_label(c)))
                    fig.update_layout(barmode='stack' if R['view'] == 'stacked' else 'group', yaxis_title=first_metric)
                else:
                    y = cdata.iloc[:, 0]
                    if R['view'] in ('bar', 'stacked'):
                        fig.add_trace(go.Bar(x=x_labels, y=y, marker_color='#0d9488'))
                    else:
                        fig.add_trace(go.Scatter(x=x_labels, y=y, mode='lines+markers', line=dict(color='#0d9488')))
                    fig.update_layout(yaxis_title=first_metric)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader(T('p7_table_subheader'))
            display_combined = combined.copy()
            display_combined.index = display_combined.index.rename(' | '.join(R['row_labels']))
            if R['has_col'] and isinstance(display_combined.columns, pd.MultiIndex):
                display_combined.columns = display_combined.columns.set_names([T('p7_values_label')] + R['col_labels'])
            else:
                display_combined.columns = display_combined.columns.rename(T('p7_values_label'))
            st.dataframe(display_combined)

            if st.button(T('pdf_generate_btn'), key="pivot_pdf_btn"):
                with st.spinner(T('pdf_generating_msg')):
                    try:
                        pdf_table = display_combined.reset_index()
                        pdf_table.columns = [_flat_label(c) for c in pdf_table.columns]
                        pdf_bytes = build_pdf_report(
                            title=T('p7_title'),
                            kpi_lines=[
                                f"{T('p7_row_label_multi')}: {', '.join(R['row_labels'])}",
                                f"{T('p7_col_label_multi')}: {', '.join(R['col_labels']) if R['col_labels'] else '-'}",
                                f"{T('p7_values_label')}: {', '.join(R['value_labels'])}",
                            ],
                            figs=[fig] if fig else [], df_table=pdf_table, table_title=T('p7_table_subheader'),
                        )
                        st.download_button(T('pdf_download_btn'), data=pdf_bytes, file_name="pivot_report.pdf", mime="application/pdf", key="pivot_pdf_dl")
                    except Exception as e:
                        st.error(T('pdf_error_msg', msg=str(e)))

# ==============================================================================
# 페이지 8: 신규사업 스코어러 (원본 HTML 도구 100% 이식 + 공용화)
# ==============================================================================
if selected == T('menu_opt_scorer'):
    st.title(T('p8_title'))
    st.caption(T('p8_intro'))

    if st.session_state.scorer_result is not None:
        st.button(T('p8_reset_btn'), on_click=reset_scorer_states)

    if st.session_state.scorer_result is None:
        scorer_file = st.file_uploader(T('p8_upload_label'), type=['csv', 'xlsx'], key="scorer_uploader")
        st.caption(T('p8_upload_caption'))
        raw_df = load_uploaded_df(scorer_file, 'scorer_raw_df', 'scorer_headers', 'scorer_fileid')

        if raw_df is not None:
            headers = st.session_state.scorer_headers
            cols = detect_scorer_columns(headers)
            missing = [k for k in ['date', 'volume'] if not cols[k]]
            if missing:
                st.error(T('p8_missing_cols_error', cols=', '.join(missing)))
                st.stop()

            DIM_MAP = {}
            if cols['hs_name']: DIM_MAP[T('p8_dim_hs_name')] = cols['hs_name']
            if cols['detailed_hs']: DIM_MAP[T('p8_dim_detailed_hs')] = cols['detailed_hs']
            if cols['product']: DIM_MAP[T('p8_dim_product')] = cols['product']
            if cols['importer']: DIM_MAP[T('p8_dim_importer')] = cols['importer']
            if cols['origin']: DIM_MAP[T('p8_dim_origin')] = cols['origin']

            if not DIM_MAP:
                st.error(T('p8_missing_cols_error', cols='HS Code Name / Detailed HS-CODE / Reported Product Name / Importer / Origin Country'))
                st.stop()

            st.markdown(f"**{T('p8_settings_header')}**")
            col1, col2, col3 = st.columns(3)
            with col1:
                months = st.slider(T('p8_months_label'), min_value=1, max_value=24, value=6, key="scorer_months")
            with col2:
                floor = st.number_input(T('p8_floor_label'), min_value=0, value=1000, step=100, key="scorer_floor")
            with col3:
                dim_label = st.selectbox(T('p8_dim_label'), options=list(DIM_MAP.keys()), key="scorer_dim")

            col4, col5, col6 = st.columns(3)
            with col4:
                minship = st.number_input(T('p8_minship_label'), min_value=1, value=2, step=1, key="scorer_minship")
            with col5:
                newkey_label = st.selectbox(T('p8_newkey_label'), options=list(DIM_MAP.keys()), key="scorer_newkey")
            with col6:
                preset_map = {T('p8_preset_growth'): 'growth', T('p8_preset_size'): 'size', T('p8_preset_easy'): 'easy'}
                preset_label = st.selectbox(T('p8_preset_label'), options=list(preset_map.keys()), key="scorer_preset")

            fitw_pct = st.slider(T('p8_fitw_label'), min_value=0, max_value=100, value=30, step=5, key="scorer_fitw")

            with st.expander(T('p8_fit_context_header'), expanded=False):
                st.caption(T('p8_fit_context_caption'))
                fit_suppliers_txt = st.text_area(T('p8_fit_suppliers_label'), key="scorer_fit_suppliers")
                fit_origins_txt = st.text_area(T('p8_fit_origins_label'), key="scorer_fit_origins")
                fit_keywords_txt = st.text_area(T('p8_fit_keywords_label'), key="scorer_fit_keywords")
                fit_moat_txt = st.text_area(T('p8_fit_moat_label'), key="scorer_fit_moat")

            if st.button(T('p8_run_btn')):
                df = raw_df.copy()
                df['_date'] = pd.to_datetime(df[cols['date']], errors='coerce')
                df['_volume'] = pd.to_numeric(df[cols['volume']], errors='coerce')
                has_price = bool(cols['price'])
                if has_price:
                    # 원본 도구는 '금액(Value)' 컬럼을 직접 쓰지만, 우리 표준 컬럼은 '단가(Unit Price)'라서
                    # 단가×물량으로 총액을 역산해 동일한 물량가중평균 단가 로직을 그대로 재사용한다.
                    df['_value'] = pd.to_numeric(df[cols['price']], errors='coerce') * df['_volume']
                else:
                    df['_value'] = None
                df = df.dropna(subset=['_date', '_volume'])

                dim_col = DIM_MAP[dim_label]
                new_key_col = DIM_MAP[newkey_label]

                fit_ctx = {
                    'suppliers': _p8_split_list(fit_suppliers_txt),
                    'origins': _p8_split_list(fit_origins_txt),
                    'keywords': _p8_split_list(fit_keywords_txt),
                    'moat': _p8_split_list(fit_moat_txt),
                }
                fit_ctx['has_ctx'] = bool(fit_ctx['suppliers'] or fit_ctx['origins'] or fit_ctx['keywords'])

                A, B, S, meta = compute_scorer(
                    df, dim_col, months, floor, minship, new_key_col, preset_map[preset_label],
                    fit_ctx, fitw_pct / 100, has_price,
                    cols['importer'], cols['exporter'], cols['origin'], cols['detailed_hs'], cols['label_kr'],
                    cols['product'],
                )

                if A.empty and B.empty:
                    st.warning(T('p8_no_data_warning'))
                else:
                    n_tx = len(df)
                    n_imp = df[cols['importer']].nunique() if cols['importer'] else 0
                    n_prod = df[cols['product']].nunique() if cols['product'] else 0
                    n_origin = df[cols['origin']].nunique() if cols['origin'] else 0
                    span_str = f"{df['_date'].min().strftime('%Y-%m')} ~ {df['_date'].max().strftime('%Y-%m')}"
                    cat_label = T('p8_cat_all')
                    if cols['category']:
                        mode_cat = _p8_mode_or_none(df[cols['category']].tolist())
                        if mode_cat:
                            cat_label = str(mode_cat)
                    meta.update({'n_tx': n_tx, 'n_imp': n_imp, 'n_prod': n_prod, 'n_origin': n_origin,
                                 'span_str': span_str, 'cat_label': cat_label})
                    st.session_state.scorer_result = {'A': A, 'B': B, 'S': S, 'meta': meta, 'dim_label': dim_label}
                    st.rerun()

    if st.session_state.scorer_result is not None:
        R = st.session_state.scorer_result
        A, B, S, meta = R['A'], R['B'], R['S'], R['meta']

        tot_yoy = ((meta['tot_rec'] - meta['tot_ly']) / meta['tot_ly'] * 100) if meta['tot_ly'] > 0 else 0
        a_growing = A[A['yoy'] > 0] if not A.empty else A

        c1, c2, c3, c4, c5 = st.columns(5)
        month_unit = '개월' if st.session_state.lang == 'ko' else ' mo'
        c1.metric(T('p8_kpi_window'), f"{meta['months']}{month_unit}")
        c2.metric(T('p8_kpi_total_yoy'), f"{tot_yoy:+.0f}%")
        c3.metric(T('p8_kpi_growing'), f"{len(a_growing)}")
        c4.metric(T('p8_kpi_new'), f"{len(B)}")
        with c5:
            st.markdown(f"**{T('p8_kpi_price')}**")
            st.caption(T('p8_kpi_price_linked') if meta['has_price'] else T('p8_kpi_price_none'))

        fig = None
        s_df = None

        st.subheader(T('p8_section_s'))
        S_top = S[:10]
        if S_top:
            names = [s['name'] for s in S_top][::-1]
            scores = [s['score'] for s in S_top][::-1]
            colors_bar = ['#e11d48' if s['type'] == 'new' else '#0d9488' for s in S_top][::-1]
            fig = go.Figure()
            fig.add_trace(go.Bar(y=names, x=scores, orientation='h', marker_color=colors_bar))
            fig.update_layout(title=T('p8_chart_title'), xaxis_title=T('p8_col_score'))
            st.plotly_chart(fig, use_container_width=True)

            s_rows = []
            for i, s in enumerate(S_top):
                r = s['row']
                reason = _p8_reason_a(r, r.get('dim_is_importer', False)) if s['type'] == 'existing' else _p8_reason_b(r)
                flags = _p8_flags_a(r) if s['type'] == 'existing' else _p8_flags_b(r)
                s_rows.append({
                    T('p8_col_rank'): i + 1,
                    T('p8_col_type'): T('p8_type_existing') if s['type'] == 'existing' else T('p8_type_new'),
                    T('p8_col_item'): s['name'],
                    T('p8_col_score'): s['score'],
                    T('p8_col_fit'): r['fit_score'] if r.get('fit_score') is not None else '-',
                    T('p8_col_quadrant'): r.get('quadrant', '-'),
                    T('p8_col_reason'): reason,
                    T('p8_col_flags'): flags,
                })
            s_df = pd.DataFrame(s_rows)
            st.dataframe(s_df, use_container_width=True)
        else:
            st.caption(T('p8_no_data_warning'))

        with st.expander(T('p8_section_a'), expanded=False):
            if not a_growing.empty:
                a_rows = []
                for _, r in a_growing.iterrows():
                    a_rows.append({
                        T('p8_col_item'): r['display_name'], T('p8_col_score'): r['final_score'],
                        T('p8_col_yoy'): _p8_fmt_pct(r['yoy']), T('p8_col_rec_vol'): f"{r['rec_vol']:,.0f}",
                        T('p8_col_ly_vol'): f"{r['ly_vol']:,.0f}", T('p8_col_delta'): f"{r['delta']:+,.0f}",
                        T('p8_col_concentration'): f"{r['top1_share']:.0f}%",
                        T('p8_col_reason'): _p8_reason_a(r, r['dim_is_importer']),
                        T('p8_col_flags'): _p8_flags_a(r),
                    })
                st.dataframe(pd.DataFrame(a_rows), use_container_width=True)
            else:
                st.caption(T('p8_no_data_warning'))

        with st.expander(T('p8_section_b'), expanded=False):
            if not B.empty:
                b_rows = []
                for _, r in B.iterrows():
                    b_rows.append({
                        T('p8_col_item'): r['display_name'], T('p8_col_score'): r['final_score'],
                        T('p8_col_vol'): f"{r['vol']:,.0f}", T('p8_col_ship'): r['ship'],
                        T('p8_col_top_importer'): f"{r['top1_imp']} {r['top1_share']:.0f}%" if r['top1_imp'] else '-',
                        T('p8_col_reason'): _p8_reason_b(r),
                        T('p8_col_flags'): _p8_flags_b(r),
                    })
                st.dataframe(pd.DataFrame(b_rows), use_container_width=True)
            else:
                st.caption(T('p8_no_data_warning'))

        if st.button(T('p8_generate_report_btn'), key="scorer_pdf_btn"):
            with st.spinner(T('pdf_generating_msg')):
                try:
                    pdf_bytes = build_scorer_report_pdf(A, B, S, meta, R['dim_label'])
                    st.download_button(T('p8_download_report_btn'), data=pdf_bytes, file_name="new_business_scorer_report.pdf", mime="application/pdf", key="scorer_pdf_dl")
                except Exception as e:
                    st.error(T('pdf_error_msg', msg=str(e)))
