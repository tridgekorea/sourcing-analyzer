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
        'menu_opt_customer': '도입 효과 분석',
        'menu_opt_health': '포트폴리오 헬스체크',
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

        'p1_title': '💲 도입 효과 분석',
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
        'p1_total_savings_caption': '※ 계약일({date}) 이후 품목군별 절감액(계약 전후 단가 차이 × 계약 후 물량)을 모두 더한 순 절감액입니다. 절감·비용 증가 합계에는 변화 미미(±${th:,.0f} 이내)와 계약 후 구매 없음 품목이 빠져 있습니다.',
        'p1_minor_threshold_label': '변화 미미 기준 ($, 절감·증가액 절대값이 이 값 이하면 변화 미미)',
        'p1_kpi_net': '순 절감액',
        'p1_kpi_saved': '절감 합계 ({n}개 품목)',
        'p1_kpi_increased': '비용 증가 합계 ({n}개 품목)',
        'p1_group_saved': '💰 절감 품목 ({n}개, ${amount:,.0f})',
        'p1_group_increased': '📈 단가 상승 품목 ({n}개, ${amount:,.0f})',
        'p1_group_minor': '➖ 변화 미미 ({n}개)',
        'p1_group_empty': '해당하는 품목이 없습니다.',
        'p1_minor_caption': '※ 절감·증가액의 절대값이 ${th:,.0f} 이하인 품목입니다. 카드의 $0은 단가 차이가 거의 없어 반올림된 값입니다.',
        'p1_show_all_btn': '전체 보기 (나머지 {n}개)',
        'p1_show_less_btn': '접기',
        'p1_no_purchase_caption': '🚫 계약 후 구매 없음 ({n}개): {names} — 계약 후 물량이 0이라 절감·비용 증가 합계에서 제외했습니다.',
        'p1_col_group': '구분',
        'p1_grp_saved_label': '절감',
        'p1_grp_increased_label': '단가 상승',
        'p1_grp_minor_label': '변화 미미',
        'p1_grp_no_purchase_label': '계약 후 구매 없음',
        'p1_savings_detail_subheader': '품목군별 상세 절감 내역',
        'p1_mix_changed_badge': '포장 단위 변경',
        'p1_mix_changed_tooltip': '계약 전후로 이 품목군의 주력 포장 단위가 바뀌었거나 포장 단위별 비중이 크게 달라졌습니다. 소포장은 kg당 단가가 높아 절감액이 협상 성과가 아니라 포장 변경 때문일 수 있습니다.',
        'p1_mix_changed_caption': '⚠ 포장 단위 변경 품목 ({n}개): {items} — 주력 포장 단위가 바뀌었거나 포장 단위별 물량 비중이 20%p 이상 움직였습니다. 절감액을 협상 성과로 해석하기 전에 확인하세요.',
        'p1_pack_none': '포장 표기 없음',
        'p1_pack_change_main': '포장 단위 변경: {b} → {a}',
        'p1_pack_change_form': '제형 변경: {b} → {a}',
        'p1_pack_change_share': '포장 단위 비중 변화: {pack} {b:.0f}% → {a:.0f}%',
        'price_mismatch_warning': '⚠ Unit Price와 금액÷중량이 5% 넘게 다른 행이 {pct:.0f}%입니다. 단가는 파일의 Unit Price를 그대로 썼습니다 — 단위(kg/톤)나 통화가 섞였는지 확인하세요.',
        'p1_col_avg_price_before': '계약전 평균단가(물량가중)',
        'p1_col_avg_price_after': '계약후 평균단가(물량가중)',
        'p1_col_volume_after': '계약후 물량',
        'p1_col_savings': '절감액',
        'p1_col_mix_detail': '포장 단위 변화',
        'p1_col_mix_changed': '구성 변화 경고',
        'p1_vwap_note': '※ 평균단가는 거래건수가 아니라 물량 기준 가중평균(VWAP)입니다. 이상치는 클러스터·기간별로 자동 제거했습니다.',
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
        'p7_dim_year': '연도별',
        'p7_dim_quarter': '분기별',
        'p7_dim_month_only': '월별(연도 무관)',
        'p7_capped_note': "※ '{field}' 항목 수가 너무 많아, 물량 상위 {n}개만 표시하고 나머지는 '기타'로 묶었습니다.",
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
        'p7_total_col': '합계',

        'pdf_generate_btn': '📄 PDF 보고서 생성',
        'pdf_download_btn': '📥 PDF 다운로드',
        'pdf_generating_msg': 'PDF를 생성하고 있습니다...',
        'pdf_error_msg': 'PDF 생성 중 오류가 발생했습니다: {msg}',
        'pdf_chart_export_warning': '차트 이미지를 PDF에 넣지 못했습니다 (차트가 빠진 채로 생성됩니다): {msg}',
        # --- ① 도입 효과 분석 PDF (새 양식) ---
        'p1_rp_eyebrow': 'Adoption Impact Report',
        'p1_rp_title2': '도입 효과 리포트',
        'p1_rp_blind_toggle': 'PDF에 고객사명 대신 블라인드 표기 (수입사 A)',
        'p1_rp_legacy_btn': '기존 양식 PDF',
        'p1_rp_s1_eyebrow': '01  Executive Summary',
        'p1_rp_head_save': '계약 후 순 ${x:,.0f} 절감했습니다',
        'p1_rp_head_loss': '계약 후 비용이 순 ${x:,.0f} 늘었습니다',
        'p1_rp_kpi_net': '순 절감액',
        'p1_rp_kpi_net_note_excl': '구성 변화 품목 제외 시 ${y:,.0f}',
        'p1_rp_kpi_net_note': '단가 차이 × 계약 후 물량',
        'p1_rp_kpi_saved': '절감 합계 ({n}개)',
        'p1_rp_kpi_increased': '비용 증가 합계 ({n}개)',
        'p1_rp_kpi_group_note': '변화 미미 ±${th:,.0f} 제외',
        'p1_rp_kpi_price': '단가 변화율',
        'p1_rp_kpi_price_note': '계약 후 물량 기준 VWAP, 계약 전 대비',
        'p1_rp_findings': '핵심 결론',
        'p1_rp_excl_suffix': ' (구성 변화 품목 제외 시 ${y:,.0f})',
        'p1_rp_f1_save': '계약 후 순 ${x:,.0f} 절감{excl}.',
        'p1_rp_f1_loss': '계약 후 비용이 순 ${x:,.0f} 늘었습니다{excl}.',
        'p1_rp_f1_d': '절감 품목 {ns}개의 절감 합계 ${s:,.0f}, 단가 상승 품목 {ni}개의 비용 증가 합계 ${i:,.0f}입니다.',
        'p1_rp_f2': '{name}이(가) 전체 절감의 {p:.0f}%를 차지합니다.',
        'p1_rp_f2_concentrated': '{name}이(가) 전체 절감의 {p:.0f}%를 차지합니다 — 절감이 한 품목에 집중되어 나머지 품목에 추가 개선 여지가 있습니다.',
        'p1_rp_f2_d': '계약 전 ${b:.2f} → 계약 후 ${a:.2f}/kg, 계약 후 물량 {v:,.0f}kg 기준입니다.',
        'p1_rp_f2_none': '변화 미미 기준(±${th:,.0f})을 넘는 절감 품목이 없습니다.',
        'p1_rp_f2_none_d': '계약 전후 단가 차이가 작거나, 계약 후 구매 물량이 적습니다.',
        'p1_rp_f3_up': '{names} 등 {n}개 품목은 단가가 올라 비용이 ${x:,.0f} 늘었습니다.',
        'p1_rp_f3_up_d': '재협상이나 대체 공급처 검토가 필요합니다.',
        'p1_rp_f3_none': '단가가 오른 품목이 없습니다.',
        'p1_rp_f3_none_d': '모든 품목을 계약 전과 같거나 낮은 단가로 구매했습니다.',
        'p1_rp_f3_mix': ' 포장 단위가 바뀐 {k}개 품목은 kg당 단가 차이가 협상 효과가 아니라 포장 단위 변화 때문일 수 있습니다.',
        'p1_rp_s2_eyebrow': '02  Savings by Item',
        'p1_rp_s2_head': '품목군별 절감과 비용 증가',
        'p1_rp_s2_sub': '계약 전후 물량가중 단가 차이 × 계약 후 물량으로 계산했습니다.',
        'p1_rp_chart': '절감·비용 증가 상위 품목군',
        'p1_rp_chart_note': 'Blue = 절감, Black = 단가 상승으로 비용 증가. 각 상위 {n}개까지 표시합니다.',
        'p1_rp_th_item': '품목군',
        'p1_rp_th_before': '계약 전<br/>USD/kg',
        'p1_rp_th_after': '계약 후<br/>USD/kg',
        'p1_rp_th_vol': '계약 후<br/>물량(kg)',
        'p1_rp_th_saving': '절감액<br/>USD',
        'p1_rp_th_mix': '구성',
        'p1_rp_badge_mix': '포장 변경',
        'p1_rp_s3_eyebrow': '03  Items to Check',
        'p1_rp_s3_head': '따로 확인할 품목',
        'p1_rp_s3_sub': '계약 후 구매가 없는 품목과 포장 단위가 크게 바뀐 품목입니다. 절감 판단에서 주의해서 보세요.',
        'p1_rp_s3_nopurchase': '계약 후 구매 없음 ({n}개)',
        'p1_rp_s3_nopurchase_note': '계약 후 물량이 0이라 절감·비용 증가 합계에서 뺐습니다.',
        'p1_rp_s3_mix': '포장 단위 변경 ({n}개)',
        'p1_rp_s3_mix_note': '같은 품목군 안에서 주력 포장 단위가 바뀌었거나 포장 단위별 물량 비중이 20%p 이상 움직였습니다. 소포장은 kg당 단가가 높아 절감액 해석에 주의가 필요합니다.',
        'p1_rp_th_overlap': '포장 단위 변화',
        'p1_rp_s3_none': '해당하는 품목이 없습니다.',
        'p1_rp_s4_eyebrow': '04  Methodology',
        'p1_rp_s4_head': '분석 기준과 데이터 한계',
        'p1_rp_lbl_data': '데이터', 'p1_rp_lbl_period': '기간', 'p1_rp_lbl_cluster': '품목 묶기', 'p1_rp_lbl_outlier': '이상치 제외',
        'p1_rp_lbl_price': '평균 단가', 'p1_rp_lbl_saving': '절감액', 'p1_rp_lbl_minor': '변화 미미', 'p1_rp_lbl_mix': '구성 변화',
        'p1_rp_lbl_blind': '블라인드',
        'p1_rp_m_data': 'Tridge TDS 한국 수입 통관 데이터, 건별 수입신고 단위. 파일에서 가장 많이 등장하는 수입사를 분석 대상으로 함',
        'p1_rp_m_period': '계약 전 {b0} – {b1} · 계약 후 {a0} – {a1} (계약일 {c})',
        'p1_rp_m_cluster': '포장 단위와 제품 코드를 뗀 품목명이 같은 거래끼리 한 품목군으로 묶음 (거래 3건 미만인 품목명은 제외). 일반 명칭 품목은 HS코드 10단위로 구분, 같은 HS 안의 세부 스펙은 구분하지 않음',
        'p1_rp_m_outlier': '품목군·기간별 단가 IQR 1.5배 밖 제외 (제외 비율 최대 7%)',
        'p1_rp_m_price': 'VWAP(물량가중평균), USD/kg',
        'p1_rp_m_saving': '(계약 전 VWAP − 계약 후 VWAP) × 계약 후 물량. 양수 = 절감, 음수 = 비용 증가',
        'p1_rp_m_minor': '절감·증가액 절대값이 ${th:,.0f} 이하인 품목은 변화 미미로 분류해 절감·비용 증가 합계에서 뺌 (순 절감액에는 포함)',
        'p1_rp_m_mix': '품목군 안의 포장 단위별 물량 비중을 계약 전후로 비교해, 주력 포장 단위가 바뀌거나 비중이 20%p 이상 움직이면 구성 변화로 표시 (품목명에 포장 표기가 없는 거래는 비교에서 제외)',
        'p1_rp_lbl_pack': '포장 단위',
        'p1_rp_m_pack': '품목명의 숫자+단위(정·포·캡슐·g·ml 등)를 포장 단위로 떼어내고, 끝에 붙은 제품 코드(PEF225, ES 2677 등)를 지운 뒤 이름이 같은 것끼리 한 품목군으로 묶음',
        'p1_rp_m_blind_on': '고객사명 비공개 (수입사 A로 표기)',
        'p1_rp_m_blind_off': '고객사명 공개',
        'p1_rp_unknown1': '같은 기간 시장 전체 시세 변화의 영향 (시장 경쟁력 분석과 함께 보세요)',
        'p1_rp_unknown2': '품질·규격 변경이나 계약 조건(결제·인코텀즈) 변화에 따른 단가 차이',
        'p1_rp_unknown3': '관세·국내 물류비 등 도착 이후 비용',
        'p1_rp_unknown4': '계약이 없었을 때의 단가 (전후 비교는 인과 효과의 추정치입니다)',
        'p1_rp_close_head': '다음 단계 제안',
        'p1_rp_close_k1': '성과 보고',
        'p1_rp_close_k2': '재협상',
        'p1_rp_close_k3': '확인 필요',
        'p1_rp_close_v1': '순 ${x:,.0f} 절감{excl} · 주요 절감 품목: {items}',
        'p1_rp_close_v1_loss': '순 ${x:,.0f} 비용 증가{excl} — 단가 상승 품목부터 원인을 확인하세요',
        'p1_rp_close_v2': '단가 상승 {n}개 품목(${x:,.0f}): {items}',
        'p1_rp_close_v2_none': '단가 상승 품목 없음 — 현재 조건 유지',
        'p1_rp_close_v3': '구성 변화 {k}개 · 계약 후 구매 없음 {m}개 품목',
        'p1_rp_close_note': '계약 전후 단가 비교는 시장 시세·품질·계약 조건 변화의 영향을 함께 담고 있습니다. 본 자료의 수치는 통관 데이터 기반 추정치입니다.',
        # --- 새 PDF 양식 공통 (report_theme) ---
        'rp_lbl_period': '분석 기간',
        'rp_lbl_data': '데이터',
        'rp_lbl_date': '작성일',
        'rp_lbl_scope': '공개 범위',
        'rp_data_source': 'Tridge TDS · 한국 수입 통관 데이터',
        'rp_blind_on': 'Blind — 한국 수입사명 비공개',
        'rp_blind_off': '내부용',
        'rp_confidential': 'Confidential · 본 자료는 통관 데이터 기반 추정치이며 투자·계약 판단의 단일 근거로 사용할 수 없습니다.',
        'rp_footer': 'Source: Tridge TDS · VWAP(물량가중평균) 기준 · USD/kg',
        'rp_period_fmt': '{start} – {end} ({months}개월)',
        'rp_meth_item': '항목',
        'rp_meth_rule': '기준',
        'rp_unknown_header': '이 리포트로 알 수 없는 것',
        'rp_pdf_btn': '📄 PDF 리포트 만들기',
        'rp_pdf_dl': '📥 PDF 리포트 다운로드',
        'rp_pdf_spinner': 'PDF 리포트를 만드는 중입니다...',
        'rp_count_unit': '{n}개',
        'rp_company_unit': '{n}곳',
        'rp_month_unit': '{n}개월',
        # --- ⑨ 스크리너 PDF ---
        'scr_rp_eyebrow': 'Commodity Screening Report',
        'scr_rp_title1': '범용상품 스크리닝',
        'scr_rp_title2': '리포트',
        'scr_rp_item': 'HS {n}개 품목 스크리닝',
        'scr_rp_s1_eyebrow': '01  Executive Summary',
        'scr_rp_s1_head_a': 'A그룹 {n}개 품목이 입찰·대체 소싱 여지가 큽니다',
        'scr_rp_s1_head_none': '범용성과 가격 격차를 함께 갖춘 품목은 아직 없습니다',
        'scr_rp_kpi_note_a': '범용성 상위 · 격차 상위',
        'scr_rp_kpi_note_b': '범용성 중간',
        'scr_rp_kpi_note_c': '범용성 상위 · 격차 하위',
        'scr_rp_kpi_note_d': '범용성 하위',
        'scr_rp_top_header': 'A그룹 상위 품목 (범용성 순)',
        'scr_rp_top_none': 'A그룹에 해당하는 품목이 없습니다.',
        'scr_rp_s2_eyebrow': '02  Quadrant',
        'scr_rp_s2_head': '범용성 × 격차 4분면',
        'scr_rp_s2_sub': '가로 = 범용성 백분위, 세로 = 격차 점수, 원 크기 = 연간 수입액',
        'scr_rp_s2_note': '점선은 분류 경계(범용성 {lo}/{hi}, 격차 {gap})입니다. A그룹 중 수입액 상위 품목에만 HS코드를 표시했습니다. 속 빈 원은 같은 달·같은 원산지 선적이 3건 미만이라 격차를 계산할 수 없는 품목으로, 0선에 표시했습니다.',
        'scr_rp_th_importers': '수입사',
        'scr_rp_legend_nogap': '격차 계산 불가',
        'scr_rp_th_exporters': '공급사',
        'scr_rp_th_origins': '원산지',
        'scr_rp_th_cr3': 'CR3',
        'scr_rp_th_uni': '범용성',
        'scr_rp_th_gap': '격차',
        'scr_rp_th_value': '연간 수입액',
        'scr_rp_s3_eyebrow': '03  Items by Class',
        'scr_rp_s3_head': '분류별 품목',
        'scr_rp_s3_sub': '분류마다 범용성 순 상위 {n}개를 보여줍니다. 전체 목록은 화면의 CSV로 내려받을 수 있습니다.',
        'scr_rp_s4_eyebrow': '04  Methodology',
        'scr_rp_s4_head': '지표 정의와 데이터 한계',
        'scr_rp_m_data': 'Tridge TDS 한국 수입 통관 데이터, 건별 수입신고 단위. HS 10단위로 묶음',
        'scr_rp_m_period': '{period}',
        'scr_rp_m_prep': '중복 행 제거(신고일·수입자·공급자·중량·금액 동일), 건당 {kg:,}kg 미만 제외, HS별 단가 P1~P99 밖 제외, 거래 {ship}건 미만 HS는 순위 제외',
        'scr_rp_m_uni': '수입사 수↑·CR3↓·해외 공급사 수↑·원산지 수↑·거래 월 수↑·단가 CV↓를 각각 데이터셋 내 백분위로 바꿔 가중 평균 (가중치 {w})',
        'scr_rp_m_gap': '같은 달·같은 원산지 안의 단가 (P90−P10)/P50을 셀 물량으로 가중 평균한 격차율의 백분위 (선적 3건 이상인 셀만)',
        'scr_rp_m_class': '범용성 백분위 {hi} 이상 & 격차 {gap} 이상 → A, 범용성 {hi} 이상 & 격차 {gap} 미만 → C, {lo}~{hi} → B, {lo} 미만 → D',
        'scr_rp_m_cr3': 'CR3 = 상위 3개 수입사 물량 비중, HHI = 수입사 물량 비중(%) 제곱합',
        'scr_rp_m_price': '평균 단가는 VWAP(물량가중평균), USD/kg. 단가 CV도 물량가중',
        'scr_rp_lbl_data': '데이터', 'scr_rp_lbl_period': '기간', 'scr_rp_lbl_prep': '전처리', 'scr_rp_lbl_uni': '범용성 점수',
        'scr_rp_lbl_gap': '격차 점수', 'scr_rp_lbl_class': '분류', 'scr_rp_lbl_conc': '집중도', 'scr_rp_lbl_price': '단가',
        'scr_rp_unknown1': '같은 HS 안에서도 스펙·등급이 섞여 있어 생기는 단가 차이 (입찰 대응 분석에서 품명 키워드로 나눠 보세요)',
        'scr_rp_unknown2': '수입자명이 유통사·대행사로 기록된 경우의 실수요자',
        'scr_rp_unknown3': '관세·국내 물류비 등 도착 이후 비용',
        'scr_rp_unknown4': '향후 수요·가격 예측',
        'scr_rp_close_head': '다음 단계 제안',
        'scr_rp_close_k1': '우선 검토',
        'scr_rp_close_k2': '심층 분석',
        'scr_rp_close_k3': '기준 확인',
        'scr_rp_close_v1_a': 'A그룹 상위 품목: {items}',
        'scr_rp_close_v1_none': 'A그룹이 없어 B그룹 상위 품목부터 검토: {items}',
        'scr_rp_close_v2': '관심 품목을 입찰 대응 분석으로 넘겨 대체 공급처 단가와 공급 안정성을 확인하세요',
        'scr_rp_close_v3': '분류 경계 범용성 {lo}/{hi}, 격차 {gap} 백분위 · 건당 {kg:,}kg 이상 거래 기준',
        'scr_rp_close_note': '점수는 업로드한 데이터셋 안에서의 상대 순위입니다. 다른 품목군과 섞어 올리면 순위가 달라집니다. 본 자료의 수치는 통관 데이터 기반 추정치입니다.',
        # --- ⑩ 입찰 대응 PDF ---
        'bd_rp_header': '📄 PDF 리포트',
        'bd_rp_item_label': '리포트 품목명',
        'bd_rp_spec_label': '스펙 (리포트 표기용)',
        'bd_rp_stab_label': '공급 안정성 페이지에 넣을 공급사 (공급사별 1페이지)',
        'bd_rp_eyebrow': 'Bid Response Report',
        'bd_rp_title2': '입찰 대응 리포트',
        'bd_rp_s1_eyebrow': '01  Executive Summary',
        'bd_rp_head_save': '검증된 대체 공급처로 단가 {pct:.0f}% 낮출 수 있습니다',
        'bd_rp_head_found': '검증된 대체 공급처 {n}곳을 확인했습니다',
        'bd_rp_head_none': '검증 기준을 통과한 공급처가 아직 없습니다',
        'bd_rp_kpi1': '최저 검증 공급처 단가',
        'bd_rp_kpi1_note': 'USD/kg · {basis} · {origin}',
        'bd_rp_kpi1_none': '검증 통과 공급사 없음',
        'bd_rp_basis_cif': 'CIF',
        'bd_rp_basis_tariff': '관세 포함',
        'bd_rp_kpi2': '현 납품 기준 대비',
        'bd_rp_kpi2_note': '기준 ${p:.2f}/kg',
        'bd_rp_kpi_no_base': '기준 공급사 미지정',
        'bd_rp_kpi3': '연간 절감 추정',
        'bd_rp_kpi3_note': '연 {t:,.0f}t 가정',
        'bd_rp_kpi4': '공급 공백',
        'bd_rp_kpi4_note': '최근 {m}개월 · 1순위 공급사',
        'bd_rp_findings': '핵심 결론',
        'bd_rp_f1_two': '{o1}산이 {o2}산보다 kg당 ${d:.2f} 낮습니다.',
        'bd_rp_f1_two_d': '같은 조건(HS·품명 키워드) 안에서 VWAP를 비교했습니다. 두 원산지 모두 연 {t:,.0f}톤 이상 들어오는 주요 원산지입니다.',
        'bd_rp_f1_one': '연 {t:,.0f}톤 이상 들어오는 주요 원산지는 {o}({tons:,.0f}t) 한 곳입니다.',
        'bd_rp_f1_one_d': '다른 원산지는 물량이 적어 단가 비교의 근거가 약합니다.',
        'bd_rp_f1_none': '연 {t:,.0f}톤 이상 들어오는 주요 원산지가 없습니다.',
        'bd_rp_f1_none_d': '원산지별 물량이 모두 적어 원산지 간 단가 비교는 참고용으로만 보세요.',
        'bd_rp_f2_base': '검증 통과 공급사 {n}곳 중 {k}곳이 현 납품 기준보다 10% 이상 저렴합니다.',
        'bd_rp_f2_nobase': '검증 통과 공급사는 {n}곳입니다.',
        'bd_rp_f2_d': '검증 기준은 최근 12개월 {ship}건 이상, {ton:,.0f}톤 이상, 한국 수입사 {imp}곳 이상 공급입니다.',
        'bd_rp_f3_cheap': '{names}는 단가는 낮지만 아직 검증 전 단계입니다.',
        'bd_rp_f3_cheap_d': '최근 12개월 거래 실적이 적어 입찰 제안의 주력 공급처로는 권하지 않습니다. 보조 옵션으로만 검토하세요.',
        'bd_rp_f3_none': '검증 전 공급사 중 최저가보다 싼 곳은 없습니다.',
        'bd_rp_f3_none_d': '현재 최저 비교단가는 검증을 통과한 공급사입니다.',
        'bd_rp_tariff_missing': '원산지별 관세율이 일부 입력되지 않아 해당 원산지는 CIF 기준으로 비교했습니다. FTA 적용 세율에 따라 순위가 바뀔 수 있습니다.',
        'bd_rp_tariff_ok': '입력한 원산지별 관세율로 비교단가 = CIF × (1 + 세율)을 계산했습니다.',
        'bd_rp_s2_eyebrow': '02  Alternative Suppliers',
        'bd_rp_s2_head': '대체 공급처 비교',
        'bd_rp_s2_sub': '같은 조건 안에서 원산지·공급사별 VWAP와 검증 지표를 비교했습니다.',
        'bd_rp_origin_chart': '원산지별 단가 분포',
        'bd_rp_origin_chart_note': '점 = VWAP, 막대 = P10~P90 범위. 속이 빈 점은 연 {t:,.0f}톤 미만 원산지입니다.',
        'bd_rp_baseline_line': '현 납품 기준 ${p:.2f}',
        'bd_rp_rank_header': '공급사 순위 (비교단가 기준)',
        'bd_rp_th_no': '#',
        'bd_rp_th_cmp': '비교단가<br/>USD/kg',
        'bd_rp_th_vs': '기준 대비',
        'bd_rp_th_ship': '건수',
        'bd_rp_th_ton': '물량(t)',
        'bd_rp_th_months': '거래월',
        'bd_rp_th_imp': '공급<br/>수입사',
        'bd_rp_th_status': '상태',
        'bd_rp_base_tag': '기준',
        'bd_rp_rank_note': '검증 = 최근 12개월 {ship}건 이상 · {ton:,.0f}톤 이상 · 한국 수입사 {imp}곳 이상 공급. 공급 수입사는 실명 없이 수만 표기합니다. 상위 {n}개 공급사만 표시합니다.',
        'bd_rp_s3_eyebrow': '03  Supply Track Record · {i}/{n}',
        'bd_rp_s3_sub': '{origin} · 최근 {m}개월 한국향 선적 실적',
        'bd_rp_s3_kpi1': '연평균 공급량',
        'bd_rp_s3_kpi1_note': '최근 {m}개월 기준',
        'bd_rp_s3_kpi2': '공백 월',
        'bd_rp_s3_kpi2_note': '선적 0건인 달',
        'bd_rp_s3_kpi3': '거래 한국 수입사',
        'bd_rp_s3_kpi3_note': '실명 비공개',
        'bd_rp_s3_kpi4': '단가 변동성',
        'bd_rp_s3_kpi4_note': '물량가중 단가 변동계수',
        'bd_rp_s3_chart': '월별 선적 물량',
        'bd_rp_s3_chart_gap': '빗금 표시는 선적이 없었던 달입니다.',
        'bd_rp_s3_chart_nogap': '{m}개월 동안 선적이 끊긴 달이 없습니다.',
        'bd_rp_s3_year': '연도별 요약',
        'bd_rp_s3_th_year': '기간',
        'bd_rp_s3_th_ton': '선적 물량(t)',
        'bd_rp_s3_th_months': '선적 월',
        'bd_rp_s3_th_vwap': 'VWAP USD/kg',
        'bd_rp_s3_th_imp': '공급 수입사',
        'bd_rp_proof': 'Proof Statement',
        'bd_rp_s4_eyebrow': '04  Methodology',
        'bd_rp_s4_head': '분석 기준과 데이터 한계',
        'bd_rp_m_data': 'Tridge TDS 한국 수입 통관 데이터, 건별 수입신고 단위',
        'bd_rp_m_period': '{period}. 공급 안정성은 기간 마지막 달 기준 최근 36개월',
        'bd_rp_m_spec': 'HS {hs} · 품명 포함 키워드: {inc} · 제외 키워드: {exc}',
        'bd_rp_m_dedup': '신고일·수입자·공급자·중량·금액이 모두 같은 행은 1건으로 처리',
        'bd_rp_m_outlier': '건당 {kg:,}kg 미만 소량 수입, 단가 P1 미만·P99 초과',
        'bd_rp_m_price': 'VWAP(물량가중평균), USD/kg, CIF 기준',
        'bd_rp_m_tariff_none': '미반영. 원산지별 세율을 입력하면 비교단가 = CIF × (1 + 세율)',
        'bd_rp_m_tariff_part': '일부 반영 ({rates}). 입력하지 않은 원산지는 CIF 그대로',
        'bd_rp_m_tariff_all': '반영 ({rates}). 비교단가 = CIF × (1 + 세율)',
        'bd_rp_m_blind_on': '한국 수입사명 비공개(물량 순 A/B/C로 치환). 해외 공급사명은 공개',
        'bd_rp_m_blind_off': '내부용 — 한국 수입사명 공개',
        'bd_rp_m_verify': '최근 12개월 {ship}건 이상 · {ton:,.0f}톤 이상 · 한국 수입사 {imp}곳 이상',
        'bd_rp_lbl_data': '데이터', 'bd_rp_lbl_period': '기간', 'bd_rp_lbl_spec': '스펙 분리', 'bd_rp_lbl_dedup': '중복 제거',
        'bd_rp_lbl_outlier': '이상치 제외', 'bd_rp_lbl_price': '평균 단가', 'bd_rp_lbl_tariff': '관세', 'bd_rp_lbl_blind': '블라인드',
        'bd_rp_lbl_verify': '검증 기준',
        'bd_rp_unknown1': '국내 통관·검역·운송·보관비 등 도착 이후 비용',
        'bd_rp_unknown2': '수입자명이 유통사·대행사로 기록된 경우의 실수요자',
        'bd_rp_unknown3': '공급사의 품질 규격 세부(수분율, 길이 분포 등)와 인증 현황',
        'bd_rp_unknown4': '향후 가격 예측. 시세 전망이 필요하면 Tridge Eye와 함께 보세요',
        'bd_rp_close_head': '입찰 제안 시 권장 구성',
        'bd_rp_close_k1': '주력 공급처',
        'bd_rp_close_k2': '백업 공급처',
        'bd_rp_close_k3': '절감 근거',
        'bd_rp_close_main': '{name} ({origin}) — ${p:.2f}/kg, 검증 통과, 공백 {gap}개월',
        'bd_rp_close_backup': '{name} ({origin}) — ${p:.2f}/kg, 수입사 {imp}곳 공급',
        'bd_rp_close_none': '검증을 통과한 공급사가 없습니다 — 최저 후보 {name}({origin}, ${p:.2f}/kg)의 실적을 더 확인하세요',
        'bd_rp_close_no_backup': '검증 통과 공급사가 1곳뿐입니다',
        'bd_rp_close_save': '현 납품 기준 대비 –{pct:.1f}%, 연 {t:,.0f}t 기준 약 ${k:,.0f}K',
        'bd_rp_close_nosave': '기준 공급사를 지정하지 않아 절감액은 계산하지 않았습니다',
        'bd_rp_close_nobetter': '현 납품 기준보다 싼 검증 공급사가 없어 절감 근거는 아직 없습니다',
        'bd_rp_close_note_missing': '관세(일부)·국내 물류비 미반영. 도착 원가 기준 최종 판단 전 원산지별 FTA 세율과 국내 비용을 확인하세요. 본 자료의 수치는 통관 데이터 기반 추정치입니다.',
        'bd_rp_close_note_ok': '국내 물류비 미반영. 도착 원가 기준 최종 판단 전 국내 비용을 확인하세요. 본 자료의 수치는 통관 데이터 기반 추정치입니다.',
        # --- 공통: 블라인드 ---
        'blind_label': '수입사 {code}',
        'blind_toggle': '한국 수입사명 블라인드 (물량 순 A/B/C…)',
        # --- ⑨ 범용상품 스크리너 ---
        'menu_opt_screener': '범용상품 스크리너',
        'sc_title': '🧭 범용상품 스크리너',
        'sc_intro': 'HS 10단위 품목별로 시장이 얼마나 열려 있는지(범용성)와 같은 시기·같은 원산지 안에서 단가 차이가 얼마나 큰지(격차)를 함께 보고, 입찰·대체 소싱 여지가 큰 품목을 찾습니다.',
        'sc_reset_btn': '새로운 스크리닝 시작 (다시하기)',
        'sc_upload_label': '스크리닝할 시장 데이터 파일을 업로드하세요',
        'sc_upload_caption': '※ 여러 HS코드·여러 수입사가 섞인 TDS raw file (날짜, 수입사, 공급사, 원산지, HS코드, 물량, 단가 필요)',
        'sc_missing_cols_error': '필수 컬럼이 없습니다: {cols}',
        'sc_period_label': '분석 기간 (기본: 최근 12개월)',
        'sc_min_kg_label': '건당 최소 중량 (kg, 미만 거래 제외)',
        'sc_min_ship_label': 'HS별 최소 거래 건수 (미만인 HS는 순위에서 제외)',
        'sc_weights_header': '범용성 점수 가중치',
        'sc_w_importers': '수입사 수 (많을수록 ↑)',
        'sc_w_cr3': 'CR3 (낮을수록 ↑)',
        'sc_w_exporters': '해외 공급사 수 (많을수록 ↑)',
        'sc_w_origins': '원산지 수 (많을수록 ↑)',
        'sc_w_months': '거래 발생 월 수 (많을수록 ↑)',
        'sc_w_cv': '단가 CV (낮을수록 ↑)',
        'sc_bounds_header': '분류 경계 (백분위)',
        'sc_bound_hi': '범용성 상위 경계',
        'sc_bound_lo': '범용성 하위 경계',
        'sc_bound_gap': '격차 상위 경계 (범용성 상위 안에서 A/C 구분)',
        'sc_bounds_warning': '하위 경계가 상위 경계보다 높습니다. 경계를 확인해주세요.',
        'sc_run_btn': '스크리닝 실행',
        'sc_spinner': '품목별 지표를 계산 중입니다...',
        'sc_no_data_warning': '조건에 맞는 거래가 없습니다. 기간·최소 중량·최소 건수를 조정해보세요.',
        'sc_prep_caption': '전처리: 원본 {n0:,}건 → 중복 {dup:,}건, 소량 거래 {small:,}건, 단가 이상치(HS별 P1~P99 밖) {outlier:,}건 제외 → {n:,}건 사용 · 거래 건수 부족으로 순위에서 제외된 HS {skipped}개',
        'sc_kpi_total': '분석 HS 수',
        'sc_class_a': 'A 공략 1순위',
        'sc_class_b': 'B 검토',
        'sc_class_c': 'C 범용·격차 작음',
        'sc_class_d': 'D 제한적',
        'sc_quadrant_header': '📍 범용성 × 격차 4분면',
        'sc_axis_uni': '범용성 (백분위)',
        'sc_axis_gap': '격차 점수 (백분위)',
        'sc_bubble_note': '※ 버블 크기 = 연간 수입액. 세로선은 범용성 경계, 가로선은 격차 경계입니다.',
        'sc_rank_header': '📋 품목 랭킹',
        'sc_filter_label': '분류 필터',
        'sc_col_hs': 'HS코드',
        'sc_col_name': '대표 품명',
        'sc_col_class': '분류',
        'sc_col_uni': '범용성 점수',
        'sc_col_uni_rank': '범용성 백분위',
        'sc_col_gap_score': '격차 점수',
        'sc_col_importers': '수입사 수',
        'sc_col_cr3': 'CR3(%)',
        'sc_col_hhi': 'HHI',
        'sc_col_exporters': '해외 공급사 수',
        'sc_col_origins': '원산지 수',
        'sc_col_months': '거래 월 수',
        'sc_col_freq': '수입사당 연간 건수',
        'sc_col_cv': '단가 CV(%)',
        'sc_col_gap': '격차율(%)',
        'sc_col_value': '연간 수입액(USD)',
        'sc_col_ship': '거래 건수',
        'sc_col_vwap': 'VWAP(USD/kg)',
        'sc_csv_btn': '📥 CSV 다운로드',
        'sc_goto_header': '➡️ 입찰 대응 분석으로 넘기기',
        'sc_goto_select': '분석할 품목(HS) 선택',
        'sc_goto_btn': '선택한 품목으로 입찰 대응 분석 열기',
        'sc_goto_done': '{hs}를 입찰 대응 분석으로 넘겼습니다. 메뉴가 자동으로 바뀌지 않으면 왼쪽 메뉴에서 [{menu}]를 눌러주세요.',
        'sc_metric_note': '※ 범용성 점수 = 수입사 수↑·CR3↓·공급사 수↑·원산지 수↑·거래 월 수↑·단가 CV↓를 데이터셋 안에서 백분위로 바꿔 가중 평균한 값. 격차율 = 같은 달·같은 원산지 안의 단가 (P90−P10)/P50을 물량으로 가중 평균한 값(선적 3건 이상인 셀만). 모든 단가는 물량가중(VWAP) 기준 USD/kg입니다.',
        # --- ⑩ 입찰 대응 분석 ---
        'menu_opt_bid': '입찰 대응 분석',
        'bd_title': '📑 입찰 대응 분석',
        'bd_intro': '한 품목(HS·스펙)에 대해 대체 공급처의 단가·실적을 비교하고, 후보 공급사의 공급 안정성을 확인합니다.',
        'bd_reset_btn': '새로운 분석 시작 (다시하기)',
        'bd_upload_label': '분석할 시장 데이터 파일을 업로드하세요',
        'bd_upload_caption': '※ TDS raw file (날짜, 수입사, 공급사, 원산지, 물량, 단가 필요. HS코드 컬럼이 있으면 HS로 걸러 볼 수 있습니다)',
        'bd_use_screener_btn': '범용상품 스크리너에 올린 파일 그대로 사용',
        'bd_missing_cols_error': '필수 컬럼이 없습니다: {cols}',
        'bd_filter_header': '🔎 분석 대상',
        'bd_hs_label': 'HS코드',
        'bd_hs_all': '(전체)',
        'bd_hs_from_screener': '범용상품 스크리너에서 넘어온 품목: {hs}',
        'bd_include_label': '품명 포함 키워드 (쉼표로 구분, 하나라도 포함)',
        'bd_exclude_label': '품명 제외 키워드 (쉼표로 구분, 하나라도 포함되면 제외)',
        'bd_period_label': '분석 기간 (기본: 최근 12개월)',
        'bd_min_kg_label': '건당 최소 중량 (kg, 미만 거래 제외)',
        'bd_prep_caption': '전처리: 원본 {n0:,}건 → 중복 {dup:,}건, 소량 거래 {small:,}건, 단가 이상치(P1~P99 밖) {outlier:,}건 제외 · 필터 적용 후 기간 내 {n:,}건',
        'bd_no_data_warning': '조건에 맞는 거래가 없습니다. HS코드·키워드·기간·최소 중량을 조정해보세요.',
        'bd_tab_alt': '대체 공급처',
        'bd_tab_stability': '공급 안정성',
        'bd_verify_header': '검증 기준 (기간 마지막 날 기준 최근 12개월)',
        'bd_verify_ship': '최소 거래 건수',
        'bd_verify_ton': '최소 물량 (톤)',
        'bd_verify_imp': '최소 거래 한국 수입사 수',
        'bd_verified_only': '검증된 공급사만 보기',
        'bd_badge_verified': '검증',
        'bd_badge_unverified': '미검증',
        'bd_tariff_header': '관세율 입력 (선택, 원산지별 %)',
        'bd_tariff_caption': '※ 관세율을 넣으면 비교단가 = CIF × (1 + 세율)로 계산합니다. 비워두면 CIF 그대로 비교합니다. DB 연동 없이 수기 입력입니다.',
        'bd_tariff_col_origin': '원산지',
        'bd_tariff_col_rate': '관세율(%)',
        'bd_badge_no_tariff': '관세 미반영',
        'bd_badge_tariff_ok': '관세 반영',
        'bd_tariff_missing_note': '관세율이 비어 있는 원산지가 있어 일부 비교단가에 관세가 반영되지 않았습니다.',
        'bd_origin_header': '🌏 원산지별 비교',
        'bd_origin_chart_title': '원산지별 단가 분포 (USD/kg)',
        'bd_supplier_header': '🏭 해외 공급사별 비교',
        'bd_base_label': '기준(현 납품) 공급사',
        'bd_base_none': '(선택 안 함)',
        'bd_annual_ton_label': '연간 예상 물량 (톤, 절감액 계산용)',
        'bd_col_origin': '원산지',
        'bd_col_exporter': '해외 공급사',
        'bd_col_status': '검증',
        'bd_col_vwap': 'VWAP(USD/kg)',
        'bd_col_cmp': '비교단가(USD/kg)',
        'bd_col_tariff': '관세율(%)',
        'bd_col_ship': '건수',
        'bd_col_ton': '총중량(톤)',
        'bd_col_months': '거래 월 수',
        'bd_col_importers': '한국 수입사 수',
        'bd_col_importer_names': '거래 한국 수입사',
        'bd_col_last': '최근 거래일',
        'bd_col_saving_pct': '절감률(%)',
        'bd_col_saving_amt': '연간 절감액(USD)',
        'bd_csv_origin_btn': '📥 원산지 표 CSV',
        'bd_csv_supplier_btn': '📥 공급사 표 CSV',
        'bd_stab_select': '공급사 선택',
        'bd_stab_chart_title': '최근 {m}개월 월별 선적 물량 (톤)',
        'bd_stab_shipped': '선적 물량',
        'bd_stab_gap': '선적 없음',
        'bd_stab_kpi_gap': '공백 월 수',
        'bd_stab_kpi_annual': '연평균 물량 (톤)',
        'bd_stab_kpi_imp': '거래 한국 수입사 수',
        'bd_stab_kpi_cv': '단가 CV (%)',
        'bd_stab_year_header': '연도별 요약 (기간 마지막 달 기준 12개월 단위)',
        'bd_col_year': '기간',
        'bd_stab_sentence': '{name}는 최근 {m}개월간 한국에 연평균 {tons:,.0f}톤을 {gap} 공급했습니다. 거래한 한국 수입사는 {imp}곳이고, 단가 변동계수(CV)는 {cv}입니다.',
        'bd_stab_no_gap': '공백 없이',
        'bd_stab_with_gap': '{n}개월의 공백을 두고',
        'bd_stab_no_data': '선택한 공급사의 최근 거래가 없습니다.',
        'bd_method_note': '※ 단가는 모두 물량가중평균(VWAP) USD/kg입니다. 전처리: 중복 행 제거(신고일·수입자·공급자·중량·금액 동일), 건당 최소 중량 미만 제외, 단가 P1~P99 밖 제외.',
        # --- 포트폴리오 헬스체크 ---
        'ph_title': '🩺 포트폴리오 헬스체크',
        'ph_intro': '한 회사의 수입 데이터로 품목(또는 원산지 등)별 단가 변화·공급 차질·수급 경색을 점검하고, 공급 구조의 취약성과 금액 비중(영향도)을 함께 따져 지금 어디가 위험한지 빠르게 확인합니다.',
        'ph_reset_btn': '새로운 점검 시작 (다시하기)',
        'ph_upload_label': '점검할 수입 데이터 파일을 업로드하세요',
        'ph_upload_caption': '※ 날짜, 수입사, 공급사, 품목, 물량, 단가 컬럼이 포함된 TDS raw file을 업로드해주세요.',
        'ph_missing_cols_error': '필수 컬럼이 없습니다: {cols}',
        'ph_importer_label': '점검할 수입사 (물량 큰 순)',
        'ph_importer_auto': '수입사: {name} (파일에 수입사가 1곳뿐이라 자동 적용)',
        'ph_unit_label': '점검 단위',
        'ph_unit_product': '품목',
        'ph_unit_origin': '원산지',
        'ph_unit_caption': '공급사는 공급 구조·공급 변화 판정의 기준 축이라 점검 단위에서 제외됩니다. 품목은 포장 단위(360정·25KG 등)와 끝의 제품 코드만 다른 품목명을 한 단위로 묶고, 정제된 이름으로 보여줍니다(도입 효과 분석과 같은 기준).',
        'ph_base_period': '기준 기간 (현재)',
        'ph_comp_period': '비교 기간 (직전)',
        'ph_period_overlap_warning': '기준 기간과 비교 기간이 겹칩니다. 겹치는 거래는 양쪽에 모두 반영됩니다.',
        'ph_period_invalid_error': '기간의 시작일이 종료일보다 늦습니다. 날짜를 확인해주세요.',
        'ph_threshold_expander': '⚙️ 기준값 조정',
        'ph_th_group_signal': '변화 신호',
        'ph_th_group_other': '취약성 · 영향도 · 신뢰도',
        'ph_th_price_red': '단가 🔴 기준 (상승률 %, 이상)',
        'ph_th_price_yellow': '단가 🟡 기준 (상승률 %, 이상)',
        'ph_th_churn_major': '주요 공급사 기준 (비교 기간 비중 %, 이상)',
        'ph_th_replace_ok': '대체 완료 기준 (빠진 물량 대비 신규 공급사 물량 %, 이상)',
        'ph_th_squeeze_volume': '수급 경색 물량 기준 (월평균 감소율 %, 이상 — 단가 🟡 이상과 겹칠 때 🔴)',
        'ph_th_conc': '집중 기준 (1위 공급사 비중 %, 이상)',
        'ph_th_spread': '분산 기준 (1위 공급사 비중 %, 미만)',
        'ph_th_min_ship': '최소 선적 건수 (두 기간 각각, 미만이면 단가 판단 보류)',
        'ph_th_min_share': '영향도 기준 (금액 비중 %, 미만이면 종합 판정 최대 주의)',
        'ph_th_order_warning': '🟡 기준이 🔴 기준보다 높거나, 분산 기준이 집중 기준보다 높습니다. 기준값을 확인해주세요.',
        'ph_run_btn': '헬스체크 실행',
        'ph_spinner': '점검 중입니다...',
        'ph_no_data_warning': '선택한 수입사·기간에 점검할 데이터가 없습니다.',
        'ph_summary_caption': '수입사: {importer} · 점검 단위: {unit} · 기준 기간 {b_start} ~ {b_end} · 비교 기간 {c_start} ~ {c_end}',
        'ph_kpi_total': '점검 대상',
        'ph_focus_header': '🔎 먼저 볼 것',
        'ph_focus_none': '위험 단위가 없습니다.',
        'ph_table_header': '📋 점검 결과',
        'ph_filter_label': '종합 판정 필터',
        'ph_col_unit': '단위',
        'ph_col_share': '금액 비중',
        'ph_col_structure': '공급 구조',
        'ph_col_supply': '공급 변화',
        'ph_col_price': '단가 변화',
        'ph_col_volume': '물량 변화',
        'ph_col_overall': '종합 판정',
        'ph_col_reason': '사유',
        'ph_status_risk': '위험',
        'ph_status_caution': '주의',
        'ph_status_safe': '안전',
        'ph_status_new': '신규',
        'ph_status_insufficient': '데이터 부족',
        'ph_new_tag': ' (신규)',
        'ph_struct_single': '단일',
        'ph_struct_conc': '집중',
        'ph_struct_normal': '보통',
        'ph_struct_spread': '분산',
        'ph_supply_disruption': '차질',
        'ph_supply_switch': '전환',
        'ph_supply_switch_better': '전환·단가 개선',
        'ph_supply_none': '없음',
        'ph_price_hold': '판단 보류',
        'ph_reason_price': '단가 {chg:+.0f}%',
        'ph_reason_vuln_price': '{struct} 공급 + 단가 {chg:+.0f}%',
        'ph_reason_disruption': '주요 공급사 이탈, 대체 물량 {ratio:.0f}%',
        'ph_reason_squeeze': '수급 경색 (물량 {vol:+.0f}%, 단가 {chg:+.0f}%)',
        'ph_reason_switch': '공급사 전환 (대체 물량 {ratio:.0f}%)',
        'ph_reason_switch_better': '공급사 전환·단가 개선 (대체 물량 {ratio:.0f}%, 단가 {chg:+.0f}%)',
        'ph_reason_hold': '선적 {n}건 — 단가 판단 보류',
        'ph_reason_low_share': '금액 비중 {share:.1f}% — 영향이 작아 주의로 조정',
        'ph_reason_new': '비교 기간 거래 없음',
        'ph_members_expander': '📄 단위별 원래 품목명 보기',
        'ph_col_members': '원래 품목명',
        'ph_col_member_count': '품목명 수',
        'ph_detail_members': '원래 품목명',
        'ph_x_members': '원래 품목명',
        'ph_table_notes': '※ 금액 = 물량 × 물량가중평균 단가(VWAP), 금액 비중은 기준 기간 전체 금액 대비입니다. 물량 변화는 월평균 비교이며 단독으로는 판정에 쓰지 않습니다. 공급 구조(단일/집중)는 단가 🟡와 겹칠 때만 판정에 반영됩니다.',
        'ph_market_note': '※ 단가 신호는 시장 전체 가격 변동 때문일 수도 있습니다 — 시장 요인인지 확인 필요 ([{season}] 메뉴 참고).',
        'ph_mix_notice': '※ 점검 단위가 품목이 아니라서 한 단위 안에 여러 품목이 섞여 있습니다. 품목 구성이 바뀌면 단가 변화에 영향을 줄 수 있습니다.',
        'ph_detail_header': '🔬 위험·주의 단위 상세',
        'ph_detail_top5': '공급사별 비중 (기준 기간 상위 5)',
        'ph_detail_lost': '이탈 공급사 (비교 기간 비중)',
        'ph_detail_new_sup': '신규 공급사 (기준 기간 비중)',
        'ph_detail_no_lost': '이탈한 공급사가 없습니다.',
        'ph_detail_no_new': '신규 공급사가 없습니다.',
        'ph_detail_supplier': '공급사',
        'ph_detail_share': '비중(%)',
        'ph_detail_vwap': '평균 단가 (VWAP, USD/kg)',
        'ph_detail_monthly': '월평균 물량 (kg)',
        'ph_detail_ship': '선적 건수',
        'ph_detail_replace': '대체율 (빠진 주요 공급사 월평균 물량 대비 신규 공급사 월평균 물량): {ratio:.0f}%',
        'ph_detail_base': '기준 기간',
        'ph_detail_comp': '비교 기간',
        'ph_detail_more': '자세히 보려면 [{risk}] · [{churn}] · [{season}] 메뉴를 참고하세요.',
        'ph_excel_btn': '📥 Excel 다운로드',
        'ph_sheet_result': '점검 결과',
        'ph_sheet_suppliers': '공급사 비중',
        'ph_x_amount': '기준 기간 금액(USD)',
        'ph_x_share': '금액 비중(%)',
        'ph_x_n_sup': '공급사 수',
        'ph_x_top1': '1위 공급사',
        'ph_x_top1_share': '1위 공급사 비중(%)',
        'ph_x_lost': '이탈 공급사',
        'ph_x_new_sup': '신규 공급사',
        'ph_x_replace': '대체율(%)',
        'ph_x_base_ship': '기준 선적 건수',
        'ph_x_comp_ship': '비교 선적 건수',
        'ph_x_base_vwap': '기준 VWAP',
        'ph_x_comp_vwap': '비교 VWAP',
        'ph_x_price_chg': '단가 변화(%)',
        'ph_x_base_monthly': '기준 월평균 물량(kg)',
        'ph_x_comp_monthly': '비교 월평균 물량(kg)',
        'ph_x_volume_chg': '물량 변화(%)',
        'ph_x_base_volume': '기준 기간 물량(kg)',
        'ph_x_base_share': '기준 기간 비중(%)',
        'ph_x_comp_share': '비교 기간 비중(%)',
        'multi_product_label': '품목 검색 (여러 개 선택 가능)',
        'group_rule_generic_hs': '※ 일반 명칭 품목(예: [혼합분유] 혼합분유)은 HS코드 10단위로 구분하고, 같은 HS 안의 세부 스펙은 구분하지 않습니다.',
        'multi_product_help': '포장 단위(25KG·360정 등)와 끝의 제품 코드만 다른 품목명은 한 품목으로 합쳐져 있어요 (도입 효과 분석과 같은 기준). 여러 품목을 고르면 하나로 합쳐서 분석합니다.',
        'p5_month_label': '{m}월',
        'p5_no_prev_year_caption': '※ 전년도에 해당하는 데이터가 없어 전년 비교선을 표시하지 못했습니다 (최소 13개월치 데이터가 필요합니다).',
        'p5_pack_marker': '포장 구성 변화: {b} {bs:.0f}% → {a} {as_:.0f}%',
        'p5_pack_marker_legend': '포장 구성 변화',
        'p5_pack_toggle': '포장 단위별로 보기',
        'p5_pack_note': '◆ 표시는 직전 달보다 주력 포장 단위가 바뀌었거나 포장 단위별 물량 비중이 20%p 이상 움직인 달입니다 (포장 표기가 없는 거래는 판정에서 제외). 합친 추세선의 단가가 포장 구성 때문에 흔들렸을 수 있습니다.',
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
        'menu_opt_customer': 'Adoption Impact',
        'menu_opt_health': 'Portfolio Health Check',
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

        'p1_title': '💲 Adoption Impact',
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
        'p1_total_savings_caption': '※ Net savings since the contract date ({date}): the sum of per-group savings (price difference before vs. after × post-contract volume). The savings and cost-increase totals exclude minor changes (within ±${th:,.0f}) and items with no purchases after the contract.',
        'p1_minor_threshold_label': 'Minor-change threshold ($ — items whose absolute savings/increase is at or below this are "minor")',
        'p1_kpi_net': 'Net savings',
        'p1_kpi_saved': 'Total savings ({n} items)',
        'p1_kpi_increased': 'Total cost increase ({n} items)',
        'p1_group_saved': '💰 Savings ({n} items, ${amount:,.0f})',
        'p1_group_increased': '📈 Price increases ({n} items, ${amount:,.0f})',
        'p1_group_minor': '➖ Minor change ({n} items)',
        'p1_group_empty': 'No items in this group.',
        'p1_minor_caption': '※ Items whose absolute savings/increase is ${th:,.0f} or less. A $0 card means the price barely changed and was rounded.',
        'p1_show_all_btn': 'Show all ({n} more)',
        'p1_show_less_btn': 'Show less',
        'p1_no_purchase_caption': '🚫 No purchases after the contract ({n}): {names} — excluded from the savings and cost-increase totals because post-contract volume is 0.',
        'p1_col_group': 'Group',
        'p1_grp_saved_label': 'Savings',
        'p1_grp_increased_label': 'Price increase',
        'p1_grp_minor_label': 'Minor change',
        'p1_grp_no_purchase_label': 'No purchases after contract',
        'p1_savings_detail_subheader': 'Detailed Savings by Product Group',
        'p1_mix_changed_badge': 'Pack size changed',
        'p1_mix_changed_tooltip': 'The main pack size of this group changed, or volume shares by pack size shifted significantly before vs. after the contract. Smaller packs cost more per kg, so the savings may reflect a pack change rather than negotiation.',
        'p1_mix_changed_caption': '⚠ Pack size changes ({n}): {items} — the main pack size changed or a pack size share moved 20%p or more. Check before reading the savings as negotiation results.',
        'p1_pack_none': 'No pack size',
        'p1_pack_change_main': 'Pack size changed: {b} → {a}',
        'p1_pack_change_form': 'Dosage form changed: {b} → {a}',
        'p1_pack_change_share': 'Pack size share changed: {pack} {b:.0f}% → {a:.0f}%',
        'price_mismatch_warning': '⚠ In {pct:.0f}% of rows, Unit Price differs from value ÷ weight by more than 5%. The file\'s Unit Price was used as is — check for mixed units (kg/ton) or currencies.',
        'p1_col_avg_price_before': 'Avg Price Before (VWAP)',
        'p1_col_avg_price_after': 'Avg Price After (VWAP)',
        'p1_col_volume_after': 'Volume After',
        'p1_col_savings': 'Savings',
        'p1_col_mix_detail': 'Pack size change',
        'p1_col_mix_changed': 'Mix warning',
        'p1_vwap_note': '※ Average price is volume-weighted (VWAP), not a simple transaction average. Outliers are removed automatically per cluster and period.',
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
        'p7_dim_year': 'By year',
        'p7_dim_quarter': 'By quarter',
        'p7_dim_month_only': 'By month (all years)',
        'p7_capped_note': "※ '{field}' had too many distinct values — showing the top {n} by volume, with the rest grouped as \"Others\".",
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
        'p7_total_col': 'Total',

        'pdf_generate_btn': '📄 Generate PDF Report',
        'pdf_download_btn': '📥 Download PDF',
        'pdf_generating_msg': 'Generating PDF...',
        'pdf_error_msg': 'An error occurred while generating the PDF: {msg}',
        'pdf_chart_export_warning': 'Could not add a chart image to the PDF (the PDF will be generated without it): {msg}',
        # --- ① Adoption Impact PDF (new template) ---
        'p1_rp_eyebrow': 'Adoption Impact Report',
        'p1_rp_title2': 'Adoption Impact Report',
        'p1_rp_blind_toggle': 'Blind the client name in the PDF (Importer A)',
        'p1_rp_legacy_btn': 'Legacy-format PDF',
        'p1_rp_s1_eyebrow': '01  Executive Summary',
        'p1_rp_head_save': 'Net savings of ${x:,.0f} since the contract',
        'p1_rp_head_loss': 'Costs rose by a net ${x:,.0f} since the contract',
        'p1_rp_kpi_net': 'Net savings',
        'p1_rp_kpi_net_note_excl': '${y:,.0f} excl. mix-changed items',
        'p1_rp_kpi_net_note': 'Price difference × post-contract volume',
        'p1_rp_kpi_saved': 'Total savings ({n})',
        'p1_rp_kpi_increased': 'Total cost increase ({n})',
        'p1_rp_kpi_group_note': 'Excl. minor changes (±${th:,.0f})',
        'p1_rp_kpi_price': 'Unit price change',
        'p1_rp_kpi_price_note': 'VWAP at post-contract volume vs. before',
        'p1_rp_findings': 'Key findings',
        'p1_rp_excl_suffix': ' (${y:,.0f} excluding mix-changed items)',
        'p1_rp_f1_save': 'Net savings of ${x:,.0f} since the contract{excl}.',
        'p1_rp_f1_loss': 'Costs rose by a net ${x:,.0f} since the contract{excl}.',
        'p1_rp_f1_d': '{ns} item group(s) saved ${s:,.0f} in total; {ni} item group(s) with higher prices added ${i:,.0f}.',
        'p1_rp_f2': '{name} accounts for {p:.0f}% of total savings.',
        'p1_rp_f2_concentrated': '{name} accounts for {p:.0f}% of total savings — savings are concentrated in one item, leaving room to improve the others.',
        'p1_rp_f2_d': '${b:.2f} → ${a:.2f}/kg before vs. after the contract, on {v:,.0f} kg bought after the contract.',
        'p1_rp_f2_none': 'No item group saved more than the minor-change threshold (±${th:,.0f}).',
        'p1_rp_f2_none_d': 'Price differences before and after the contract are small, or post-contract volumes are small.',
        'p1_rp_f3_up': 'Prices rose for {n} item group(s) such as {names}, adding ${x:,.0f} in cost.',
        'p1_rp_f3_up_d': 'Consider renegotiating or reviewing alternative suppliers.',
        'p1_rp_f3_none': 'No item group saw a price increase.',
        'p1_rp_f3_none_d': 'All item groups were bought at the same or lower prices than before the contract.',
        'p1_rp_f3_mix': ' For {k} item group(s) whose pack size changed, the per-kg price difference may come from the pack change rather than negotiation.',
        'p1_rp_s2_eyebrow': '02  Savings by Item',
        'p1_rp_s2_head': 'Savings and cost increases by item group',
        'p1_rp_s2_sub': 'Volume-weighted price difference before vs. after the contract × post-contract volume.',
        'p1_rp_chart': 'Top item groups by savings and cost increase',
        'p1_rp_chart_note': 'Blue = savings, Black = cost increase from higher prices. Up to the top {n} of each.',
        'p1_rp_th_item': 'Item group',
        'p1_rp_th_before': 'Before<br/>USD/kg',
        'p1_rp_th_after': 'After<br/>USD/kg',
        'p1_rp_th_vol': 'After<br/>vol. (kg)',
        'p1_rp_th_saving': 'Savings<br/>USD',
        'p1_rp_th_mix': 'Mix',
        'p1_rp_badge_mix': 'Pack change',
        'p1_rp_s3_eyebrow': '03  Items to Check',
        'p1_rp_s3_head': 'Items to check separately',
        'p1_rp_s3_sub': 'Item groups with no purchases after the contract, and item groups whose pack sizes changed significantly. Treat their savings with care.',
        'p1_rp_s3_nopurchase': 'No purchases after contract ({n})',
        'p1_rp_s3_nopurchase_note': 'Excluded from the savings and cost-increase totals because post-contract volume is 0.',
        'p1_rp_s3_mix': 'Pack size changes ({n})',
        'p1_rp_s3_mix_note': 'Within the group, the main pack size changed or a pack size share moved 20%p or more. Smaller packs cost more per kg, so read the savings with care.',
        'p1_rp_th_overlap': 'Pack size change',
        'p1_rp_s3_none': 'None.',
        'p1_rp_s4_eyebrow': '04  Methodology',
        'p1_rp_s4_head': 'Criteria and data limits',
        'p1_rp_lbl_data': 'Data', 'p1_rp_lbl_period': 'Period', 'p1_rp_lbl_cluster': 'Item grouping', 'p1_rp_lbl_outlier': 'Outliers',
        'p1_rp_lbl_price': 'Average price', 'p1_rp_lbl_saving': 'Savings', 'p1_rp_lbl_minor': 'Minor change', 'p1_rp_lbl_mix': 'Mix change',
        'p1_rp_lbl_blind': 'Blind',
        'p1_rp_m_data': 'Tridge TDS Korea import customs data, per declaration. The most frequent importer in the file is analyzed',
        'p1_rp_m_period': 'Before {b0} – {b1} · after {a0} – {a1} (contract date {c})',
        'p1_rp_m_cluster': 'Trades whose product names match after removing pack sizes and product codes form one item group (names with fewer than 3 trades excluded). Generic-name products are separated by 10-digit HS code; detailed specs within the same HS code are not distinguished',
        'p1_rp_m_outlier': 'Unit prices outside 1.5× IQR per item group and period excluded (at most 7%)',
        'p1_rp_m_price': 'VWAP (volume-weighted), USD/kg',
        'p1_rp_m_saving': '(VWAP before − VWAP after) × post-contract volume. Positive = savings, negative = cost increase',
        'p1_rp_m_minor': 'Item groups with absolute savings/increase of ${th:,.0f} or less are minor and excluded from the savings and cost-increase totals (included in net savings)',
        'p1_rp_m_mix': 'Volume shares by pack size compared before vs. after; flagged when the main pack size changes or a share moves 20%p or more (trades without a pack size in the name are excluded)',
        'p1_rp_lbl_pack': 'Pack size',
        'p1_rp_m_pack': 'Number + unit in the product name (tablets, sachets, capsules, g, ml, etc.) is split off as the pack size, trailing product codes (PEF225, ES 2677, etc.) are removed, and identical names form one item group',
        'p1_rp_m_blind_on': 'Client name withheld (shown as Importer A)',
        'p1_rp_m_blind_off': 'Client name shown',
        'p1_rp_unknown1': 'The effect of market-wide price moves in the same period (see Market Competitiveness Analysis)',
        'p1_rp_unknown2': 'Price differences from quality/spec changes or contract terms (payment, Incoterms)',
        'p1_rp_unknown3': 'Costs after arrival, such as tariffs and domestic logistics',
        'p1_rp_unknown4': 'What prices would have been without the contract (before/after comparison is an estimate of the effect)',
        'p1_rp_close_head': 'Suggested next steps',
        'p1_rp_close_k1': 'Report results',
        'p1_rp_close_k2': 'Renegotiate',
        'p1_rp_close_k3': 'To check',
        'p1_rp_close_v1': 'Net savings ${x:,.0f}{excl} · main items: {items}',
        'p1_rp_close_v1_loss': 'Net cost increase ${x:,.0f}{excl} — start with the item groups whose prices rose',
        'p1_rp_close_v2': '{n} item group(s) with higher prices (${x:,.0f}): {items}',
        'p1_rp_close_v2_none': 'No price increases — keep current terms',
        'p1_rp_close_v3': '{k} mix-change warning(s) · {m} item group(s) with no purchases after the contract',
        'p1_rp_close_note': 'Before/after price comparisons also reflect market prices, quality, and contract terms. Figures are estimates based on customs data.',
        # --- New PDF template common (report_theme) ---
        'rp_lbl_period': 'Period',
        'rp_lbl_data': 'Data',
        'rp_lbl_date': 'Date',
        'rp_lbl_scope': 'Disclosure',
        'rp_data_source': 'Tridge TDS · Korea import customs data',
        'rp_blind_on': 'Blind — Korean importer names withheld',
        'rp_blind_off': 'Internal use',
        'rp_confidential': 'Confidential · Figures are estimates based on customs data and must not be the sole basis for investment or contract decisions.',
        'rp_footer': 'Source: Tridge TDS · VWAP (volume-weighted) · USD/kg',
        'rp_period_fmt': '{start} – {end} ({months} months)',
        'rp_meth_item': 'Item',
        'rp_meth_rule': 'Rule',
        'rp_unknown_header': 'What this report cannot tell you',
        'rp_pdf_btn': '📄 Create PDF report',
        'rp_pdf_dl': '📥 Download PDF report',
        'rp_pdf_spinner': 'Creating the PDF report...',
        'rp_count_unit': '{n}',
        'rp_company_unit': '{n}',
        'rp_month_unit': '{n} mo',
        # --- ⑨ Screener PDF ---
        'scr_rp_eyebrow': 'Commodity Screening Report',
        'scr_rp_title1': 'Commodity Screening',
        'scr_rp_title2': 'Report',
        'scr_rp_item': 'Screening of {n} HS codes',
        'scr_rp_s1_eyebrow': '01  Executive Summary',
        'scr_rp_s1_head_a': '{n} Class A items offer room for tendering and alternative sourcing',
        'scr_rp_s1_head_none': 'No item combines high universality with a large price gap yet',
        'scr_rp_kpi_note_a': 'High universality · large gap',
        'scr_rp_kpi_note_b': 'Medium universality',
        'scr_rp_kpi_note_c': 'High universality · small gap',
        'scr_rp_kpi_note_d': 'Low universality',
        'scr_rp_top_header': 'Top Class A items (by universality)',
        'scr_rp_top_none': 'No items fall into Class A.',
        'scr_rp_s2_eyebrow': '02  Quadrant',
        'scr_rp_s2_head': 'Universality × Gap quadrant',
        'scr_rp_s2_sub': 'X = universality percentile, Y = gap score, bubble size = annual import value',
        'scr_rp_s2_note': 'Dotted lines are class boundaries (universality {lo}/{hi}, gap {gap}). HS codes are labeled only for the top Class A items by import value. Hollow circles are items whose gap cannot be calculated (fewer than 3 shipments per month and origin), shown on the zero line.',
        'scr_rp_th_importers': 'Imp.',
        'scr_rp_legend_nogap': 'Gap not computable',
        'scr_rp_th_exporters': 'Supp.',
        'scr_rp_th_origins': 'Orig.',
        'scr_rp_th_cr3': 'CR3',
        'scr_rp_th_uni': 'Univ.',
        'scr_rp_th_gap': 'Gap',
        'scr_rp_th_value': 'Annual value',
        'scr_rp_s3_eyebrow': '03  Items by Class',
        'scr_rp_s3_head': 'Items by class',
        'scr_rp_s3_sub': 'Top {n} items per class by universality. The full list is available as CSV on screen.',
        'scr_rp_s4_eyebrow': '04  Methodology',
        'scr_rp_s4_head': 'Metric definitions and data limits',
        'scr_rp_m_data': 'Tridge TDS Korea import customs data, per declaration, grouped by 10-digit HS code',
        'scr_rp_m_period': '{period}',
        'scr_rp_m_prep': 'Duplicates removed (same date, importer, supplier, weight, value); shipments under {kg:,} kg excluded; unit prices outside P1–P99 per HS excluded; HS codes with fewer than {ship} shipments not ranked',
        'scr_rp_m_uni': 'Weighted average of dataset percentiles for importers↑, CR3↓, overseas suppliers↑, origins↑, months with trade↑, unit price CV↓ (weights {w})',
        'scr_rp_m_gap': 'Percentile of the gap rate: volume-weighted (P90−P10)/P50 unit price within the same month and origin (cells with 3+ shipments only)',
        'scr_rp_m_class': 'Universality ≥ {hi} & gap ≥ {gap} → A, universality ≥ {hi} & gap < {gap} → C, {lo}–{hi} → B, below {lo} → D',
        'scr_rp_m_cr3': 'CR3 = volume share of the top 3 importers; HHI = sum of squared importer shares (%)',
        'scr_rp_m_price': 'Average prices are VWAP (volume-weighted), USD/kg; unit price CV is also volume-weighted',
        'scr_rp_lbl_data': 'Data', 'scr_rp_lbl_period': 'Period', 'scr_rp_lbl_prep': 'Preprocessing', 'scr_rp_lbl_uni': 'Universality',
        'scr_rp_lbl_gap': 'Gap score', 'scr_rp_lbl_class': 'Class', 'scr_rp_lbl_conc': 'Concentration', 'scr_rp_lbl_price': 'Price',
        'scr_rp_unknown1': 'Price differences caused by mixed specs or grades within the same HS code (split them with product keywords in Tender Response Analysis)',
        'scr_rp_unknown2': 'The actual buyer when the importer of record is a distributor or agent',
        'scr_rp_unknown3': 'Costs after arrival, such as tariffs and domestic logistics',
        'scr_rp_unknown4': 'Future demand or price forecasts',
        'scr_rp_close_head': 'Suggested next steps',
        'scr_rp_close_k1': 'Review first',
        'scr_rp_close_k2': 'Deep dive',
        'scr_rp_close_k3': 'Criteria',
        'scr_rp_close_v1_a': 'Top Class A items: {items}',
        'scr_rp_close_v1_none': 'No Class A items; start with top Class B items: {items}',
        'scr_rp_close_v2': 'Send items of interest to Tender Response Analysis to check alternative supplier prices and supply stability',
        'scr_rp_close_v3': 'Class boundaries: universality {lo}/{hi}, gap {gap} percentile · shipments of {kg:,} kg or more',
        'scr_rp_close_note': 'Scores are relative ranks within the uploaded dataset; mixing other product groups changes the ranking. Figures are estimates based on customs data.',
        # --- ⑩ Tender PDF ---
        'bd_rp_header': '📄 PDF report',
        'bd_rp_item_label': 'Item name for the report',
        'bd_rp_spec_label': 'Spec (shown in the report)',
        'bd_rp_stab_label': 'Suppliers for supply stability pages (one page each)',
        'bd_rp_eyebrow': 'Bid Response Report',
        'bd_rp_title2': 'Bid Response Report',
        'bd_rp_s1_eyebrow': '01  Executive Summary',
        'bd_rp_head_save': 'Verified alternative suppliers can cut the unit price by {pct:.0f}%',
        'bd_rp_head_found': '{n} verified alternative supplier(s) identified',
        'bd_rp_head_none': 'No supplier has passed the verification criteria yet',
        'bd_rp_kpi1': 'Lowest verified price',
        'bd_rp_kpi1_note': 'USD/kg · {basis} · {origin}',
        'bd_rp_kpi1_none': 'No verified supplier',
        'bd_rp_basis_cif': 'CIF',
        'bd_rp_basis_tariff': 'incl. tariff',
        'bd_rp_kpi2': 'vs. current supply',
        'bd_rp_kpi2_note': 'Baseline ${p:.2f}/kg',
        'bd_rp_kpi_no_base': 'No baseline supplier',
        'bd_rp_kpi3': 'Estimated annual savings',
        'bd_rp_kpi3_note': 'Assuming {t:,.0f} t/year',
        'bd_rp_kpi4': 'Supply gaps',
        'bd_rp_kpi4_note': 'Last {m} mo · top supplier',
        'bd_rp_findings': 'Key findings',
        'bd_rp_f1_two': '{o1} is ${d:.2f}/kg cheaper than {o2}.',
        'bd_rp_f1_two_d': 'VWAP compared under the same conditions (HS and product keywords). Both are major origins shipping {t:,.0f}+ tons a year.',
        'bd_rp_f1_one': '{o} ({tons:,.0f} t) is the only major origin shipping {t:,.0f}+ tons a year.',
        'bd_rp_f1_one_d': 'Other origins have small volumes, so price comparisons with them are weak.',
        'bd_rp_f1_none': 'No origin ships {t:,.0f}+ tons a year.',
        'bd_rp_f1_none_d': 'All origins have small volumes; treat origin price comparisons as reference only.',
        'bd_rp_f2_base': '{k} of {n} verified suppliers are at least 10% cheaper than the current supply.',
        'bd_rp_f2_nobase': '{n} supplier(s) passed verification.',
        'bd_rp_f2_d': 'Verification: {ship}+ shipments, {ton:,.0f}+ tons, and {imp}+ Korean importers in the last 12 months.',
        'bd_rp_f3_cheap': '{names}: low prices, but not yet verified.',
        'bd_rp_f3_cheap_d': 'Too few trades in the last 12 months to recommend as a main supplier in a bid. Consider only as a backup option.',
        'bd_rp_f3_none': 'No unverified supplier is cheaper than the lowest verified price.',
        'bd_rp_f3_none_d': 'The lowest comparison price comes from a verified supplier.',
        'bd_rp_tariff_missing': 'Tariff rates were not entered for some origins, so those are compared on a CIF basis. Rankings may change with FTA rates.',
        'bd_rp_tariff_ok': 'Comparison price = CIF × (1 + rate) using the tariff rates entered per origin.',
        'bd_rp_s2_eyebrow': '02  Alternative Suppliers',
        'bd_rp_s2_head': 'Alternative suppliers',
        'bd_rp_s2_sub': 'VWAP and verification metrics compared by origin and supplier under the same conditions.',
        'bd_rp_origin_chart': 'Unit price range by origin',
        'bd_rp_origin_chart_note': 'Dot = VWAP, bar = P10–P90. Hollow dots are origins under {t:,.0f} t/year.',
        'bd_rp_baseline_line': 'Current supply ${p:.2f}',
        'bd_rp_rank_header': 'Supplier ranking (by comparison price)',
        'bd_rp_th_no': '#',
        'bd_rp_th_cmp': 'Price<br/>USD/kg',
        'bd_rp_th_vs': 'vs. base',
        'bd_rp_th_ship': 'Ship.',
        'bd_rp_th_ton': 'Vol. (t)',
        'bd_rp_th_months': 'Mo.',
        'bd_rp_th_imp': 'KR<br/>imp.',
        'bd_rp_th_status': 'Status',
        'bd_rp_base_tag': 'Baseline',
        'bd_rp_rank_note': 'Verified = {ship}+ shipments · {ton:,.0f}+ tons · {imp}+ Korean importers in the last 12 months. Importers are shown as counts only. Top {n} suppliers shown.',
        'bd_rp_s3_eyebrow': '03  Supply Track Record · {i}/{n}',
        'bd_rp_s3_sub': '{origin} · shipments to Korea, last {m} months',
        'bd_rp_s3_kpi1': 'Average annual supply',
        'bd_rp_s3_kpi1_note': 'Based on the last {m} months',
        'bd_rp_s3_kpi2': 'Gap months',
        'bd_rp_s3_kpi2_note': 'Months with no shipment',
        'bd_rp_s3_kpi3': 'Korean importers',
        'bd_rp_s3_kpi3_note': 'Names withheld',
        'bd_rp_s3_kpi4': 'Price volatility',
        'bd_rp_s3_kpi4_note': 'Volume-weighted CV',
        'bd_rp_s3_chart': 'Monthly shipped volume',
        'bd_rp_s3_chart_gap': 'Hatched bars are months with no shipment.',
        'bd_rp_s3_chart_nogap': 'No month without shipments in {m} months.',
        'bd_rp_s3_year': 'Yearly summary',
        'bd_rp_s3_th_year': 'Period',
        'bd_rp_s3_th_ton': 'Shipped (t)',
        'bd_rp_s3_th_months': 'Shipping months',
        'bd_rp_s3_th_vwap': 'VWAP USD/kg',
        'bd_rp_s3_th_imp': 'Korean importers',
        'bd_rp_proof': 'Proof Statement',
        'bd_rp_s4_eyebrow': '04  Methodology',
        'bd_rp_s4_head': 'Criteria and data limits',
        'bd_rp_m_data': 'Tridge TDS Korea import customs data, per declaration',
        'bd_rp_m_period': '{period}. Supply stability covers the last 36 months up to the period end',
        'bd_rp_m_spec': 'HS {hs} · product name includes: {inc} · excludes: {exc}',
        'bd_rp_m_dedup': 'Rows with the same date, importer, supplier, weight, and value counted once',
        'bd_rp_m_outlier': 'Shipments under {kg:,} kg; unit prices below P1 or above P99',
        'bd_rp_m_price': 'VWAP (volume-weighted), USD/kg, CIF',
        'bd_rp_m_tariff_none': 'Not applied. With rates per origin, comparison price = CIF × (1 + rate)',
        'bd_rp_m_tariff_part': 'Partly applied ({rates}). Origins without a rate use CIF as is',
        'bd_rp_m_tariff_all': 'Applied ({rates}). Comparison price = CIF × (1 + rate)',
        'bd_rp_m_blind_on': 'Korean importer names withheld (replaced by A/B/C by volume). Overseas supplier names shown',
        'bd_rp_m_blind_off': 'Internal use — Korean importer names shown',
        'bd_rp_m_verify': '{ship}+ shipments · {ton:,.0f}+ tons · {imp}+ Korean importers in the last 12 months',
        'bd_rp_lbl_data': 'Data', 'bd_rp_lbl_period': 'Period', 'bd_rp_lbl_spec': 'Spec split', 'bd_rp_lbl_dedup': 'Duplicates',
        'bd_rp_lbl_outlier': 'Outliers', 'bd_rp_lbl_price': 'Average price', 'bd_rp_lbl_tariff': 'Tariff', 'bd_rp_lbl_blind': 'Blind',
        'bd_rp_lbl_verify': 'Verification',
        'bd_rp_unknown1': 'Costs after arrival such as customs, quarantine, transport, and storage in Korea',
        'bd_rp_unknown2': 'The actual buyer when the importer of record is a distributor or agent',
        'bd_rp_unknown3': 'Detailed quality specs (moisture, length distribution, etc.) and certifications of suppliers',
        'bd_rp_unknown4': 'Future price forecasts; use Tridge Eye for market outlooks',
        'bd_rp_close_head': 'Recommended bid setup',
        'bd_rp_close_k1': 'Main supplier',
        'bd_rp_close_k2': 'Backup supplier',
        'bd_rp_close_k3': 'Savings basis',
        'bd_rp_close_main': '{name} ({origin}) — ${p:.2f}/kg, verified, {gap} gap month(s)',
        'bd_rp_close_backup': '{name} ({origin}) — ${p:.2f}/kg, supplies {imp} importer(s)',
        'bd_rp_close_none': 'No verified supplier — check the track record of the lowest candidate {name} ({origin}, ${p:.2f}/kg)',
        'bd_rp_close_no_backup': 'Only one supplier passed verification',
        'bd_rp_close_save': '–{pct:.1f}% vs. current supply, about ${k:,.0f}K at {t:,.0f} t/year',
        'bd_rp_close_nosave': 'No baseline supplier was selected, so savings were not calculated',
        'bd_rp_close_nobetter': 'No verified supplier is cheaper than the current supply yet',
        'bd_rp_close_note_missing': 'Tariffs (partly) and domestic logistics not included. Check FTA rates per origin and domestic costs before a final landed-cost decision. Figures are estimates based on customs data.',
        'bd_rp_close_note_ok': 'Domestic logistics not included. Check domestic costs before a final landed-cost decision. Figures are estimates based on customs data.',
        # --- Common: blind ---
        'blind_label': 'Importer {code}',
        'blind_toggle': 'Blind Korean importer names (A/B/C… by volume)',
        # --- ⑨ Commodity Screener ---
        'menu_opt_screener': 'Commodity Screener',
        'sc_title': '🧭 Commodity Screener',
        'sc_intro': 'For each 10-digit HS code, compares how open the market is (universality) with how widely unit prices vary within the same month and origin (gap), to find items with room for tendering or alternative sourcing.',
        'sc_reset_btn': 'Start a New Screening (Reset)',
        'sc_upload_label': 'Upload the market data file to screen',
        'sc_upload_caption': '※ A TDS raw file with multiple HS codes and importers (date, importer, supplier, origin, HS code, volume, and unit price required)',
        'sc_missing_cols_error': 'Required columns are missing: {cols}',
        'sc_period_label': 'Analysis period (default: last 12 months)',
        'sc_min_kg_label': 'Minimum weight per shipment (kg; smaller shipments excluded)',
        'sc_min_ship_label': 'Minimum shipments per HS (HS codes below this are not ranked)',
        'sc_weights_header': 'Universality score weights',
        'sc_w_importers': 'Number of importers (more ↑)',
        'sc_w_cr3': 'CR3 (lower ↑)',
        'sc_w_exporters': 'Number of overseas suppliers (more ↑)',
        'sc_w_origins': 'Number of origins (more ↑)',
        'sc_w_months': 'Months with trade (more ↑)',
        'sc_w_cv': 'Unit price CV (lower ↑)',
        'sc_bounds_header': 'Class boundaries (percentile)',
        'sc_bound_hi': 'Universality upper boundary',
        'sc_bound_lo': 'Universality lower boundary',
        'sc_bound_gap': 'Gap upper boundary (splits A/C within high universality)',
        'sc_bounds_warning': 'The lower boundary is above the upper boundary. Please check the boundaries.',
        'sc_run_btn': 'Run Screening',
        'sc_spinner': 'Calculating metrics per item...',
        'sc_no_data_warning': 'No trades match the conditions. Try adjusting the period, minimum weight, or minimum shipments.',
        'sc_prep_caption': 'Preprocessing: {n0:,} raw rows → excluded {dup:,} duplicates, {small:,} small shipments, {outlier:,} price outliers (outside P1–P99 per HS) → {n:,} rows used · {skipped} HS code(s) not ranked due to too few shipments',
        'sc_kpi_total': 'HS codes analyzed',
        'sc_class_a': 'A Top target',
        'sc_class_b': 'B Review',
        'sc_class_c': 'C Universal, small gap',
        'sc_class_d': 'D Limited',
        'sc_quadrant_header': '📍 Universality × Gap quadrant',
        'sc_axis_uni': 'Universality (percentile)',
        'sc_axis_gap': 'Gap score (percentile)',
        'sc_bubble_note': '※ Bubble size = annual import value. Vertical lines are universality boundaries; the horizontal line is the gap boundary.',
        'sc_rank_header': '📋 Item ranking',
        'sc_filter_label': 'Class filter',
        'sc_col_hs': 'HS code',
        'sc_col_name': 'Representative product name',
        'sc_col_class': 'Class',
        'sc_col_uni': 'Universality score',
        'sc_col_uni_rank': 'Universality percentile',
        'sc_col_gap_score': 'Gap score',
        'sc_col_importers': 'Importers',
        'sc_col_cr3': 'CR3 (%)',
        'sc_col_hhi': 'HHI',
        'sc_col_exporters': 'Overseas suppliers',
        'sc_col_origins': 'Origins',
        'sc_col_months': 'Months with trade',
        'sc_col_freq': 'Shipments per importer per year',
        'sc_col_cv': 'Unit price CV (%)',
        'sc_col_gap': 'Gap rate (%)',
        'sc_col_value': 'Annual import value (USD)',
        'sc_col_ship': 'Shipments',
        'sc_col_vwap': 'VWAP (USD/kg)',
        'sc_csv_btn': '📥 Download CSV',
        'sc_goto_header': '➡️ Send to Tender Response Analysis',
        'sc_goto_select': 'Item (HS) to analyze',
        'sc_goto_btn': 'Open Tender Response Analysis for this item',
        'sc_goto_done': '{hs} was sent to Tender Response Analysis. If the menu does not switch automatically, click [{menu}] in the left menu.',
        'sc_metric_note': '※ Universality score = weighted average of dataset percentiles for importers↑, CR3↓, suppliers↑, origins↑, months with trade↑, and unit price CV↓. Gap rate = volume-weighted average of (P90−P10)/P50 unit price within the same month and origin (cells with 3+ shipments only). All prices are volume-weighted (VWAP) USD/kg.',
        # --- ⑩ Tender Response Analysis ---
        'menu_opt_bid': 'Tender Response Analysis',
        'bd_title': '📑 Tender Response Analysis',
        'bd_intro': 'For one item (HS and spec), compares prices and track records of alternative suppliers and checks the supply stability of candidates.',
        'bd_reset_btn': 'Start a New Analysis (Reset)',
        'bd_upload_label': 'Upload the market data file to analyze',
        'bd_upload_caption': '※ TDS raw file (date, importer, supplier, origin, volume, and unit price required; an HS code column lets you filter by HS)',
        'bd_use_screener_btn': 'Use the file uploaded to the Commodity Screener',
        'bd_missing_cols_error': 'Required columns are missing: {cols}',
        'bd_filter_header': '🔎 Analysis target',
        'bd_hs_label': 'HS code',
        'bd_hs_all': '(All)',
        'bd_hs_from_screener': 'Item sent from the Commodity Screener: {hs}',
        'bd_include_label': 'Product name must include (comma-separated, any match)',
        'bd_exclude_label': 'Product name must exclude (comma-separated, any match excludes)',
        'bd_period_label': 'Analysis period (default: last 12 months)',
        'bd_min_kg_label': 'Minimum weight per shipment (kg; smaller shipments excluded)',
        'bd_prep_caption': 'Preprocessing: {n0:,} raw rows → excluded {dup:,} duplicates, {small:,} small shipments, {outlier:,} price outliers (outside P1–P99) · {n:,} rows in the period after filters',
        'bd_no_data_warning': 'No trades match the conditions. Try adjusting the HS code, keywords, period, or minimum weight.',
        'bd_tab_alt': 'Alternative suppliers',
        'bd_tab_stability': 'Supply stability',
        'bd_verify_header': 'Verification criteria (last 12 months up to the period end)',
        'bd_verify_ship': 'Minimum shipments',
        'bd_verify_ton': 'Minimum volume (tons)',
        'bd_verify_imp': 'Minimum Korean importers',
        'bd_verified_only': 'Show verified suppliers only',
        'bd_badge_verified': 'Verified',
        'bd_badge_unverified': 'Unverified',
        'bd_tariff_header': 'Tariff rates (optional, % by origin)',
        'bd_tariff_caption': '※ With a tariff rate, comparison price = CIF × (1 + rate). Left blank, CIF is compared as is. Entered manually (no tariff database).',
        'bd_tariff_col_origin': 'Origin',
        'bd_tariff_col_rate': 'Tariff (%)',
        'bd_badge_no_tariff': 'Tariff not applied',
        'bd_badge_tariff_ok': 'Tariff applied',
        'bd_tariff_missing_note': 'Some origins have no tariff rate, so tariffs are not reflected in some comparison prices.',
        'bd_origin_header': '🌏 Comparison by origin',
        'bd_origin_chart_title': 'Unit price distribution by origin (USD/kg)',
        'bd_supplier_header': '🏭 Comparison by overseas supplier',
        'bd_base_label': 'Baseline (current) supplier',
        'bd_base_none': '(None)',
        'bd_annual_ton_label': 'Expected annual volume (tons, for savings)',
        'bd_col_origin': 'Origin',
        'bd_col_exporter': 'Overseas supplier',
        'bd_col_status': 'Verification',
        'bd_col_vwap': 'VWAP (USD/kg)',
        'bd_col_cmp': 'Comparison price (USD/kg)',
        'bd_col_tariff': 'Tariff (%)',
        'bd_col_ship': 'Shipments',
        'bd_col_ton': 'Total weight (tons)',
        'bd_col_months': 'Months with trade',
        'bd_col_importers': 'Korean importers',
        'bd_col_importer_names': 'Korean importers supplied',
        'bd_col_last': 'Latest trade',
        'bd_col_saving_pct': 'Savings (%)',
        'bd_col_saving_amt': 'Annual savings (USD)',
        'bd_csv_origin_btn': '📥 Origin table CSV',
        'bd_csv_supplier_btn': '📥 Supplier table CSV',
        'bd_stab_select': 'Supplier',
        'bd_stab_chart_title': 'Monthly shipped volume, last {m} months (tons)',
        'bd_stab_shipped': 'Shipped volume',
        'bd_stab_gap': 'No shipment',
        'bd_stab_kpi_gap': 'Months without shipments',
        'bd_stab_kpi_annual': 'Average annual volume (tons)',
        'bd_stab_kpi_imp': 'Korean importers supplied',
        'bd_stab_kpi_cv': 'Unit price CV (%)',
        'bd_stab_year_header': 'Yearly summary (12-month blocks ending at the period end)',
        'bd_col_year': 'Period',
        'bd_stab_sentence': 'Over the last {m} months, {name} supplied Korea an average of {tons:,.0f} tons a year {gap}. It supplied {imp} Korean importer(s), with a unit price coefficient of variation (CV) of {cv}.',
        'bd_stab_no_gap': 'without a gap',
        'bd_stab_with_gap': 'with {n} month(s) without shipments',
        'bd_stab_no_data': 'The selected supplier has no recent trades.',
        'bd_method_note': '※ All prices are volume-weighted average (VWAP) USD/kg. Preprocessing: duplicate rows removed (same date, importer, supplier, weight, and value), shipments under the minimum weight excluded, unit prices outside P1–P99 excluded.',
        # --- Portfolio Health Check ---
        'ph_title': '🩺 Portfolio Health Check',
        'ph_intro': "Checks one company's import data per product (or origin, etc.) for unit price changes, supply disruption, and supply squeeze, weighing supply-structure vulnerability and spend share (impact) to quickly show where the risk is right now.",
        'ph_reset_btn': 'Start a New Check (Reset)',
        'ph_upload_label': 'Upload the import data file to check',
        'ph_upload_caption': '※ Please upload a TDS raw file with date, importer, supplier, product, volume, and unit price columns.',
        'ph_missing_cols_error': 'Required columns are missing: {cols}',
        'ph_importer_label': 'Importer to check (largest volume first)',
        'ph_importer_auto': 'Importer: {name} (applied automatically — the file has only one importer)',
        'ph_unit_label': 'Check unit',
        'ph_unit_product': 'Product',
        'ph_unit_origin': 'Origin',
        'ph_unit_caption': 'Supplier is excluded as a check unit because it is the basis of the supply structure and supply change checks. Product names that differ only in pack size (360 tablets, 25 KG, etc.) or a trailing product code form one unit, shown under a cleaned name (same rule as Adoption Impact).',
        'ph_base_period': 'Base period (current)',
        'ph_comp_period': 'Comparison period (previous)',
        'ph_period_overlap_warning': 'The base and comparison periods overlap. Overlapping transactions are counted in both.',
        'ph_period_invalid_error': 'A period starts after it ends. Please check the dates.',
        'ph_threshold_expander': '⚙️ Adjust thresholds',
        'ph_th_group_signal': 'Change signals',
        'ph_th_group_other': 'Vulnerability · Impact · Reliability',
        'ph_th_price_red': 'Unit price 🔴 (increase %, at or above)',
        'ph_th_price_yellow': 'Unit price 🟡 (increase %, at or above)',
        'ph_th_churn_major': 'Major supplier (comparison-period share %, at or above)',
        'ph_th_replace_ok': 'Replacement complete (new-supplier volume vs. lost volume %, at or above)',
        'ph_th_squeeze_volume': 'Supply squeeze volume (monthly avg decrease %, at or above — 🔴 together with price 🟡 or higher)',
        'ph_th_conc': 'Concentrated (top supplier share %, at or above)',
        'ph_th_spread': 'Diversified (top supplier share %, below)',
        'ph_th_min_ship': 'Minimum shipments (each period; below this, price is on hold)',
        'ph_th_min_share': 'Impact (spend share %, below this the overall status is capped at Caution)',
        'ph_th_order_warning': 'A 🟡 threshold is above its 🔴 threshold, or the diversified threshold is above the concentrated threshold. Please check the thresholds.',
        'ph_run_btn': 'Run Health Check',
        'ph_spinner': 'Checking...',
        'ph_no_data_warning': 'No data to check for the selected importer and periods.',
        'ph_summary_caption': 'Importer: {importer} · Check unit: {unit} · Base period {b_start} ~ {b_end} · Comparison period {c_start} ~ {c_end}',
        'ph_kpi_total': 'Units checked',
        'ph_focus_header': '🔎 Look at these first',
        'ph_focus_none': 'No units at risk.',
        'ph_table_header': '📋 Check results',
        'ph_filter_label': 'Filter by overall status',
        'ph_col_unit': 'Unit',
        'ph_col_share': 'Spend share',
        'ph_col_structure': 'Supply structure',
        'ph_col_supply': 'Supply change',
        'ph_col_price': 'Unit price change',
        'ph_col_volume': 'Volume change',
        'ph_col_overall': 'Overall',
        'ph_col_reason': 'Reason',
        'ph_status_risk': 'At risk',
        'ph_status_caution': 'Caution',
        'ph_status_safe': 'Safe',
        'ph_status_new': 'New',
        'ph_status_insufficient': 'Insufficient data',
        'ph_new_tag': ' (new)',
        'ph_struct_single': 'Single',
        'ph_struct_conc': 'Concentrated',
        'ph_struct_normal': 'Moderate',
        'ph_struct_spread': 'Diversified',
        'ph_supply_disruption': 'Disruption',
        'ph_supply_switch': 'Switched',
        'ph_supply_switch_better': 'Switched · better price',
        'ph_supply_none': 'None',
        'ph_price_hold': 'On hold',
        'ph_reason_price': 'unit price {chg:+.0f}%',
        'ph_reason_vuln_price': '{struct} supply + unit price {chg:+.0f}%',
        'ph_reason_disruption': 'major supplier lost, {ratio:.0f}% of volume replaced',
        'ph_reason_squeeze': 'supply squeeze (volume {vol:+.0f}%, unit price {chg:+.0f}%)',
        'ph_reason_switch': 'supplier switch ({ratio:.0f}% of volume replaced)',
        'ph_reason_switch_better': 'supplier switch with better price ({ratio:.0f}% replaced, unit price {chg:+.0f}%)',
        'ph_reason_hold': '{n} shipment(s) — unit price on hold',
        'ph_reason_low_share': 'spend share {share:.1f}% — small impact, lowered to Caution',
        'ph_reason_new': 'no transactions in the comparison period',
        'ph_members_expander': '📄 Original product names by unit',
        'ph_col_members': 'Original product names',
        'ph_col_member_count': 'Names',
        'ph_detail_members': 'Original product names',
        'ph_x_members': 'Original product names',
        'ph_table_notes': '※ Spend = volume × volume-weighted average price (VWAP); spend share is relative to total base-period spend. Volume change compares monthly averages and is not used on its own. Supply structure (single/concentrated) only affects the status when combined with a price 🟡.',
        'ph_market_note': '※ A unit price signal may come from market-wide price moves — check whether it is a market factor (see [{season}]).',
        'ph_mix_notice': '※ The check unit is not a product, so each unit mixes several products. A change in product mix can affect the unit price change.',
        'ph_detail_header': '🔬 Details for at-risk and caution units',
        'ph_detail_top5': 'Supplier share (top 5, base period)',
        'ph_detail_lost': 'Lost suppliers (share in comparison period)',
        'ph_detail_new_sup': 'New suppliers (share in base period)',
        'ph_detail_no_lost': 'No suppliers were lost.',
        'ph_detail_no_new': 'No new suppliers.',
        'ph_detail_supplier': 'Supplier',
        'ph_detail_share': 'Share (%)',
        'ph_detail_vwap': 'Average unit price (VWAP, USD/kg)',
        'ph_detail_monthly': 'Monthly average volume (kg)',
        'ph_detail_ship': 'Shipments',
        'ph_detail_replace': 'Replacement rate (new-supplier monthly volume vs. lost major suppliers): {ratio:.0f}%',
        'ph_detail_base': 'Base period',
        'ph_detail_comp': 'Comparison period',
        'ph_detail_more': 'For more detail, see the [{risk}] · [{churn}] · [{season}] menus.',
        'ph_excel_btn': '📥 Download Excel',
        'ph_sheet_result': 'Results',
        'ph_sheet_suppliers': 'Supplier share',
        'ph_x_amount': 'Base-period spend (USD)',
        'ph_x_share': 'Spend share (%)',
        'ph_x_n_sup': 'Suppliers',
        'ph_x_top1': 'Top supplier',
        'ph_x_top1_share': 'Top supplier share (%)',
        'ph_x_lost': 'Lost suppliers',
        'ph_x_new_sup': 'New suppliers',
        'ph_x_replace': 'Replacement rate (%)',
        'ph_x_base_ship': 'Base shipments',
        'ph_x_comp_ship': 'Comparison shipments',
        'ph_x_base_vwap': 'Base VWAP',
        'ph_x_comp_vwap': 'Comparison VWAP',
        'ph_x_price_chg': 'Unit price change (%)',
        'ph_x_base_monthly': 'Base monthly avg volume (kg)',
        'ph_x_comp_monthly': 'Comparison monthly avg volume (kg)',
        'ph_x_volume_chg': 'Volume change (%)',
        'ph_x_base_volume': 'Base period volume (kg)',
        'ph_x_base_share': 'Base period share (%)',
        'ph_x_comp_share': 'Comparison period share (%)',
        'multi_product_label': 'Search products (multi-select)',
        'group_rule_generic_hs': '※ Generic-name products (e.g. [Mixed Milk Powder] Mixed Milk Powder) are separated by 10-digit HS code; detailed specs within the same HS code are not distinguished.',
        'multi_product_help': "Product names that differ only in pack size (25 KG, 360 tablets, etc.) or a trailing product code are combined into one product (same rule as Adoption Impact). Selecting several products combines them into one analysis.",
        'p5_month_label': 'Month {m}',
        'p5_no_prev_year_caption': '※ No prior-year data available to overlay (needs at least 13 months of history).',
        'p5_pack_marker': 'Pack mix change: {b} {bs:.0f}% → {a} {as_:.0f}%',
        'p5_pack_marker_legend': 'Pack mix change',
        'p5_pack_toggle': 'Show by pack size',
        'p5_pack_note': '◆ marks months where the main pack size changed or a pack size share moved 20%p or more vs. the previous month (trades without a pack size are excluded from this check). The combined price line may have moved because of the pack mix.',
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

PACK_UNIT_RE = re.compile(r'(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*(정|포|캡슐|개입|입|개|매|봉|병|캔|팩|EA|PCS|KG|ML|T|G|L)(?![A-Za-z가-힣])', re.I)
# 끝의 숫자 코드: 영문 1~4자 + 숫자 3자리 이상(ES 2677, A 65200) 또는 숫자만 4자리 이상. 숫자만 3자리(밸런스 위드인 365)는 제품명으로 둔다
_TRAIL_NUM_CODE_RE = re.compile(r'\s+(?:[A-Z]{1,4}\s*\d{3,}|\d{4,})$')
_TRAIL_ALNUM_CODE_RE = re.compile(r'\s+(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]+$')


def normalize_product_name(name):
    """① 품목군 묶기용 품목명 정제 (정제 규칙은 이 함수 하나에서만 관리).
    반환: (정제된 이름, 포장 단위). 정제된 이름이 같은 거래끼리 같은 품목군이 된다.
    - [분류] 대괄호와 '__' 뒤 영문명 제거
    - 숫자+단위(360정·30포·90캡슐·900G·500ML 등)는 포장 단위로 떼어내 따로 돌려준다
    - 끝에 붙은 제품 코드(PEF225, ES 2677, 19A) 제거
    - 단어 경계(공백)는 남기고 대문자로 통일 — 맛·브랜드·대상이 다른 이름은 그대로 다른 품목군"""
    if not isinstance(name, str):
        return '', ''
    text = re.sub(r'\[.*?\]', '', name).split('_')[0]
    packs = [f"{float(num):g}{unit.upper() if unit.isascii() else unit}" for num, unit in PACK_UNIT_RE.findall(text)]
    text = PACK_UNIT_RE.sub(' ', text)
    text = re.sub(r'[^A-Za-z0-9가-힣]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().upper()
    while True:
        trimmed = _TRAIL_ALNUM_CODE_RE.sub('', _TRAIL_NUM_CODE_RE.sub('', text)).strip()
        if trimmed == text or not trimmed:
            break
        text = trimmed
    return text, ' '.join(packs)


def _split_pack(label):
    """포장 단위 라벨 → (숫자들, 단위 이름). 예: '360정' → (('360',), '정'), '900G 30포' → (('900', '30'), 'G 포')."""
    nums = tuple(re.findall(r'\d+(?:\.\d+)?', label))
    units = ' '.join(re.sub(r'[\d.]+', ' ', label).split())
    return nums, units


def pack_mix_change(before_g, after_g, shift_pp=20.0):
    """같은 품목군의 포장 단위별 물량 비중을 계약 전후로 비교한다.
    주력 포장 단위가 바뀌거나 어떤 포장 단위의 비중이 shift_pp(%p) 이상 움직이면 구성 변화.
    품목명에 포장 표기가 없는 거래는 비교에서 뺀다 — 계약 전에는 용량을 안 적다가 이후 적기 시작한 것처럼
    표기 방식만 바뀐 경우를 포장 변경으로 오인하지 않기 위해서다. (이 제외는 경고 판정에만 적용되고,
    절감액·VWAP 계산에는 포장 표기가 없는 거래도 그대로 들어간다.)
    주력 포장 단위의 개수·용량은 같고 단위 이름만 다르면(360정 → 360캡슐) 종류를 'form'(제형 변경)으로 구분한다.
    반환: (경고 여부, 종류 'main'/'form'/'share'/'', 포장 단위 또는 전 주력, 후 주력 또는 '', 전 비중, 후 비중)"""
    def shares(g):
        g = g[g['pack_unit'] != '']
        v = g.groupby('pack_unit')['volume'].sum()
        return v / v.sum() * 100 if v.sum() > 0 else v
    b, a = shares(before_g), shares(after_g)
    if b.empty or a.empty:
        return False, '', '', '', np.nan, np.nan
    top_b, top_a = b.idxmax(), a.idxmax()
    if top_b != top_a:
        (nums_b, units_b), (nums_a, units_a) = _split_pack(top_b), _split_pack(top_a)
        if nums_b and nums_b == nums_a and units_b != units_a:
            return True, 'form', units_b, units_a, float(b[top_b]), float(a[top_a])
        return True, 'main', top_b, top_a, float(b[top_b]), float(a[top_a])
    labels = b.index.union(a.index)
    diff = a.reindex(labels, fill_value=0) - b.reindex(labels, fill_value=0)
    big = diff.abs().idxmax()
    if abs(diff[big]) >= shift_pp:
        return True, 'share', big, '', float(b.get(big, 0)), float(a.get(big, 0))
    return False, '', '', '', np.nan, np.nan


def pack_change_text(kind, frm, to, b, a):
    """포장 단위 변화 설명 문구 (화면·PDF 공용)."""
    label = lambda p: p if p else T('p1_pack_none')
    if kind == 'main':
        return T('p1_pack_change_main', b=label(frm), a=label(to))
    if kind == 'form':
        return T('p1_pack_change_form', b=frm, a=to)
    if kind == 'share':
        return T('p1_pack_change_share', pack=label(frm), b=b, a=a)
    return ''


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
def weighted_avg(df, value_col, weight_col='volume'):
    """물량가중평균(VWAP). 거래건수가 아니라 실제 물량 기준으로 가중해야
    '소량 거래 1건'이 '대량 거래 1건'과 똑같은 비중으로 평균을 왜곡하지 않는다.
    weight_col 합이 0/NaN이면 NaN을 반환한다."""
    if df.empty:
        return np.nan
    w = df[weight_col]
    tw = w.sum()
    if tw == 0 or pd.isna(tw):
        return np.nan
    return (df[value_col] * w).sum() / tw


def weighted_avg_groupby(df, group_cols, value_col, weight_col='volume', out_name=None):
    """group_cols 기준으로 물량가중평균을 계산해 Series(또는 group_cols가 여러 개면
    MultiIndex Series)로 반환하는 공통 헬퍼. groupby().apply()를 중첩 사용하지 않아
    최신 pandas에서 그룹 기준 컬럼이 사라지는 문제를 피한다."""
    name = out_name or value_col
    if df.empty:
        # 빈 데이터프레임에 groupby().apply()를 하면 pandas가 Series가 아니라
        # 빈 DataFrame을 반환해서 이후 .rename(문자열)이 깨짐 — 미리 빈 Series로 처리.
        return pd.Series(dtype=float, name=name)
    result = df.groupby(group_cols).apply(lambda g: weighted_avg(g, value_col, weight_col))
    if isinstance(result, pd.DataFrame):
        # 일부 pandas 버전/엣지케이스(그룹이 1개뿐이거나 특이한 조합)에서
        # Series 대신 DataFrame이 나올 수 있어 명시적으로 Series로 강제 변환.
        result = result.iloc[:, 0] if result.shape[1] >= 1 else pd.Series(dtype=float)
    result.name = name
    return result


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

P1_SAVINGS_GROUP_LABEL_KEY = {'saved': 'p1_grp_saved_label', 'increased': 'p1_grp_increased_label',
                              'minor': 'p1_grp_minor_label', 'no_purchase': 'p1_grp_no_purchase_label'}
P1_CARDS_PER_GROUP = 12


def split_savings_groups(savings_df, threshold):
    """도입 효과 분석 절감액을 그룹으로 나눈다.
    - saved: 절감액 > threshold (절감액 큰 순)
    - increased: 절감액 < -threshold, 즉 단가 상승으로 비용이 늘어난 품목 (증가액 큰 순)
    - minor: |절감액| <= threshold (카드에 $0으로 보이던 품목은 대부분 여기)
    - no_purchase: 계약 후 물량 0 — 절감·증가 합계에서 제외"""
    bought = savings_df['volume_after'] > 0
    valid, no_purchase = savings_df[bought], savings_df[~bought]
    return {
        'saved': valid[valid['savings'] > threshold].sort_values('savings', ascending=False),
        'increased': valid[valid['savings'] < -threshold].sort_values('savings'),
        'minor': valid[valid['savings'].abs() <= threshold].sort_values('savings', ascending=False),
        'no_purchase': no_purchase,
    }


def savings_table_with_group(groups):
    """상세 절감 내역 표: '구분' 열을 붙여 절감 → 단가 상승 → 변화 미미 → 계약 후 구매 없음 순으로 합친다."""
    parts = []
    for key in ('saved', 'increased', 'minor', 'no_purchase'):
        g = groups[key].copy()
        g.insert(0, 'group', T(P1_SAVINGS_GROUP_LABEL_KEY[key]))
        parts.append(g)
    return pd.concat(parts)


def _toggle_session_flag(key):
    st.session_state[key] = not st.session_state.get(key, False)


def render_savings_cards(group_df, group_key):
    """기존 품목군별 절감 카드 디자인 그대로 그린다. 처음에는 상위 P1_CARDS_PER_GROUP개만, '전체 보기'로 나머지를 펼친다."""
    if group_df.empty:
        st.caption(T('p1_group_empty'))
        return
    state_key = f'p1_show_all_{group_key}'
    show_all = st.session_state.get(state_key, False)
    visible = group_df if show_all else group_df.head(P1_CARDS_PER_GROUP)
    cols = st.columns(4)
    for i, row in enumerate(visible.itertuples()):
        col = cols[i % 4]
        color, arrow, val = ("blue", "▼", row.savings) if row.savings >= 0 else ("red", "▲", -row.savings)
        mix_txt = pack_change_text(row.mix_kind, row.mix_from, row.mix_to, row.mix_b, row.mix_a) if row.mix_changed else ''
        warn_html = f"""<div style="font-size:11px;color:#b3261e;margin-top:4px;" title="{mix_txt} — {T('p1_mix_changed_tooltip')}">⚠ {T('p1_mix_changed_badge')}</div>""" if row.mix_changed else ""
        col.markdown(f"""<div style="border: 1px solid #e6e6e6; border-radius: 0.5rem; padding: 1rem; text-align: center; height: 120px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 1rem;"><strong>{row.Index}</strong><p style="font-size: 1.5rem; font-weight: bold; color: {color}; margin-top: 8px; margin-bottom: 0;">{arrow} ${val:,.0f}</p>{warn_html}</div>""", unsafe_allow_html=True)
    hidden = len(group_df) - P1_CARDS_PER_GROUP
    if hidden > 0:
        label = T('p1_show_less_btn') if show_all else T('p1_show_all_btn', n=hidden)
        st.button(label, key=f'{state_key}_btn', on_click=_toggle_session_flag, args=(state_key,))


def reset_analysis_states():
    """모든 분석 상태를 초기화하는 함수"""
    st.session_state.analysis_done = False
    st.session_state.market_analysis_done = False
    keys_to_reset = ['customer_name', 'plot_df', 'customer_df', 'contract_date', 'p1_price_mismatch', 
                     'tfidf_matrix', 'savings_df', 'total_savings', 'market_df',
                     'analyzed_product_name', 'p1_minor_threshold',
                     'p1_show_all_saved', 'p1_show_all_increased', 'p1_show_all_minor']
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


def reset_health_states():
    """포트폴리오 헬스체크 상태만 초기화하는 함수"""
    st.session_state.health_raw_df = None
    st.session_state.health_headers = None
    st.session_state.health_result = None


def reset_screener_states():
    """범용상품 스크리너 상태만 초기화하는 함수"""
    st.session_state.scr_raw_df = None
    st.session_state.scr_headers = None
    st.session_state.scr_result = None


def reset_bid_states():
    """입찰 대응 분석 상태만 초기화하는 함수"""
    st.session_state.bid_raw_df = None
    st.session_state.bid_headers = None
    st.session_state.bid_hs_preset = None


_HS_COL_RE = re.compile(r'(?<![a-z])hs(?![a-z])')
_CODE_COL_RE = re.compile(r'(?<![a-z])hs(?![a-z])|code|코드')


def is_code_like_column(name):
    """컬럼명이 HS코드 등 '코드' 컬럼인지 판별 (예: 'HS Code', 'Detailed HS-CODE', 'HS코드').
    이런 컬럼은 0303890000처럼 숫자로만 되어 있어도 식별자이므로 문자열로 다뤄야 앞자리 0이 보존된다.
    'hs'는 단어 단위로만 인식해서 'Months' 같은 컬럼명은 제외한다."""
    return isinstance(name, str) and bool(_CODE_COL_RE.search(name.lower()))


def is_hs_column(name):
    """코드 컬럼 중 컬럼명에 'hs'가 단어로 들어간 HS코드 컬럼인지 판별 (앞자리 0 복원 대상)."""
    return isinstance(name, str) and bool(_HS_COL_RE.search(name.lower()))


def _pad_hs_code(value):
    """HS코드는 항상 짝수 자리(6/8/10자리)라서, 엑셀 숫자 셀 등으로 앞자리 0이 사라져 홀수 자리가 된 값을 복원한다.
    예: '303890000' → '0303890000', '303890000.0' → '0303890000'. 숫자가 아닌 값(HS Code Name 등)은 그대로."""
    if not isinstance(value, str):
        return value
    code = re.sub(r'\.0+$', '', value.strip())
    if code.isdigit() and len(code) % 2 == 1:
        code = '0' + code
    return code if code.isdigit() else value


def read_uploaded_table(uploaded_file):
    """CSV(인코딩 자동 판별)/XLSX 파일을 읽어 DataFrame으로 반환하는 공통 헬퍼.
    코드 컬럼(is_code_like_column)은 문자열로 읽어 앞자리 0을 보존하고,
    HS코드 컬럼(is_hs_column)은 이미 사라진 앞자리 0까지 복원한다 (_pad_hs_code).
    실패 시 None을 반환한다 (호출부에서 오류 메시지 처리)."""
    if uploaded_file is None:
        return None
    df = None
    if uploaded_file.name.endswith('.csv'):
        for enc in ('utf-8', 'euc-kr', 'cp949'):
            try:
                uploaded_file.seek(0)
                header = pd.read_csv(uploaded_file, encoding=enc, nrows=0).columns
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc, dtype={c: str for c in header if is_code_like_column(c)})
                break
            except UnicodeDecodeError:
                continue
    elif uploaded_file.name.endswith('.xlsx'):
        uploaded_file.seek(0)
        header = pd.read_excel(uploaded_file, nrows=0).columns
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, dtype={c: str for c in header if is_code_like_column(c)})
    if df is not None:
        for c in df.columns:
            if is_hs_column(c):
                df[c] = df[c].map(_pad_hs_code)
    return df


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
        # pandas 3부터 문자열 컬럼이 object가 아니라 StringDtype으로 읽히므로 둘 다 허용.
        # HS코드처럼 숫자로만 된 코드 컬럼은 숫자형이어도 축 후보에 포함한다.
        if is_code_like_column(c) or pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
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
# 포트폴리오 헬스체크 — 단위(품목/원산지 등)별 변화 신호·취약성·영향도·신뢰도 점검
# ============================================================
HEALTH_DEFAULT_THRESHOLDS = {'price_red': 15, 'price_yellow': 8, 'churn_major': 10, 'replace_ok': 50,
                             'squeeze_volume': 15, 'conc': 70, 'spread': 50, 'min_ship': 3, 'min_share': 1.0}
HEALTH_OVERALL_ORDER = {'risk': 0, 'caution': 1, 'safe': 2, 'new': 3, 'insufficient': 4}
HEALTH_OVERALL_KEY = {'risk': 'ph_status_risk', 'caution': 'ph_status_caution', 'safe': 'ph_status_safe',
                      'new': 'ph_status_new', 'insufficient': 'ph_status_insufficient'}
HEALTH_OVERALL_EMOJI = {'risk': '🔴', 'caution': '🟡', 'safe': '🟢', 'new': '⚪', 'insufficient': '⏸️'}
HEALTH_LEVEL_EMOJI = {'red': '🔴', 'yellow': '🟡', 'green': '🟢'}
HEALTH_STRUCT_KEY = {'single': 'ph_struct_single', 'conc': 'ph_struct_conc', 'normal': 'ph_struct_normal', 'spread': 'ph_struct_spread'}
HEALTH_SUPPLY_KEY = {'disruption': 'ph_supply_disruption', 'switch': 'ph_supply_switch',
                     'switch_better': 'ph_supply_switch_better', 'none': 'ph_supply_none'}
# 사유 중 판정 근거가 되는 신호 (먼저 볼 것 문장에 사용). 나머지(전환·판단 보류 등)는 참고 정보.
HEALTH_SIGNAL_REASONS = ('ph_reason_disruption', 'ph_reason_squeeze', 'ph_reason_price', 'ph_reason_vuln_price')


def prepare_health_df(raw_df, cols, unit_col, product_mode):
    """헬스체크용 표준 데이터프레임(_date/_volume/_price/_exporter/_importer/_unit)을 만든다.
    품목 단위면 product_group_labels(①⑥과 같은 기준)로 표기가 조금씩 다른 품목명을 대표명으로 묶는다."""
    imp = raw_df[cols['importer']]
    df = pd.DataFrame({
        '_date': pd.to_datetime(raw_df[cols['date']], errors='coerce'),
        '_volume': pd.to_numeric(raw_df[cols['volume']], errors='coerce'),
        '_price': derive_unit_price(raw_df, cols)[0],  # Unit Price 우선, 없으면 금액 ÷ 중량 (①⑨⑩ 공통)
        '_exporter': raw_df[cols['exporter']].fillna('(N/A)').astype(str).str.strip(),
        '_importer': imp.where(imp.isna(), imp.astype(str).str.strip()),
        '_unit': raw_df[unit_col],
    })
    df = df.dropna(subset=['_date', '_volume', '_unit', '_importer'])
    df = df[df['_volume'] > 0]
    df['_unit'] = df['_unit'].astype(str).str.strip()
    df['_raw_name'] = df['_unit']  # 묶기 전 원래 이름 (단위별 원래 품목명 확인용)
    if product_mode:
        hs = raw_df.loc[df.index, cols['hs']] if cols.get('hs') else None
        df['_unit'] = product_group_labels(df['_unit'], hs)[0]
    return df


def _period_months(start, end):
    """기간 길이를 개월 수로 환산 (거래가 없던 달도 포함한 달력 기준)."""
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    return max(days, 1) / 30.44


def compute_health_check(df, base_period, comp_period, thresholds):
    """prepare_health_df 결과(수입사 필터 후)로 단위별 판정을 계산한다.
    - 변화 신호: 단가(VWAP) 변화, 공급 차질(주요 공급사 이탈 + 대체 미흡), 수급 경색(물량 감소 + 단가 상승)
    - 취약성: 공급 구조(단일/집중/보통/분산). 단일·집중 + 단가 🟡이면 위험으로 격상
    - 영향도: 금액(물량 × VWAP) 비중이 기준 미만이면 종합 판정 최대 '주의'
    - 신뢰도: 두 기간 중 하나라도 선적이 기준 미만이면 단가 신호는 판단 보류 (공급 차질 판정은 유지)
    사유는 화면 언어가 바뀌어도 다시 그릴 수 있도록 (TEXTS 키, 인자) 형태로 저장한다."""
    th = thresholds
    day = df['_date'].dt.normalize()

    def _in(period):
        return df[(day >= pd.Timestamp(period[0])) & (day <= pd.Timestamp(period[1]))]

    base_df, comp_df = _in(base_period), _in(comp_period)
    base_months, comp_months = _period_months(*base_period), _period_months(*comp_period)
    comp_groups = dict(tuple(comp_df.groupby('_unit')))
    rows = []
    for unit, b in base_df.groupby('_unit'):
        c = comp_groups.get(unit, comp_df.iloc[0:0])
        b_vol, c_vol = float(b['_volume'].sum()), float(c['_volume'].sum())
        bp = weighted_avg(b[b['_price'] > 0], '_price', '_volume')
        cp = weighted_avg(c[c['_price'] > 0], '_price', '_volume')
        base_vwap = None if pd.isna(bp) else float(bp)
        comp_vwap = None if pd.isna(cp) else float(cp)

        sup_b = b.groupby('_exporter')['_volume'].sum().sort_values(ascending=False)
        share_b = sup_b / b_vol * 100
        top1_share = float(share_b.iloc[0])
        if len(sup_b) == 1:
            structure = 'single'
        elif top1_share >= th['conc']:
            structure = 'conc'
        elif top1_share < th['spread']:
            structure = 'spread'
        else:
            structure = 'normal'

        members = sorted(set(b['_raw_name']) | set(c['_raw_name'])) if '_raw_name' in b.columns else [unit]
        row = {
            'unit': unit, 'members': members, 'is_new': c_vol <= 0, 'base_volume': b_vol,
            'amount': b_vol * base_vwap if base_vwap is not None else 0.0, 'share': 0.0,
            'n_suppliers': len(sup_b), 'top1': sup_b.index[0], 'top1_share': top1_share, 'structure': structure,
            'top5': [(s, float(v)) for s, v in share_b.head(5).items()],
            'base_ship': len(b), 'comp_ship': len(c),
            'base_vwap': base_vwap, 'comp_vwap': comp_vwap, 'price_chg': None, 'price': None,
            'base_monthly': b_vol / base_months, 'comp_monthly': c_vol / comp_months, 'volume_chg': None,
            'supply': None, 'replace_ratio': None, 'lost': [], 'new_suppliers': [], 'squeeze': False,
            'reasons': [], 'downgraded': False,
        }

        sup_c = pd.Series(dtype=float)
        if row['is_new']:
            row['overall'] = 'new'
            row['reasons'].append(('ph_reason_new', {}))
        else:
            sup_c = c.groupby('_exporter')['_volume'].sum()
            share_c = sup_c / c_vol * 100
            lost = share_c[~share_c.index.isin(sup_b.index)].sort_values(ascending=False)
            new_sup = sup_b[~sup_b.index.isin(sup_c.index)]
            row['lost'] = [(s, float(v)) for s, v in lost.items()]
            row['new_suppliers'] = [(s, float(share_b[s])) for s in new_sup.index]

            # 단가: 선적이 적으면 판단 보류, 하락은 green
            if base_vwap is not None and comp_vwap:
                row['price_chg'] = (base_vwap / comp_vwap - 1) * 100
            if min(row['base_ship'], row['comp_ship']) < th['min_ship']:
                row['price'] = 'hold'
            elif row['price_chg'] is None:
                row['price'] = 'na'
            else:
                chg = row['price_chg']
                row['price'] = 'red' if chg >= th['price_red'] else 'yellow' if chg >= th['price_yellow'] else 'green'
            price_ok = row['price'] in ('red', 'yellow', 'green')

            # 물량: 월평균 비교 (단독으로는 판정에 쓰지 않음)
            row['volume_chg'] = (row['base_monthly'] / row['comp_monthly'] - 1) * 100
            row['squeeze'] = price_ok and row['volume_chg'] <= -th['squeeze_volume'] and row['price_chg'] >= th['price_yellow']

            # 공급 변화: 주요 공급사 이탈 시 신규 공급사가 빠진 물량을 얼마나 채웠는지 (월평균 기준)
            major_lost = lost[lost >= th['churn_major']]
            if len(major_lost):
                lost_monthly = float(sup_c[major_lost.index].sum()) / comp_months
                new_monthly = float(new_sup.sum()) / base_months
                row['replace_ratio'] = new_monthly / lost_monthly * 100 if lost_monthly > 0 else 0.0
                if row['replace_ratio'] < th['replace_ok']:
                    row['supply'] = 'disruption'
                elif price_ok and row['price_chg'] < 0:
                    row['supply'] = 'switch_better'
                else:
                    row['supply'] = 'switch'
            else:
                row['supply'] = 'none'

            red = yellow = False
            reasons = row['reasons']
            if row['supply'] == 'disruption':
                red = True
                reasons.append(('ph_reason_disruption', {'ratio': row['replace_ratio']}))
            if row['squeeze']:
                red = True
                reasons.append(('ph_reason_squeeze', {'vol': row['volume_chg'], 'chg': row['price_chg']}))
            elif row['price'] == 'red':
                red = True
                reasons.append(('ph_reason_price', {'chg': row['price_chg']}))
            elif row['price'] == 'yellow':
                if structure in ('single', 'conc'):
                    red = True
                    reasons.append(('ph_reason_vuln_price', {'structure': structure, 'top1': top1_share, 'chg': row['price_chg']}))
                else:
                    yellow = True
                    reasons.append(('ph_reason_price', {'chg': row['price_chg']}))
            if row['supply'] in ('switch', 'switch_better'):
                key = 'ph_reason_switch_better' if row['supply'] == 'switch_better' else 'ph_reason_switch'
                reasons.append((key, {'ratio': row['replace_ratio'], 'chg': row['price_chg']}))
            if row['price'] == 'hold':
                reasons.append(('ph_reason_hold', {'n': min(row['base_ship'], row['comp_ship'])}))

            if red:
                row['overall'] = 'risk'
            elif yellow:
                row['overall'] = 'caution'
            elif row['price'] == 'hold' and row['supply'] == 'none':
                row['overall'] = 'insufficient'
            else:
                row['overall'] = 'safe'

        all_sup = sorted(set(sup_b.index) | set(sup_c.index), key=lambda s: (-sup_b.get(s, 0), -sup_c.get(s, 0)))
        c_total = float(sup_c.sum()) or 1.0
        row['suppliers'] = [(s, float(sup_b.get(s, 0)), float(share_b.get(s, 0)), float(sup_c.get(s, 0)) / c_total * 100) for s in all_sup]
        rows.append(row)

    # 영향도: 기준 기간 전체 금액 대비 비중. 작은 단위는 위험 → 주의로 내림
    total_amount = sum(r['amount'] for r in rows)
    for r in rows:
        r['share'] = r['amount'] / total_amount * 100 if total_amount > 0 else 0.0
        if r['overall'] == 'risk' and r['share'] < th['min_share']:
            r['overall'] = 'caution'
            r['downgraded'] = True
            r['reasons'].append(('ph_reason_low_share', {'share': r['share']}))

    rows.sort(key=lambda r: (HEALTH_OVERALL_ORDER[r['overall']], -r['share']))
    return {'rows': rows}


def _health_reason_text(key, kwargs):
    """저장된 (TEXTS 키, 인자) 사유를 현재 언어 문장으로 만든다."""
    if key == 'ph_reason_vuln_price':
        struct = T('ph_struct_single') if kwargs['structure'] == 'single' else f"{T('ph_struct_conc')}({kwargs['top1']:.0f}%)"
        return T(key, struct=struct, chg=kwargs['chg'])
    return T(key, **kwargs)


def health_reasons(row, signals_only=False):
    """사유 문장 목록. signals_only=True면 판정 근거가 된 신호만."""
    return [_health_reason_text(k, kw) for k, kw in row['reasons'] if not signals_only or k in HEALTH_SIGNAL_REASONS]


def health_display_table(rows, plain=False):
    """결과 표: 단위 | 금액 비중 | 공급 구조 | 공급 변화 | 단가 변화 | 물량 변화 | 종합 판정 | 사유
    plain=True면 이모지 없이 (PDF용)."""
    def mark(emoji, text):
        return text if plain else f"{emoji} {text}"

    out = []
    for r in rows:
        if r['is_new']:
            supply_txt = price_txt = volume_txt = '-'
        else:
            supply_txt = T(HEALTH_SUPPLY_KEY[r['supply']])
            if r['supply'] == 'disruption':
                supply_txt = mark('🔴', supply_txt)
            if r['price'] == 'hold':
                price_txt = mark('⏸️', T('ph_price_hold'))
            elif r['price'] == 'na':
                price_txt = '-'
            else:
                price_txt = mark(HEALTH_LEVEL_EMOJI[r['price']], f"{r['price_chg']:+.1f}%")
            volume_txt = f"{r['volume_chg']:+.1f}%"
        overall = T(HEALTH_OVERALL_KEY[r['overall']])
        out.append({
            T('ph_col_unit'): r['unit'] + (T('ph_new_tag') if r['is_new'] else ''),
            T('ph_col_share'): f"{r['share']:.1f}%",
            T('ph_col_structure'): f"{T(HEALTH_STRUCT_KEY[r['structure']])} ({r['top1_share']:.0f}%)",
            T('ph_col_supply'): supply_txt,
            T('ph_col_price'): price_txt,
            T('ph_col_volume'): volume_txt,
            T('ph_col_overall'): mark(HEALTH_OVERALL_EMOJI[r['overall']], overall),
            T('ph_col_reason'): ' / '.join(health_reasons(r)),
        })
    return pd.DataFrame(out)


def health_focus_lines(rows, n=3):
    """'먼저 볼 것': 위험 단위 중 금액 비중 상위 n개를 (단위명, 사유 문장)으로 반환."""
    risk_rows = sorted([r for r in rows if r['overall'] == 'risk'], key=lambda r: -r['share'])[:n]
    return [(r['unit'], ' / '.join(health_reasons(r, signals_only=True))) for r in risk_rows]


def build_health_excel(rows):
    """점검 결과 시트(표시용 칸 + 수치) + 공급사 비중 시트를 담은 xlsx 바이트를 반환."""
    import io
    _r = lambda v, n: None if v is None else round(v, n)
    numeric = pd.DataFrame([{
        T('ph_x_amount'): round(r['amount'], 0),
        T('ph_x_share'): round(r['share'], 2),
        T('ph_x_n_sup'): r['n_suppliers'],
        T('ph_x_top1'): r['top1'],
        T('ph_x_top1_share'): round(r['top1_share'], 1),
        T('ph_x_lost'): ', '.join(f"{s} ({sh:.1f}%)" for s, sh in r['lost']),
        T('ph_x_new_sup'): ', '.join(f"{s} ({sh:.1f}%)" for s, sh in r['new_suppliers']),
        T('ph_x_replace'): _r(r['replace_ratio'], 1),
        T('ph_x_base_ship'): r['base_ship'],
        T('ph_x_comp_ship'): r['comp_ship'],
        T('ph_x_base_vwap'): r['base_vwap'],
        T('ph_x_comp_vwap'): r['comp_vwap'],
        T('ph_x_price_chg'): _r(r['price_chg'], 1),
        T('ph_x_base_monthly'): round(r['base_monthly'], 0),
        T('ph_x_comp_monthly'): round(r['comp_monthly'], 0),
        T('ph_x_volume_chg'): _r(r['volume_chg'], 1),
        T('ph_x_members'): ' / '.join(r.get('members', [])),
    } for r in rows])
    summary = pd.concat([health_display_table(rows), numeric], axis=1)
    suppliers = pd.DataFrame([{
        T('ph_col_unit'): r['unit'],
        T('ph_detail_supplier'): s,
        T('ph_x_base_volume'): vol,
        T('ph_x_base_share'): round(sh_b, 1),
        T('ph_x_comp_share'): round(sh_c, 1),
    } for r in rows for s, vol, sh_b, sh_c in r['suppliers']])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name=T('ph_sheet_result')[:31], index=False)
        suppliers.to_excel(writer, sheet_name=T('ph_sheet_suppliers')[:31], index=False)
    return buf.getvalue()


# ============================================================
# 공통: 블라인드 처리 (한국 수입사명 → 수입사 A/B/C, 물량 순)
# 화면·PDF·CSV로 나가는 수입사명은 모두 apply_blind를 거친다. 해외 공급사명은 그대로 둔다.
# ============================================================
def _blind_code(i):
    """0 → A, 25 → Z, 26 → AA ..."""
    code, i = '', i + 1
    while i:
        i, r = divmod(i - 1, 26)
        code = chr(65 + r) + code
    return code


def blind_importer_map(df, importer_col, volume_col):
    """수입사명 → '수입사 A' 매핑 (물량 큰 순으로 A, B, C ...)."""
    vol = df.groupby(importer_col)[volume_col].sum().sort_values(ascending=False)
    return {name: T('blind_label', code=_blind_code(i)) for i, name in enumerate(vol.index)}


def apply_blind(df, importer_col, enabled, volume_col=None, mapping=None):
    """블라인드가 켜져 있으면 importer_col의 값을 매핑으로 치환한 복사본을 반환한다.
    importer_col이 '이름1, 이름2' 같은 목록 문자열이어도 각각 치환한다."""
    if not enabled or importer_col not in df.columns:
        return df
    if mapping is None:
        mapping = blind_importer_map(df, importer_col, volume_col)
    out = df.copy()
    out[importer_col] = out[importer_col].map(
        lambda v: ', '.join(mapping.get(x.strip(), x.strip()) for x in v.split(',')) if isinstance(v, str) else v)
    return out


# ============================================================
# 공통: 거래 데이터 전처리 (⑨ 범용상품 스크리너, ⑩ 입찰 대응 분석)
# ============================================================
TRADE_HS_CANDIDATES = ['Detailed HS-CODE', 'Detailed HS Code', 'HS Code', 'HS-CODE', 'hs_code', 'HS코드']


def detect_trade_columns(headers):
    """표준 컬럼 + HS코드 + 금액 컬럼."""
    cols = detect_standard_columns(headers)
    cols['hs'] = find_column(headers, TRADE_HS_CANDIDATES)
    cols['value'] = find_column(headers, ['Value', 'Total Value', 'Amount', 'value', '금액'])
    return cols


def _clean_text(series):
    return series.where(series.isna(), series.astype(str).str.strip()).fillna('(N/A)')


def derive_unit_price(raw_df, cols, tol=0.05):
    """단가(USD/kg) 공통 계산 (①⑨⑩).
    Unit Price가 있으면 그 값을 쓰고, 없거나 빈 행은 금액(Value) ÷ 중량으로 채운다.
    둘 다 있으면 금액÷중량과 tol(5%)보다 크게 다른 행의 비율도 돌려준다.
    반환: (단가 Series, 금액 Series 또는 None, 불일치 비율 또는 None)"""
    vol = pd.to_numeric(raw_df[cols['volume']], errors='coerce')
    price = pd.to_numeric(raw_df[cols['price']], errors='coerce') if cols.get('price') else pd.Series(np.nan, index=raw_df.index)
    value = pd.to_numeric(raw_df[cols['value']], errors='coerce') if cols.get('value') else None
    derived = (value / vol).where(vol > 0) if value is not None else pd.Series(np.nan, index=raw_df.index)
    mismatch = None
    if cols.get('price') and value is not None:
        both = (price > 0) & (derived > 0)
        mismatch = float(((price[both] / derived[both] - 1).abs() > tol).mean()) if both.any() else 0.0
    return price.where(price.notna(), derived), value, mismatch


def prepare_trade_df(raw_df, cols, period=None, min_shipment_kg=1000, trim=(0.01, 0.99)):
    """⑨⑩ 공통 전처리. (df, stats)를 반환한다.
    1) 날짜·중량·단가가 유효한 행만 (단가가 없으면 금액/중량으로 계산)
    2) 중복 제거: 신고일·수입자·공급자·중량·금액이 모두 같은 행
    3) 기간 필터 → 건당 최소 중량 미만 제외 → HS별 단가 P1~P99 밖 제외"""
    n0 = len(raw_df)
    vol = pd.to_numeric(raw_df[cols['volume']], errors='coerce')
    price, value, _ = derive_unit_price(raw_df, cols)
    if value is None:
        value = price * vol
    d = pd.DataFrame({
        '_date': pd.to_datetime(raw_df[cols['date']], errors='coerce'),
        '_importer': _clean_text(raw_df[cols['importer']]) if cols.get('importer') else '(N/A)',
        '_exporter': _clean_text(raw_df[cols['exporter']]) if cols.get('exporter') else '(N/A)',
        '_origin': _clean_text(raw_df[cols['origin']]) if cols.get('origin') else '(N/A)',
        '_product': _clean_text(raw_df[cols['product']]) if cols.get('product') else '',
        '_hs': _clean_text(raw_df[cols['hs']]) if cols.get('hs') else '(N/A)',
        '_volume': vol, '_price': price, '_amount': value.where(value.notna(), price * vol),
    })
    d = d[d['_date'].notna() & (d['_volume'] > 0) & (d['_price'] > 0)]
    d['_date'] = d['_date'].dt.normalize()
    before = len(d)
    d = d.drop_duplicates(subset=['_date', '_importer', '_exporter', '_volume', '_amount'])
    stats = {'n0': n0, 'dup': before - len(d)}
    if period is not None:
        d = d[(d['_date'] >= pd.Timestamp(period[0])) & (d['_date'] <= pd.Timestamp(period[1]))]
    before = len(d)
    d = d[d['_volume'] >= min_shipment_kg]
    stats['small'] = before - len(d)
    before = len(d)
    if len(d):
        lo = d.groupby('_hs')['_price'].transform(lambda p: p.quantile(trim[0]))
        hi = d.groupby('_hs')['_price'].transform(lambda p: p.quantile(trim[1]))
        d = d[(d['_price'] >= lo) & (d['_price'] <= hi)]
    stats['outlier'] = before - len(d)
    stats['n'] = len(d)
    d = d.copy()
    d['_ym'] = d['_date'].dt.to_period('M')
    return d, stats


def default_last_12m(dates):
    """데이터 마지막 날짜 기준 최근 12개월 (시작일, 종료일)."""
    dates = pd.to_datetime(dates, errors='coerce').dropna()
    end = dates.max().date() if len(dates) else datetime.date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(months=12)).date() + datetime.timedelta(days=1)
    return start, end


def _weighted_cv(g):
    """물량가중 단가 변동계수(%)."""
    w, p = g['_volume'], g['_price']
    tw = w.sum()
    if len(g) < 2 or tw <= 0:
        return np.nan
    m = (p * w).sum() / tw
    return float(np.sqrt(((p - m) ** 2 * w).sum() / tw) / m * 100) if m > 0 else np.nan


def _gap_rate(g, min_cell=3):
    """같은 달·같은 원산지 셀마다 단가 격차율 (P90−P10)/P50을 구해 셀 물량으로 가중 평균(%)."""
    rates, weights = [], []
    for _, c in g.groupby(['_ym', '_origin']):
        if len(c) < min_cell:
            continue
        p10, p50, p90 = c['_price'].quantile([0.1, 0.5, 0.9])
        if p50 > 0:
            rates.append((p90 - p10) / p50 * 100)
            weights.append(c['_volume'].sum())
    return float(np.average(rates, weights=weights)) if rates else np.nan


# ============================================================
# ⑨ 범용상품 스크리너
# ============================================================
SCREENER_COMPONENTS = [('importers', True), ('cr3', False), ('exporters', True), ('origins', True), ('months', True), ('cv', False)]
SCREENER_DEFAULT_WEIGHTS = {k: 1.0 for k, _ in SCREENER_COMPONENTS}
SCREENER_CLASS_KEY = {'A': 'sc_class_a', 'B': 'sc_class_b', 'C': 'sc_class_c', 'D': 'sc_class_d'}
SCREENER_CLASS_COLOR = {'A': '#171CFD', 'B': '#3E50FE', 'C': '#9BB3FF', 'D': '#0A0A18'}


def compute_screener(df, period_months, weights, bound_hi=60, bound_lo=30, bound_gap=50, min_shipments=5):
    """HS별 지표 → 범용성·격차 점수 → A/B/C/D 분류. 반환: (결과 DataFrame, 순위에서 제외된 HS 수)."""
    years = max(period_months, 1) / 12
    rows, skipped = [], 0
    for hs, g in df.groupby('_hs'):
        if len(g) < min_shipments:
            skipped += 1
            continue
        imp_vol = g.groupby('_importer')['_volume'].sum().sort_values(ascending=False)
        share = imp_vol / imp_vol.sum() * 100
        n_imp = len(imp_vol)
        rows.append({
            'hs': hs,
            'name': g['_product'].mode().iloc[0] if g['_product'].notna().any() else '',
            'importers': n_imp,
            'cr3': float(share.head(3).sum()),
            'hhi': float((share ** 2).sum()),
            'exporters': g['_exporter'].nunique(),
            'origins': g['_origin'].nunique(),
            'months': g['_ym'].nunique(),
            'freq': len(g) / n_imp / years,
            'cv': _weighted_cv(g),
            'gap': _gap_rate(g),
            'value': float(g['_amount'].sum()) / years,
            'shipments': len(g),
            'vwap': float(weighted_avg(g, '_price', '_volume')),
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res, skipped
    num, den = pd.Series(0.0, index=res.index), pd.Series(0.0, index=res.index)
    for key, higher_better in SCREENER_COMPONENTS:
        pct = res[key].rank(pct=True, ascending=higher_better) * 100
        w = float(weights.get(key, 0))
        num += pct.fillna(0) * w
        den += pct.notna() * w
    res['uni'] = (num / den.where(den > 0)).round(1)
    res['uni_rank'] = (res['uni'].rank(pct=True) * 100).round(1)
    res['gap_score'] = (res['gap'].rank(pct=True) * 100).round(1)

    def _cls(r):
        if r['uni_rank'] >= bound_hi:
            return 'A' if pd.notna(r['gap_score']) and r['gap_score'] >= bound_gap else 'C'
        return 'B' if r['uni_rank'] >= bound_lo else 'D'
    res['class'] = res.apply(_cls, axis=1)
    res = res.sort_values(['class', 'uni_rank'], ascending=[True, False]).reset_index(drop=True)
    return res, skipped


def screener_display_table(res):
    """랭킹 표 (화면·CSV 공용)."""
    cols = [('hs', 'sc_col_hs'), ('name', 'sc_col_name'), ('class', 'sc_col_class'), ('uni', 'sc_col_uni'),
            ('uni_rank', 'sc_col_uni_rank'), ('gap_score', 'sc_col_gap_score'), ('importers', 'sc_col_importers'),
            ('cr3', 'sc_col_cr3'), ('hhi', 'sc_col_hhi'), ('exporters', 'sc_col_exporters'), ('origins', 'sc_col_origins'),
            ('months', 'sc_col_months'), ('freq', 'sc_col_freq'), ('cv', 'sc_col_cv'), ('gap', 'sc_col_gap'),
            ('value', 'sc_col_value'), ('shipments', 'sc_col_ship'), ('vwap', 'sc_col_vwap')]
    out = res[[c for c, _ in cols]].copy()
    out['class'] = out['class'].map(lambda c: T(SCREENER_CLASS_KEY[c]))
    for c in ('cr3', 'freq', 'cv', 'gap'):
        out[c] = out[c].round(1)
    out['hhi'] = out['hhi'].round(0)
    out['value'] = out['value'].round(0)
    out['vwap'] = out['vwap'].round(3)
    return out.rename(columns={c: T(k) for c, k in cols})


def screener_quadrant_fig(res, bound_hi, bound_lo, bound_gap):
    """범용성 백분위 × 격차 점수 4분면 산점도 (버블 = 연간 수입액)."""
    fig = go.Figure()
    max_val = max(float(res['value'].max()), 1.0)
    for cls in ['A', 'B', 'C', 'D']:
        sub = res[res['class'] == cls]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub['uni_rank'], y=sub['gap_score'].fillna(0), mode='markers', name=T(SCREENER_CLASS_KEY[cls]),
            marker=dict(size=sub['value'], sizemode='area', sizeref=2.0 * max_val / (48 ** 2), sizemin=5,
                        color=SCREENER_CLASS_COLOR[cls], line=dict(width=0.5, color='#FFFFFF')),
            text=sub['hs'] + ' ' + sub['name'].astype(str),
            hovertemplate='%{text}<br>' + T('sc_axis_uni') + ': %{x:.0f}<br>' + T('sc_axis_gap') + ': %{y:.0f}<extra></extra>',
        ))
    for x in (bound_lo, bound_hi):
        fig.add_vline(x=x, line_width=1, line_dash='dot', line_color='rgba(10,10,24,0.35)')
    fig.add_hline(y=bound_gap, line_width=1, line_dash='dot', line_color='rgba(10,10,24,0.35)')
    fig.update_layout(xaxis_title=T('sc_axis_uni'), yaxis_title=T('sc_axis_gap'), xaxis_range=[-3, 103], yaxis_range=[-3, 103],
                      height=520, legend_title_text='', plot_bgcolor='#FFFFFF')
    fig.update_xaxes(gridcolor='rgba(10,10,24,0.08)')
    fig.update_yaxes(gridcolor='rgba(10,10,24,0.08)')
    return fig


# ============================================================
# ⑩ 입찰 대응 분석
# ============================================================
BID_DEFAULT_VERIFY = {'min_ship': 3, 'min_ton': 10.0, 'min_imp': 2}


def _split_keywords(text):
    return [k.strip().lower() for k in re.split(r'[,\n]+', text or '') if k.strip()]


def bid_filter(df, hs=None, include='', exclude=''):
    """HS + 품명 포함/제외 키워드(스펙 분리) 필터."""
    d = df if not hs else df[df['_hs'] == hs]
    name = d['_product'].astype(str).str.lower()
    inc, exc = _split_keywords(include), _split_keywords(exclude)
    if inc:
        d = d[name.apply(lambda n: any(k in n for k in inc))]
        name = d['_product'].astype(str).str.lower()
    if exc:
        d = d[~name.apply(lambda n: any(k in n for k in exc))]
    return d


def bid_supplier_table(df, by, period_end, verify, tariffs=None):
    """원산지별(by='_origin') 또는 해외 공급사별(by='_exporter') 비교표.
    검증 = 기간 마지막 날 기준 최근 12개월 건수·물량·수입사 수가 모두 기준 이상.
    비교단가 = VWAP × (1 + 원산지 관세율), 관세율이 없으면 VWAP 그대로."""
    if df.empty:
        return pd.DataFrame()
    tariffs = tariffs or {}
    recent = df[df['_date'] > pd.Timestamp(period_end) - pd.DateOffset(months=12)]
    rows = []
    for key, g in df.groupby(by):
        r = recent[recent[by] == key]
        origin = key if by == '_origin' else g['_origin'].mode().iloc[0]
        rate = tariffs.get(origin)
        vwap = float(weighted_avg(g, '_price', '_volume'))
        imp_vol = g.groupby('_importer')['_volume'].sum().sort_values(ascending=False)
        rows.append({
            'key': key, 'origin': origin, 'vwap': vwap,
            'tariff': rate, 'cmp': vwap * (1 + rate / 100) if rate is not None else vwap,
            'shipments': len(g), 'tons': float(g['_volume'].sum()) / 1000, 'months': g['_ym'].nunique(),
            'importers': len(imp_vol), 'importer_names': ', '.join(imp_vol.index), 'last': g['_date'].max().date(),
            'verified': (len(r) >= verify['min_ship']) and (r['_volume'].sum() >= verify['min_ton'] * 1000)
                        and (r['_importer'].nunique() >= verify['min_imp']),
        })
    return pd.DataFrame(rows).sort_values('cmp').reset_index(drop=True)


def bid_add_savings(table, base_key, annual_tons):
    """기준 공급사 대비 절감률(%)·연간 절감액(USD = 단가차 × 연간 물량)."""
    table = table.copy()
    if base_key is None or base_key not in set(table['key']):
        table['saving_pct'] = np.nan
        table['saving_amt'] = np.nan
        return table
    base = float(table.loc[table['key'] == base_key, 'cmp'].iloc[0])
    table['saving_pct'] = (base - table['cmp']) / base * 100 if base > 0 else np.nan
    table['saving_amt'] = (base - table['cmp']) * annual_tons * 1000
    return table


def bid_stability(df, exporter, end_date, months=36):
    """공급사의 최근 N개월 월별 선적·공백 월·연평균 물량·수입사 수·단가 CV·연도별 요약."""
    end = pd.Timestamp(end_date).to_period('M')
    first = max(end - (months - 1), df['_ym'].min()) if len(df) else end - (months - 1)
    g = df[(df['_exporter'] == exporter) & (df['_ym'] >= first) & (df['_ym'] <= end)]
    month_index = pd.period_range(first, end, freq='M')
    monthly = g.groupby('_ym')['_volume'].sum().reindex(month_index, fill_value=0) / 1000
    n_months = len(month_index)
    # 연도별 요약: 기간 마지막 달에서 거꾸로 12개월씩 끊는다 (예: 2023.10 – 2024.09). 맨 앞 구간은 12개월보다 짧을 수 있다.
    blocks = []
    blk_end = end
    while blk_end >= first:
        blk_start = max(blk_end - 11, first)
        y = g[(g['_ym'] >= blk_start) & (g['_ym'] <= blk_end)]
        blocks.append({'label': f"{blk_start.strftime('%Y.%m')} – {blk_end.strftime('%Y.%m')}", 'span': (blk_end - blk_start).n + 1,
                       'shipments': len(y), 'tons': y['_volume'].sum() / 1000,
                       'vwap': weighted_avg(y, '_price', '_volume') if len(y) else np.nan,
                       'months': y['_ym'].nunique(), 'importers': y['_importer'].nunique()})
        blk_end = blk_start - 1
    yearly = pd.DataFrame(blocks[::-1]).set_index('label') if blocks and len(g) else pd.DataFrame()
    return {
        'monthly': monthly, 'n_months': n_months, 'gap_months': int((monthly == 0).sum()),
        'annual_tons': float(monthly.sum()) / (n_months / 12) if n_months else 0.0,
        'importers': int(g['_importer'].nunique()), 'cv': _weighted_cv(g), 'yearly': yearly, 'empty': g.empty,
    }


def bid_stability_sentence(name, stab):
    gap = T('bd_stab_no_gap') if stab['gap_months'] == 0 else T('bd_stab_with_gap', n=stab['gap_months'])
    cv = '-' if pd.isna(stab['cv']) else f"{stab['cv']:.0f}%"
    return T('bd_stab_sentence', name=name, m=stab['n_months'], tons=stab['annual_tons'], gap=gap, imp=stab['importers'], cv=cv)


def bid_stability_fig(stab):
    """월별 선적 막대. 선적 없는 달은 빗금 막대로 표시."""
    months = [str(p) for p in stab['monthly'].index]
    vals = stab['monthly'].values
    top = max(float(vals.max()) if len(vals) else 0, 1.0)
    fig = go.Figure()
    fig.add_bar(x=months, y=vals, name=T('bd_stab_shipped'), marker_color='#171CFD')
    gaps = [top if v == 0 else None for v in vals]
    fig.add_bar(x=months, y=gaps, name=T('bd_stab_gap'), marker=dict(color='rgba(0,0,0,0)', line=dict(color='#9BB3FF', width=1),
                pattern=dict(shape='/', fgcolor='#9BB3FF', size=6)))
    fig.update_layout(barmode='overlay', height=360, plot_bgcolor='#FFFFFF', xaxis_title='', yaxis_title='ton', legend_title_text='')
    fig.update_yaxes(gridcolor='rgba(10,10,24,0.08)')
    return fig


def _badge_html(text, dark):
    bg = '#0A0A18' if dark else '#171CFD'
    return f"<span style='background:{bg};color:#FFFFFF;padding:2px 8px;border-radius:4px;font-size:12px;'>{text}</span>"


# ============================================================
# 새 PDF 양식 (report_theme) — ⑨ 범용상품 스크리너 / ⑩ 입찰 대응 분석
# ============================================================
def _rp_meta(eyebrow, title_lines, item, spec, hs, period, blind, doc_title):
    """make_report에 넘길 meta. 표지·푸터 문구는 현재 언어의 TEXTS에서 가져온다."""
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    months = max(1, round((end - start).days / 30.44))
    return {
        'eyebrow': eyebrow, 'title_lines': title_lines, 'item': item, 'spec': spec, 'hs': hs,
        'period': T('rp_period_fmt', start=start.strftime('%Y.%m'), end=end.strftime('%Y.%m'), months=months),
        'date': datetime.date.today().strftime('%Y.%m.%d'), 'blind': blind, 'doc_title': doc_title,
        'labels': {'period': T('rp_lbl_period'), 'data': T('rp_lbl_data'), 'date': T('rp_lbl_date'), 'scope': T('rp_lbl_scope'),
                   'data_source': T('rp_data_source'), 'blind_on': T('rp_blind_on'), 'blind_off': T('rp_blind_off'),
                   'confidential': T('rp_confidential'), 'footer': T('rp_footer')},
    }


def adoption_pdf_chart(groups, n=10):
    """① 절감·비용 증가 상위 품목군 가로 막대 (절감 Blue, 증가 Black). 값 라벨은 막대 끝 바깥에 둬 겹치지 않게."""
    import report_theme as rt
    import matplotlib.ticker as mticker
    rows = list(groups['saved'].head(n)['savings'].items()) + list(groups['increased'].head(n)['savings'].items())
    if not rows:
        return None
    rows = rows[::-1]  # 가로 막대는 아래부터 그리므로, 절감 1위가 맨 위에 오게 뒤집는다
    labels = [str(k)[:28] for k, _ in rows]
    vals = [float(v) for _, v in rows]
    fig, ax = rt.new_chart(0.27 * len(rows) + 0.7)
    rt.apply_pdf_chart_style(ax, grid_axis='x')
    y = np.arange(len(rows))
    ax.barh(y, vals, height=0.62, color=[rt.HEX['blue'] if v > 0 else rt.HEX['black'] for v in vals], linewidth=0)
    span = max(max(abs(v) for v in vals), 1.0)
    for yi, v in zip(y, vals):
        txt = f"${v:,.0f}" if v > 0 else f"–${-v:,.0f}"
        ax.text(v + (span * 0.015 if v > 0 else -span * 0.015), yi, txt, va='center', ha='left' if v > 0 else 'right',
                fontsize=7, color=rt.HEX['black'])
    ax.axvline(0, color=rt.HEX['black'], linewidth=0.6)
    lo = min(min(vals), 0)
    hi = max(max(vals), 0)
    ax.set_xlim(lo - span * 0.22 if lo < 0 else -span * 0.02, hi + span * 0.22 if hi > 0 else span * 0.02)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5, color=rt.HEX['black'])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 0 else f"–${-v:,.0f}"))
    fig.subplots_adjust(left=0.27, right=0.99, top=0.97, bottom=0.08 if len(rows) > 6 else 0.16)
    return rt.chart_image(fig)


def build_adoption_report(ctx):
    """① 도입 효과 분석 PDF: 표지 → 요약(KPI 4 + 결론 3) → 품목군별 절감(차트 + 그룹별 표)
    → 따로 확인할 품목(계약 후 구매 없음·구성 변화) → 방법론·데이터 한계 → 클로징."""
    import report_theme as rt
    from reportlab.platypus import Spacer
    from reportlab.lib.units import mm
    groups, th = ctx['groups'], ctx['minor_th']
    sv = ctx['savings_df']
    valid = sv[sv['volume_after'] > 0]
    net = float(valid['savings'].sum())
    net_excl = float(valid.loc[~valid['mix_changed'], 'savings'].sum())
    differ = round(net) != round(net_excl)
    saved_total = float(groups['saved']['savings'].sum())
    inc_total = float(-groups['increased']['savings'].sum())
    before_cost = float((valid['avg_price_before'] * valid['volume_after']).sum())
    after_cost = float((valid['avg_price_after'] * valid['volume_after']).sum())
    price_chg = (after_cost / before_cost - 1) * 100 if before_cost > 0 else None
    customer = ctx['customer_label']
    plot_df, contract = ctx['plot_df'], pd.Timestamp(ctx['contract_date'])
    d0, d1 = plot_df['date'].min(), plot_df['date'].max()
    meta = _rp_meta(T('p1_rp_eyebrow'), [customer, T('p1_rp_title2')], customer, None, None, (d0, d1), ctx['blind'],
                    f"{customer} {T('p1_rp_title2')}")
    excl_txt = T('p1_rp_excl_suffix', y=abs(net_excl)) if differ else ''

    # 01 요약
    headline = T('p1_rp_head_save', x=net) if net > 0 else T('p1_rp_head_loss', x=-net)
    cards = [
        (T('p1_rp_kpi_net'), (f"${net:,.0f}" if net >= 0 else f"–${-net:,.0f}"),
         T('p1_rp_kpi_net_note_excl', y=net_excl) if differ else T('p1_rp_kpi_net_note')),
        (T('p1_rp_kpi_saved', n=len(groups['saved'])), f"${saved_total:,.0f}", T('p1_rp_kpi_group_note', th=th)),
        (T('p1_rp_kpi_increased', n=len(groups['increased'])), f"${inc_total:,.0f}", T('p1_rp_kpi_group_note', th=th)),
        (T('p1_rp_kpi_price'), '-' if price_chg is None else (f"–{-price_chg:.1f}%" if price_chg < 0 else f"+{price_chg:.1f}%"),
         T('p1_rp_kpi_price_note')),
    ]
    f1 = ((T('p1_rp_f1_save', x=net, excl=excl_txt) if net > 0 else T('p1_rp_f1_loss', x=-net, excl=excl_txt)),
          T('p1_rp_f1_d', ns=len(groups['saved']), s=saved_total, ni=len(groups['increased']), i=inc_total))
    if len(groups['saved']):
        top = groups['saved'].iloc[0]
        top_name = str(groups['saved'].index[0])
        share = float(top['savings']) / saved_total * 100 if saved_total > 0 else 0
        f2 = ((T('p1_rp_f2_concentrated', name=top_name, p=share) if share >= 50 else T('p1_rp_f2', name=top_name, p=share)),
              T('p1_rp_f2_d', b=float(top['avg_price_before']), a=float(top['avg_price_after']), v=float(top['volume_after'])))
    else:
        f2 = (T('p1_rp_f2_none', th=th), T('p1_rp_f2_none_d'))
    n_mix = int(sv['mix_changed'].sum())
    mix_txt = T('p1_rp_f3_mix', k=n_mix) if n_mix else ''
    if len(groups['increased']):
        names = ', '.join(str(i) for i in groups['increased'].index[:2])
        f3 = (T('p1_rp_f3_up', names=names, n=len(groups['increased']), x=inc_total), T('p1_rp_f3_up_d') + mix_txt)
    else:
        f3 = (T('p1_rp_f3_none'), T('p1_rp_f3_none_d') + mix_txt)
    s1 = [rt.SectionHeader(T('p1_rp_s1_eyebrow'), headline), Spacer(1, 3 * mm), rt.kpi_row(cards), Spacer(1, 9 * mm),
          rt.h3(T('p1_rp_findings')), Spacer(1, 2 * mm), rt.Rule()]
    s1 += rt.findings_block([('01',) + f1, ('02',) + f2, ('03',) + f3])

    # 02 품목군별 절감
    header = [T('p1_rp_th_item'), T('p1_rp_th_before'), T('p1_rp_th_after'), T('p1_rp_th_vol'), T('p1_rp_th_saving'), T('p1_rp_th_mix')]
    widths = rt.col_widths([34, 13, 13, 15, 15, 13])

    def _rows(g):
        out = []
        for name, r in g.iterrows():
            sav = float(r['savings'])
            out.append([str(name), f"{r['avg_price_before']:.2f}", '-' if pd.isna(r['avg_price_after']) else f"{r['avg_price_after']:.2f}",
                        f"{r['volume_after']:,.0f}", f"{sav:,.0f}" if sav >= 0 else f"–{-sav:,.0f}",
                        rt.status_badge(T('p1_rp_badge_mix'), False) if r['mix_changed'] else ''])
        return out

    s2 = [rt.SectionHeader(T('p1_rp_s2_eyebrow'), T('p1_rp_s2_head'), T('p1_rp_s2_sub')), Spacer(1, 2 * mm)]
    chart = adoption_pdf_chart(groups)
    if chart is not None:
        s2 += [rt.h3(T('p1_rp_chart')), rt.small(T('p1_rp_chart_note', n=10)), Spacer(1, 2 * mm), chart, Spacer(1, 7 * mm)]
    group_titles = {'saved': T('p1_group_saved', n=len(groups['saved']), amount=saved_total),
                    'increased': T('p1_group_increased', n=len(groups['increased']), amount=inc_total),
                    'minor': T('p1_group_minor', n=len(groups['minor']))}
    for key in ('saved', 'increased', 'minor'):
        if groups[key].empty:
            continue
        s2 += [rt.h3(group_titles[key]), Spacer(1, 2.5 * mm),
               rt.brand_table(header, _rows(groups[key]), widths, right_cols=(1, 2, 3, 4), header_markup=True), Spacer(1, 7 * mm)]

    # 03 따로 확인할 품목
    nop = groups['no_purchase']
    mixed = valid[valid['mix_changed']]
    s3 = [rt.SectionHeader(T('p1_rp_s3_eyebrow'), T('p1_rp_s3_head'), T('p1_rp_s3_sub')), Spacer(1, 2 * mm),
          rt.h3(T('p1_rp_s3_nopurchase', n=len(nop))), rt.small(T('p1_rp_s3_nopurchase_note')), Spacer(1, 2.5 * mm)]
    if nop.empty:
        s3.append(rt.small(T('p1_rp_s3_none')))
    else:
        s3.append(rt.brand_table([T('p1_rp_th_item'), T('p1_rp_th_before')],
                                 [[str(n_), f"{r['avg_price_before']:.2f}"] for n_, r in nop.iterrows()],
                                 rt.col_widths([70, 30]), right_cols=(1,), header_markup=True))
    s3 += [Spacer(1, 8 * mm), rt.h3(T('p1_rp_s3_mix', n=len(mixed))), rt.small(T('p1_rp_s3_mix_note')), Spacer(1, 2.5 * mm)]
    if mixed.empty:
        s3.append(rt.small(T('p1_rp_s3_none')))
    else:
        s3.append(rt.brand_table([T('p1_rp_th_item'), T('p1_rp_th_overlap'), T('p1_rp_th_saving')],
                                 [[str(n_), pack_change_text(r['mix_kind'], r['mix_from'], r['mix_to'], r['mix_b'], r['mix_a']),
                                   f"{r['savings']:,.0f}" if r['savings'] >= 0 else f"–{-r['savings']:,.0f}"] for n_, r in mixed.iterrows()],
                                 rt.col_widths([40, 42, 18]), right_cols=(2,), header_markup=True))

    # 04 방법론
    fmt = lambda t: pd.Timestamp(t).strftime('%Y.%m.%d')
    meth = [(T('p1_rp_lbl_data'), T('p1_rp_m_data')),
            (T('p1_rp_lbl_period'), T('p1_rp_m_period', b0=fmt(d0), b1=fmt(contract - pd.Timedelta(days=1)), a0=fmt(contract), a1=fmt(d1), c=fmt(contract))),
            (T('p1_rp_lbl_cluster'), T('p1_rp_m_cluster')), (T('p1_rp_lbl_pack'), T('p1_rp_m_pack')),
            (T('p1_rp_lbl_outlier'), T('p1_rp_m_outlier')),
            (T('p1_rp_lbl_price'), T('p1_rp_m_price')), (T('p1_rp_lbl_saving'), T('p1_rp_m_saving')),
            (T('p1_rp_lbl_minor'), T('p1_rp_m_minor', th=th)), (T('p1_rp_lbl_mix'), T('p1_rp_m_mix')),
            (T('p1_rp_lbl_blind'), T('p1_rp_m_blind_on') if ctx['blind'] else T('p1_rp_m_blind_off'))]
    s4 = [rt.SectionHeader(T('p1_rp_s4_eyebrow'), T('p1_rp_s4_head')), Spacer(1, 2 * mm),
          rt.methodology_table(meth, header=(T('rp_meth_item'), T('rp_meth_rule'))), Spacer(1, 8 * mm),
          rt.h3(T('rp_unknown_header')), Spacer(1, 2 * mm), rt.Rule()]
    s4 += rt.dash_list([T(f'p1_rp_unknown{i}') for i in range(1, 5)])

    # 클로징
    top_items = ', '.join(str(i) for i in groups['saved'].index[:3]) or '-'
    v1 = T('p1_rp_close_v1', x=net, excl=excl_txt, items=top_items) if net > 0 else T('p1_rp_close_v1_loss', x=-net, excl=excl_txt)
    v2 = (T('p1_rp_close_v2', n=len(groups['increased']), x=inc_total, items=', '.join(str(i) for i in groups['increased'].index[:3]))
          if len(groups['increased']) else T('p1_rp_close_v2_none'))
    closing = {'kind': 'closing', 'eyebrow': 'Summary', 'headline': T('p1_rp_close_head'), 'note': T('p1_rp_close_note'),
               'items': [('01', T('p1_rp_close_k1'), v1), ('02', T('p1_rp_close_k2'), v2),
                         ('03', T('p1_rp_close_k3'), T('p1_rp_close_v3', k=n_mix, m=len(nop)))]}
    return rt.make_report(meta, [{'kind': 'body', 'flow': f} for f in (s1, s2, s3, s4)] + [closing])


def _rp_money_k(v):
    return f"${v / 1000:,.0f}K"


def screener_pdf_quadrant(res, bound_hi, bound_lo, bound_gap):
    """⑨ 4분면 산점도 (matplotlib, 브랜드 색). 라벨은 A그룹 수입액 상위 몇 개만, 서로 겹치지 않게 엇갈려 배치."""
    import report_theme as rt
    fig, ax = rt.new_chart(3.6)
    max_val = max(float(res['value'].max()), 1.0)
    for cls in ['D', 'C', 'B', 'A']:
        sub = res[res['class'] == cls]
        if sub.empty:
            continue
        face = rt.CLASS_HEX[cls]
        has_gap = sub['gap_score'].notna()
        ax.scatter(sub.loc[has_gap, 'uni_rank'], sub.loc[has_gap, 'gap_score'], s=18 + 420 * (sub.loc[has_gap, 'value'] / max_val),
                   color=face, edgecolor='white', linewidth=0.6, label=T(SCREENER_CLASS_KEY[cls]), zorder=3)
        if (~has_gap).any():  # 격차를 계산할 수 없는 품목 (같은 달·원산지 셀의 선적이 3건 미만): 0선에 속 빈 원
            ax.scatter(sub.loc[~has_gap, 'uni_rank'], [0] * int((~has_gap).sum()), s=18 + 420 * (sub.loc[~has_gap, 'value'] / max_val),
                       color='white', edgecolor=face, linewidth=1.2, zorder=3)
    if res['gap_score'].isna().any():  # 범례용 표본 하나: 속 빈 원 = 격차 계산 불가
        ax.scatter([], [], s=30, color='white', edgecolor=rt.HEX['black'], linewidth=1.2, label=T('scr_rp_legend_nogap'))
    for x in (bound_lo, bound_hi):
        ax.axvline(x, color=rt.HEX['black'], linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax.axhline(bound_gap, color=rt.HEX['black'], linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    top_a = res[res['class'] == 'A'].sort_values('value', ascending=False).head(5)
    placed = []
    for _, r in top_a.iterrows():
        x, y = float(r['uni_rank']), float(r['gap_score'] if pd.notna(r['gap_score']) else 0)
        dy = 4
        while any(abs(x - px) < 9 and abs(y + dy - py) < 4 for px, py in placed):
            dy += 4
        placed.append((x, y + dy))
        ax.annotate(str(r['hs']), (x, y), xytext=(min(x, 96), min(y + dy, 106)), fontsize=6.6, color=rt.HEX['black'],
                    ha='right' if x > 90 else 'center', va='bottom',
                    arrowprops=dict(arrowstyle='-', color=rt.HEX['black'], lw=0.4) if dy > 4 else None)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 112)
    ax.set_xlabel(T('sc_axis_uni'), fontsize=7.5, color=rt.HEX['black'])
    ax.set_ylabel(T('sc_axis_gap'), fontsize=7.5, color=rt.HEX['black'])
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5, frameon=False, fontsize=7.2, markerscale=0.6)
    for t in leg.get_texts():
        t.set_color(rt.HEX['black'])
    for h in leg.legend_handles:
        h.set_sizes([30])
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.2)
    return rt.chart_image(fig)


def build_screener_report(R, weights, min_kg, min_ship, top_n=15):
    """⑨ PDF: 표지 → 요약(분류별 품목 수, A그룹 Top) → 4분면 → 분류별 품목 표 → 지표 정의 → 클로징."""
    import report_theme as rt
    from reportlab.platypus import Spacer
    from reportlab.lib.units import mm
    res, (bound_hi, bound_lo, bound_gap) = R['res'], R['bounds']
    counts = res['class'].value_counts()
    meta = _rp_meta(T('scr_rp_eyebrow'), [T('scr_rp_title1'), T('scr_rp_title2')], T('scr_rp_item', n=len(res)), None, None,
                    R['period'], False, T('sc_title'))

    n_a = int(counts.get('A', 0))
    s1 = [rt.SectionHeader(T('scr_rp_s1_eyebrow'), T('scr_rp_s1_head_a', n=n_a) if n_a else T('scr_rp_s1_head_none')), Spacer(1, 3 * mm)]
    s1.append(rt.kpi_row([(T(SCREENER_CLASS_KEY[c]), T('rp_count_unit', n=int(counts.get(c, 0))), T(f'scr_rp_kpi_note_{c.lower()}'))
                          for c in 'ABCD']))
    s1 += [Spacer(1, 9 * mm), rt.h3(T('scr_rp_top_header')), Spacer(1, 2.5 * mm)]
    top_a = res[res['class'] == 'A'].sort_values('uni_rank', ascending=False).head(10)
    if top_a.empty:
        s1.append(rt.small(T('scr_rp_top_none')))
    else:
        rows = [[str(i), r['hs'], r['name'], f"{r['uni_rank']:.0f}", f"{r['gap_score']:.0f}", f"{r['importers']}",
                 f"{r['cr3']:.0f}%", _rp_money_k(r['value'])] for i, (_, r) in enumerate(top_a.iterrows(), 1)]
        s1.append(rt.brand_table([T('bd_rp_th_no'), T('sc_col_hs'), T('sc_col_name'), T('scr_rp_th_uni'), T('scr_rp_th_gap'),
                                  T('scr_rp_th_importers'), T('scr_rp_th_cr3'), T('scr_rp_th_value')], rows,
                                 rt.col_widths([5, 15, 31, 11, 9, 10, 10, 14]), right_cols=(3, 4, 5, 6, 7)))

    s2 = [rt.SectionHeader(T('scr_rp_s2_eyebrow'), T('scr_rp_s2_head'), T('scr_rp_s2_sub')), Spacer(1, 2 * mm),
          screener_pdf_quadrant(res, bound_hi, bound_lo, bound_gap), Spacer(1, 3 * mm),
          rt.small(T('scr_rp_s2_note', lo=bound_lo, hi=bound_hi, gap=bound_gap))]

    s3 = [rt.SectionHeader(T('scr_rp_s3_eyebrow'), T('scr_rp_s3_head'), T('scr_rp_s3_sub', n=top_n)), Spacer(1, 2 * mm)]
    header = [T('sc_col_hs'), T('sc_col_name'), T('scr_rp_th_uni'), T('scr_rp_th_gap'), T('scr_rp_th_importers'),
              T('scr_rp_th_cr3'), T('scr_rp_th_exporters'), T('scr_rp_th_origins'), T('scr_rp_th_value')]
    for cls in 'ABCD':
        sub = res[res['class'] == cls].sort_values('uni_rank', ascending=False)
        if sub.empty:
            continue
        s3 += [rt.badge_note(rt.class_badge(cls, T(SCREENER_CLASS_KEY[cls])), T('rp_count_unit', n=len(sub)), badge_w=38 * mm),
               Spacer(1, 2 * mm)]
        rows = [[r['hs'], r['name'], f"{r['uni_rank']:.0f}", '-' if pd.isna(r['gap_score']) else f"{r['gap_score']:.0f}",
                 f"{r['importers']}", f"{r['cr3']:.0f}%", f"{r['exporters']}", f"{r['origins']}", _rp_money_k(r['value'])]
                for _, r in sub.head(top_n).iterrows()]
        s3 += [rt.brand_table(header, rows, rt.col_widths([15, 24, 11, 8, 9, 10, 9, 9, 14]), right_cols=(2, 3, 4, 5, 6, 7, 8)),
               Spacer(1, 7 * mm)]

    w_txt = ', '.join(f"{T(f'sc_w_{k}').split(' (')[0]} {weights[k]:g}" for k, _ in SCREENER_COMPONENTS)
    meth = [(T('scr_rp_lbl_data'), T('scr_rp_m_data')), (T('scr_rp_lbl_period'), meta['period']),
            (T('scr_rp_lbl_prep'), T('scr_rp_m_prep', kg=int(min_kg), ship=int(min_ship))),
            (T('scr_rp_lbl_uni'), T('scr_rp_m_uni', w=w_txt)), (T('scr_rp_lbl_gap'), T('scr_rp_m_gap')),
            (T('scr_rp_lbl_class'), T('scr_rp_m_class', hi=bound_hi, lo=bound_lo, gap=bound_gap)),
            (T('scr_rp_lbl_conc'), T('scr_rp_m_cr3')), (T('scr_rp_lbl_price'), T('scr_rp_m_price'))]
    s4 = [rt.SectionHeader(T('scr_rp_s4_eyebrow'), T('scr_rp_s4_head')), Spacer(1, 2 * mm),
          rt.methodology_table(meth, header=(T('rp_meth_item'), T('rp_meth_rule'))), Spacer(1, 8 * mm),
          rt.h3(T('rp_unknown_header')), Spacer(1, 2 * mm), rt.Rule()]
    s4 += rt.dash_list([T(f'scr_rp_unknown{i}') for i in range(1, 5)])

    pick = res[res['class'] == 'A'] if n_a else res[res['class'] == 'B']
    items_txt = ', '.join(f"{r['hs']} {r['name']}" for _, r in pick.sort_values('uni_rank', ascending=False).head(3).iterrows()) or '-'
    closing = {'kind': 'closing', 'eyebrow': 'Summary', 'headline': T('scr_rp_close_head'), 'note': T('scr_rp_close_note'),
               'items': [('01', T('scr_rp_close_k1'), T('scr_rp_close_v1_a', items=items_txt) if n_a else T('scr_rp_close_v1_none', items=items_txt)),
                         ('02', T('scr_rp_close_k2'), T('scr_rp_close_v2')),
                         ('03', T('scr_rp_close_k3'), T('scr_rp_close_v3', lo=bound_lo, hi=bound_hi, gap=bound_gap, kg=int(min_kg)))]}
    return rt.make_report(meta, [{'kind': 'body', 'flow': f} for f in (s1, s2, s3, s4)] + [closing])


def bid_pdf_origin_chart(df, origin_tbl, base_price, major_tons):
    """⑩ 원산지별 단가 분포: 점 = VWAP, 막대 = P10~P90, 연 major_tons 미만 원산지는 속 빈 점. 현 납품 기준 점선."""
    import report_theme as rt
    import matplotlib.ticker as mticker
    years = max((df['_date'].max() - df['_date'].min()).days / 365.25, 1 / 12)
    top = df.groupby('_origin')['_volume'].sum().sort_values(ascending=False).head(10).index
    stats = []
    for o in top:
        g = df[df['_origin'] == o]
        p10, p90 = g['_price'].quantile([0.1, 0.9])
        stats.append((o, float(weighted_avg(g, '_price', '_volume')), float(p10), float(p90), g['_volume'].sum() / 1000 / years))
    stats.sort(key=lambda x: -x[1])
    fig, ax = rt.new_chart(0.42 * len(stats) + 0.75)
    rt.apply_pdf_chart_style(ax, grid_axis='x')
    for yi, (o, vwap, p10, p90, tons) in enumerate(stats):
        thin = tons < major_tons
        ax.plot([p10, p90], [yi, yi], color=rt.HEX['black'], alpha=0.25, linewidth=2.2, solid_capstyle='round')
        ax.scatter([vwap], [yi], s=46, color='#FFFFFF' if thin else rt.HEX['blue'], edgecolor=rt.HEX['blue'], linewidth=1.4, zorder=3)
        ax.text(vwap, yi + 0.3, f"${vwap:.2f}", va='bottom', ha='center', fontsize=7.2, color=rt.HEX['black'])
    lo = min(s[2] for s in stats)
    hi = max(s[3] for s in stats)
    if base_price is not None:
        hi = max(hi, base_price)
        lo = min(lo, base_price)
        ax.axvline(base_price, color=rt.HEX['black'], linewidth=1, linestyle=(0, (3, 2)))
        ax.text(base_price + (hi - lo) * 0.01, len(stats) - 0.45, T('bd_rp_baseline_line', p=base_price), fontsize=7.2,
                color=rt.HEX['black'], va='bottom')
    pad = (hi - lo) * 0.08 or hi * 0.05
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.6, len(stats) - 0.1 + (0.4 if base_price is not None else 0))
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([f"{o}  ({t:,.0f}t)" for o, _, _, _, t in stats], fontsize=8, color=rt.HEX['black'])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    fig.subplots_adjust(left=0.22, right=0.99, top=0.95, bottom=0.12 if len(stats) > 3 else 0.2)
    return rt.chart_image(fig)


def bid_pdf_monthly_chart(stab):
    """⑩ 월별 선적 막대 (톤). 선적 없는 달은 빗금 막대."""
    import report_theme as rt
    import matplotlib.ticker as mticker
    vals = stab['monthly'].values.astype(float)
    labels = [f"{str(p.year)[2:]}.{p.month:02d}" for p in stab['monthly'].index]
    fig, ax = rt.new_chart(2.1)
    rt.apply_pdf_chart_style(ax, grid_axis='y')
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.72, color=rt.HEX['blue'], linewidth=0)
    base = vals[vals > 0].mean() if (vals > 0).any() else 1.0
    gaps = x[vals == 0]
    if len(gaps):
        ax.bar(gaps, [base * 0.12] * len(gaps), width=0.72, color='white', edgecolor=rt.HEX['black'], linewidth=0.8, hatch='////')
    step = 3 if len(vals) > 12 else 1
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}t"))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.96, bottom=0.14)
    return rt.chart_image(fig)


def build_bid_report(ctx):
    """⑩ PDF: 표지 → 요약(KPI 4 + 핵심 결론 3 + 관세 배지) → 대체 공급처(원산지 차트 + 공급사 순위표)
    → 공급 안정성(공급사별 1페이지) → 방법론·데이터 한계 → Black 클로징. 목업(build_report.py)과 같은 구성."""
    import report_theme as rt
    from reportlab.platypus import Spacer, PageBreak, TableStyle
    from reportlab.lib.units import mm
    sup, verify, annual_t = ctx['sup_tbl'], ctx['verify'], ctx['annual_tons']
    base_key = ctx['base_key']
    base_row = sup[sup['key'] == base_key].iloc[0] if base_key is not None and base_key in set(sup['key']) else None
    base_cmp = float(base_row['cmp']) if base_row is not None else None
    cand = sup[sup['key'] != base_key]
    verified = cand[cand['verified']].sort_values('cmp')
    basis = T('bd_rp_basis_cif') if ctx['tariff_missing'] else T('bd_rp_basis_tariff')
    stab_cache = {}

    def _stab(name):
        if name not in stab_cache:
            stab_cache[name] = bid_stability(ctx['scoped'], name, ctx['period'][1])
        return stab_cache[name]

    meta = _rp_meta(T('bd_rp_eyebrow'), [ctx['item'], T('bd_rp_title2')], ctx['item'], ctx['spec'] or None,
                    ctx['hs'], ctx['period'], ctx['blind'], f"{ctx['item']} {T('bd_rp_title2')}")

    # 01 요약
    best = verified.iloc[0] if len(verified) else None
    save_pct = (base_cmp - float(best['cmp'])) / base_cmp * 100 if (best is not None and base_cmp) else None
    if save_pct is not None and save_pct > 0:
        headline = T('bd_rp_head_save', pct=save_pct)
    elif len(verified):
        headline = T('bd_rp_head_found', n=len(verified))
    else:
        headline = T('bd_rp_head_none')
    best_gap = _stab(best['key'])['gap_months'] if best is not None else None
    m36 = _stab(best['key'])['n_months'] if best is not None else 36
    cards = [
        (T('bd_rp_kpi1'), f"${best['cmp']:.2f}" if best is not None else '-',
         T('bd_rp_kpi1_note', basis=basis, origin=best['origin']) if best is not None else T('bd_rp_kpi1_none')),
        (T('bd_rp_kpi2'), f"–{save_pct:.1f}%" if save_pct is not None and save_pct > 0 else (f"+{-save_pct:.1f}%" if save_pct is not None else '-'),
         T('bd_rp_kpi2_note', p=base_cmp) if base_cmp else T('bd_rp_kpi_no_base')),
        (T('bd_rp_kpi3'), _rp_money_k((base_cmp - float(best['cmp'])) * annual_t * 1000) if (save_pct is not None and save_pct > 0) else '-',
         T('bd_rp_kpi3_note', t=annual_t) if base_cmp else T('bd_rp_kpi_no_base')),
        (T('bd_rp_kpi4'), T('rp_month_unit', n=best_gap) if best_gap is not None else '-', T('bd_rp_kpi4_note', m=m36)),
    ]
    major_t = 500.0
    df = ctx['df']
    years = max((df['_date'].max() - df['_date'].min()).days / 365.25, 1 / 12)
    o_vol = (df.groupby('_origin')['_volume'].sum() / 1000 / years)
    o_vwap = weighted_avg_groupby(df, '_origin', '_price', '_volume')
    major = [o for o in o_vol.index if o_vol[o] >= major_t]
    if len(major) >= 2:
        cheap, dear = min(major, key=lambda o: o_vwap[o]), max(major, key=lambda o: o_vwap[o])
        f1 = (T('bd_rp_f1_two', o1=cheap, o2=dear, d=float(o_vwap[dear] - o_vwap[cheap])), T('bd_rp_f1_two_d', t=major_t))
    elif len(major) == 1:
        f1 = (T('bd_rp_f1_one', o=major[0], tons=float(o_vol[major[0]]), t=major_t), T('bd_rp_f1_one_d'))
    else:
        f1 = (T('bd_rp_f1_none', t=major_t), T('bd_rp_f1_none_d'))
    if base_cmp:
        k = int((verified['cmp'] <= base_cmp * 0.9).sum())
        f2 = (T('bd_rp_f2_base', n=len(verified), k=k), T('bd_rp_f2_d', ship=verify['min_ship'], ton=verify['min_ton'], imp=verify['min_imp']))
    else:
        f2 = (T('bd_rp_f2_nobase', n=len(verified)), T('bd_rp_f2_d', ship=verify['min_ship'], ton=verify['min_ton'], imp=verify['min_imp']))
    floor = float(best['cmp']) if best is not None else np.inf
    cheap_unv = cand[(~cand['verified']) & (cand['cmp'] < floor)].sort_values('cmp')
    if len(cheap_unv):
        names = ', '.join(f"{r['key']}({r['origin']})" for _, r in cheap_unv.head(2).iterrows())
        f3 = (T('bd_rp_f3_cheap', names=names), T('bd_rp_f3_cheap_d'))
    else:
        f3 = (T('bd_rp_f3_none'), T('bd_rp_f3_none_d'))
    s1 = [rt.SectionHeader(T('bd_rp_s1_eyebrow'), headline), Spacer(1, 3 * mm), rt.kpi_row(cards), Spacer(1, 9 * mm),
          rt.h3(T('bd_rp_findings')), Spacer(1, 2 * mm), rt.Rule()]
    s1 += rt.findings_block([('01',) + f1, ('02',) + f2, ('03',) + f3])
    s1 += [Spacer(1, 7 * mm),
           rt.badge_note(rt.status_badge(T('bd_badge_no_tariff'), False) if ctx['tariff_missing'] else rt.status_badge(T('bd_badge_tariff_ok'), True),
                         T('bd_rp_tariff_missing') if ctx['tariff_missing'] else T('bd_rp_tariff_ok'))]

    # 02 대체 공급처
    s2 = [rt.SectionHeader(T('bd_rp_s2_eyebrow'), T('bd_rp_s2_head'), T('bd_rp_s2_sub')), Spacer(1, 2 * mm),
          rt.h3(T('bd_rp_origin_chart')), rt.small(T('bd_rp_origin_chart_note', t=major_t)), Spacer(1, 2 * mm),
          bid_pdf_origin_chart(df, ctx['origin_tbl'], float(base_row['vwap']) if base_row is not None else None, major_t),
          Spacer(1, 7 * mm), rt.h3(T('bd_rp_rank_header')), Spacer(1, 2.5 * mm)]
    top_n = 12
    ranked = pd.concat([cand[cand['verified']].sort_values('cmp'), cand[~cand['verified']].sort_values('cmp')]).head(top_n)
    rows = []
    for i, (_, r) in enumerate(ranked.iterrows(), 1):
        vs = f"{(r['cmp'] - base_cmp) / base_cmp * 100:+.1f}%" if base_cmp else '-'
        rows.append([str(i), r['key'], r['origin'], f"{r['cmp']:.2f}", vs, f"{r['shipments']:,}", f"{r['tons']:,.0f}",
                     f"{r['months']}", T('rp_company_unit', n=r['importers']),
                     rt.status_badge(T('bd_badge_verified'), True) if r['verified'] else rt.status_badge(T('bd_badge_unverified'), False)])
    if base_row is not None:
        rows.append(['–', base_row['key'], base_row['origin'], f"{base_cmp:.2f}", T('bd_rp_base_tag'), '–', '–', '–', '–', ''])
    tbl = rt.brand_table([T('bd_rp_th_no'), T('bd_col_exporter'), T('bd_col_origin'), T('bd_rp_th_cmp'), T('bd_rp_th_vs'),
                          T('bd_rp_th_ship'), T('bd_rp_th_ton'), T('bd_rp_th_months'), T('bd_rp_th_imp'), T('bd_rp_th_status')],
                         rows, rt.col_widths([7, 44, 17, 17, 17, 11, 15, 13, 14, 19]), right_cols=(3, 4, 5, 6, 7, 8), header_markup=True)
    if base_row is not None:
        tbl.setStyle(TableStyle([("LINEABOVE", (0, len(rows)), (-1, len(rows)), 0.8, rt.BLACK)]))
    s2 += [tbl, Spacer(1, 3 * mm),
           rt.small(T('bd_rp_rank_note', ship=verify['min_ship'], ton=verify['min_ton'], imp=verify['min_imp'], n=top_n))]

    # 03 공급 안정성 (공급사별 1페이지)
    s3 = []
    picks = [p for p in ctx['stab_suppliers'] if p in set(sup['key'])]
    for idx, name in enumerate(picks):
        st_ = _stab(name)
        if st_['empty']:
            continue
        if s3:
            s3.append(PageBreak())
        origin = sup.loc[sup['key'] == name, 'origin'].iloc[0]
        m = st_['n_months']
        s3 += [rt.SectionHeader(T('bd_rp_s3_eyebrow', i=idx + 1, n=len(picks)), name, T('bd_rp_s3_sub', origin=origin, m=m)),
               Spacer(1, 2 * mm),
               rt.kpi_row([(T('bd_rp_s3_kpi1'), f"{st_['annual_tons']:,.0f}t", T('bd_rp_s3_kpi1_note', m=m)),
                           (T('bd_rp_s3_kpi2'), T('rp_month_unit', n=st_['gap_months']), T('bd_rp_s3_kpi2_note')),
                           (T('bd_rp_s3_kpi3'), T('rp_company_unit', n=st_['importers']), T('bd_rp_s3_kpi3_note')),
                           (T('bd_rp_s3_kpi4'), '-' if pd.isna(st_['cv']) else f"{st_['cv']:.0f}%", T('bd_rp_s3_kpi4_note'))]),
               Spacer(1, 8 * mm), rt.h3(T('bd_rp_s3_chart')),
               rt.small(T('bd_rp_s3_chart_gap') if st_['gap_months'] else T('bd_rp_s3_chart_nogap', m=m)),
               Spacer(1, 2 * mm), bid_pdf_monthly_chart(st_), Spacer(1, 8 * mm), rt.h3(T('bd_rp_s3_year')), Spacer(1, 2.5 * mm)]
        yrows = [[label, f"{r['tons']:,.0f}", f"{int(r['months'])}/{int(r['span'])}", '-' if pd.isna(r['vwap']) else f"{r['vwap']:.2f}",
                  T('rp_company_unit', n=int(r['importers']))] for label, r in st_['yearly'].iterrows()]
        s3 += [rt.brand_table([T('bd_rp_s3_th_year'), T('bd_rp_s3_th_ton'), T('bd_rp_s3_th_months'), T('bd_rp_s3_th_vwap'), T('bd_rp_s3_th_imp')],
                              yrows, rt.col_widths([28, 18, 18, 18, 18]), right_cols=(1, 2, 3, 4)),
               Spacer(1, 8 * mm), rt.proof_box(T('bd_rp_proof'), bid_stability_sentence(name, st_))]

    # 04 방법론
    rates = ', '.join(f"{o} {r:g}%" for o, r in ctx['tariffs'].items())
    if not ctx['tariffs']:
        tariff_txt = T('bd_rp_m_tariff_none')
    elif ctx['tariff_missing']:
        tariff_txt = T('bd_rp_m_tariff_part', rates=rates)
    else:
        tariff_txt = T('bd_rp_m_tariff_all', rates=rates)
    meth = [(T('bd_rp_lbl_data'), T('bd_rp_m_data')), (T('bd_rp_lbl_period'), T('bd_rp_m_period', period=meta['period'])),
            (T('bd_rp_lbl_spec'), T('bd_rp_m_spec', hs=ctx['hs'] or T('bd_hs_all'), inc=ctx['include'] or '-', exc=ctx['exclude'] or '-')),
            (T('bd_rp_lbl_dedup'), T('bd_rp_m_dedup')), (T('bd_rp_lbl_outlier'), T('bd_rp_m_outlier', kg=int(ctx['min_kg']))),
            (T('bd_rp_lbl_price'), T('bd_rp_m_price')), (T('bd_rp_lbl_tariff'), tariff_txt),
            (T('bd_rp_lbl_verify'), T('bd_rp_m_verify', ship=verify['min_ship'], ton=verify['min_ton'], imp=verify['min_imp'])),
            (T('bd_rp_lbl_blind'), T('bd_rp_m_blind_on') if ctx['blind'] else T('bd_rp_m_blind_off'))]
    s4 = [rt.SectionHeader(T('bd_rp_s4_eyebrow'), T('bd_rp_s4_head')), Spacer(1, 2 * mm),
          rt.methodology_table(meth, header=(T('rp_meth_item'), T('rp_meth_rule'))), Spacer(1, 8 * mm),
          rt.h3(T('rp_unknown_header')), Spacer(1, 2 * mm), rt.Rule()]
    s4 += rt.dash_list([T(f'bd_rp_unknown{i}') for i in range(1, 5)])

    # 클로징
    if best is not None:
        main_txt = T('bd_rp_close_main', name=best['key'], origin=best['origin'], p=float(best['cmp']), gap=best_gap)
    else:
        low = cand.sort_values('cmp').iloc[0]
        main_txt = T('bd_rp_close_none', name=low['key'], origin=low['origin'], p=float(low['cmp']))
    if len(verified) >= 2:
        b2 = verified.iloc[1]
        backup_txt = T('bd_rp_close_backup', name=b2['key'], origin=b2['origin'], p=float(b2['cmp']), imp=int(b2['importers']))
    else:
        backup_txt = T('bd_rp_close_no_backup')
    if save_pct is not None and save_pct > 0:
        save_txt = T('bd_rp_close_save', pct=save_pct, t=annual_t, k=(base_cmp - float(best['cmp'])) * annual_t)
    else:
        save_txt = T('bd_rp_close_nosave') if not base_cmp else T('bd_rp_close_nobetter')
    closing = {'kind': 'closing', 'eyebrow': 'Summary', 'headline': T('bd_rp_close_head'),
               'items': [('01', T('bd_rp_close_k1'), main_txt), ('02', T('bd_rp_close_k2'), backup_txt), ('03', T('bd_rp_close_k3'), save_txt)],
               'note': T('bd_rp_close_note_missing') if ctx['tariff_missing'] else T('bd_rp_close_note_ok')}
    sections = [{'kind': 'body', 'flow': s1}, {'kind': 'body', 'flow': s2}]
    if s3:
        sections.append({'kind': 'body', 'flow': s3})
    sections += [{'kind': 'body', 'flow': s4}, closing]
    return rt.make_report(meta, sections)


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


def product_group_labels(names, hs=None):
    """품목 묶기 (①②⑥ 공통, 행 단위). 정제 규칙은 normalize_product_name 하나에서 관리한다.
    - 기본: 포장 단위·끝 제품 코드만 다른 품목명은 한 품목, 맛·브랜드·대상이 다르면 다른 품목
    - 일반 명칭 품목(정제한 이름이 대괄호 분류명과 같거나 분류명 단어 외에 남는 단어가 없는 품목, 예: [혼합분유] 혼합분유)은
      같은 이름 아래 다른 스펙이 섞이므로 HS코드 10단위가 2개 이상이면 HS별로 나눈다. 같은 HS 안의 세부 스펙은 구분하지 않는다.
    반환: (표시 이름, 묶기 키, 포장 단위) — names와 같은 인덱스의 Series. 키는 공백 없는 한 단어(① TF-IDF용)."""
    names = pd.Series(names).map(lambda v: v if isinstance(v, str) else ('' if pd.isna(v) else str(v)))
    if hs is None:
        hs_s = pd.Series('', index=names.index)
    else:
        hs_s = pd.Series(np.asarray(hs, dtype=object), index=names.index).map(
            lambda v: '' if v is None or (not isinstance(v, str) and pd.isna(v)) else str(_pad_hs_code(str(v).strip())))
        hs_s = hs_s.where(~hs_s.isin(['nan', 'None']), '')

    def _info(n):
        clean, pack = normalize_product_name(n)
        m = re.match(r'\s*\[(.*?)\]', n)
        cat = normalize_product_name(m.group(1))[0] if m else ''
        return clean, pack, bool(clean and cat) and set(clean.split()) <= set(cat.split())

    info = {n: _info(n) for n in names.unique()}
    clean = names.map(lambda n: info[n][0])
    pack = names.map(lambda n: info[n][1])
    base = clean.str.replace(' ', '', regex=False)
    base = base.where(base != '', names)
    generic = set(base[names.map(lambda n: info[n][2])])
    has_hs = hs_s != ''
    hs_n = hs_s[has_hs].groupby(base[has_hs]).nunique()
    split = base.isin(generic) & has_hs & base.map(hs_n).fillna(0).gt(1)
    key = base.where(~split, base + '_' + hs_s)
    label = clean.where(clean != '', names)
    label = label.where(~split, label + ' (HS ' + hs_s + ')')
    disp = label.groupby(key).agg(lambda s: s.value_counts().index[0])
    dup = disp.duplicated(keep=False)
    disp[dup] = disp[dup] + ' (' + disp.index[dup] + ')'  # 표시 이름이 겹치는 드문 경우 구분
    return key.map(disp), key, pack


def pack_month_markers(df, month_col, pack_col, volume_col):
    """⑥ 가격 추세: 직전 달(데이터가 있는 달)보다 포장 단위 구성이 크게 바뀐 달과 설명 문구.
    판정은 ①의 구성 변화 기준(pack_mix_change: 주력 단위 변경 또는 20%p 이상 이동)을 그대로 쓴다."""
    def shares(g):
        g = g[g['pack_unit'] != '']
        v = g.groupby('pack_unit')['volume'].sum()
        return v / v.sum() * 100 if v.sum() > 0 else v
    d = df[[month_col, pack_col, volume_col]].rename(columns={pack_col: 'pack_unit', volume_col: 'volume'})
    months = sorted(d[month_col].dropna().unique())
    out = []
    for prev, cur in zip(months, months[1:]):
        g_prev, g_cur = d[d[month_col] == prev], d[d[month_col] == cur]
        changed = pack_mix_change(g_prev, g_cur)
        if not changed[0]:
            continue
        b, a = shares(g_prev), shares(g_cur)
        if changed[1] == 'share':
            p = changed[2]
            text = T('p5_pack_marker', b=p, bs=float(b.get(p, 0)), a=p, as_=float(a.get(p, 0)))
        else:
            text = T('p5_pack_marker', b=b.idxmax(), bs=float(b.max()), a=a.idxmax(), as_=float(a.max()))
        out.append((cur, text))
    return out


def fig_to_png_bytes(fig, width=900, height=500, scale=2):
    """Plotly figure를 PNG 바이트로 변환 (PDF에 삽입하기 위함). kaleido 필요.
    kaleido는 시스템 폰트를 쓰므로 한글이 깨지지 않도록 복사본에 NanumGothic을 지정한다 (원본 fig는 그대로)."""
    try:
        export_fig = go.Figure(fig)
        export_fig.update_layout(font_family='NanumGothic')
        return export_fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception as e:
        st.warning(T('pdf_chart_export_warning', msg=e))
        return None


def _find_korean_font_path():
    """시스템에서 한글을 지원하는 .ttf 폰트 파일 경로를 찾는다 (없으면 None).
    matplotlib.font_manager는 자체 캐시가 낡아있을 수 있어 신뢰하지 않고, 파일시스템에서 직접 찾는다.
    reportlab의 TTFont는 .ttc(트루타입 컬렉션)를 지원하지 않으므로 .ttf 파일만 대상으로 한다.
    리포에 번들된 assets/fonts/NanumGothic-Regular.ttf를 최우선으로 쓰고, 없을 때만 시스템 경로를 찾는다."""
    import glob
    bundled_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts', 'NanumGothic-Regular.ttf')
    if os.path.isfile(bundled_path):
        return bundled_path
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


def install_bundled_font_for_charts():
    """번들 폰트를 ~/.fonts/에 복사해 kaleido(차트 PNG 내보내기)가 한글 폰트를 찾을 수 있게 한다.
    이미 있으면 건너뛰고, 실패해도 앱은 계속 동작한다."""
    try:
        import shutil
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts', 'NanumGothic-Regular.ttf')
        dst_dir = os.path.join(os.path.expanduser('~'), '.fonts')
        dst = os.path.join(dst_dir, 'NanumGothic-Regular.ttf')
        if os.path.isfile(src) and not os.path.isfile(dst):
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copyfile(src, dst)
    except Exception:
        pass


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


_PDF_EMOJI_RE = re.compile('[\U0001F000-\U0001FAFF\u2300-\u23FF\u2600-\u27BF\u2B00-\u2BFF\uFE00-\uFE0F\u200D]+ ?')  # 이모지(+ 뒤따르는 공백 1칸)


def _pdf_safe_text(text):
    """PDF 전용 글자 정리: 번들 폰트(NanumGothic)에 글리프가 없어 빈칸으로 찍히는 글자를 처리한다.
    ①~⑳은 '1.'~'20.'으로 바꾸고 이모지는 제거한다 (화면 문구는 그대로 두고 PDF에만 적용)."""
    if not isinstance(text, str):
        return text
    text = re.sub('[①-⑳]', lambda m: f"{ord(m.group()) - 0x245F}.", text)
    text = _PDF_EMOJI_RE.sub('', text)
    return re.sub(' {2,}', ' ', text).strip()


def _pdf_safe_paragraph(paragraph_cls):
    """reportlab Paragraph를 감싸 모든 PDF 텍스트가 _pdf_safe_text를 거치게 한다."""
    return lambda text, *args, **kwargs: paragraph_cls(_pdf_safe_text(text), *args, **kwargs)


def build_pdf_report(title, kpi_lines, figs, df_table=None, table_title=None, tables=None):
    """제목 + KPI 텍스트 + Plotly 차트(이미지로 변환) + 표를 하나의 PDF로 조립해 바이트로 반환.
    tables에 [(섹션 제목, DataFrame), ...]을 넘기면 df_table 뒤에 섹션별 표를 차례로 붙인다.
    reportlab + kaleido 필요."""
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    Paragraph = _pdf_safe_paragraph(Paragraph)  # ①~⑨·이모지 빈칸 방지

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

    sections = ([(table_title, df_table)] if df_table is not None else []) + list(tables or [])
    for sec_title, sec_df in sections:
        if sec_df is None or sec_df.empty:
            continue
        if sec_title:
            story.append(Paragraph(sec_title, styles['Heading3']))

        display_table = sec_df.copy()
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
        story.append(Spacer(1, 10))

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
    Paragraph = _pdf_safe_paragraph(Paragraph)  # ①~⑨·이모지 빈칸 방지

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
                    'HS코드, Incoterm처럼 표준 항목 외의 컬럼도 파일에 있으면 자동으로 인식되어, 일부 메뉴(포트폴리오 헬스체크·공급망 흐름도·집중도 진단·신규 이탈 거래처·피벗 빌더·신규사업 스코어러·범용상품 스크리너·입찰 대응 분석)의 축/필터 선택지에 그대로 추가됩니다.',
                ],
            },
            {
                'heading': '💲 ① 도입 효과 분석',
                'image': 'p1_customer.png',
                'intro': '계약을 시작한 뒤로 우리(또는 특정 수입사)의 구매 단가가 실제로 얼마나 절감됐는지 숫자로 확인하고 싶을 때 씁니다. 계약 갱신이나 성과 보고 자료를 만들 때 가장 먼저 여는 화면이에요. 트릿지 도입(계약일) 전후 비교용이며, 현재 상태 점검은 포트폴리오 헬스체크를 사용하세요.',
                'steps': [
                    '거래 내역 파일을 업로드합니다 (여러 회사 데이터가 섞여 있어도 자동으로 가장 많이 등장하는 회사를 기준으로 분석합니다).',
                    '계약 시작일을 선택합니다.',
                    '필요하면 "변화 미미 기준"($, 기본 10)을 조정합니다.',
                    '"분석 실행" 버튼을 클릭합니다.',
                ],
                'results': [
                    '순 절감액 / 절감 합계 / 비용 증가 합계 — 품목군별 절감액(계약 전후 단가 차이 × 계약 후 물량)의 순합계와, 절감·비용 증가 각각의 합계(품목 수 포함).',
                    '품목군별 카드 — 절감 품목, 단가 상승 품목, 변화 미미(절감·증가액이 ±$10 이내, 기준 조정 가능) 세 그룹으로 나뉘어 금액 큰 순으로 보입니다. 그룹마다 처음 12개만 보이고 "전체 보기"로 나머지를 펼칩니다. 계약 후 구매가 없는 품목은 "계약 후 구매 없음"으로 따로 표시하고 합계에서 뺍니다.',
                    '상세 절감 내역 표 — 같은 그룹을 "구분" 열로 보여주며, 결과는 새 양식 PDF 리포트(표지 → KPI 4개·핵심 결론 3줄 → 품목군별 절감 차트·표 → 따로 확인할 품목 → 방법론 → 클로징)로 내려받을 수 있습니다. 구성 변화 품목이 있으면 "구성 변화 품목 제외 시" 순 절감액을 함께 표기합니다. 이전 양식이 필요하면 "기존 양식 PDF" 버튼을 쓰세요.',
                    '수입 품목군 — 포장 단위(360정·30포·900g 등)와 끝의 제품 코드만 다른 품목명을 같은 품목군으로 묶습니다. 맛·브랜드·대상이 다르면 다른 품목군입니다. 계약 전후로 품목군 안의 주력 포장 단위가 바뀌거나 비중이 20%p 이상 움직이면 "포장 단위 변경"(구성 변화)으로 표시하고, 이 품목을 뺀 순 절감액도 함께 보여줍니다. 일반 명칭 품목(예: [혼합분유] 혼합분유)은 HS코드 10단위로 구분하고, 같은 HS 안의 세부 스펙은 구분하지 않습니다.',
                    '월별 수입 추이, 최근 3개월 비중 — 최근 구매 패턴 파악.',
                    '계약 이후 새로 생기거나 사라진 품목·원산지·공급사 목록.',
                ],
            },
            {
                'heading': '🩺 ② 포트폴리오 헬스체크',
                'intro': '계약 여부와 상관없이, 지금 어느 품목이 위험한지 빠르게 훑어보고 싶을 때 씁니다. 한 회사의 수입 데이터로 품목(또는 원산지·HS코드 등)별 변화 신호를 찾고, 공급 구조의 취약성·금액 비중(영향도)·선적 건수(신뢰도)를 함께 따져 위험 / 주의 / 안전으로 판정합니다.',
                'steps': [
                    '거래 내역 파일을 업로드합니다 (날짜·수입사·공급사·품목·물량·단가 컬럼이 필요합니다).',
                    '점검할 수입사를 선택합니다 (물량 큰 순으로 정렬되며, 파일에 수입사가 1곳뿐이면 자동 적용됩니다).',
                    '점검 단위를 고릅니다 — 품목(기본), 원산지, 또는 파일에 있는 추가 컬럼(HS코드 등). 품목은 포장 단위(360정·25KG 등)와 끝의 제품 코드만 다른 품목명을 한 단위로 묶고(도입 효과 분석과 같은 기준) 정제된 이름으로 보여줍니다. 묶인 원래 품목명은 결과 표 아래 "단위별 원래 품목명 보기"에서 확인합니다. 일반 명칭 품목(예: [혼합분유] 혼합분유)은 HS코드 10단위로 구분하고, 같은 HS 안의 세부 스펙은 구분하지 않습니다.',
                    '기준 기간(기본: 데이터 마지막 날짜 기준 최근 6개월)과 비교 기간(기본: 그 직전 6개월)을 선택합니다. 필요하면 "기준값 조정"에서 기준값을 바꿉니다.',
                    '"헬스체크 실행"을 클릭합니다.',
                ],
                'results': [
                    '단가 변화(변화 신호) — 두 기간의 물량가중평균 단가(VWAP) 변화율. +15% 이상이면 위험, +8% 이상이면 주의, 하락은 안전. 시장 전체 가격 변동 때문일 수 있으니 시장 요인인지 확인이 필요합니다.',
                    '공급 차질(변화 신호) — 비교 기간 비중 10% 이상인 주요 공급사가 기준 기간에 사라졌는데, 신규 공급사가 빠진 물량(월평균)의 50%도 채우지 못했으면 위험. 50% 이상 채웠으면 신호가 아니라 "전환"으로 표시하고, 단가까지 내려갔으면 "전환·단가 개선"으로 표시합니다.',
                    '물량 변화 — 두 기간의 월평균 물량 변화율. 단독으로는 판정에 쓰지 않고 수치만 보여줍니다. 단, 물량 -15% 이하이면서 단가 +8% 이상이면 "수급 경색"으로 위험.',
                    '공급 구조(취약성) — 공급사 1곳이면 "단일", 1위 비중 70% 이상이면 "집중", 50% 미만이면 "분산", 그 사이는 "보통". 태그만으로는 판정에 영향이 없지만, 단일·집중이면서 단가 주의(+8% 이상)이면 위험으로 올립니다.',
                    '금액 비중(영향도) — 금액 = 물량 × VWAP, 기준 기간 전체 금액 대비 비중. 1% 미만인 단위는 신호가 있어도 종합 판정이 최대 "주의"입니다.',
                    '신뢰도 — 두 기간 중 하나라도 선적이 3건 미만이면 단가 신호는 "판단 보류"로 두고, 공급 차질 판정은 그대로 합니다.',
                    '종합 판정 — 위험 신호가 하나라도 있거나 단일·집중 + 단가 주의이면 위험, 주의 신호만 있거나 금액 비중이 작아 위험에서 내려온 경우 주의, 나머지는 안전. 비교 기간에 거래가 없던 단위는 "신규", 단가가 판단 보류이고 공급 변화도 없으면 "데이터 부족".',
                    '결과 읽는 법 — "먼저 볼 것"에서 금액 비중이 큰 위험 단위 3개와 사유를 확인 → 결과 표(판정 순, 같은 판정은 금액 비중 순)에서 전체를 훑어보기 → 위험·주의 단위의 상세를 펼쳐 공급사 비중·이탈/신규 공급사·대체율·단가·물량·선적 건수 확인 → 더 깊이 볼 때는 집중도 리스크 진단 / 신규·이탈 거래처 추적 / 가격 추세 & 계절성 메뉴로 이동.',
                ],
                'tips': [
                    '기간 길이가 달라도 공정하도록 물량과 대체율은 월평균으로 비교합니다.',
                    '점검 단위가 품목이 아닐 때(원산지 등)는 한 단위 안에 여러 품목이 섞여 있어, 품목 구성 변화가 단가 변화에 영향을 줄 수 있습니다.',
                    '모든 기준값은 "기준값 조정"에서 바꿀 수 있고, 결과는 Excel(점검 결과 + 공급사 비중 시트)과 PDF로 내려받을 수 있습니다.',
                ],
            },
            {
                'heading': '🏆 ③ 시장 경쟁력 분석',
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
                'heading': '🔀 ④ 공급망 흐름도 (Sankey)',
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
                'heading': '⚠️ ⑤ 집중도 리스크 진단',
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
                'heading': '📈 ⑥ 가격 추세 & 계절성',
                'image': 'p5_season.png',
                'intro': '특정 품목의 단가가 시기별로 어떻게 움직이는지, 언제 사는 게 유리한지 알고 싶을 때 씁니다. 구매 시점을 계획할 때 참고하세요.',
                'steps': [
                    '전체 시장 데이터를 업로드합니다.',
                    '품목을 검색해 하나 이상 선택합니다 (포장 단위·끝의 제품 코드만 다른 품목명은 한 품목으로 합쳐져 있습니다 — 도입 효과 분석과 같은 기준). 일반 명칭 품목(예: [혼합분유] 혼합분유)은 HS코드 10단위로 구분하고, 같은 HS 안의 세부 스펙은 구분하지 않습니다.',
                    '전년 비교 여부와 비교 축(원산지별/공급사별)을 설정합니다.',
                    '"추세 그리기"를 클릭합니다.',
                ],
                'results': [
                    '최근월 평균단가, 전년 동월 대비, 연중 최고가 대비 현재 위치.',
                    '계절 고점 자동 감지 — 매년 특정 시기에 비싸지는 패턴이 있는지.',
                    '"○월에 사면 가장 저렴했다" 같은 인사이트 문장이 자동 생성됩니다.',
                    '포장 구성 변화 표시 — 직전 달보다 주력 포장 단위가 바뀌었거나 비중이 20%p 이상 움직인 달에 ◆ 마커가 찍히고, 마우스를 올리면 "포장 구성 변화: 25KG 80% → 600KG 60%"처럼 보여줍니다. 대포장·소포장이 섞이면 합친 단가가 포장 구성 때문에 흔들릴 수 있습니다.',
                    '"포장 단위별로 보기"를 켜면 같은 차트에 포장 단위별 단가 선이 따로 그려집니다 (포장 표기가 없는 거래는 "표기 없음").',
                ],
                'tips': ['데이터가 1년치뿐이면 "이 계절 패턴은 우연일 수 있다"는 경고가 함께 표시됩니다 — 2년 이상 데이터가 쌓이면 신뢰도가 높아집니다.'],
            },
            {
                'heading': '🔀 ⑦ 신규·이탈 거래처 추적',
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
                'heading': '🔬 ⑧ 신규사업 스코어러',
                'image': 'p7_scorer.png',
                'intro': '시장 전체 데이터에서 "최근 급상승한 기존 품목"과 "완전히 새로 등장한 품목"을 찾아, 새로운 소싱 기회의 우선순위를 매기고 싶을 때 씁니다. 다른 메뉴들이 "우리가 이미 하는 거래"를 들여다본다면, 이 메뉴는 "아직 안 하고 있지만 해볼 만한 것"을 찾아줍니다.',
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
                'heading': '🧭 ⑨ 범용상품 스크리너',
                'intro': '여러 HS코드가 섞인 시장 데이터에서, 시장이 열려 있어 공급사를 바꾸기 쉽고(범용성) 같은 시기·같은 원산지 안에서도 단가 차이가 큰(격차) 품목을 찾고 싶을 때 씁니다. 입찰이나 대체 소싱으로 단가를 낮출 여지가 큰 품목을 고르는 첫 단계입니다.',
                'steps': [
                    '여러 HS코드·여러 수입사가 섞인 시장 데이터를 업로드합니다 (날짜·수입사·공급사·원산지·HS코드·물량·단가 필요).',
                    '분석 기간(기본: 최근 12개월), 건당 최소 중량(기본 1,000kg), HS별 최소 거래 건수를 정합니다.',
                    '필요하면 사이드바에서 범용성 점수 가중치와 분류 경계(기본 60/30, 격차 50)를 조정합니다.',
                    '"스크리닝 실행"을 클릭합니다.',
                ],
                'results': [
                    '지표(HS 10단위별) — 수입사 수, CR3, HHI, 해외 공급사 수, 원산지 수, 거래 발생 월 수, 수입사당 연간 건수, 단가 CV, 격차율, 연간 수입액.',
                    '범용성 점수 — 수입사 수↑·CR3↓·공급사 수↑·원산지 수↑·거래 월 수↑·단가 CV↓를 데이터셋 안에서 백분위로 바꿔 가중 평균한 값.',
                    '격차 점수 — 같은 달·같은 원산지 안의 단가 (P90−P10)/P50을 물량으로 가중 평균한 격차율의 백분위.',
                    '분류 — 범용성 상위 & 격차 상위 = A(공략 1순위), 범용성 상위 & 격차 하위 = C, 범용성 중간 = B, 범용성 하위 = D.',
                    '4분면 산점도(버블 = 연간 수입액), 랭킹 표, CSV와 PDF 리포트 다운로드.',
                    '품목을 고르고 "입찰 대응 분석 열기"를 누르면 같은 파일과 HS코드가 입찰 대응 분석으로 넘어갑니다.',
                ],
                'tips': [
                    '전처리: 중복 행(신고일·수입자·공급자·중량·금액 동일) 제거, 건당 최소 중량 미만 제외, HS별 단가 P1~P99 밖 제외. 평균 단가는 모두 물량가중(VWAP)입니다.',
                    '점수는 업로드한 데이터셋 안에서의 상대 순위라, 다른 품목군과 섞어 올리면 순위가 달라집니다.',
                ],
            },
            {
                'heading': '📑 ⑩ 입찰 대응 분석',
                'intro': '한 품목(HS·스펙)에 대해 입찰 제안서를 준비할 때 씁니다. 원산지·해외 공급사별 단가와 거래 실적을 비교해 검증된 대체 공급처를 찾고, 후보 공급사가 한국에 얼마나 꾸준히 공급해왔는지 확인합니다.',
                'steps': [
                    '시장 데이터를 업로드하거나, 범용상품 스크리너에서 품목을 넘겨받습니다 (HS코드가 자동 선택됩니다).',
                    'HS코드, 품명 포함/제외 키워드(같은 HS 안의 스펙 분리), 기간(기본: 최근 12개월), 건당 최소 중량을 정합니다.',
                    '[대체 공급처] 탭에서 검증 기준(최근 12개월 3건·10톤·한국 수입사 2곳 이상)을 확인하고, 필요하면 원산지별 관세율을 입력합니다.',
                    '기준(현 납품) 공급사와 연간 물량(톤)을 넣으면 후보별 절감률과 연간 절감액이 계산됩니다.',
                    '[공급 안정성] 탭에서 공급사를 골라 최근 36개월 선적 실적을 확인하고, 아래에서 PDF 리포트를 만듭니다.',
                ],
                'results': [
                    '원산지별·공급사별 비교표 — VWAP, 비교단가, 건수, 총중량, 거래 월 수, 공급받는 한국 수입사 수, 최근 거래일, 검증 여부.',
                    '관세 — 관세율을 넣으면 비교단가 = CIF × (1 + 세율), 비워두면 CIF 그대로. 하나라도 비어 있으면 "관세 미반영" 배지가 화면과 PDF에 표시됩니다.',
                    '공급 안정성 — 월별 선적 막대(선적 없는 달은 빗금), 공백 월 수, 연평균 물량, 거래 한국 수입사 수, 단가 CV, 연도별 요약과 자동 요약 문장.',
                    'PDF 리포트 — 표지 → 요약(KPI 4개 + 핵심 결론 3줄 + 관세 배지) → 대체 공급처 → 공급 안정성(공급사별 1페이지) → 방법론·데이터 한계 → 클로징.',
                ],
                'tips': [
                    '"한국 수입사명 블라인드"를 켜두면 화면·CSV·PDF에서 한국 수입사명이 물량 순으로 수입사 A/B/C로 바뀝니다. 해외 공급사명은 그대로 보입니다.',
                ],
            },
            {
                'heading': '🧩 ⑪ 자유 피벗 빌더',
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
                'tips': [
                    '도입 효과 분석, 범용상품 스크리너, 입찰 대응 분석의 PDF는 새 리포트 양식(표지 → 요약 KPI·핵심 결론 → 본문 → 방법론·데이터 한계 → 클로징, Pretendard 글꼴, Tridge 브랜드 색)으로 만들어집니다. 다른 메뉴도 순서대로 새 양식으로 바뀔 예정입니다.',
                    '지금 읽고 계신 이 가이드 자체도 사이드바의 "📖 사용법 가이드" 버튼으로 언제든 다시 받을 수 있습니다.',
                ],
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
                    'Columns beyond the standard set (e.g. HS Code, Incoterm) are auto-detected and added as axis/filter options in several menus (Portfolio Health Check, Supply Chain Flow, Concentration Risk, New/Lost Partners, Pivot Builder, New Business Scorer, Commodity Screener, Tender Response Analysis).',
                ],
            },
            {
                'heading': '💲 ① Adoption Impact',
                'image': 'p1_customer.png',
                'intro': "Use this when you want to prove, in numbers, how much a company's purchase price dropped after a contract started. It's usually the first screen opened for renewal or performance reporting. It is meant for before/after comparison around the Tridge adoption (contract) date — to check the current state, use Portfolio Health Check.",
                'steps': [
                    'Upload transaction data (if multiple companies are mixed in, the most frequent one is used automatically).',
                    'Select the contract start date.',
                    'Adjust the "Minor-change threshold" ($, default 10) if needed.',
                    'Click "Run Analysis".',
                ],
                'results': [
                    'Net savings / total savings / total cost increase — the net sum of per-group savings (price difference before vs. after × post-contract volume), plus the savings and cost-increase totals with item counts.',
                    'Cards by product group — split into Savings, Price increases, and Minor change (within ±$10, adjustable), largest amounts first. Each group shows the first 12 cards; "Show all" expands the rest. Items with no purchases after the contract are listed separately as "No purchases after contract" and excluded from the totals.',
                    'Detailed savings table — the same groups appear in a "Group" column, and the results can be downloaded as a new-format PDF report (cover → 4 KPIs and 3 key findings → savings chart and tables by item group → items to check → methodology → closing). When mix-changed items exist, net savings excluding them is shown as well. Use "Legacy-format PDF" if you need the previous format.',
                    'Import item groups — product names that differ only in pack size (360 tablets, 30 sachets, 900 g, etc.) or a trailing product code form one item group; different flavors, brands, or targets stay separate. When the main pack size changes or a pack size share moves 20%p or more before vs. after the contract, the group is flagged as a "pack size change" (mix change), and net savings excluding it is also shown. Generic-name products (e.g. [Mixed Milk Powder] Mixed Milk Powder) are separated by 10-digit HS code; detailed specs within the same HS code are not distinguished.',
                    'Monthly import trend and last-3-month share.',
                    'New/lost products, origins, and suppliers since the contract.',
                ],
            },
            {
                'heading': '🩺 ② Portfolio Health Check',
                'intro': "Use this when you want a quick scan of which products are at risk right now, regardless of any contract. It looks for change signals per product (or origin, HS code, etc.) in one company's import data, then weighs supply-structure vulnerability, spend share (impact), and shipment count (reliability) to rate each unit At risk / Caution / Safe.",
                'steps': [
                    'Upload transaction data (date, importer, supplier, product, volume, and unit price columns are required).',
                    'Select the importer to check (sorted by volume; applied automatically if the file has only one importer).',
                    'Choose the check unit — Product (default), Origin, or an extra column in the file (e.g. HS code). Product names that differ only in pack size (360 tablets, 25 KG, etc.) or a trailing product code form one unit (same rule as Adoption Impact), shown under a cleaned name; the original names are listed under "Original product names by unit" below the results table. Generic-name products (e.g. [Mixed Milk Powder] Mixed Milk Powder) are separated by 10-digit HS code; detailed specs within the same HS code are not distinguished.',
                    'Select the base period (default: last 6 months up to the latest date in the data) and the comparison period (default: the 6 months before that). Adjust thresholds under "Adjust thresholds" if needed.',
                    'Click "Run Health Check".',
                ],
                'results': [
                    'Unit price change (change signal) — change in volume-weighted average price (VWAP) between the periods. At risk at +15% or more, Caution at +8% or more; decreases are Safe. It may reflect market-wide price moves, so check whether it is a market factor.',
                    'Supply disruption (change signal) — a major supplier (10%+ share in the comparison period) is gone in the base period and new suppliers replaced less than 50% of its monthly volume: At risk. If 50% or more was replaced it is not a signal but tagged "Switched", or "Switched · better price" if the unit price also fell.',
                    'Volume change — change in monthly average volume. Shown as a number only and not used on its own; but volume -15% or less together with unit price +8% or more is a "supply squeeze" (At risk).',
                    'Supply structure (vulnerability) — one supplier is "Single", top supplier 70%+ is "Concentrated", below 50% is "Diversified", otherwise "Moderate". The tag alone does not change the status, but Single/Concentrated with a price Caution (+8% or more) is raised to At risk.',
                    'Spend share (impact) — spend = volume × VWAP, as a share of total base-period spend. Units under 1% are capped at "Caution" even with a risk signal.',
                    'Reliability — if either period has fewer than 3 shipments, the unit price signal is put "On hold"; the supply disruption check still applies.',
                    'Overall — At risk if any risk signal exists or Single/Concentrated + price Caution; Caution if only caution signals exist or the unit was lowered from At risk because of small spend share; otherwise Safe. Units with no transactions in the comparison period are "New"; units whose price is on hold with no supply change are "Insufficient data".',
                    'How to read the results — check the top 3 at-risk units by spend share and their reasons in "Look at these first" → scan the results table (by status, then spend share) → expand details for at-risk and caution units (supplier share, lost/new suppliers, replacement rate, price, volume, shipments) → for a deeper look, go to Concentration Risk / New/Lost Trading Partners / Price Trend & Seasonality.',
                ],
                'tips': [
                    'Volume and replacement rate are compared as monthly averages so periods of different length stay comparable.',
                    'When the check unit is not a product (e.g. origin), each unit mixes several products, so a change in product mix can affect the unit price change.',
                    'All thresholds can be changed under "Adjust thresholds", and results can be downloaded as Excel (results + supplier share sheets) and PDF.',
                ],
            },
            {
                'heading': '🏆 ③ Market Competitiveness Analysis',
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
                'heading': '🔀 ④ Supply Chain Flow (Sankey)',
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
                'heading': '⚠️ ⑤ Concentration Risk Diagnosis',
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
                'heading': '📈 ⑥ Price Trend & Seasonality',
                'image': 'p5_season.png',
                'intro': "Use this to see how a product's price moves over time and when it is cheapest to buy — useful for planning purchase timing.",
                'steps': [
                    'Upload the full market data.',
                    'Search and select one or more products (names that differ only in pack size or a trailing product code are combined into one product — same rule as Adoption Impact). Generic-name products (e.g. [Mixed Milk Powder] Mixed Milk Powder) are separated by 10-digit HS code; detailed specs within the same HS code are not distinguished.',
                    'Set the previous-year overlay and comparison axis.',
                    'Click "Draw trend".',
                ],
                'results': [
                    "Latest month's average price, year-over-year change, and position vs. the yearly peak.",
                    'Automatic seasonal-peak detection.',
                    'Auto-generated insight sentences like "buying in March was historically cheapest."',
                    'Pack mix changes — a ◆ marker appears in months where the main pack size changed or a share moved 20%p or more vs. the previous month; hover to see e.g. "Pack mix change: 25KG 80% → 600KG 60%". When bulk and small packs mix, the combined price can move because of the pack mix.',
                    'Turn on "Show by pack size" to draw a separate price line per pack size on the same chart (trades without a pack size appear as "No pack size").',
                ],
                'tips': ['With only one year of data, a warning notes the seasonal pattern could be coincidental — confidence improves with 2+ years of history.'],
            },
            {
                'heading': '🔀 ⑦ New/Lost Trading Partners',
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
                'heading': '🔬 ⑧ New Business Scorer',
                'image': 'p7_scorer.png',
                'intro': 'Use this to find "recently surging existing items" and "brand-new entrants" across the whole market, ranked by opportunity. While the other menus examine trade you already do, this one surfaces opportunities you are not yet pursuing.',
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
                'heading': '🧭 ⑨ Commodity Screener',
                'intro': 'Use this on market data with many HS codes to find items where the market is open enough to switch suppliers (universality) and unit prices vary widely within the same month and origin (gap) — the first step in picking items where tendering or alternative sourcing can lower prices.',
                'steps': [
                    'Upload market data with multiple HS codes and importers (date, importer, supplier, origin, HS code, volume, and unit price required).',
                    'Set the period (default: last 12 months), minimum weight per shipment (default 1,000 kg), and minimum shipments per HS.',
                    'If needed, adjust universality weights and class boundaries (default 60/30, gap 50) in the sidebar.',
                    'Click "Run Screening".',
                ],
                'results': [
                    'Metrics per 10-digit HS — importers, CR3, HHI, overseas suppliers, origins, months with trade, shipments per importer per year, unit price CV, gap rate, annual import value.',
                    'Universality score — weighted average of dataset percentiles for importers↑, CR3↓, suppliers↑, origins↑, months with trade↑, and unit price CV↓.',
                    'Gap score — percentile of the volume-weighted (P90−P10)/P50 unit price within the same month and origin.',
                    'Classes — high universality & high gap = A (top target), high universality & low gap = C, medium universality = B, low universality = D.',
                    'Quadrant scatter (bubble = annual import value), ranking table, CSV and PDF report downloads.',
                    'Pick an item and click "Open Tender Response Analysis" to send the same file and HS code there.',
                ],
                'tips': [
                    'Preprocessing: duplicates removed (same date, importer, supplier, weight, value), shipments under the minimum weight excluded, unit prices outside P1–P99 per HS excluded. All average prices are volume-weighted (VWAP).',
                    'Scores are relative ranks within the uploaded dataset; mixing other product groups changes the ranking.',
                ],
            },
            {
                'heading': '📑 ⑩ Tender Response Analysis',
                'intro': 'Use this when preparing a bid for one item (HS and spec). It compares prices and track records by origin and overseas supplier to find verified alternative suppliers, and shows how consistently candidates have supplied Korea.',
                'steps': [
                    'Upload market data, or receive an item from the Commodity Screener (the HS code is selected automatically).',
                    'Set the HS code, product keywords to include/exclude (to split specs within an HS code), the period (default: last 12 months), and the minimum weight per shipment.',
                    'In [Alternative suppliers], check the verification criteria (3+ shipments, 10+ tons, 2+ Korean importers in the last 12 months) and enter tariff rates per origin if needed.',
                    'Enter the baseline (current) supplier and annual volume (tons) to see savings rates and annual savings per candidate.',
                    'In [Supply stability], pick a supplier to review its last 36 months of shipments, then create the PDF report below.',
                ],
                'results': [
                    'Comparison tables by origin and supplier — VWAP, comparison price, shipments, total weight, months with trade, Korean importers supplied, latest trade, verification.',
                    'Tariffs — with a rate, comparison price = CIF × (1 + rate); blank means CIF as is. If any rate is blank, a "Tariff not applied" badge appears on screen and in the PDF.',
                    'Supply stability — monthly shipment bars (hatched for months without shipments), gap months, average annual volume, Korean importers supplied, unit price CV, yearly summary, and an auto-generated summary sentence.',
                    'PDF report — cover → summary (4 KPIs + 3 key findings + tariff badge) → alternative suppliers → supply stability (one page per supplier) → methodology and data limits → closing.',
                ],
                'tips': [
                    'With "Blind Korean importer names" on, Korean importer names become Importer A/B/C (by volume) on screen, in CSV, and in the PDF. Overseas supplier names stay visible.',
                ],
            },
            {
                'heading': '🧩 ⑪ Free Pivot Builder',
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
                'tips': [
                    'PDFs from Adoption Impact, the Commodity Screener, and Tender Response Analysis use the new report template (cover → summary KPIs and key findings → body → methodology and data limits → closing, Pretendard font, Tridge brand colors). Other menus will move to the new template step by step.',
                    'This very guide can be re-downloaded anytime from the "📖 User Guide" button in the sidebar.',
                ],
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
    Paragraph = _pdf_safe_paragraph(Paragraph)  # ①~⑨·이모지 빈칸 방지

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
install_bundled_font_for_charts()

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

if 'health_raw_df' not in st.session_state:
    st.session_state.health_raw_df = None
    st.session_state.health_headers = None
    st.session_state.health_result = None

if 'scr_raw_df' not in st.session_state:
    st.session_state.scr_raw_df = None
    st.session_state.scr_headers = None
    st.session_state.scr_result = None

if 'bid_raw_df' not in st.session_state:
    st.session_state.bid_raw_df = None
    st.session_state.bid_headers = None
    st.session_state.bid_hs_preset = None

# --- [비밀번호 기능] 비밀번호가 맞을 때까지 이 아래 앱 본문을 그리지 않음 ---
if not check_password():
    st.stop()
# --- [비밀번호 기능] 끝 ---

# --- 사이드바 메뉴 (원본 UI + 언어 토글) ---
with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.25rem 0 1rem 0;">
            <svg width="130" height="22" viewBox="0 0 600 101.665" role="img" aria-label="Tridge logo">
                <g fill="#FFFFFF">
                    <path fill-rule="nonzero" d="M 84.8125 29.132812 C 92.011719 29.132812 97.847656 23.296875 97.847656 16.097656 C 97.847656 8.898438 92.011719 3.0625 84.8125 3.0625 C 77.613281 3.0625 71.777344 8.898438 71.777344 16.097656 C 71.777344 18.289062 72.324219 20.351562 73.277344 22.164062 L 56.109375 39.335938 C 54.292969 38.375 52.226562 37.824219 50.027344 37.824219 C 42.828125 37.824219 36.992188 43.660156 36.992188 50.859375 C 36.992188 53.058594 37.542969 55.125 38.503906 56.941406 L 21.34375 74.097656 C 19.527344 73.136719 17.460938 72.589844 15.261719 72.589844 C 8.066406 72.589844 2.230469 78.425781 2.230469 85.625 C 2.230469 92.824219 8.066406 98.660156 15.261719 98.660156 C 22.464844 98.660156 28.296875 92.824219 28.296875 85.625 C 28.296875 83.445312 27.757812 81.394531 26.8125 79.589844 L 43.992188 62.410156 C 45.800781 63.355469 47.847656 63.894531 50.027344 63.894531 C 52.222656 63.894531 54.285156 63.347656 56.101562 62.390625 L 73.277344 79.566406 C 72.324219 81.375 71.777344 83.4375 71.777344 85.625 C 71.777344 92.824219 77.613281 98.660156 84.8125 98.660156 C 92.011719 98.660156 97.847656 92.824219 97.847656 85.625 C 97.847656 78.425781 92.011719 72.589844 84.8125 72.589844 C 82.625 72.589844 80.566406 73.132812 78.757812 74.085938 L 61.570312 56.902344 C 62.519531 55.09375 63.0625 53.042969 63.0625 50.859375 C 63.0625 48.683594 62.519531 46.632812 61.574219 44.828125 L 78.761719 27.640625 C 80.574219 28.589844 82.625 29.132812 84.8125 29.132812 " />
                    <path fill-rule="nonzero" d="M 128.640625 19.433594 L 158.070312 19.535156 L 158.070312 98.851562 L 178.164062 98.851562 L 178.164062 19.433594 L 207.59375 19.433594 L 207.59375 2.675781 L 128.640625 2.675781 Z M 128.640625 19.433594 " />
                    <path fill-rule="nonzero" d="M 267.769531 41.996094 C 266.417969 44.1875 264.371094 45.847656 261.957031 46.742188 C 258.871094 47.917969 255.585938 48.476562 252.289062 48.394531 L 238.554688 48.394531 L 238.554688 19.363281 L 252.269531 19.363281 C 255.585938 19.242188 258.902344 19.800781 261.996094 21.015625 C 264.410156 22 266.429688 23.746094 267.769531 25.976562 C 269.109375 28.460938 269.777344 31.242188 269.6875 34.0625 C 269.777344 36.832031 269.117188 39.570312 267.769531 41.996094 M 274.484375 61.066406 C 279.445312 58.84375 283.605469 55.160156 286.433594 50.527344 C 289.292969 45.503906 290.726562 39.785156 290.5625 34 C 290.683594 28.230469 289.285156 22.527344 286.496094 17.476562 C 283.757812 12.738281 279.648438 8.925781 274.71875 6.542969 C 268.933594 3.84375 262.59375 2.515625 256.214844 2.675781 L 218.207031 2.675781 L 218.207031 98.832031 L 238.554688 98.832031 L 238.554688 64.71875 L 253.328125 64.71875 L 271.554688 98.851562 L 293.992188 98.851562 L 273.570312 61.433594 C 273.875 61.308594 274.183594 61.195312 274.484375 61.066406 " />
                    <path fill-rule="nonzero" d="M 304.59375 98.851562 L 324.921875 98.851562 L 324.921875 2.675781 L 304.59375 2.675781 Z M 304.59375 98.851562 " />
                    <path fill-rule="nonzero" d="M 399.320312 68.207031 C 397.464844 72.507812 394.1875 76.027344 390.039062 78.210938 C 385.308594 80.492188 380.097656 81.597656 374.851562 81.433594 L 362 81.433594 L 361.929688 20.144531 L 374.882812 20.144531 C 380.117188 19.980469 385.320312 21.078125 390.046875 23.351562 C 394.175781 25.5 397.441406 29 399.320312 33.261719 C 401.589844 38.78125 402.65625 44.714844 402.445312 50.679688 C 402.65625 56.671875 401.589844 62.648438 399.320312 68.207031 M 401.003906 8.429688 C 393.304688 4.421875 384.710938 2.445312 376.039062 2.675781 L 341.667969 2.675781 L 341.667969 98.851562 L 375.765625 98.851562 C 384.519531 99.054688 393.183594 97.035156 400.921875 92.980469 C 407.859375 89.226562 413.5 83.464844 417.109375 76.453125 C 421.066406 68.441406 423.003906 59.585938 422.761719 50.648438 C 423.003906 41.75 421.078125 32.9375 417.132812 24.953125 C 413.53125 17.953125 407.910156 12.191406 401.003906 8.429688 " />
                    <path fill-rule="nonzero" d="M 478.167969 62.648438 L 498.8125 62.648438 C 498.800781 66.164062 497.921875 69.621094 496.222656 72.703125 C 494.46875 75.722656 491.851562 78.148438 488.707031 79.671875 C 485.023438 81.40625 480.976562 82.257812 476.910156 82.136719 C 472.230469 82.257812 467.628906 80.980469 463.691406 78.453125 C 459.816406 75.816406 456.824219 72.082031 455.070312 67.730469 C 452.898438 62.304688 451.863281 56.492188 452.015625 50.648438 C 451.851562 44.816406 452.90625 39.023438 455.121094 33.625 C 456.90625 29.316406 459.917969 25.621094 463.78125 23.007812 C 467.628906 20.519531 472.128906 19.242188 476.707031 19.332031 C 479.140625 19.304688 481.5625 19.636719 483.90625 20.328125 C 485.996094 20.9375 487.964844 21.898438 489.742188 23.167969 C 491.445312 24.394531 492.914062 25.917969 494.082031 27.660156 C 495.320312 29.527344 496.261719 31.578125 496.863281 33.738281 L 496.863281 33.757812 L 517.527344 33.757812 C 516.917969 29.144531 515.394531 24.699219 513.0625 20.683594 C 510.757812 16.726562 507.726562 13.246094 504.125 10.417969 C 500.371094 7.484375 496.140625 5.222656 491.617188 3.722656 C 486.707031 2.117188 481.574219 1.316406 476.410156 1.359375 C 470.34375 1.289062 464.339844 2.414062 458.710938 4.664062 C 453.292969 6.875 448.414062 10.214844 444.398438 14.464844 C 440.15625 18.925781 436.910156 24.222656 434.839844 30.015625 C 432.476562 36.710938 431.320312 43.78125 431.433594 50.882812 C 431.199219 59.96875 433.15625 68.976562 437.132812 77.15625 C 440.683594 84.253906 446.222656 90.167969 453.082031 94.1875 C 460.3125 98.285156 468.519531 100.34375 476.828125 100.171875 C 484.292969 100.292969 491.667969 98.589844 498.3125 95.210938 C 504.449219 92.046875 509.5625 87.195312 513.050781 81.230469 C 516.773438 74.597656 518.621094 67.082031 518.40625 59.472656 L 518.40625 47.363281 L 478.167969 47.363281 Z M 478.167969 62.648438 " />
                    <path fill-rule="nonzero" d="M 552.578125 82.0625 L 552.578125 59.097656 L 593.769531 59.097656 L 593.769531 42.339844 L 552.65625 42.339844 L 552.65625 19.433594 L 597.109375 19.433594 L 597.109375 2.675781 L 532.320312 2.675781 L 532.320312 98.851562 L 597.1875 98.851562 L 597.1875 82.0625 Z M 552.578125 82.0625 " />
                </g>
            </svg>
            <div style="color:rgba(255,255,255,0.55);font-size:10px;letter-spacing:0.4px;margin-top:5px;">SOURCING ANALYZER</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    lang_choice = st.radio(
        "🌐 Language / 언어",
        options=['한국어', 'English'],
        index=0 if st.session_state.lang == 'ko' else 1,
        horizontal=True,
    )
    st.session_state.lang = 'ko' if lang_choice == '한국어' else 'en'
    st.markdown("---")
    _menu_keys = ['menu_opt_customer', 'menu_opt_health', 'menu_opt_market', 'menu_opt_flow', 'menu_opt_risk', 'menu_opt_season',
                  'menu_opt_churn', 'menu_opt_scorer', 'menu_opt_screener', 'menu_opt_bid', 'menu_opt_pivot']
    selected = option_menu(
        menu_title=T('app_menu_title'),
        options=[T(k) for k in _menu_keys],
        icons=["person-bounding-box", "heart-pulse", "graph-up-arrow", "diagram-3", "exclamation-triangle", "calendar3", "arrow-left-right", "binoculars", "funnel", "clipboard-check", "grid-3x3"],
        menu_icon="cast",
        default_index=0,
        manual_select=st.session_state.pop('_menu_jump', None),  # 다른 메뉴에서 코드로 이동할 때만 값이 있음
        key='main_menu',
        styles={
            "container": {"padding": "0", "background-color": "#26224F"},
            "icon": {"color": "#C8C5E8", "font-size": "15px"},
            "nav-link": {"color": "#C8C5E8", "font-size": "13px", "text-align": "left", "margin": "2px 0", "border-radius": "6px", "--hover-color": "#332E68"},
            "nav-link-selected": {"background-color": "#4F46E5", "color": "#FFFFFF", "font-weight": "500"},
            "menu-title": {"color": "#FFFFFF"},
        },
    )
    # key를 고정하면 언어를 바꾼 직후 이전 언어 라벨이 돌아올 수 있어, 현재 언어 라벨로 맞춘다.
    for _k in _menu_keys:
        if selected in (TEXTS['ko'][_k], TEXTS['en'][_k]):
            selected = T(_k)
            break
    st.markdown("---")
    try:
        _guide_pdf_bytes = build_user_guide_pdf(st.session_state.lang)
        st.download_button(T('guide_download_btn'), data=_guide_pdf_bytes, file_name="user_guide.pdf", mime="application/pdf", width="stretch")
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
                    # 공통 읽기 함수 (CSV 인코딩 자동 판별, HS 등 코드 컬럼 앞자리 0 보존)
                    df_for_check = read_uploaded_table(uploaded_file)

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
                minor_threshold_input = st.number_input(T('p1_minor_threshold_label'), min_value=0.0, value=10.0, step=5.0)
                submitted = st.form_submit_button(T('p1_run_btn'))

            if submitted:
                with st.spinner(T('p1_spinner')):
                    df = df_for_check.copy()

                    # 단가: Unit Price가 없으면 금액(Value) ÷ 중량 (⑨⑩과 같은 공통 함수)
                    _tcols = detect_trade_columns(list(df.columns))
                    st.session_state.p1_price_mismatch = None
                    if _tcols['volume'] and (_tcols['price'] or _tcols['value']):
                        _price, _, _mismatch = derive_unit_price(df, _tcols)
                        df[_tcols['price'] or 'Unit Price'] = _price
                        st.session_state.p1_price_mismatch = _mismatch
                    
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
                    # 품목군 묶기 (F안): 포장 단위·제품 코드를 뗀 이름이 같은 거래끼리만 묶는다.
                    # 유사도로 묶으면 맛·브랜드·대상만 다른 제품까지 섞여(과병합) 공백 없는 키로 정확히 일치시키고,
                    # 떼어낸 포장 단위는 구성 변화(포장 단위 변경) 판정에 쓴다.
                    # 일반 명칭 품목(예: [혼합분유] 혼합분유)은 HS코드 10단위로 한 번 더 나눈다 (product_group_labels).
                    _hs = customer_df[_tcols['hs']] if _tcols['hs'] and _tcols['hs'] in customer_df.columns else None
                    _disp, _key, _pack = product_group_labels(customer_df['product_name'], _hs)
                    customer_df['product_preprocessed'] = _disp
                    customer_df['pack_unit'] = _pack
                    customer_df['product_key'] = _key
                    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2), token_pattern=r'(?u)\b\w+\b')
                    tfidf_matrix = vectorizer.fit_transform(customer_df['product_key'])
                    # 키가 한 단어라 코사인거리 0(같은 키)끼리만 묶인다. min_samples=3: 거래 3건 미만 품목명은 제외
                    dbscan = DBSCAN(eps=0.6, min_samples=3, metric='cosine')
                    cluster_labels = dbscan.fit_predict(tfidf_matrix)
                    cluster_name_map = get_cluster_name(cluster_labels, customer_df['product_preprocessed'])
                    customer_df['cluster'] = cluster_labels
                    customer_df['cluster_name'] = customer_df['cluster'].map(cluster_name_map)
                    plot_df = customer_df[customer_df['cluster'] != -1].copy()

                    contract_date = pd.to_datetime(contract_date_input)
                    before_contract_df = plot_df[plot_df['date'] < contract_date]
                    after_contract_df = plot_df[plot_df['date'] >= contract_date]

                    def _vwap(g):
                        """물량가중평균단가. 거래건수가 아니라 실제 산 물량 기준으로 가중해야
                        '소량 고가 거래 1건'이 '대량 저가 거래 1건'과 똑같은 비중으로 평균을 왜곡하지 않는다."""
                        tv = g['volume'].sum()
                        return (g['unit_price'] * g['volume']).sum() / tv if tv > 0 else np.nan

                    def _clean_and_vwap(g):
                        """클러스터+기간별로 이상치를 먼저 제거한 뒤 물량가중평균을 계산한다
                        (전체를 한번에 돌리면 원래 비싼 품목이 다른 클러스터들 사이에서 '이상치'로 잘못 걸러질 수 있음)."""
                        g_clean = remove_outliers_iqr(g, 'unit_price', cap_percent=0.07, iqr_multiplier=1.5)
                        return _vwap(g_clean)

                    def _clean_volume_sum(g):
                        g_clean = remove_outliers_iqr(g, 'unit_price', cap_percent=0.07, iqr_multiplier=1.5)
                        return g_clean['volume'].sum()

                    if not before_contract_df.empty:
                        avg_price_before = before_contract_df.groupby('cluster_name').apply(_clean_and_vwap).rename('avg_price_before')
                    else:
                        avg_price_before = pd.Series(dtype=float, name='avg_price_before')

                    if not after_contract_df.empty:
                        avg_price_after = after_contract_df.groupby('cluster_name').apply(_clean_and_vwap).rename('avg_price_after')
                        volume_after = after_contract_df.groupby('cluster_name').apply(_clean_volume_sum).rename('volume_after')
                    else:
                        avg_price_after = pd.Series(dtype=float, name='avg_price_after')
                        volume_after = pd.Series(dtype=float, name='volume_after')

                    savings_df = pd.concat([avg_price_before, avg_price_after, volume_after], axis=1)
                    # 계약 전 단가가 있는 품목군만 비교 대상. 계약 후 구매가 없으면(물량 0) 버리지 않고
                    # '계약 후 구매 없음'으로 남겨 따로 표시한다 (절감액 0, 합계에서는 제외).
                    savings_df = savings_df[savings_df['avg_price_before'].notna()].copy()
                    no_after = savings_df['avg_price_after'].isna() | ~(savings_df['volume_after'] > 0)
                    savings_df.loc[no_after, 'volume_after'] = 0.0
                    savings_df['savings'] = ((savings_df['avg_price_before'] - savings_df['avg_price_after']) * savings_df['volume_after']).where(~no_after, 0.0)

                    # 구성 변화 체크 (포장 단위 기준): 계약 전후로 품목군 안의 포장 단위별 물량 비중이 크게 바뀌면
                    # kg당 단가 차이가 "협상 성과"가 아니라 "포장 변경" 때문일 수 있어 경고한다.
                    _mix = [pack_mix_change(before_contract_df[before_contract_df['cluster_name'] == idx],
                                            after_contract_df[after_contract_df['cluster_name'] == idx]) for idx in savings_df.index]
                    for _i, _col in enumerate(['mix_changed', 'mix_kind', 'mix_from', 'mix_to', 'mix_b', 'mix_a']):
                        savings_df[_col] = [m[_i] for m in _mix] if _mix else []
                    savings_df['mix_changed'] = savings_df['mix_changed'].astype(bool)
                    savings_df = savings_df.sort_values('savings', ascending=False)
                    total_savings = savings_df.loc[savings_df['volume_after'] > 0, 'savings'].sum()

                    st.session_state.customer_name = customer_name
                    st.session_state.plot_df = plot_df
                    st.session_state.customer_df = customer_df
                    st.session_state.contract_date = contract_date
                    st.session_state.tfidf_matrix = tfidf_matrix
                    st.session_state.savings_df = savings_df
                    st.session_state.total_savings = total_savings
                    st.session_state.p1_minor_threshold = float(minor_threshold_input)
                    st.session_state.analysis_done = True
                    
                st.success(T('p1_analysis_complete_success', customer_name=customer_name))
                st.rerun()

    if st.session_state.analysis_done:
        # 안쪽에 그룹별 expander를 넣어야 해서(Streamlit은 expander 중첩 불가) 이 섹션만 테두리 컨테이너로 감싼다.
        with st.container(border=True):
            st.subheader(T('p1_exp1_title'))
            savings_df = st.session_state.savings_df
            minor_th = st.session_state.get('p1_minor_threshold', 10.0)
            groups = split_savings_groups(savings_df, minor_th)
            saved_total = float(groups['saved']['savings'].sum())
            increased_total = float(-groups['increased']['savings'].sum())
            net_total = float(st.session_state.total_savings)
            kpi_net = (T('p1_kpi_net'), f"${net_total:,.2f}")
            kpi_saved = (T('p1_kpi_saved', n=len(groups['saved'])), f"${saved_total:,.2f}")
            kpi_increased = (T('p1_kpi_increased', n=len(groups['increased'])), f"${increased_total:,.2f}")
            for kcol, (label, value) in zip(st.columns(3), [kpi_net, kpi_saved, kpi_increased]):
                kcol.metric(label, value)
            st.caption(T('p1_total_savings_caption', date=st.session_state.contract_date.date(), th=minor_th))

            st.subheader(T('p1_savings_detail_subheader'))
            group_titles = {
                'saved': T('p1_group_saved', n=len(groups['saved']), amount=saved_total),
                'increased': T('p1_group_increased', n=len(groups['increased']), amount=increased_total),
                'minor': T('p1_group_minor', n=len(groups['minor'])),
            }
            for gkey, expanded in [('saved', True), ('increased', True), ('minor', False)]:
                with st.expander(group_titles[gkey], expanded=expanded):
                    render_savings_cards(groups[gkey], gkey)
                    if gkey == 'minor' and not groups['minor'].empty:
                        st.caption(T('p1_minor_caption', th=minor_th))
            no_purchase_names = [str(n) for n in groups['no_purchase'].index]
            if no_purchase_names:
                st.caption(T('p1_no_purchase_caption', n=len(no_purchase_names), names=', '.join(no_purchase_names)))
            if savings_df['mix_changed'].any():
                _mixed = savings_df[savings_df['mix_changed']]
                _items = ', '.join(f"{n} ({pack_change_text(r.mix_kind, r.mix_from, r.mix_to, r.mix_b, r.mix_a)})" for n, r in _mixed.iterrows())
                st.caption(T('p1_mix_changed_caption', n=len(_mixed), items=_items))
            if st.session_state.get('p1_price_mismatch') and st.session_state.p1_price_mismatch > 0.1:
                st.warning(T('price_mismatch_warning', pct=st.session_state.p1_price_mismatch * 100))

            p1_col_names = {
                'group': T('p1_col_group'),
                'avg_price_before': T('p1_col_avg_price_before'),
                'avg_price_after': T('p1_col_avg_price_after'),
                'volume_after': T('p1_col_volume_after'),
                'savings': T('p1_col_savings'),
                'mix_detail': T('p1_col_mix_detail'),
                'mix_changed': T('p1_col_mix_changed'),
            }
            _table = savings_table_with_group(groups)
            _table['mix_detail'] = [pack_change_text(r.mix_kind, r.mix_from, r.mix_to, r.mix_b, r.mix_a) for r in _table.itertuples()]
            display_savings_df = _table[list(p1_col_names)].rename(columns=p1_col_names)
            st.dataframe(display_savings_df.style.format({
                T('p1_col_avg_price_before'): '${:,.2f}',
                T('p1_col_avg_price_after'): '${:,.2f}',
                T('p1_col_volume_after'): '{:,.0f} KG',
                T('p1_col_savings'): '${:,.2f}',
            }, na_rep='-'))
            st.caption(T('p1_vwap_note'))
            st.caption(T('group_rule_generic_hs'))

            p1_blind = st.checkbox(T('p1_rp_blind_toggle'), value=False, key="p1_rp_blind")
            pdf_c1, pdf_c2 = st.columns([1, 1])
            if pdf_c1.button(T('rp_pdf_btn'), key="p1_pdf_btn"):
                with st.spinner(T('rp_pdf_spinner')):
                    try:
                        _cust = st.session_state.customer_name
                        _label = apply_blind(pd.DataFrame({'c': [_cust]}), 'c', p1_blind, mapping={_cust: T('blind_label', code='A')})['c'].iloc[0]
                        pdf_bytes = build_adoption_report({
                            'savings_df': savings_df, 'groups': groups, 'minor_th': minor_th, 'customer_label': str(_label),
                            'plot_df': st.session_state.plot_df, 'contract_date': st.session_state.contract_date, 'blind': p1_blind,
                        })
                        st.download_button(T('rp_pdf_dl'), data=pdf_bytes, file_name="adoption_impact_report.pdf", mime="application/pdf", key="p1_pdf_dl")
                    except Exception as e:
                        st.error(T('pdf_error_msg', msg=str(e)))
            if pdf_c2.button(T('p1_rp_legacy_btn'), key="p1_pdf_legacy_btn"):
                with st.spinner(T('pdf_generating_msg')):
                    try:
                        def _pdf_rows(g):
                            fmt = lambda v, f: '-' if pd.isna(v) else f.format(v)
                            return pd.DataFrame({
                                T('legend_cluster'): [str(i) for i in g.index],
                                T('p1_col_avg_price_before'): [fmt(v, '${:,.2f}') for v in g['avg_price_before']],
                                T('p1_col_avg_price_after'): [fmt(v, '${:,.2f}') for v in g['avg_price_after']],
                                T('p1_col_volume_after'): [fmt(v, '{:,.0f} KG') for v in g['volume_after']],
                                T('p1_col_savings'): [fmt(v, '${:,.2f}') for v in g['savings']],
                                T('p1_col_mix_changed'): ['⚠ ' + T('p1_mix_changed_badge') if v else '' for v in g['mix_changed']],
                            })
                        pdf_sections = [(group_titles[k], _pdf_rows(groups[k])) for k in ('saved', 'increased', 'minor')]
                        pdf_sections.append((f"{T('p1_grp_no_purchase_label')} ({len(no_purchase_names)})", _pdf_rows(groups['no_purchase'])))
                        pdf_bytes = build_pdf_report(
                            title=f"{T('p1_title')} — {st.session_state.customer_name}",
                            kpi_lines=[f"{label}: {value}" for label, value in [kpi_net, kpi_saved, kpi_increased]]
                                      + [T('p1_total_savings_caption', date=st.session_state.contract_date.date(), th=minor_th)],
                            figs=[], tables=pdf_sections,
                        )
                        st.download_button(T('pdf_download_btn'), data=pdf_bytes, file_name="adoption_impact.pdf", mime="application/pdf", key="p1_pdf_legacy_dl")
                    except Exception as e:
                        st.error(T('pdf_error_msg', msg=str(e)))

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
                st.plotly_chart(fig1, width="stretch")
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
            st.plotly_chart(fig2, width="stretch")
            
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
                    st.plotly_chart(fig_pie, width="stretch")
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
# 페이지 1-2: 포트폴리오 헬스체크
# ==============================================================================
if selected == T('menu_opt_health'):
    st.title(T('ph_title'))
    st.write(T('ph_intro'))

    if st.session_state.health_result is not None:
        st.button(T('ph_reset_btn'), on_click=reset_health_states)

    if st.session_state.health_result is None:
        health_file = st.file_uploader(T('ph_upload_label'), type=['csv', 'xlsx'], key="health_uploader")
        st.caption(T('ph_upload_caption'))
        raw_df = load_uploaded_df(health_file, 'health_raw_df', 'health_headers', 'health_fileid')

        if raw_df is not None:
            headers = st.session_state.health_headers
            cols = detect_trade_columns(headers)
            missing = [k for k in ['date', 'importer', 'exporter', 'product', 'volume'] if not cols[k]]
            if not cols['price'] and not cols['value']:
                missing.append('price')
            if missing:
                st.error(T('ph_missing_cols_error', cols=', '.join(missing)))
                st.stop()
            _mismatch = derive_unit_price(raw_df, cols)[2]
            if _mismatch and _mismatch > 0.1:
                st.warning(T('price_mismatch_warning', pct=_mismatch * 100))

            _imp = raw_df[cols['importer']].dropna().astype(str).str.strip()
            _imp_vol = pd.to_numeric(raw_df.loc[_imp.index, cols['volume']], errors='coerce').groupby(_imp).sum().sort_values(ascending=False)
            importer_order = list(_imp_vol.index)
            if not importer_order:
                st.warning(T('ph_no_data_warning'))
                st.stop()
            if len(importer_order) >= 2:
                health_importer = st.selectbox(T('ph_importer_label'), options=importer_order,
                                               format_func=lambda n: f"{n} ({_imp_vol[n]:,.0f} kg)", key="health_importer")
            else:
                health_importer = importer_order[0]
                st.caption(T('ph_importer_auto', name=health_importer))

            AXIS_MAP = build_axis_map([
                (T('ph_unit_product'), cols['product']),
                (T('ph_unit_origin'), cols['origin']),
            ], raw_df, cols)
            unit_label = st.selectbox(T('ph_unit_label'), options=list(AXIS_MAP.keys()), key="health_unit")
            unit_col = AXIS_MAP[unit_label]
            product_mode = unit_col == cols['product']
            st.caption(T('ph_unit_caption'))

            _parsed_dates = pd.to_datetime(raw_df[cols['date']], errors='coerce').dropna()
            _max_date = _parsed_dates.max().date() if len(_parsed_dates) else datetime.date.today()
            _base_start = (pd.Timestamp(_max_date) - pd.DateOffset(months=6)).date() + datetime.timedelta(days=1)
            _comp_end = _base_start - datetime.timedelta(days=1)
            _comp_start = (pd.Timestamp(_comp_end) - pd.DateOffset(months=6)).date() + datetime.timedelta(days=1)

            st.markdown(f"**{T('ph_base_period')}**")
            colB1, colB2 = st.columns(2)
            with colB1:
                base_start = st.date_input(T('p4_date_start'), value=_base_start, key="health_base_start")
            with colB2:
                base_end = st.date_input(T('p4_date_end'), value=_max_date, key="health_base_end")

            st.markdown(f"**{T('ph_comp_period')}**")
            colC1, colC2 = st.columns(2)
            with colC1:
                comp_start = st.date_input(T('p4_date_start'), value=_comp_start, key="health_comp_start")
            with colC2:
                comp_end = st.date_input(T('p4_date_end'), value=_comp_end, key="health_comp_end")

            if base_start <= comp_end and comp_start <= base_end:
                st.warning(T('ph_period_overlap_warning'))

            d = HEALTH_DEFAULT_THRESHOLDS
            with st.expander(T('ph_threshold_expander'), expanded=False):
                th_c1, th_c2 = st.columns(2)
                with th_c1:
                    st.markdown(f"**{T('ph_th_group_signal')}**")
                    price_red = st.slider(T('ph_th_price_red'), 1, 100, d['price_red'], 1, key="health_price_red")
                    price_yellow = st.slider(T('ph_th_price_yellow'), 1, 100, d['price_yellow'], 1, key="health_price_yellow")
                    churn_major = st.slider(T('ph_th_churn_major'), 1, 50, d['churn_major'], 1, key="health_churn_major")
                    replace_ok = st.slider(T('ph_th_replace_ok'), 10, 100, d['replace_ok'], 5, key="health_replace_ok")
                    squeeze_volume = st.slider(T('ph_th_squeeze_volume'), 5, 90, d['squeeze_volume'], 5, key="health_squeeze_volume")
                with th_c2:
                    st.markdown(f"**{T('ph_th_group_other')}**")
                    conc = st.slider(T('ph_th_conc'), 30, 100, d['conc'], 5, key="health_conc")
                    spread = st.slider(T('ph_th_spread'), 10, 100, d['spread'], 5, key="health_spread")
                    min_ship = st.number_input(T('ph_th_min_ship'), min_value=1, value=d['min_ship'], step=1, key="health_min_ship")
                    min_share = st.number_input(T('ph_th_min_share'), min_value=0.0, max_value=100.0, value=d['min_share'], step=0.5, key="health_min_share")
                if price_yellow > price_red or spread > conc:
                    st.warning(T('ph_th_order_warning'))
            thresholds = {'price_red': price_red, 'price_yellow': price_yellow, 'churn_major': churn_major,
                          'replace_ok': replace_ok, 'squeeze_volume': squeeze_volume, 'conc': conc, 'spread': spread,
                          'min_ship': min_ship, 'min_share': min_share}

            if st.button(T('ph_run_btn')):
                if base_start > base_end or comp_start > comp_end:
                    st.error(T('ph_period_invalid_error'))
                    st.stop()
                with st.spinner(T('ph_spinner')):
                    df = prepare_health_df(raw_df, cols, unit_col, product_mode)
                    df = df[df['_importer'] == health_importer]
                    result = compute_health_check(df, (base_start, base_end), (comp_start, comp_end), thresholds)
                if not result['rows']:
                    st.warning(T('ph_no_data_warning'))
                else:
                    result.update({
                        'importer': health_importer, 'unit_label': unit_label, 'product_mode': product_mode,
                        'periods': (base_start, base_end, comp_start, comp_end), 'thresholds': thresholds,
                    })
                    st.session_state.health_result = result
                    st.rerun()

    if st.session_state.health_result is not None:
        res = st.session_state.health_result
        rows = res['rows']
        b_start, b_end, c_start, c_end = res['periods']
        summary_caption = T('ph_summary_caption', importer=res['importer'], unit=res['unit_label'],
                            b_start=b_start, b_end=b_end, c_start=c_start, c_end=c_end)
        st.caption(summary_caption)

        status_order = ['risk', 'caution', 'safe', 'new', 'insufficient']
        counts = Counter(r['overall'] for r in rows)
        kpi_items = [(T('ph_kpi_total'), len(rows))] + [(T(HEALTH_OVERALL_KEY[k]), counts.get(k, 0)) for k in status_order]
        for kcol, (label, value) in zip(st.columns(len(kpi_items)), kpi_items):
            kcol.metric(label, value)

        st.subheader(T('ph_focus_header'))
        focus_lines = health_focus_lines(rows)
        if focus_lines:
            for unit, reasons in focus_lines:
                st.markdown(f"- 🔴 **{unit}**: {reasons}")
        else:
            st.info(T('ph_focus_none'))

        st.subheader(T('ph_table_header'))
        status_labels = {f"{HEALTH_OVERALL_EMOJI[k]} {T(HEALTH_OVERALL_KEY[k])}": k for k in status_order if counts.get(k)}
        selected_labels = st.multiselect(T('ph_filter_label'), options=list(status_labels), default=list(status_labels), key="health_filter")
        selected_status = {status_labels[l] for l in selected_labels if l in status_labels}
        shown_rows = [r for r in rows if r['overall'] in selected_status]
        st.dataframe(health_display_table(shown_rows), hide_index=True, width="stretch")
        st.caption(T('ph_table_notes'))
        if res['product_mode']:
            st.caption(T('group_rule_generic_hs'))
        st.caption(T('ph_market_note', season=T('menu_opt_season')))
        if not res['product_mode']:
            st.info(T('ph_mix_notice'))
        else:
            with st.expander(T('ph_members_expander'), expanded=False):
                st.dataframe(pd.DataFrame({
                    T('ph_col_unit'): [r['unit'] for r in shown_rows],
                    T('ph_col_member_count'): [len(r.get('members', [])) for r in shown_rows],
                    T('ph_col_members'): [' / '.join(r.get('members', [])) for r in shown_rows],
                }), hide_index=True, width="stretch")

        detail_rows = [r for r in shown_rows if r['overall'] in ('risk', 'caution')]
        if detail_rows:
            st.subheader(T('ph_detail_header'))
            share_fmt = {T('ph_detail_share'): '{:.1f}'}
            share_cols = [T('ph_detail_supplier'), T('ph_detail_share')]
            for r in detail_rows:
                label = f"{HEALTH_OVERALL_EMOJI[r['overall']]} {r['unit']}{T('ph_new_tag') if r['is_new'] else ''} — {T(HEALTH_OVERALL_KEY[r['overall']])}"
                with st.expander(label):
                    reasons = health_reasons(r)
                    if reasons:
                        st.markdown(f"**{T('ph_col_reason')}**: " + ' / '.join(reasons))
                    if res['product_mode'] and r.get('members'):
                        st.caption(f"{T('ph_detail_members')}: " + ' / '.join(r['members']))
                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        st.markdown(f"**{T('ph_detail_top5')}**")
                        st.dataframe(pd.DataFrame(r['top5'], columns=share_cols).style.format(share_fmt), hide_index=True, width="stretch")
                    with dc2:
                        st.markdown(f"**{T('ph_detail_lost')}**")
                        if r['lost']:
                            st.dataframe(pd.DataFrame(r['lost'], columns=share_cols).style.format(share_fmt), hide_index=True, width="stretch")
                        else:
                            st.caption(T('ph_detail_no_lost'))
                    with dc3:
                        st.markdown(f"**{T('ph_detail_new_sup')}**")
                        if r['new_suppliers']:
                            st.dataframe(pd.DataFrame(r['new_suppliers'], columns=share_cols).style.format(share_fmt), hide_index=True, width="stretch")
                        else:
                            st.caption(T('ph_detail_no_new'))
                    if r['replace_ratio'] is not None:
                        st.write(T('ph_detail_replace', ratio=r['replace_ratio']))
                    _fmt = lambda v, f: '-' if v is None else f.format(v)
                    mc1, mc2, mc3 = st.columns(3)
                    with mc1:
                        st.markdown(f"**{T('ph_detail_vwap')}**")
                        st.write(f"{T('ph_detail_base')}: {_fmt(r['base_vwap'], '{:,.2f}')} · {T('ph_detail_comp')}: {_fmt(r['comp_vwap'], '{:,.2f}')}")
                    with mc2:
                        st.markdown(f"**{T('ph_detail_monthly')}**")
                        st.write(f"{T('ph_detail_base')}: {r['base_monthly']:,.0f} · {T('ph_detail_comp')}: {r['comp_monthly']:,.0f}")
                    with mc3:
                        st.markdown(f"**{T('ph_detail_ship')}**")
                        st.write(f"{T('ph_detail_base')}: {r['base_ship']} · {T('ph_detail_comp')}: {r['comp_ship']}")
                    if r['price'] in ('red', 'yellow') or r['squeeze']:
                        st.caption(T('ph_market_note', season=T('menu_opt_season')))
                    st.caption(T('ph_detail_more', risk=T('menu_opt_risk'), churn=T('menu_opt_churn'), season=T('menu_opt_season')))

        dl1, dl2 = st.columns(2)
        with dl1:
            try:
                st.download_button(T('ph_excel_btn'), data=build_health_excel(rows), file_name="portfolio_health_check.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="health_excel_dl")
            except Exception as e:
                st.error(str(e))
        with dl2:
            if st.button(T('pdf_generate_btn'), key="health_pdf_btn"):
                with st.spinner(T('pdf_generating_msg')):
                    try:
                        kpi_lines = [summary_caption, ' · '.join(f"{label}: {value}" for label, value in kpi_items)]
                        kpi_lines += [T('ph_focus_header') + ' — ' + f"{unit}: {reasons}" for unit, reasons in focus_lines]
                        kpi_lines += [T('ph_table_notes'), T('ph_market_note', season=T('menu_opt_season'))]
                        if res['product_mode']:
                            kpi_lines.append(T('group_rule_generic_hs'))
                        if not res['product_mode']:
                            kpi_lines.append(T('ph_mix_notice'))
                        pdf_bytes = build_pdf_report(
                            title=T('ph_title'), kpi_lines=kpi_lines, figs=[],
                            df_table=health_display_table(rows, plain=True), table_title=T('ph_table_header'),
                        )
                        st.download_button(T('pdf_download_btn'), data=pdf_bytes, file_name="portfolio_health_check.pdf", mime="application/pdf", key="health_pdf_dl")
                    except Exception as e:
                        st.error(T('pdf_error_msg', msg=str(e)))

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
                    
                    all_competitors_ranked = weighted_avg_groupby(market_df, 'importer_name', 'competitiveness_index', 'volume').sort_values(ascending=False).reset_index()
                    
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
        filtered_competitors_ranked = weighted_avg_groupby(market_df, 'importer_name', 'competitiveness_index', 'volume').sort_values(ascending=False).reset_index()
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
            st.plotly_chart(fig_comp, width="stretch")
            
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
            monthly_competitiveness = weighted_avg_groupby(market_df, ['year_month', 'importer_name'], 'competitiveness_index', 'volume').unstack()
            
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
            st.plotly_chart(fig_comp_trend, width="stretch")
            st.caption(T('p2_comp_trend_caption'))
            st.markdown("---")

            st.markdown(f"##### {T('p2_price_trend_subheader')}")
            market_avg_price = weighted_avg_groupby(market_df, 'year_month', 'unit_price', 'volume', out_name='market_avg_price')
            customer_market_df = market_df[market_df['importer_name'] == customer_name]
            customer_avg_price = weighted_avg_groupby(customer_market_df, 'year_month', 'unit_price', 'volume', out_name='customer_avg_price')
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=market_avg_price.index.to_timestamp(), y=market_avg_price, mode='lines+markers', name=T('legend_market_avg_price'), line=dict(width=3)))
            fig4.add_trace(go.Scatter(x=customer_avg_price.index.to_timestamp(), y=customer_avg_price, mode='lines+markers', name=T('legend_customer_avg_price', customer=customer_name), line=dict(color='red')))
            
            # --- [수정 20] 필터링된 랭킹 사용 ---
            if top_competitors_list_filtered: 
                st.info(T('p2_benchmark_info'))
                st.caption(T('p2_benchmark_caption'))
                top_competitors_df = market_df[market_df['importer_name'].isin(top_competitors_list_filtered)] 
                top_competitors_avg_price = weighted_avg_groupby(top_competitors_df, 'year_month', 'unit_price', 'volume', out_name='top_competitors_avg_price')
                fig4.add_trace(go.Scatter(x=top_competitors_avg_price.index.to_timestamp(), y=top_competitors_avg_price, mode='lines+markers', name=T('legend_top_group_avg_price'), line=dict(color='green', dash='dash')))
            else:
                st.success(T('p2_benchmark_success', customer=customer_name))

            fig4.update_layout(title=f"<b>{T('p2_price_trend_chart_title', product=analyzed_product_name)}</b>", xaxis_title=T('axis_yearmonth'), yaxis_title=T('axis_avg_unit_price'))
            st.plotly_chart(fig4, width="stretch")

            st.markdown(f"##### {T('p2_price_compare_subheader')}")
            col1, col2, col3 = st.columns(3)
            col1.metric(T('metric_market_avg'), f"${weighted_avg(market_df, 'unit_price', 'volume'):.2f}")
            col2.metric(T('metric_customer_avg', customer=customer_name), f"${weighted_avg(customer_market_df, 'unit_price', 'volume'):.2f}")
            # --- [수정 21] 필터링된 랭킹 사용 ---
            if top_competitors_list_filtered: 
                col3.metric(T('metric_top_group_avg'), f"${weighted_avg(top_competitors_df, 'unit_price', 'volume'):.2f}")

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
                    st.plotly_chart(fig5, width="stretch")
            with col2:
                years_with_data_price = sorted(market_df['year'].unique(), reverse=True)
                if years_with_data_price:
                    selected_year_price = st.selectbox(T('p2_price_year_select'), options=years_with_data_price, key=f"price_year_{analyzed_product_name}")
                    price_comp_df = market_df[market_df['year'] == selected_year_price]
                    top_importers_by_vol = price_comp_df.groupby('importer_name')['volume'].sum().nlargest(5).index.tolist()
                    if customer_name not in top_importers_by_vol: top_importers_by_vol.append(customer_name)
                    price_comp_data = price_comp_df[price_comp_df['importer_name'].isin(top_importers_by_vol)]
                    avg_price_by_importer = weighted_avg_groupby(price_comp_data, 'importer_name', 'unit_price', 'volume').sort_values().reset_index()
                    
                    competitors = [imp for imp in avg_price_by_importer['importer_name'] if imp != customer_name]
                    blue_shades = px.colors.sequential.Blues_r[::(len(px.colors.sequential.Blues_r)//(len(competitors)+1)) if competitors else 1]
                    color_map_bar = {comp: blue_shades[i % len(blue_shades)] for i, comp in enumerate(competitors)}
                    color_map_bar[customer_name] = 'red'

                    fig6 = px.bar(avg_price_by_importer, x='importer_name', y='unit_price', title=f"<b>{T('p2_price_bar_title', year=selected_year_price)}</b><br><span style='font-size: 0.8em; color:grey;'>{T('p2_price_bar_subtitle')}</span>", labels={'importer_name': T('axis_importer'), 'unit_price': T('axis_avg_unit_price')}, color='importer_name', color_discrete_map=color_map_bar)
                    st.plotly_chart(fig6, width="stretch")
        
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
                    st.plotly_chart(fig9, width="stretch")
                    with st.expander(T('p2_detail_data_expander')):
                        summary_df_exp = exporter_analysis_df_top10.groupby('Exporter')['unit_price'].agg(['max', 'min']).reset_index()
                        exp_vwap = weighted_avg_groupby(exporter_analysis_df_top10, 'Exporter', 'unit_price', 'volume')
                        summary_df_exp['mean'] = summary_df_exp['Exporter'].map(exp_vwap)
                        summary_df_exp = summary_df_exp[['Exporter', 'max', 'mean', 'min']]
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
                            ).reset_index()
                            importer_vwap = weighted_avg_groupby(single_exporter_df, 'importer_name', 'unit_price', 'volume')
                            importer_summary['avg_unit_price'] = importer_summary['importer_name'].map(importer_vwap)
                            importer_summary = importer_summary.sort_values('total_volume', ascending=False)

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
                            st.plotly_chart(fig8, width="stretch")

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
                            st.plotly_chart(fig10, width="stretch")
                            with st.expander(T('p2_detail_data_expander')):
                                summary_df_imp = single_exporter_df_top10.groupby('importer_name')['unit_price'].agg(['max', 'min']).reset_index()
                                imp_vwap = weighted_avg_groupby(single_exporter_df_top10, 'importer_name', 'unit_price', 'volume')
                                summary_df_imp['mean'] = summary_df_imp['importer_name'].map(imp_vwap)
                                summary_df_imp = summary_df_imp[['importer_name', 'max', 'mean', 'min']]
                                summary_df_imp.columns = [T('col_importer'), T('col_max_price'), T('col_avg_price'), T('col_min_price')]
                                st.dataframe(summary_df_imp.style.format({T('col_max_price'): '${:,.2f}', T('col_avg_price'): '${:,.2f}', T('col_min_price'): '${:,.2f}'}))

                    st.subheader(T('p2_alt_sourcing_subheader', year=selected_year_exporter))
                    
                    # --- [수정 24] 'export_country'를 사용하도록 groupby 컬럼 변경 (+ 물량가중평균으로 변경) ---
                    avg_prices = exporter_analysis_df.groupby(['quarter', 'Exporter', 'export_country']).agg(
                        representative_product=('product_name', 'first'),
                        volume=('volume', 'sum'),
                    ).reset_index()
                    _avg_price_vwap = weighted_avg_groupby(exporter_analysis_df, ['quarter', 'Exporter', 'export_country'], 'unit_price', 'volume', out_name='avg_price').reset_index()
                    avg_prices = avg_prices.merge(_avg_price_vwap, on=['quarter', 'Exporter', 'export_country'])
                    
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
                            
                            # --- [수정 26] 'export_country' 기준으로 현재 원산지 옵션 표시 (물량가중평균) ---
                            customer_origins_q_sub = q_df[q_df['export_country'].isin(customer_origins)]
                            customer_origins_q_df = weighted_avg_groupby(customer_origins_q_sub, 'export_country', 'avg_price', 'volume').reset_index().sort_values('avg_price')
                            if not customer_origins_q_df.empty:
                                st.dataframe(customer_origins_q_df.rename(columns={'export_country': T('col_origin'), 'avg_price': T('col_avg_price2')}).style.format({T('col_avg_price2'): '${:,.2f}'}))
                            else:
                                st.write(T('p2_no_origin_deals'))

                            st.markdown(T('p2_alt_recommend_md'))
                            customer_avg_price_q = weighted_avg(q_df[q_df['Exporter'].isin(customer_exporters_in_year)], 'avg_price', 'volume')
                            if not pd.isna(customer_avg_price_q):
                                cheaper_exporters = q_df[(~q_df['Exporter'].isin(customer_exporters_in_year)) & (q_df['avg_price'] < customer_avg_price_q)].sort_values('avg_price')
                                if not cheaper_exporters.empty:
                                    st.dataframe(cheaper_exporters[['Exporter', 'representative_product', 'avg_price']].rename(columns={'Exporter': T('col_recommend_supplier'), 'representative_product': T('col_rep_item'), 'avg_price': T('col_avg_price2')}).style.format({T('col_avg_price2'): '${:,.2f}'}))
                                else:
                                    st.write(T('p2_no_cheaper_exporter'))
                            
                            # --- [수정 27] 'export_country' 기준으로 대안 원산지 탐색 (물량가중평균) ---
                            customer_origin_avg_price_q = weighted_avg(customer_origins_q_sub, 'avg_price', 'volume')
                            if not pd.isna(customer_origin_avg_price_q):
                                cheaper_origins = weighted_avg_groupby(q_df, 'export_country', 'avg_price', 'volume').reset_index()
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
                    g = s.groupby(right_col).agg(volume=('_volume', 'sum')).reset_index()
                    g['avg_price'] = g[right_col].map(weighted_avg_groupby(s, right_col, '_price', '_volume'))
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
            st.plotly_chart(fig, width="stretch")
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
                    st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")

        figs_for_pdf = [fig]
        if len(R['trend']) > 1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=R['trend']['_ym'], y=R['trend']['share'], mode='lines+markers', line=dict(color='#0d9488')))
            fig2.update_layout(title=T('p4_trend_chart_title'), xaxis_title=T('axis_yearmonth'), yaxis_title=T('p4_axis_share'))
            st.plotly_chart(fig2, width="stretch")
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
            cols = detect_trade_columns(headers)
            missing = [k for k in ['date', 'product', 'volume'] if not cols[k]]
            if not cols['price'] and not cols['value']:
                missing.append('price')
            if missing:
                st.error(T('p5_missing_cols_error', cols=', '.join(missing)))
                st.stop()
            _mismatch = derive_unit_price(raw_df, cols)[2]
            if _mismatch and _mismatch > 0.1:
                st.warning(T('price_mismatch_warning', pct=_mismatch * 100))

            _prod = raw_df[cols['product']].dropna()
            row_labels = product_group_labels(_prod.astype(str), raw_df.loc[_prod.index, cols['hs']] if cols.get('hs') else None)[0]
            selected_reps = st.multiselect(T('multi_product_label'), options=sorted(row_labels.unique()), key="season_products")
            st.caption(T('multi_product_help'))
            st.caption(T('group_rule_generic_hs'))

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
                    selected_idx = row_labels.index[row_labels.isin(selected_reps)]

                    df = raw_df.copy()
                    df['_date'] = pd.to_datetime(df[cols['date']], errors='coerce')
                    df['_price'] = derive_unit_price(df, cols)[0]  # Unit Price 우선, 없으면 금액 ÷ 중량
                    df['_volume'] = pd.to_numeric(df[cols['volume']], errors='coerce')
                    df = df.dropna(subset=['_date', '_price', cols['product']])
                    sub = df[df.index.isin(selected_idx)]

                    if sub.empty:
                        st.warning(T('p5_no_data_warning'))
                    else:
                        sub = sub.copy()
                        sub['_ym'] = sub['_date'].dt.to_period('M')
                        sub['_pack'] = [normalize_product_name(n)[1] for n in sub[cols['product']].astype(str)]
                        pack_markers = [(str(ym), text) for ym, text in pack_month_markers(sub, '_ym', '_pack', '_volume')]
                        pack_series = {}
                        for pack in sub.groupby('_pack')['_volume'].sum().sort_values(ascending=False).head(5).index:
                            ps = weighted_avg_groupby(sub[sub['_pack'] == pack], '_ym', '_price', '_volume').reset_index().sort_values('_ym')
                            ps['_ym_str'] = ps['_ym'].astype(str)
                            pack_series[pack] = ps
                        sub['_month'] = sub['_date'].dt.month
                        sub['_year'] = sub['_date'].dt.year

                        overall = weighted_avg_groupby(sub, '_ym', '_price', '_volume').reset_index().sort_values('_ym')
                        overall['_ym_str'] = overall['_ym'].astype(str)

                        month_avg = weighted_avg_groupby(sub, '_month', '_price', '_volume')
                        years_count = sub['_year'].nunique()
                        overall_avg = weighted_avg(sub, '_price', '_volume')
                        cheapest_month = int(month_avg.idxmin())
                        cheapest_month_pct = (overall_avg - month_avg.min()) / overall_avg * 100

                        latest_year = sub['_year'].max()
                        monthly_avg_latest_year = weighted_avg_groupby(sub[sub['_year'] == latest_year], '_month', '_price', '_volume')
                        peak_months = []
                        if len(monthly_avg_latest_year) >= 3:
                            lyr_mean = monthly_avg_latest_year.mean()
                            peak_months = sorted(monthly_avg_latest_year[monthly_avg_latest_year > lyr_mean * 1.05].index.tolist())

                        breakdown_series = {}
                        if breakdown != 'none':
                            bcol = cols[breakdown]
                            for name, g in sub.groupby(bcol):
                                s = weighted_avg_groupby(g, '_ym', '_price', '_volume').reset_index().sort_values('_ym')
                                s['_ym_str'] = s['_ym'].astype(str)
                                breakdown_series[str(name)] = s

                        st.session_state.season_result = {
                            'product_label': ', '.join(selected_reps), 'overall': overall,
                            'show_prev_year': show_prev_year, 'breakdown': breakdown,
                            'breakdown_series': breakdown_series, 'peak_months': peak_months,
                            'cheapest_month': cheapest_month, 'cheapest_month_pct': cheapest_month_pct,
                            'years_count': years_count, 'overall_avg': overall_avg,
                            'pack_markers': pack_markers, 'pack_series': pack_series,
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
            months_str = ', '.join(T('p5_month_label', m=m) for m in R['peak_months'])
            st.info(T('p5_season_badge', months=months_str))
        else:
            st.caption(T('p5_no_season'))

        by_pack = st.toggle(T('p5_pack_toggle'), value=False, key="season_by_pack") if len(R.get('pack_series', {})) > 1 else False
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

        if by_pack:  # 같은 차트에 포장 단위별 선 (브랜드 색 순서, 포장 표기 없는 거래는 '표기 없음')
            pack_colors = ['#171CFD', '#0A0A18', '#3E50FE', '#9BB3FF', '#0706CE']
            for i, (pack, ps) in enumerate(R['pack_series'].items()):
                fig.add_trace(go.Scatter(x=ps['_ym_str'], y=ps['_price'], mode='lines+markers', name=pack or T('p1_pack_none'),
                                         line=dict(color=pack_colors[i % len(pack_colors)], width=1.5), marker=dict(size=5)))
        if R.get('pack_markers'):
            m_x = [m for m, _ in R['pack_markers']]
            m_y = [ym_to_price.get(pd.Period(m, freq='M')) for m in m_x]
            fig.add_trace(go.Scatter(x=m_x, y=m_y, mode='markers', name=T('p5_pack_marker_legend'),
                                     marker=dict(symbol='diamond', size=13, color='#0A0A18', line=dict(color='#FFFFFF', width=1)),
                                     hovertext=[t for _, t in R['pack_markers']], hoverinfo='text'))
        fig.update_layout(title=T('p5_chart_title', product=R['product_label']), xaxis_title=T('p5_axis_month'), yaxis_title=T('p5_axis_price'))
        st.plotly_chart(fig, width="stretch")
        if R.get('pack_markers'):
            st.caption(T('p5_pack_note'))

        if R['show_prev_year'] and not prev_year_added:
            st.caption(T('p5_no_prev_year_caption'))

        st.markdown(f"### {T('insight_box_title')}")
        month_label = T('p5_month_label', m=R['cheapest_month'])
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
                            T('group_rule_generic_hs'),
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

                    # --- ① 가격 비교: 신규/유지/이탈 거래처의 물량가중평균단가 ---
                    new_price = lost_price = kept_price = None
                    if cols['price']:
                        if new_items:
                            v = weighted_avg(b_df[b_df[axis_col].astype(str).isin(new_items)], '_price', '_volume')
                            new_price = float(v) if pd.notna(v) else None
                        if lost_items:
                            v = weighted_avg(a_df[a_df[axis_col].astype(str).isin(lost_items)], '_price', '_volume')
                            lost_price = float(v) if pd.notna(v) else None
                        if kept_items:
                            v = weighted_avg(b_df[b_df[axis_col].astype(str).isin(kept_items)], '_price', '_volume')
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

        DATE_DIM_LABELS = [T('p7_row_month'), T('p7_dim_year'), T('p7_dim_quarter'), T('p7_dim_month_only')]
        ROW_OPTIONS = DATE_DIM_LABELS + list(DIM_MAP.keys())
        COL_OPTIONS = DATE_DIM_LABELS + list(DIM_MAP.keys())

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
            col_labels = st.multiselect(T('p7_col_label_multi'), options=COL_OPTIONS, key="pivot_cols")

        _parsed_dates = pd.to_datetime(raw_df[cols['date']], errors='coerce').dropna()
        _min_date = _parsed_dates.min().date() if len(_parsed_dates) else datetime.date.today()
        _max_date = _parsed_dates.max().date() if len(_parsed_dates) else datetime.date.today()
        colD1, colD2 = st.columns(2)
        with colD1:
            pivot_date_start = st.date_input(T('p4_date_start'), value=_min_date, key="pivot_date_start")
        with colD2:
            pivot_date_end = st.date_input(T('p4_date_end'), value=_max_date, key="pivot_date_end")

        value_labels = st.multiselect(T('p7_values_label'), options=list(METRIC_OPTIONS.keys()), default=[T('p7_metric_volume_sum')], key="pivot_values")
        view_label = st.selectbox(T('p7_view_label'), options=list(VIEW_MAP.keys()), key="pivot_view")

        st.markdown(f"**{T('p7_filter_label')}**")
        filter_col_labels = st.multiselect(T('p7_filter_cols_label'), options=list(DIM_MAP.keys()), key="pivot_filter_cols")
        filter_selections = {}
        for fc_label in filter_col_labels:
            fcol = DIM_MAP[fc_label]
            with st.expander(fc_label):
                value_counts = raw_df[fcol].dropna().astype(str).value_counts()  # 건수 많은 순
                unit = '건' if st.session_state.lang == 'ko' else ''
                label_to_value = {f"{val} ({cnt:,}{unit})": val for val, cnt in value_counts.items()}
                selected_labels = st.multiselect(T('p7_filter_values'), options=list(label_to_value.keys()), key=f"pivot_filter_vals_{fcol}")
                filter_selections[fcol] = [label_to_value[l] for l in selected_labels]

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
                df = df[(df['_date'] >= pd.to_datetime(pivot_date_start)) & (df['_date'] <= pd.to_datetime(pivot_date_end))]

                for fcol, vals in filter_selections.items():
                    if vals:
                        df = df[df[fcol].astype(str).isin(vals)]

                MAX_CATEGORIES = 20
                capped_fields = []

                def _build_dim_field(label):
                    """행/열로 고른 라벨(날짜류 또는 DIM_MAP의 일반 컬럼) 하나에 대해 문자열 컬럼을 만든다.
                    고유값이 MAX_CATEGORIES를 넘으면 물량 상위 항목만 남기고 나머지는 '기타'로 묶어서,
                    품목명처럼 종류가 수천 개인 컬럼을 열로 골랐을 때 표가 통째로 깨지는 걸 방지한다."""
                    if label == T('p7_row_month'):
                        s = df['_date'].dt.to_period('M').astype(str)
                    elif label == T('p7_dim_year'):
                        s = df['_date'].dt.year.astype('Int64').astype(str)
                    elif label == T('p7_dim_quarter'):
                        s = df['_date'].dt.year.astype(str) + '-Q' + df['_date'].dt.quarter.astype('Int64').astype(str)
                    elif label == T('p7_dim_month_only'):
                        month_unit = '월' if st.session_state.lang == 'ko' else ''
                        s = df['_date'].dt.month.astype('Int64').astype(str) + month_unit
                    else:
                        s = df[DIM_MAP[label]].astype(str)

                    if s.nunique(dropna=True) > MAX_CATEGORIES:
                        totals = df.assign(_tmp_dim=s).groupby('_tmp_dim')['_volume'].sum()
                        top_vals = set(totals.nlargest(MAX_CATEGORIES).index)
                        s = s.where(s.isin(top_vals), T('p2_others_label'))
                        capped_fields.append(label)
                    return s

                row_fields = []
                for i, rl in enumerate(row_labels):
                    fname = f'_row_{i}'
                    df[fname] = _build_dim_field(rl)
                    row_fields.append(fname)

                col_fields = []
                for i, cl in enumerate(col_labels):
                    fname = f'_col_{i}'
                    df[fname] = _build_dim_field(cl)
                    col_fields.append(fname)

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
                            'capped_fields': capped_fields,
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
                st.plotly_chart(fig, width="stretch")
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
                st.plotly_chart(fig, width="stretch")

            st.subheader(T('p7_table_subheader'))
            for capped_label in R.get('capped_fields', []):
                st.caption(T('p7_capped_note', field=capped_label, n=20))
            display_combined = combined.copy()
            if isinstance(display_combined.index, pd.MultiIndex):
                display_combined.index = display_combined.index.set_names(R['row_labels'])
            else:
                display_combined.index = display_combined.index.rename(' | '.join(R['row_labels']))

            # 열이 여러 칸으로 나뉜 경우, 지표별로 '합계' 컬럼을 맨 앞에 추가 (연간 합계 등 한눈에 보기용)
            if R['has_col'] and isinstance(display_combined.columns, pd.MultiIndex):
                total_label = T('p7_total_col')
                pieces = []
                for metric_name in R['value_labels']:
                    sub = display_combined[metric_name]
                    total_series = sub.sum(axis=1, skipna=True)
                    sub_with_total = pd.concat([total_series.rename(total_label), sub], axis=1)
                    sub_with_total.columns = pd.MultiIndex.from_product([[metric_name], sub_with_total.columns])
                    pieces.append(sub_with_total)
                display_combined = pd.concat(pieces, axis=1)
                display_combined.columns = display_combined.columns.set_names([T('p7_values_label')] + R['col_labels'])
            else:
                display_combined.columns = display_combined.columns.rename(T('p7_values_label'))

            # 지표 종류에 맞춰 숫자 포맷(천단위 콤마·단가는 $ 소수점) 적용 — 단위를 못 알아보는 문제 방지
            def _is_price_metric(name):
                return ('단가' in name) or ('price' in name.lower())

            def _fmt_cell(v, is_price):
                if pd.isna(v):
                    return '-'
                return f"${v:,.2f}" if is_price else f"{v:,.0f}"

            display_str = display_combined.copy()
            for c in display_combined.columns:
                metric_name = c[0] if isinstance(c, tuple) else c
                is_price = _is_price_metric(metric_name)
                display_str[c] = display_combined[c].apply(lambda v: _fmt_cell(v, is_price))
            st.dataframe(display_str)

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
            st.plotly_chart(fig, width="stretch")

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
            st.dataframe(s_df, width="stretch")
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
                st.dataframe(pd.DataFrame(a_rows), width="stretch")
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
                st.dataframe(pd.DataFrame(b_rows), width="stretch")
            else:
                st.caption(T('p8_no_data_warning'))

        if st.button(T('p8_generate_report_btn'), key="scorer_pdf_btn"):
            with st.spinner(T('pdf_generating_msg')):
                try:
                    pdf_bytes = build_scorer_report_pdf(A, B, S, meta, R['dim_label'])
                    st.download_button(T('p8_download_report_btn'), data=pdf_bytes, file_name="new_business_scorer_report.pdf", mime="application/pdf", key="scorer_pdf_dl")
                except Exception as e:
                    st.error(T('pdf_error_msg', msg=str(e)))


# ==============================================================================
# 페이지 ⑨: 범용상품 스크리너
# ==============================================================================
if selected == T('menu_opt_screener'):
    st.title(T('sc_title'))
    st.write(T('sc_intro'))

    if st.session_state.scr_result is not None:
        st.button(T('sc_reset_btn'), on_click=reset_screener_states)

    if st.session_state.scr_result is None:
        scr_file = st.file_uploader(T('sc_upload_label'), type=['csv', 'xlsx'], key="scr_uploader")
        st.caption(T('sc_upload_caption'))
        raw_df = load_uploaded_df(scr_file, 'scr_raw_df', 'scr_headers', 'scr_fileid')

        if raw_df is not None:
            cols = detect_trade_columns(st.session_state.scr_headers)
            missing = [k for k in ['date', 'importer', 'exporter', 'origin', 'hs', 'volume'] if not cols[k]]
            if not cols['price'] and not cols['value']:
                missing.append('price')
            if missing:
                st.error(T('sc_missing_cols_error', cols=', '.join(missing)))
                st.stop()

            d_start, d_end = default_last_12m(raw_df[cols['date']])
            c1, c2 = st.columns(2)
            with c1:
                period = st.date_input(T('sc_period_label'), value=(d_start, d_end), key="scr_period")
            with c2:
                min_kg = st.slider(T('sc_min_kg_label'), 0, 10000, 1000, 100, key="scr_min_kg")
            min_ship = st.number_input(T('sc_min_ship_label'), min_value=1, value=5, step=1, key="scr_min_ship")

            with st.sidebar.expander(T('sc_weights_header'), expanded=False):
                weights = {k: st.slider(T(f'sc_w_{k}'), 0.0, 3.0, SCREENER_DEFAULT_WEIGHTS[k], 0.5, key=f"scr_w_{k}")
                           for k, _ in SCREENER_COMPONENTS}
            with st.sidebar.expander(T('sc_bounds_header'), expanded=False):
                bound_hi = st.slider(T('sc_bound_hi'), 10, 95, 60, 5, key="scr_bound_hi")
                bound_lo = st.slider(T('sc_bound_lo'), 5, 90, 30, 5, key="scr_bound_lo")
                bound_gap = st.slider(T('sc_bound_gap'), 10, 90, 50, 5, key="scr_bound_gap")
                if bound_lo >= bound_hi:
                    st.warning(T('sc_bounds_warning'))

            if st.button(T('sc_run_btn')):
                if not isinstance(period, (tuple, list)) or len(period) != 2:
                    st.warning(T('sc_no_data_warning'))
                    st.stop()
                with st.spinner(T('sc_spinner')):
                    df, stats = prepare_trade_df(raw_df, cols, period=period, min_shipment_kg=min_kg)
                    months = max(1, round((pd.Timestamp(period[1]) - pd.Timestamp(period[0])).days / 30.44))
                    res, skipped = compute_screener(df, months, weights, bound_hi, bound_lo, bound_gap, min_ship)
                if res.empty:
                    st.warning(T('sc_no_data_warning'))
                else:
                    st.session_state.scr_result = {'res': res, 'stats': stats, 'skipped': skipped,
                                                   'bounds': (bound_hi, bound_lo, bound_gap), 'period': tuple(period),
                                                   'weights': dict(weights), 'min_kg': min_kg, 'min_ship': min_ship}
                    st.rerun()

    if st.session_state.scr_result is not None:
        R = st.session_state.scr_result
        res, (bound_hi, bound_lo, bound_gap) = R['res'], R['bounds']
        st.caption(T('sc_prep_caption', skipped=R['skipped'], **R['stats']))
        counts = res['class'].value_counts()
        kpis = [(T('sc_kpi_total'), len(res))] + [(T(SCREENER_CLASS_KEY[c]), int(counts.get(c, 0))) for c in 'ABCD']
        for kcol, (label, value) in zip(st.columns(len(kpis)), kpis):
            kcol.metric(label, value)

        st.subheader(T('sc_quadrant_header'))
        st.plotly_chart(screener_quadrant_fig(res, bound_hi, bound_lo, bound_gap), width="stretch")
        st.caption(T('sc_bubble_note'))

        st.subheader(T('sc_rank_header'))
        class_labels = {T(SCREENER_CLASS_KEY[c]): c for c in 'ABCD' if counts.get(c)}
        picked = st.multiselect(T('sc_filter_label'), options=list(class_labels), default=list(class_labels), key="scr_filter")
        shown = res[res['class'].isin({class_labels[p] for p in picked if p in class_labels})]
        table = screener_display_table(shown)
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(T('sc_metric_note'))
        st.download_button(T('sc_csv_btn'), data=table.to_csv(index=False).encode('utf-8-sig'),
                           file_name="commodity_screener.csv", mime="text/csv", key="scr_csv")
        if st.button(T('rp_pdf_btn'), key="scr_pdf_btn"):
            with st.spinner(T('rp_pdf_spinner')):
                try:
                    pdf_bytes = build_screener_report(R, R.get('weights', SCREENER_DEFAULT_WEIGHTS), R.get('min_kg', 1000), R.get('min_ship', 5))
                    st.download_button(T('rp_pdf_dl'), data=pdf_bytes, file_name="commodity_screener_report.pdf", mime="application/pdf", key="scr_pdf_dl")
                except Exception as e:
                    st.error(T('pdf_error_msg', msg=str(e)))

        st.subheader(T('sc_goto_header'))
        hs_options = list(res['hs'])
        hs_pick = st.selectbox(T('sc_goto_select'), options=hs_options,
                               format_func=lambda h: f"{h} · {res.loc[res['hs'] == h, 'name'].iloc[0]} ({res.loc[res['hs'] == h, 'class'].iloc[0]})",
                               key="scr_goto_hs")
        if st.button(T('sc_goto_btn'), key="scr_goto_btn"):
            st.session_state.bid_hs_preset = hs_pick
            if st.session_state.get('bid_raw_df') is None:
                st.session_state.bid_raw_df = st.session_state.scr_raw_df
                st.session_state.bid_headers = st.session_state.scr_headers
            st.session_state['_menu_jump'] = 9  # 메뉴 순서상 ⑩ 입찰 대응 분석
            st.success(T('sc_goto_done', hs=hs_pick, menu=T('menu_opt_bid')))
            st.rerun()


# ==============================================================================
# 페이지 ⑩: 입찰 대응 분석
# ==============================================================================
if selected == T('menu_opt_bid'):
    st.title(T('bd_title'))
    st.write(T('bd_intro'))

    if st.session_state.bid_raw_df is not None:
        st.button(T('bd_reset_btn'), on_click=reset_bid_states)

    if st.session_state.bid_raw_df is None:
        bid_file = st.file_uploader(T('bd_upload_label'), type=['csv', 'xlsx'], key="bid_uploader")
        st.caption(T('bd_upload_caption'))
        if st.session_state.get('scr_raw_df') is not None and st.button(T('bd_use_screener_btn')):
            st.session_state.bid_raw_df = st.session_state.scr_raw_df
            st.session_state.bid_headers = st.session_state.scr_headers
            st.rerun()
        load_uploaded_df(bid_file, 'bid_raw_df', 'bid_headers', 'bid_fileid')
        if st.session_state.bid_raw_df is not None:
            st.rerun()

    raw_df = st.session_state.bid_raw_df
    if raw_df is not None:
        cols = detect_trade_columns(st.session_state.bid_headers)
        missing = [k for k in ['date', 'importer', 'exporter', 'origin', 'volume'] if not cols[k]]
        if not cols['price'] and not cols['value']:
            missing.append('price')
        if missing:
            st.error(T('bd_missing_cols_error', cols=', '.join(missing)))
            st.stop()

        st.subheader(T('bd_filter_header'))
        full_df, stats = prepare_trade_df(raw_df, cols, period=None, min_shipment_kg=st.session_state.get('bid_min_kg', 1000))
        hs_list = sorted(full_df['_hs'].unique()) if cols['hs'] else []
        preset = st.session_state.get('bid_hs_preset')
        if preset:
            st.caption(T('bd_hs_from_screener', hs=preset))
        f1, f2 = st.columns(2)
        with f1:
            hs_opts = [T('bd_hs_all')] + hs_list
            hs_sel = st.selectbox(T('bd_hs_label'), options=hs_opts, index=hs_opts.index(preset) if preset in hs_opts else 0, key="bid_hs")
            include_kw = st.text_input(T('bd_include_label'), key="bid_include")
        with f2:
            d_start, d_end = default_last_12m(full_df['_date'])
            period = st.date_input(T('bd_period_label'), value=(d_start, d_end), key="bid_period")
            exclude_kw = st.text_input(T('bd_exclude_label'), key="bid_exclude")
        g1, g2 = st.columns(2)
        with g1:
            st.slider(T('bd_min_kg_label'), 0, 10000, 1000, 100, key="bid_min_kg")
        with g2:
            blind_on = st.checkbox(T('blind_toggle'), value=True, key="bid_blind")

        if not isinstance(period, (tuple, list)) or len(period) != 2:
            st.stop()
        scoped = bid_filter(full_df, None if hs_sel == T('bd_hs_all') else hs_sel, include_kw, exclude_kw)
        in_period = scoped[(scoped['_date'] >= pd.Timestamp(period[0])) & (scoped['_date'] <= pd.Timestamp(period[1]))]
        st.caption(T('bd_prep_caption', n0=stats['n0'], dup=stats['dup'], small=stats['small'], outlier=stats['outlier'], n=len(in_period)))
        if in_period.empty:
            st.warning(T('bd_no_data_warning'))
            st.stop()
        blind_map = blind_importer_map(in_period, '_importer', '_volume') if blind_on else None

        tab_alt, tab_stab = st.tabs([T('bd_tab_alt'), T('bd_tab_stability')])
        with tab_alt:
            with st.expander(T('bd_verify_header'), expanded=False):
                v1, v2, v3 = st.columns(3)
                verify = {
                    'min_ship': v1.number_input(T('bd_verify_ship'), min_value=0, value=BID_DEFAULT_VERIFY['min_ship'], step=1, key="bid_v_ship"),
                    'min_ton': v2.number_input(T('bd_verify_ton'), min_value=0.0, value=BID_DEFAULT_VERIFY['min_ton'], step=1.0, key="bid_v_ton"),
                    'min_imp': v3.number_input(T('bd_verify_imp'), min_value=0, value=BID_DEFAULT_VERIFY['min_imp'], step=1, key="bid_v_imp"),
                }
            origins = sorted(in_period['_origin'].unique())
            st.markdown(f"**{T('bd_tariff_header')}**")
            tariff_df = st.data_editor(pd.DataFrame({T('bd_tariff_col_origin'): origins, T('bd_tariff_col_rate'): [None] * len(origins)}),
                                       hide_index=True, disabled=[T('bd_tariff_col_origin')], key="bid_tariff_editor",
                                       column_config={T('bd_tariff_col_rate'): st.column_config.NumberColumn(min_value=0.0, max_value=1000.0, step=0.1)})
            st.caption(T('bd_tariff_caption'))
            tariffs = {o: float(r) for o, r in zip(tariff_df[T('bd_tariff_col_origin')], tariff_df[T('bd_tariff_col_rate')]) if pd.notna(r)}
            tariff_missing = len(tariffs) < len(origins)
            st.markdown(_badge_html(T('bd_badge_no_tariff'), dark=True) if tariff_missing else _badge_html(T('bd_badge_tariff_ok'), dark=False), unsafe_allow_html=True)
            if tariff_missing:
                st.caption(T('bd_tariff_missing_note'))

            def _view(table, key_label):
                by_origin = key_label == T('bd_col_origin')
                out = pd.DataFrame({
                    key_label: table['key'],
                    T('bd_col_status'): table['verified'].map(lambda v: T('bd_badge_verified') if v else T('bd_badge_unverified')),
                    **({} if by_origin else {T('bd_col_origin'): table['origin']}),
                    T('bd_col_vwap'): table['vwap'].round(3),
                    T('bd_col_tariff'): table['tariff'],
                    T('bd_col_cmp'): table['cmp'].round(3),
                    T('bd_col_ship'): table['shipments'],
                    T('bd_col_ton'): table['tons'].round(1),
                    T('bd_col_months'): table['months'],
                    T('bd_col_importers'): table['importers'],
                    T('bd_col_importer_names'): table['importer_names'],
                    T('bd_col_last'): table['last'],
                })
                if 'saving_pct' in table:
                    out[T('bd_col_saving_pct')] = table['saving_pct'].round(1)
                    out[T('bd_col_saving_amt')] = table['saving_amt'].round(0)
                return apply_blind(out, T('bd_col_importer_names'), blind_on, mapping=blind_map)

            st.subheader(T('bd_origin_header'))
            origin_tbl = bid_supplier_table(in_period, '_origin', period[1], verify, tariffs)
            top_origins = in_period.groupby('_origin')['_volume'].sum().sort_values(ascending=False).head(12).index
            box = px.box(in_period[in_period['_origin'].isin(top_origins)], x='_origin', y='_price', points=False,
                         category_orders={'_origin': list(top_origins)}, color_discrete_sequence=['#171CFD'])
            box.update_layout(title=T('bd_origin_chart_title'), xaxis_title='', yaxis_title='USD/kg', height=380, plot_bgcolor='#FFFFFF')
            box.update_yaxes(gridcolor='rgba(10,10,24,0.08)')
            st.plotly_chart(box, width="stretch")
            origin_view = _view(origin_tbl, T('bd_col_origin'))
            st.dataframe(origin_view, hide_index=True, width="stretch")
            st.download_button(T('bd_csv_origin_btn'), data=origin_view.to_csv(index=False).encode('utf-8-sig'),
                               file_name="tender_origins.csv", mime="text/csv", key="bid_csv_origin")

            st.subheader(T('bd_supplier_header'))
            sup_tbl = bid_supplier_table(in_period, '_exporter', period[1], verify, tariffs)
            s1, s2, s3 = st.columns([2, 1, 1])
            with s1:
                base_opts = [T('bd_base_none')] + list(sup_tbl['key'])
                base_sel = st.selectbox(T('bd_base_label'), options=base_opts, key="bid_base")
            with s2:
                annual_tons = st.number_input(T('bd_annual_ton_label'), min_value=0.0, value=100.0, step=10.0, key="bid_annual_tons")
            with s3:
                verified_only = st.checkbox(T('bd_verified_only'), value=False, key="bid_verified_only")
            sup_tbl = bid_add_savings(sup_tbl, None if base_sel == T('bd_base_none') else base_sel, annual_tons)
            shown_sup = sup_tbl[sup_tbl['verified'] | (sup_tbl['key'] == base_sel)] if verified_only else sup_tbl
            sup_view = _view(shown_sup, T('bd_col_exporter'))
            st.dataframe(sup_view, hide_index=True, width="stretch")
            st.download_button(T('bd_csv_supplier_btn'), data=sup_view.to_csv(index=False).encode('utf-8-sig'),
                               file_name="tender_suppliers.csv", mime="text/csv", key="bid_csv_supplier")
            st.caption(T('bd_method_note'))

        with tab_stab:
            sup_rank = in_period.groupby('_exporter')['_volume'].sum().sort_values(ascending=False)
            stab_sel = st.selectbox(T('bd_stab_select'), options=list(sup_rank.index), key="bid_stab_sup")
            stab = bid_stability(scoped, stab_sel, period[1])
            if stab['empty']:
                st.info(T('bd_stab_no_data'))
            else:
                st.info(bid_stability_sentence(stab_sel, stab))
                k1, k2, k3, k4 = st.columns(4)
                k1.metric(T('bd_stab_kpi_gap'), stab['gap_months'])
                k2.metric(T('bd_stab_kpi_annual'), f"{stab['annual_tons']:,.0f}")
                k3.metric(T('bd_stab_kpi_imp'), stab['importers'])
                k4.metric(T('bd_stab_kpi_cv'), '-' if pd.isna(stab['cv']) else f"{stab['cv']:.1f}")
                fig = bid_stability_fig(stab)
                fig.update_layout(title=T('bd_stab_chart_title', m=stab['n_months']))
                st.plotly_chart(fig, width="stretch")
                st.markdown(f"**{T('bd_stab_year_header')}**")
                yearly = stab['yearly'].drop(columns=['span']).reset_index().rename(columns={
                    'label': T('bd_col_year'), 'shipments': T('bd_col_ship'), 'tons': T('bd_col_ton'),
                    'vwap': T('bd_col_vwap'), 'months': T('bd_col_months'), 'importers': T('bd_col_importers')})
                st.dataframe(yearly.style.format({T('bd_col_ship'): '{:,.0f}', T('bd_col_ton'): '{:,.1f}', T('bd_col_vwap'): '{:,.3f}',
                                                  T('bd_col_months'): '{:,.0f}', T('bd_col_importers'): '{:,.0f}'}),
                             hide_index=True, width="stretch")

        st.divider()
        st.subheader(T('bd_rp_header'))
        default_item = in_period['_product'].mode().iloc[0] if in_period['_product'].astype(str).str.len().gt(0).any() else (hs_sel or '')
        r1, r2 = st.columns(2)
        with r1:
            rp_item = st.text_input(T('bd_rp_item_label'), value=str(default_item)[:40], key="bid_rp_item")
        with r2:
            rp_spec = st.text_input(T('bd_rp_spec_label'), value=include_kw, key="bid_rp_spec")
        default_stab = list(sup_tbl[sup_tbl['verified']].sort_values('cmp')['key'].head(2)) or list(sup_tbl.sort_values('cmp')['key'].head(1))
        rp_stab = st.multiselect(T('bd_rp_stab_label'), options=list(sup_tbl['key']), default=default_stab, key="bid_rp_stab")
        if st.button(T('rp_pdf_btn'), key="bid_pdf_btn"):
            with st.spinner(T('rp_pdf_spinner')):
                try:
                    pdf_bytes = build_bid_report({
                        'df': in_period, 'scoped': scoped, 'origin_tbl': origin_tbl, 'sup_tbl': sup_tbl, 'verify': verify,
                        'annual_tons': annual_tons, 'base_key': None if base_sel == T('bd_base_none') else base_sel,
                        'tariffs': tariffs, 'tariff_missing': tariff_missing, 'period': tuple(period), 'item': rp_item or '-',
                        'spec': rp_spec, 'hs': None if hs_sel == T('bd_hs_all') else hs_sel, 'include': include_kw, 'exclude': exclude_kw,
                        'min_kg': st.session_state.get('bid_min_kg', 1000), 'blind': blind_on, 'stab_suppliers': rp_stab,
                    })
                    st.download_button(T('rp_pdf_dl'), data=pdf_bytes, file_name="tender_response_report.pdf", mime="application/pdf", key="bid_pdf_dl")
                except Exception as e:
                    st.error(T('pdf_error_msg', msg=str(e)))
