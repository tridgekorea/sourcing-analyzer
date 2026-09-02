import streamlit as st
import pandas as pd
import numpy as np
import re
import time
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
    },
    'en': {
        'app_menu_title': 'Menu',
        'menu_opt_customer': 'Customer Efficiency Analysis',
        'menu_opt_market': 'Market Competitiveness Analysis',
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
    return name.strip()

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

# --- [추가 4] 'find_column' 함수 정의 (NameError 해결) ---
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
        options=[T('menu_opt_customer'), T('menu_opt_market')],
        icons=["person-bounding-box", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0,
    )
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
