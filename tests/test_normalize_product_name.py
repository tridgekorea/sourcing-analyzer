"""① 도입 효과 분석의 품목명 정제(normalize_product_name)와 포장 단위 구성 변화(pack_mix_change) 고정 테스트.

실행: python tests/test_normalize_product_name.py   (pytest가 있으면 pytest로도 실행 가능)
analysis_tool.py는 import하면 Streamlit 화면까지 실행되므로, 필요한 함수만 소스에서 읽어 검사한다.
"""
import ast
import os
import re

import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis_tool.py")


def _load():
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    keep = {"normalize_product_name", "pack_mix_change", "pack_change_text", "_split_pack", "T"}
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in keep)
            or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("PACK_UNIT_RE", "_TRAIL_NUM_CODE_RE", "_TRAIL_ALNUM_CODE_RE", "TEXTS"))]

    class _St:
        session_state = {"lang": "ko"}

    ns = {"re": re, "np": np, "pd": pd, "st": _St()}
    exec(compile(ast.Module(body=body, type_ignores=[]), SRC, "exec"), ns)
    return ns


NS = _load()
norm = NS["normalize_product_name"]


def name(raw):
    return norm(raw)[0]


# 같은 제품: 포장 단위나 끝의 제품 코드만 다르면 같은 품목군 (실제 TDS 품목명에서 확인한 사례 10개)
SAME_PRODUCT = [
    ["[프로바이오틱스] 밸런스 위드인 365 (10포)__BALANCE WITHIN 365", "[프로바이오틱스] 밸런스 위드인 365 (30포)__BALANCE WITHIN 365",
     "[프로바이오틱스] 밸런스 위드인 365 (90포)__BALANCE WITHIN 365"],
    ["[EPA 및 DHA 함유 유지] 오메가 3 밸런스 (20정)", "오메가 3 밸런스 (360정)", "오메가 3 밸런스 (360캡슐)"],
    ["올데이 비타민 C 플러스 (10정)", "올데이 비타민 C 플러스 (60정)", "올데이 비타민 C 플러스(120정)"],
    ["[비타민 C] 아세로라 C(100정)__ACEROLA C", "[비타민 C] 아세로라 C(300정)__ACEROLA C"],
    ["비젼 헬스 (30 캡슐)", "비젼 헬스 (90 캡슐)"],
    ["[복합영양소제품] 츄어블 비타민 비 (85정)__CHEWABLE VITAMIN B", "[복합영양소제품] 츄어블 비타민 비 (270정)__CHEWABLE VITAMIN B"],
    ["카스타드향", "카스타드향 ES 2677"],
    ["프리페어드 에더블 펫", "프리페어드 에더블 펫 PEF225"],
    ["바닐라향", "바닐라향 PD1754W"],
    ["김맛 오차즈케", "김맛 오차즈케 19A", "김맛 오차즈케 20A"],
]

# 다른 제품: 맛·브랜드·대상·제품이 다르면 다른 품목군 (유사도 방식에서 과병합됐던 사례)
DIFFERENT_PRODUCT = [
    ("모닌 제로 바닐라", "모닌 제로 카라멜"),                          # 맛
    ("모닌 제로 카라멜", "모닌 제로 헤이즐넛"),                        # 맛
    ("바디키 식사대용 쉐이크 밀크티", "바디키 식사대용 쉐이크 카페라떼"),  # 맛
    ("에너지 팩", "맨즈 에너지 팩"),                                  # 대상
    ("바이탈 팩", "우먼즈 바이탈 팩"),                                # 대상
    ("애플 시럽", "그린 애플 시럽"),                                  # 종류
    ("모닌 카라멜 플레이버", "소스 카라멜 플레이버"),                  # 브랜드
    ("이스트 비 (270정)", "츄어블 비타민 비 (270정)"),                # 다른 제품
    ("데일리 플러스 비젼", "데일리 플러스 액티브"),                    # 다른 제품
    ("블루베리 프룻 믹스", "코코넛 프룻 믹스"),                        # 맛
]


def test_same_product_grouped():
    for group in SAME_PRODUCT:
        keys = {name(n) for n in group}
        assert len(keys) == 1, (group, keys)


def test_different_products_not_grouped():
    for a, b in DIFFERENT_PRODUCT:
        assert name(a) != name(b), (a, b, name(a))


def test_meaningful_numbers_and_letters_kept():
    assert name("아세로라 C(300정)") == "아세로라 C"          # 끝의 한 글자 C는 코드가 아님
    assert name("오메가 3 밸런스 (360정)") == "오메가 3 밸런스"  # 숫자 3은 제품명의 일부
    assert name("밸런스 위드인 365 (30포)") == "밸런스 위드인 365"
    assert name("뉴트리 파이토 푸로틴 (900G)__NUTRI-PHYTO PROTEIN") == "뉴트리 파이토 푸로틴"


def test_pack_unit_extracted():
    assert norm("오메가 3 밸런스 (360정)")[1] == "360정"
    assert norm("비젼 헬스 (30 캡슐)")[1] == "30캡슐"
    assert norm("뉴트리 파이토 푸로틴 (900G)")[1] == "900G"
    assert norm("카스타드향 ES 2677")[1] == ""


def _g(rows):
    return pd.DataFrame(rows, columns=["pack_unit", "volume"])


def test_pack_mix_change():
    f = NS["pack_mix_change"]
    # 주력 포장 단위가 바뀜: 360정 → 120정
    r = f(_g([("360정", 900), ("120정", 100)]), _g([("360정", 200), ("120정", 800)]))
    assert r[0] and r[1] == "main" and (r[2], r[3]) == ("360정", "120정")
    assert NS["pack_change_text"](*r[1:]) == "포장 단위 변경: 360정 → 120정"
    # 주력은 같지만 비중이 20%p 이상 이동: 360정 90% → 60%
    r = f(_g([("360정", 900), ("120정", 100)]), _g([("360정", 600), ("120정", 400)]))
    assert r[0] and r[1] == "share"
    # 소폭 변화(10%p)는 경고 아님
    r = f(_g([("360정", 900), ("120정", 100)]), _g([("360정", 800), ("120정", 200)]))
    assert not r[0]
    # 포장 단위가 하나뿐이면 경고 아님
    assert not f(_g([("360정", 500)]), _g([("360정", 700)]))[0]
    # 개수는 같고 단위 이름만 다름 → 경고는 유지하되 '제형 변경'
    r = f(_g([("360정", 900)]), _g([("360캡슐", 900)]))
    assert r[0] and r[1] == "form" and (r[2], r[3]) == ("정", "캡슐")
    assert NS["pack_change_text"](*r[1:]) == "제형 변경: 정 → 캡슐"
    # 개수·용량이 바뀌면 기존대로 '포장 단위 변경'
    r = f(_g([("360정", 900)]), _g([("120캡슐", 900)]))
    assert r[0] and r[1] == "main"
    # 표기 방식만 바뀐 경우(계약 전 포장 표기 없음 → 계약 후 1000ML 표기)는 경고 아님
    assert not f(_g([("", 900)]), _g([("1000ML", 900)]))[0]
    assert not f(_g([("250ML", 450), ("", 550)]), _g([("", 1000)]))[0]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
