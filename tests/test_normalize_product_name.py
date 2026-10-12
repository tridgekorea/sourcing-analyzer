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
    keep = {"normalize_product_name", "pack_mix_change", "pack_change_text", "_split_pack", "product_group_labels", "_pad_hs_code", "pack_month_markers", "T"}
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in keep)
            or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("PACK_UNIT_RE", "_TRAIL_NUM_CODE_RE", "_TRAIL_ALNUM_CODE_RE", "TEXTS"))]

    class _St:
        session_state = {"lang": "ko"}

    from collections import Counter
    ns = {"re": re, "np": np, "pd": pd, "st": _St(), "Counter": Counter}
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


def test_product_group_labels_same_rule_as_adoption():
    """①②⑥ 공통 품목 묶기: 정제된 이름이 표시 이름 (HS 없이 호출하면 정제 이름만으로 묶음)."""
    disp, key, pack = NS["product_group_labels"](["[혼합분유] 혼합분유 (25KG)__MIXED MILK POWDER", "[혼합분유] 혼합분유 (600KG)",
                                                  "칼-맥.디__CAL-MAG.D", "모닌 제로 바닐라", "모닌 제로 카라멜"])
    assert set(disp) == {"혼합분유", "칼 맥 디", "모닌 제로 바닐라", "모닌 제로 카라멜"}, set(disp)
    assert key[0] == key[1] and list(pack[:2]) == ["25KG", "600KG"]


def test_generic_name_split_by_hs():
    """일반 명칭 품목(분류명 = 품목명)은 HS 10단위로 나눈다. 같은 HS 안의 세부 스펙은 구분하지 않는다."""
    f = NS["product_group_labels"]
    # 혼합분유: HS 2개 → 2개 품목 (엑셀에서 앞자리 0이 빠진 9자리 코드도 10자리로 복원)
    disp, key, _ = f(["[혼합분유] 혼합분유 (25KG)", "[혼합분유] 혼합분유 (25KG)", "혼합분유 (600KG)", "[혼합분유] 혼합분유"],
                     ["404901000", "0404901000", "404902000", "404902000"])
    assert key[0] == key[1] and key[2] == key[3] and key[0] != key[2], list(key)
    assert disp[0] == "혼합분유 (HS 0404901000)" and disp[2] == "혼합분유 (HS 0404902000)", list(disp)
    # 황기·당귀: HS가 하나뿐이면 나누지 않음
    disp, key, _ = f(["[황기] 황기 (20KG)", "[황기] 황기", "[당귀] 당귀 (10KG)", "[당귀] 당귀"],
                     ["1211909099", "1211909099", "1211909099", "1211909099"])
    assert key[0] == key[1] and key[2] == key[3] and key[0] != key[2], list(key)
    assert list(disp) == ["황기", "황기", "당귀", "당귀"], list(disp)
    # 일반 명칭이 아닌 품목(분류명 외 단어가 남음)은 HS가 달라도 그대로
    _, key, _ = f(["[비타민 C] 아세로라 C(100정)", "[비타민 C] 아세로라 C(300정)"], ["2936270000", "2106909099"])
    assert key[0] == key[1], list(key)


def test_pack_month_markers():
    """⑥: 직전 달보다 주력 포장 단위가 바뀐 달에 마커 (25KG 80% → 600KG 60%)."""
    df = pd.DataFrame({"ym": ["2026-01", "2026-01", "2026-02", "2026-02", "2026-03", "2026-03"],
                       "pack": ["25KG", "600KG", "25KG", "600KG", "25KG", "600KG"],
                       "vol": [800, 200, 400, 600, 380, 620]})
    marks = NS["pack_month_markers"](df, "ym", "pack", "vol")
    assert [m for m, _ in marks] == ["2026-02"], marks
    assert marks[0][1] == "포장 구성 변화: 25KG 80% → 600KG 60%", marks


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")
