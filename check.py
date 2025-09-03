# streamlit_vat_checker.py
import io
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="계산서 발행 체크 (부가세 10%)", layout="wide")
st.title("계산서 발행 체크 - 부가세 10% 대상자 정리 & 매칭")

st.markdown("""
**사용 흐름**
1) 정산내역서(여러 날짜 시트) 업로드 → 날짜(시트) 선택 → 1차 추출  
2) (선택) 매입전자세금계산서목록 업로드 → **매칭 실행**(금액 ±100원) → `계산서발행여부` + `상호명` 표시  
- 추출 규칙(고정): 좌(M)=**H/G**, 우(L)=**U/T**, 주변에 **‘부가세’ 포함 & ‘3.3/원천’ 배제**를 기본으로 판단  
- 대표자 중복은 **합산** → 대표자 고유 인원 기준
""")

# -----------------------------
# 공통 유틸
# -----------------------------
def to_amount(s: pd.Series) -> pd.Series:
    s = (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace("\n", "", regex=False)
         .str.replace("\r", "", regex=False)
    )
    return pd.to_numeric(s, errors="coerce")

def looks_like_korean_name(x: str) -> bool:
    if not isinstance(x, str): return False
    s = x.strip()
    if not s: return False
    if s.lower() in {"nan", "none", "nil"}: return False
    return bool(re.fullmatch(r"[가-힣]{2,4}", s))

def side_text_window(df: pd.DataFrame, row_idx: int, start_col: int, end_col: int,
                     w_before: int = 12, w_after: int = 4) -> str:
    ncols = df.shape[1]
    r0 = max(0, row_idx - w_before)
    r1 = min(len(df)-1, row_idx + w_after)
    parts: List[str] = []
    for r in range(r0, r1+1):
        row = df.iloc[r, :]
        for j in range(start_col, min(end_col+1, ncols)):
            v = row.iloc[j]
            if pd.isna(v): 
                continue
            parts.append(str(v))
    return " ".join(parts)

def collect_section_ranges(df: pd.DataFrame, start_col: int, end_col: int,
                           vat_keys: List[str], excl_keys: List[str]) -> List[Tuple[int, int]]:
    """‘부가세’가 있는 행 직후부터 다음 ‘3.3/원천’ 전까지를 VAT 10% 섹션으로 인식."""
    ncols = df.shape[1]
    def has_any_in_row(i, keys):
        row = df.iloc[i, :]
        for j in range(start_col, min(end_col+1, ncols)):
            v = row.iloc[j]
            if pd.isna(v): 
                continue
            s = str(v)
            if any(k in s for k in keys):
                return True
        return False

    vat_rows  = [i for i in range(len(df)) if has_any_in_row(i, vat_keys)]
    excl_rows = [i for i in range(len(df)) if has_any_in_row(i, excl_keys)]

    ranges = []
    for v0 in sorted(vat_rows):
        nxt = None
        for e in sorted(excl_rows):
            if e > v0:
                nxt = e
                break
        a = v0 + 1
        b = (nxt - 1) if nxt is not None else (len(df) - 1)
        if a <= b:
            ranges.append((a, b))
    # merge overlaps
    merged: List[List[int]] = []
    for a, b in sorted(ranges):
        if not merged or a > merged[-1][1] + 1:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return [(a, b) for a, b in merged]

def in_any_range(i: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(a <= i <= b for a, b in ranges)

# -----------------------------
# 1) 추출 로직 (요청한 기준 고정, 약간 완화 허용)
# -----------------------------
def extract_vat10_reps(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    - 좌(뮤즈): H(대표자), G(부가세 포함 합계); 컨텍스트 A..N
    - 우(르오): U(대표자), T(부가세 포함 합계); 컨텍스트 O..끝
    - 컨텍스트에 '부가세' 포함 & '3.3/원천' 미포함이면 포함
      (섹션 탐지 + 근접 윈도우 판단을 함께 사용: 누락 최소화, 약간의 오탐 허용)
    - 대표자 중복 합산 → 날짜 | 대표자 | 부가세10% 금액 | 계산서발행여부
    - 금액은 소수 2자리까지
    """
    ncols = df.shape[1]
    G = to_amount(df.iloc[:, 6])    if ncols > 6  else pd.Series([np.nan]*len(df))
    H = df.iloc[:, 7].astype(str).str.strip() if ncols > 7  else pd.Series([""]*len(df))
    T = to_amount(df.iloc[:, 19])   if ncols > 19 else pd.Series([np.nan]*len(df))
    U = df.iloc[:, 20].astype(str).str.strip() if ncols > 20 else pd.Series([""]*len(df))

    LEFT_START, LEFT_END   = 0, 13   # A..N
    RIGHT_START, RIGHT_END = 14, ncols-1  # O..끝

    VAT_KEYS  = ["부가세", "VAT"]
    EXCL_KEYS = ["3.3", "3,3", "3 . 3", "3%", "원천", "원천징수"]

    rows: List[Tuple[str, str, float]] = []

    # 섹션 탐지
    left_ranges  = collect_section_ranges(df, LEFT_START,  LEFT_END,  VAT_KEYS, EXCL_KEYS)
    right_ranges = collect_section_ranges(df, RIGHT_START, RIGHT_END, VAT_KEYS, EXCL_KEYS)

    for i in range(len(df)):
        # LEFT (뮤즈)
        if pd.notna(G.iloc[i]) and float(G.iloc[i]) > 0 and looks_like_korean_name(H.iloc[i]):
            ctx = side_text_window(df, i, LEFT_START, LEFT_END, w_before=12, w_after=4)
            if (any(k in ctx for k in VAT_KEYS) and not any(k in ctx for k in EXCL_KEYS)) or in_any_range(i, left_ranges):
                rows.append((sheet_name, H.iloc[i], float(G.iloc[i])))

        # RIGHT (르오)
        if pd.notna(T.iloc[i]) and float(T.iloc[i]) > 0 and looks_like_korean_name(U.iloc[i]):
            ctx = side_text_window(df, i, RIGHT_START, RIGHT_END, w_before=12, w_after=4)
            if (any(k in ctx for k in VAT_KEYS) and not any(k in ctx for k in EXCL_KEYS)) or in_any_range(i, right_ranges):
                rows.append((sheet_name, U.iloc[i], float(T.iloc[i])))

    if not rows:
        return pd.DataFrame(columns=["날짜","대표자","부가세10% 금액","계산서발행여부","상호명"])

    out = pd.DataFrame(rows, columns=["날짜","대표자","부가세10% 금액"])
    out["부가세10% 금액"] = out["부가세10% 금액"].round(2)
    out = (
        out.groupby(["날짜","대표자"], as_index=False)["부가세10% 금액"]
           .sum()
           .sort_values(["대표자"])
           .reset_index(drop=True)
    )
    out["계산서발행여부"] = ""  # 매칭 전 공란
    out["상호명"] = ""          # 매칭 후 채움
    return out

# -----------------------------
# 2) 매입전자세금계산서목록 로딩 & 금액/상호 보존
# -----------------------------
def detect_amount_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        s = str(c)
        if any(k in s for k in ["합계", "합계금액", "총액", "공급가액+세액", "세액포함금액", "공급대가"]):
            return c
    # 없으면 숫자비율 가장 높은 열
    best_col, best_ratio = None, -1
    for c in df.columns:
        r = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce").notna().mean()
        if r > best_ratio:
            best_ratio = r
            best_col = c
    return best_col

def load_hometax_amounts(upload) -> pd.DataFrame:
    """
    매입전자세금계산서목록에서 금액(O열)과 상호명(G열)을 보존.
    """
    # 첫 번째 시트 읽기
    raw = pd.read_excel(upload, sheet_name=0, header=0)

    # 금액(O열)과 상호명(G열) 추출
    amt = pd.to_numeric(raw.iloc[:, 14]  # O열 (0-base 인덱스라 14번째가 O열)
                        .astype(str).str.replace(",", "", regex=False),
                        errors="coerce").fillna(0.0)
    supplier = raw.iloc[:, 6].astype(str).str.strip()  # G열

    out = pd.DataFrame({
        "원본행": np.arange(len(raw)),
        "합계금액": amt,
        "상호명": supplier
    })
    return out


# -----------------------------
# 3) 매칭 (금액 ±100원, 단건→2건 합산) + 상호명 표기
# -----------------------------
def _display_name_for_rows(ht: pd.DataFrame, idx_list: List[int]) -> str:
    """
    홈택스 DF에서 행 인덱스 리스트에 대해 표시할 '상호명' 생성.
    단건: 해당 행의 상호 필드 중 우선순위 컬럼
    2건: 두 행의 이름을 ' / '로 연결 (중복 제거)
    """
    name_cols_priority = ["상호","상호명","공급자상호","공급자상호명","공급자명","거래처","공급자","공급받는자"]
    names: List[str] = []
    for i in idx_list:
        label = ""
        for c in name_cols_priority:
            if c in ht.columns:
                v = ht.loc[i, c]
                if pd.notna(v) and str(v).strip():
                    label = str(v).strip()
                    break
        names.append(label if label else f"행{i}")
    # 중복 제거
    uniq: List[str] = []
    for n in names:
        if n not in uniq:
            uniq.append(n)
    return " / ".join(uniq)

def match_by_amount(extracted: pd.DataFrame, hometax: pd.DataFrame, tol: float = 100.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    반환 result 컬럼:
    날짜 | 대표자 | 부가세10% 금액 | 계산서발행여부 | 매칭유형 | 매칭금액합 | 사용행 | 상호명
    """
    ex = extracted.copy().sort_values(["날짜","부가세10% 금액"], ascending=[True, False]).reset_index(drop=True)
    ht = hometax.copy().reset_index(drop=True)

    used = set()
    matches = []
    ht_list = [(i, float(ht.loc[i, "합계금액"])) for i in range(len(ht))]

    def find_single(target: float):
        cand = [(i, a, abs(a - target)) for (i, a) in ht_list if i not in used and abs(a - target) <= tol]
        if not cand: return None
        cand.sort(key=lambda x: (x[2], abs(x[1])))
        return cand[0]  # (idx, amt, diff)

    def find_pair(target: float):
        best = None
        n = len(ht_list)
        for j in range(n):
            i1, a1 = ht_list[j]
            if i1 in used: continue
            for k in range(j+1, n):
                i2, a2 = ht_list[k]
                if i2 in used: continue
                s = a1 + a2
                d = abs(s - target)
                if d <= tol and (best is None or d < best[4]):
                    best = (i1, a1, i2, a2, d)
        return best  # (i1, a1, i2, a2, diff)

    for _, r in ex.iterrows():
        date = r["날짜"]; name = r["대표자"]; amt = float(r["부가세10% 금액"])

        # 1) 단건
        sg = find_single(amt)
        if sg is not None:
            i, a, _ = sg
            used.add(i)
            supplier = _display_name_for_rows(ht, [int(i)])
            matches.append([date, name, amt, "발행완료", "단건", a, [int(i)], supplier])
            continue

        # 2) 2건 합산
        pr = find_pair(amt)
        if pr is not None:
            i1, a1, i2, a2, _ = pr
            used.add(i1); used.add(i2)
            supplier = _display_name_for_rows(ht, [int(i1), int(i2)])
            matches.append([date, name, amt, "발행완료", "2건합산", a1 + a2, [int(i1), int(i2)], supplier])
            continue

        # 3) 미발행
        matches.append([date, name, amt, "미발행", "", np.nan, [], ""])

    result = pd.DataFrame(matches, columns=["날짜","대표자","부가세10% 금액","계산서발행여부","매칭유형","매칭금액합","사용행","상호명"])
    leftover = ht.loc[[i for (i, _) in ht_list if i not in used]].reset_index(drop=True)
    return result, leftover

# -----------------------------
# UI ① 정산내역서 업로드 & 날짜 선택
# -----------------------------
st.sidebar.header("① 정산내역서 업로드 & 날짜(시트) 선택")
settle_file = st.sidebar.file_uploader("정산내역서 엑셀 (여러 날짜 시트)", type=["xlsx","xls"])
date_hint = st.sidebar.text_input("기본 표시 시트 (예: 25.08.29)", value="25.08.29")

if not settle_file:
    st.info("정산내역서를 업로드하세요.")
    st.stop()

xls = pd.ExcelFile(settle_file)
all_sheets = xls.sheet_names
date_like = [s for s in all_sheets if re.fullmatch(r"\d{2}\.\d{2}\.\d{2}", str(s))]
default_sel = date_hint if date_hint in all_sheets else (date_like[-1] if date_like else all_sheets[0])
sheet_name = st.sidebar.selectbox("추출할 날짜(시트)", options=all_sheets, index=all_sheets.index(default_sel))

# -----------------------------
# 1차 추출
# -----------------------------
st.header("1) 부가세 10% 대상자 추출")
src = pd.read_excel(settle_file, sheet_name=sheet_name)
extracted = extract_vat10_reps(src, sheet_name)

if extracted.empty:
    st.error("추출 결과가 없습니다. (부가세 구간/열 위치/3.3 표기 등을 확인하세요)")
    st.stop()

display_df = extracted[["날짜","대표자","부가세10% 금액"]].copy()
display_df["계산서발행여부"] = ""  # 매칭 전 공란
st.dataframe(display_df, use_container_width=True)

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    display_df.to_excel(w, index=False, sheet_name="정리결과(매칭전)")
st.download_button("엑셀 다운로드 - 정리결과(매칭 전)", data=buf.getvalue(),
                   file_name=f"{sheet_name}_정리결과_부가세10.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# -----------------------------
# UI ② 매칭
# -----------------------------
st.header("2) 매입전자세금계산서목록 업로드 & 매칭")
ht_file = st.file_uploader("매입전자세금계산서목록 (최근 1달)", type=["xlsx","xls"])
tol = st.number_input("금액 오차 허용(원)", min_value=0, max_value=10000, value=100, step=3)

if ht_file:
    hometax_df = load_hometax_amounts(ht_file)
    st.caption(f"매입전자세금계산서목록 행수: {len(hometax_df)}")
    if st.button("매칭 실행"):
        matched, leftover = match_by_amount(extracted[["날짜","대표자","부가세10% 금액"]], hometax_df, tol=tol)

        st.subheader("매칭 결과 (날짜 | 대표자 | 부가세10% 금액 | 계산서발행여부 | 상호명)")
        final = matched[["날짜","대표자","부가세10% 금액","계산서발행여부","상호명"]].copy()
        st.dataframe(final, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**미사용(남은) 매입 세금계산서**")
            st.dataframe(leftover, use_container_width=True)
        with c2:
            done = (final["계산서발행여부"]=="발행완료").sum()
            total = len(final)
            st.metric("발행완료 / 총 건수", f"{done} / {total}")

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            final.to_excel(w, index=False, sheet_name="매칭결과(요약)")             # 상호명 포함
            matched.to_excel(w, index=False, sheet_name="매칭결과(세부)")           # 사용행/매칭유형 등 전체
            leftover.to_excel(w, index=False, sheet_name="남은세금계산서")
        st.download_button("엑셀 다운로드 - 매칭결과(상호명 포함)", data=out.getvalue(),
                           file_name=f"{sheet_name}_부가세10_매칭결과_상호명포함.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("매칭하려면 매입전자세금계산서목록 파일을 업로드하세요.")

