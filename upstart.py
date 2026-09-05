"""실행: python -m streamlit run upstart.py"""

import csv
from datetime import datetime
from io import StringIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
IWM_URL = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
IWM_CSV_URL = IWM_URL + "/latest-holdings.csv"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
RESULT_COLUMNS = ["Ticker", "종목명", "현재가", "고가일수", "저가일수", "거래량 비율", "최대 낙폭 (%)", "3개월 가격변동폭 (%)"]


def fetch_text(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    return response.content.decode("utf-8-sig")


def parse_iwm(text):
    """CSV 앞의 메타데이터와 뒤의 면책문구를 제외하고 주식만 추출한다."""
    rows = list(csv.reader(StringIO(text)))
    header_index = next(
        (i for i, row in enumerate(rows) if {"Ticker", "Name", "Asset Class"} <= set(row)),
        None,
    )
    if header_index is None:
        raise ValueError("IWM CSV에서 종목 목록 헤더를 찾지 못했습니다.")
    header = rows[header_index]
    records = [dict(zip(header, row)) for row in rows[header_index + 1:] if len(row) == len(header)]
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("IWM 종목 목록이 비어 있습니다.")
    frame = frame.loc[frame["Asset Class"].eq("Equity"), ["Ticker", "Name"]]
    asof = next((row[1] for row in rows[:header_index] if len(row) > 1 and "as of" in row[0].lower()), "미표시")
    return frame.rename(columns={"Name": "종목명"}), asof


@st.cache_data(ttl=86400, show_spinner=False)
def load_universe(index_name):
    if index_name == "SP500":
        tables = pd.read_html(StringIO(fetch_text(SP500_URL)))
        table = next((t for t in tables if {"Symbol", "Security"} <= set(t.columns)), None)
        if table is None:
            raise ValueError("S&P 500 목록의 형식이 변경되었습니다.")
        frame = table[["Symbol", "Security"]].rename(columns={"Symbol": "Ticker", "Security": "종목명"})
        source = f"[S&P 500 종목 목록]({SP500_URL})"
    else:
        frame, asof = parse_iwm(fetch_text(IWM_CSV_URL))
        source = f"[iShares IWM 주식 보유 목록]({IWM_URL}) · 기준일: {asof}"
    frame = frame.dropna().copy()
    frame["Ticker"] = frame["Ticker"].str.strip().str.upper().str.replace(".", "-", regex=False)
    frame = frame.loc[frame["Ticker"].str.fullmatch(r"[A-Z][A-Z0-9-]*", na=False)]
    frame = frame.drop_duplicates("Ticker").sort_values("Ticker").reset_index(drop=True)
    if frame.empty:
        raise ValueError("사용 가능한 종목이 없습니다.")
    return frame, source


def extract_prices(raw, ticker):
    if raw is None or raw.empty:
        return pd.DataFrame(columns=OHLCV)
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            if ticker in frame.columns.get_level_values(level):
                frame = frame.xs(ticker, axis=1, level=level)
                break
        else:
            return pd.DataFrame(columns=OHLCV)
    if not set(OHLCV) <= set(frame.columns):
        return pd.DataFrame(columns=OHLCV)
    frame = frame[OHLCV].apply(pd.to_numeric, errors="coerce")
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    # 다른 종목의 거래일 정렬로 생긴 빈 행만 제거한다. 부분 결측은 검증에서 제외한다.
    return frame.dropna(how="all")


def download_batch(tickers, period):
    return yf.download(
        tickers=list(tickers), period=period, interval="1d", group_by="ticker",
        auto_adjust=False, progress=False, threads=4, timeout=20,
    )


def prepare_window(frame):
    """전일 종가용 1일을 포함한 31개 봉을 검증한 뒤 30거래일을 반환한다."""
    recent = frame.tail(31).copy()
    if len(recent) < 31:
        raise ValueError("전일 종가 포함 31거래일 부족")
    if not np.isfinite(recent[OHLCV].to_numpy()).all():
        raise ValueError("가격 또는 거래량 결측")
    if (recent[["Open", "High", "Low", "Close"]] <= 0).any().any() or (recent["Volume"] < 0).any():
        raise ValueError("가격 또는 거래량 비정상")
    recent["전일Close"] = recent["Close"].shift(1)
    recent["고가비"] = recent["High"] / recent["전일Close"] - 1
    recent["저가비"] = recent["Low"] / recent["전일Close"] - 1
    return recent.tail(30)


def receive_prices(tickers, progress):
    frames, failures = {}, {}
    batches = [tickers[i:i + 50] for i in range(0, len(tickers), 50)]
    for number, batch in enumerate(batches, 1):
        progress.progress((number - 1) / len(batches), text=f"가격 수신: {number}/{len(batches)} 묶음")
        try:
            raw = download_batch(batch, "3mo")
            missing = []
            for ticker in batch:
                frame = extract_prices(raw, ticker)
                if frame.empty:
                    missing.append(ticker)
                else:
                    frames[ticker] = frame
        except Exception:
            missing = batch
        if missing:
            time.sleep(1)
            try:
                retry = download_batch(missing, "3mo")
                for ticker in missing:
                    frame = extract_prices(retry, ticker)
                    if frame.empty:
                        failures[ticker] = "수신 실패 (재시도 포함)"
                    else:
                        frames[ticker] = frame
            except Exception as exc:
                for ticker in missing:
                    failures[ticker] = f"수신 실패: {exc}"
    windows = {}
    latest_date = max((f.index[-1] for f in frames.values()), default=None)
    for ticker, frame in frames.items():
        try:
            if frame.index[-1] != latest_date:
                raise ValueError(f"최신 거래일 누락 (마지막: {frame.index[-1]:%Y-%m-%d})")
            window = prepare_window(frame)
            history = frame.loc[frame.index >= frame.index[-1] - pd.DateOffset(months=3)]
            values = history[["High", "Low"]]
            valid_range = np.isfinite(values.to_numpy()).all() and (values > 0).all().all()
            window.attrs["price_range_pct"] = (
                float((history["High"].max() - history["Low"].min()) / frame["Close"].iloc[-1] * 100)
                if valid_range else np.nan
            )
            windows[ticker] = window
        except ValueError as exc:
            failures[ticker] = str(exc)
    progress.progress(1.0, text=f"완료: 분석 가능 {len(windows):,}개 / 제외 {len(failures):,}개")
    return windows, failures


def condition_masks(frame, use_high, high_pct, use_low, low_pct):
    # 부동소수점 오차로 정확히 경계에 있는 거래일이 탈락하지 않도록 허용 오차 적용.
    high = frame["고가비"].ge(high_pct / 100 - 1e-12) & use_high
    low = frame["저가비"].le(-low_pct / 100 + 1e-12) & use_low
    return high, low


def screen_prices(windows, names, use_high, high_pct, use_low, low_pct, match_all, cancel_high=False,
                  price_range=None, drawdown_range=None):
    rows = []
    for ticker, frame in windows.items():
        high, low = condition_masks(frame, use_high, high_pct, use_low, low_pct)
        checks = ([bool(high.any())] if use_high else []) + ([bool(low.any())] if use_low else [])
        if checks and not (all(checks) if match_all else any(checks)):
            continue
        if cancel_high and use_high and high.any():
            # 고가비가 같은 최대 상승일이 여러 개면 가장 최근 거래일을 사용한다.
            max_rise_day = frame.loc[frame["고가비"].eq(frame["고가비"].max())].iloc[-1]
            if frame["Close"].iloc[-1] < max_rise_day["Low"]:
                continue
        event = high | low
        baseline_volume = frame.loc[~event, "Volume"].sum()
        ratio = frame.loc[event, "Volume"].sum() / baseline_volume if baseline_volume > 0 else np.nan
        close = frame["Close"]
        max_drawdown_pct = float((close / close.cummax() - 1).min() * 100)
        price_range_pct = frame.attrs.get("price_range_pct", np.nan)
        if price_range is not None and not (price_range[0] - 1e-10 <= price_range_pct <= price_range[1] + 1e-10):
            continue
        if drawdown_range is not None and not (drawdown_range[0] - 1e-10 <= -max_drawdown_pct <= drawdown_range[1] + 1e-10):
            continue
        rows.append([ticker, names.get(ticker, ticker), close.iloc[-1], int(high.sum()), int(low.sum()), ratio, max_drawdown_pct, price_range_pct])
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values(
        ["거래량 비율", "Ticker"], ascending=[False, True], na_position="last",
    ).reset_index(drop=True)


@st.cache_data(ttl=900, max_entries=100, show_spinner=False)
def load_chart_prices(ticker, received_at):
    # 1년 표시 구간 시작점에서도 110거래일 이동평균이 나오도록 여유분 수신.
    frame = extract_prices(download_batch([ticker], "2y"), ticker)
    frame = frame.dropna(subset=OHLCV)
    if frame.empty:
        raise ValueError("차트 데이터를 받지 못했습니다.")
    for days in (28, 55, 110):
        frame[f"MA{days}"] = frame["Close"].rolling(days).mean()
    return frame.loc[frame.index >= frame.index[-1] - pd.DateOffset(years=1)]


def make_chart(frame, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=frame.index, open=frame["Open"], high=frame["High"], low=frame["Low"], close=frame["Close"],
        name=ticker, increasing_line_color="#ef5350", decreasing_line_color="#2979ff",
    ), row=1, col=1)
    for days, color in zip((28, 55, 110), ("#ff9800", "#26a69a", "#ab47bc")):
        fig.add_trace(go.Scatter(x=frame.index, y=frame[f"MA{days}"], name=f"{days}일 이동평균", line={"color": color, "width": 1.5}), row=1, col=1)
    colors = np.where(frame["Close"] >= frame["Open"], "#ef5350", "#2979ff")
    fig.add_trace(go.Bar(x=frame.index, y=frame["Volume"], name="거래량", marker_color=colors), row=2, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False, hovermode="x unified", legend_orientation="h", margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="가격 (USD)", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    return fig


def main():
    st.set_page_config(page_title="Upstart · 변동성 종목 검색", layout="wide")
    st.title("Upstart · 변동성 종목 검색")
    st.caption("최근 30거래일의 고가·저가 변동률로 종목을 검색합니다.")
    index_name = st.selectbox("대상 지수", ["SP500", "Russell2000"])
    if index_name == "Russell2000":
        st.info("Russell 2000 추종 ETF인 IWM의 주식 보유 목록을 사용합니다. 공식 지수 구성 종목과 차이가 있을 수 있습니다.")
    if st.button("데이터 수신", type="primary"):
        try:
            # 버튼을 누를 때 목록과 가격을 새로 수신한다.
            load_universe.clear(index_name)
            with st.spinner("인터넷에서 종목 목록을 가져오는 중..."):
                universe, source = load_universe(index_name)
            progress = st.progress(0, text=f"{len(universe):,}개 종목 가격 수신 준비")
            windows, failures = receive_prices(universe["Ticker"].tolist(), progress)
            st.session_state["upstart_data"] = dict(
                index_name=index_name, windows=windows, failures=failures, source=source,
                names=universe.set_index("Ticker")["종목명"].to_dict(),
                received_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            st.error(f"데이터 수신에 실패했습니다: {exc}")
            st.info("기존 수신 데이터가 있으면 아래에 유지됩니다.")

    st.subheader("검색 조건")
    left, right = st.columns(2)
    with left:
        use_high = st.checkbox("최대상승(%) 이상", value=True)
        high_pct = st.number_input("상승률 (%)", min_value=0.0, value=5.0, step=0.5, disabled=not use_high)
    with right:
        use_low = st.checkbox("최대하락(%) 이상", value=False)
        low_pct = st.number_input("하락률 (%) — 양수 입력", min_value=0.0, max_value=100.0, value=5.0, step=0.5, disabled=not use_low)
    mode = st.radio("두 조건 적용 방식", ["모두 충족 (AND)", "하나 이상 충족 (OR)"], horizontal=True, disabled=not (use_high and use_low))
    st.caption("상승: 고가비 ≥ 상승률 · 하락: 저가비 ≤ −하락률. AND는 각 조건을 충족한 거래일이 한 번 이상 있는 종목입니다(같은 날일 필요 없음).")
    cancel_high = st.checkbox(
        "취소 조건 · 최대상승일 저가보다 현재 종가가 낮으면 제외",
        value=False, disabled=not use_high,
        help="최대상승 조건을 충족한 종목에 적용합니다. 최근 30거래일 중 고가비가 가장 큰 날을 사용하며, 동률이면 가장 최근 날을 사용합니다. 종가가 해당 저가와 같으면 유지합니다.",
    )
    range_left, range_right = st.columns(2)
    with range_left:
        use_price_range = st.checkbox("가격변동폭 범위 적용 (최근 3개월)")
        price_min = st.number_input("가격변동폭 최소 (%)", min_value=0.0, value=0.0, step=1.0, disabled=not use_price_range)
        price_max = st.number_input("가격변동폭 최대 (%)", min_value=0.0, value=50.0, step=1.0, disabled=not use_price_range)
    with range_right:
        use_drawdown_range = st.checkbox("최대낙폭 범위 적용 (최근 30거래일)")
        drawdown_min = st.number_input("최대낙폭 최소 (%) — 양수 입력", min_value=0.0, max_value=100.0, value=0.0, step=1.0, disabled=not use_drawdown_range)
        drawdown_max = st.number_input("최대낙폭 최대 (%) — 양수 입력", min_value=0.0, max_value=100.0, value=30.0, step=1.0, disabled=not use_drawdown_range)
    price_range = (price_min, price_max) if use_price_range else None
    drawdown_range = (drawdown_min, drawdown_max) if use_drawdown_range else None
    st.caption("가격변동폭 = (3개월 최고 High − 최저 Low) / 최신 Close × 100. 최대낙폭은 하락폭 크기를 양수로 입력합니다(예: 5~20% → MDD −20~−5%). 범위 양 끝을 포함하며 기존 조건에 모두 추가 적용합니다.")
    if (use_price_range and price_min > price_max) or (use_drawdown_range and drawdown_min > drawdown_max):
        st.error("범위의 최솟값은 최댓값보다 클 수 없습니다.")
        return
    with st.expander("계산 기준"):
        st.markdown(
            "- 고가비 = 금일 High / 전일 Close − 1, 저가비 = 금일 Low / 전일 Close − 1\n"
            "- 고가일수·저가일수는 체크한 조건을 충족한 거래일수입니다. 해제한 조건은 0일로 표시합니다.\n"
            "- 거래량 비율 = 조건 충족일 거래량 합계 / 나머지 거래일 거래량 합계 (배). 양쪽 조건 충족일은 한 번만 합산합니다.\n"
            "- 최대 낙폭 (%) = 최근 30거래일에서 (종가 / 해당일까지의 기간 내 최고 종가 − 1)의 최솟값 × 100입니다. 하락은 음수, 낙폭이 없으면 0%로 표시합니다.\n"
            "- 3개월 가격변동폭은 수신한 3개월 전체 일봉으로 계산합니다. 신규 상장 등으로 이력이 짧으면 수신 가능한 구간을 사용합니다.\n"
            "- 분모가 0이면 거래량 비율은 빈칸입니다. 모든 검색 옵션을 해제하면 전체 종목을 표시합니다.\n"
            "- 현재가는 수신한 마지막 일봉의 Close이며 실시간 호가가 아닙니다. 장중 수신하면 당일 봉은 미완성일 수 있습니다.\n"
            "- Yahoo Finance의 auto_adjust=False 가격을 사용합니다. 배당 조정은 적용하지 않으며 분할은 공급자의 처리 기준을 따릅니다.\n"
            "- 전일 종가용 1일을 포함한 31거래일이 부족하거나 결측·최신 거래일 누락이 있는 종목은 제외합니다."
        )
    data = st.session_state.get("upstart_data")
    if data is None:
        st.info("지수를 선택한 뒤 ‘데이터 수신’을 눌러 주세요.")
        return
    if data["index_name"] != index_name:
        st.info("선택한 지수가 바뀌었습니다. ‘데이터 수신’을 눌러 주세요.")
        return
    st.caption(f"수신 시각 (서버): {data['received_at']} · 출처: {data['source']} · 가격: Yahoo Finance")
    if data["failures"]:
        with st.expander(f"수신 실패·분석 제외 {len(data['failures']):,}개"):
            st.dataframe(pd.DataFrame(data["failures"].items(), columns=["Ticker", "사유"]), hide_index=True)
    if not data["windows"]:
        st.warning("분석 가능한 데이터가 없습니다. 잠시 후 다시 수신해 주세요.")
        return
    if use_price_range and any("price_range_pct" not in frame.attrs for frame in data["windows"].values()):
        st.warning("기존 수신 데이터에는 3개월 가격변동폭이 없습니다. ‘데이터 수신’을 다시 눌러 주세요.")
        return
    result = screen_prices(data["windows"], data["names"], use_high, high_pct, use_low, low_pct, mode.startswith("모두"), cancel_high=cancel_high,
                           price_range=price_range, drawdown_range=drawdown_range)
    st.subheader(f"검색 결과 · {len(result):,} / {len(data['windows']):,}개 종목")
    if result.empty:
        st.info("조건을 충족한 종목이 없습니다.")
        return
    editor_key = f"upstart_selection_{data['received_at']}_{index_name}_{use_high}_{high_pct}_{use_low}_{low_pct}_{mode}_{cancel_high}_{price_range}_{drawdown_range}"
    selection = st.dataframe(
        result, key=editor_key, hide_index=True, use_container_width=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "현재가": st.column_config.NumberColumn("현재가 (USD)", format="%.2f"),
            "거래량 비율": st.column_config.NumberColumn("거래량 비율 (배)", format="%.4f"),
            "3개월 가격변동폭 (%)": st.column_config.NumberColumn("3개월 가격변동폭 (%)", format="%.2f%%"),
            "최대 낙폭 (%)": st.column_config.NumberColumn(
                "최대 낙폭 (%)", format="%.2f%%",
                help="최근 30거래일 종가 기준 최대 낙폭(MDD). 예: -12.50%",
            ),
        },
    )
    st.download_button("검색 결과 CSV 다운로드", result.to_csv(index=False).encode("utf-8-sig"), "upstart_results.csv", "text/csv")
    selected = selection["selection"]["rows"]
    if not selected:
        st.info("표에서 Ticker 행을 선택하면 해당 종목의 1년 캔들차트와 거래량을 바로 표시합니다.")
        return
    ticker = result.iloc[selected[0]]["Ticker"]
    st.subheader(f"{ticker} · {data['names'][ticker]}")
    try:
        with st.spinner("1년 차트 데이터를 가져오는 중..."):
            chart_data = load_chart_prices(ticker, data["received_at"])
        st.plotly_chart(make_chart(chart_data, ticker), use_container_width=True)
        st.caption(f"차트: {chart_data.index[0]:%Y-%m-%d} ~ {chart_data.index[-1]:%Y-%m-%d} · 이동평균은 거래일 기준 단순평균입니다.")
    except Exception as exc:
        st.error(f"{ticker} 차트 수신 실패: {exc}")
    with st.expander("30거래일 원본 및 고가비·저가비", expanded=False):
        detail = data["windows"][ticker].copy()
        high, low = condition_masks(detail, use_high, high_pct, use_low, low_pct)
        detail["고가조건 충족"], detail["저가조건 충족"] = high, low
        detail.index.name = "Date"
        display = detail.rename(columns={"고가비": "고가비 (%)", "저가비": "저가비 (%)"})
        display[["고가비 (%)", "저가비 (%)"]] *= 100
        st.dataframe(display, use_container_width=True)
        st.download_button("30거래일 데이터 CSV 다운로드", detail.to_csv().encode("utf-8-sig"), f"{ticker}_30days.csv", "text/csv")


if __name__ == "__main__":
    # Yahoo 캐시도 프로젝트 안에 두어 실행 계정의 홈 쓰기 권한에 의존하지 않는다.
    yf.set_tz_cache_location(str(Path(__file__).resolve().parent / ".cache" / "upstart_yfinance"))
    main()
