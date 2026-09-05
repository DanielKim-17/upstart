import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import gspread
from google.oauth2.service_account import Credentials
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.set_page_config(page_title="My Favorite Stock", layout="wide")


SHEET_NAME = os.getenv("MYFAVORITE_SHEET_NAME", "myfavorite")
CATEGORIES = ("매수", "매도", "매수대기", "매도대기", "기타")
ADD_BUY_COLUMNS = ("1차 추가매수", "2차 추가매수")

CREDENTIAL_ENV_KEYS = (
    "GOOGLE_SERVICE_ACCOUNT_JSON",
    "GOOGLE_SHEET_CREDENTIALS",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def normalize_ticker(raw: str) -> str:
    """Yahoo Finance 티커를 정규화한다."""
    if raw is None:
        return ""
    ticker = str(raw).strip()
    if not ticker:
        return ""
    ticker = ticker.replace(" ", "")
    if "." in ticker:
        return ticker.upper()
    if ticker.isdigit():
        return f"{ticker}.KS"
    return ticker.upper()


def _read_json_from_string(value: str) -> Optional[dict]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


def get_service_account_info() -> Optional[dict]:
    for key in CREDENTIAL_ENV_KEYS:
        value = os.getenv(key)
        if value:
            if value.startswith("{"):
                info = _read_json_from_string(value)
                if info:
                    return info
            path = Path(value)
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    if isinstance(info, dict):
                        return info
                except Exception:
                    pass

    default_candidates = [
        Path(__file__).resolve().parent / "google-service-account.json",
        Path.cwd() / "google-service-account.json",
        Path.home() / "google-service-account.json",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    info = json.load(f)
                if isinstance(info, dict):
                    return info
            except Exception:
                pass

    try:
        secrets = st.secrets
        secret_map = {}
        try:
            secret_map = st.secrets.to_dict()
        except Exception:
            try:
                secret_map = dict(secrets)
            except Exception:
                secret_map = {}

        def normalize_key(name: str) -> str:
            return str(name).strip().lower().replace("-", "_")

        normalized = {normalize_key(k): v for k, v in secret_map.items()}
        direct_keys = (
            "google_service_account",
            "gcp_service_account",
            "service_account",
            "google_service_account_json",
            "google_service_account_info",
            "google_service_account_credentials",
            "google_service_account_key",
        )
        for key in direct_keys:
            value = normalized.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                info = _read_json_from_string(value)
                if info:
                    return info

        if "connections" in normalized:
            conn = normalized["connections"]
            if isinstance(conn, dict):
                for key in ("gsheets", "google_sheets", "service_account"):
                    if key in conn and isinstance(conn[key], dict):
                        return conn[key]

        if "gsheets" in normalized and isinstance(normalized["gsheets"], dict):
            return normalized["gsheets"]

        if all(k in normalized for k in ("type", "project_id", "private_key", "client_email")):
            return {k: normalized[k] for k in (
                "type",
                "project_id",
                "private_key_id",
                "private_key",
                "client_email",
                "client_id",
                "auth_uri",
                "token_uri",
                "auth_provider_x509_cert_url",
                "client_x509_cert_url",
                "universe_domain",
            ) if k in normalized}
    except Exception:
        pass

    return None


@st.cache_data(ttl=600)
def load_favorite_sheet() -> pd.DataFrame:
    """Google Sheet의 myfavorite 시트를 읽어 카테고리와 종목 정보를 반환한다."""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    service_account_info = get_service_account_info()
    if not service_account_info:
        raise RuntimeError(
            "Google Service Account 정보가 없습니다. "
            ".streamlit/secrets.toml, GOOGLE_SERVICE_ACCOUNT_JSON, "
            "또는 google-service-account.json 경로를 설정해 주세요."
        )

    creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
    client = gspread.authorize(creds)

    workbook = client.open(SHEET_NAME)
    worksheet = workbook.sheet1
    rows = worksheet.get_all_records()

    if not rows:
        return pd.DataFrame(columns=["Category", "Ticker", "Ticker Name", "Name", *ADD_BUY_COLUMNS])

    df = pd.DataFrame(rows)
    rename_map = {}
    for col in list(df.columns):
        key = str(col).strip().lower().replace(" ", "")
        if key in {"1차추가매수", "2차추가매수"}:
            rename_map[col] = key.replace("차", "차 ", 1)
        if key in {"category", "ticker", "name", "tickername"}:
            rename_map[col] = {"category": "Category", "ticker": "Ticker", "name": "Ticker Name", "tickername": "Ticker Name"}.get(key, col)
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Category" not in df.columns:
        raise ValueError("Google Sheet에 'Category' 컬럼이 없습니다.")
    if "Ticker" not in df.columns:
        raise ValueError("Google Sheet에 'Ticker' 컬럼이 없습니다.")

    if "Ticker Name" not in df.columns:
        df["Ticker Name"] = ""

    df["Category"] = df["Category"].fillna("").astype(str).str.strip()
    df["Category"] = df["Category"].where(df["Category"].isin(CATEGORIES), "기타")
    for col in ADD_BUY_COLUMNS:
        values = df[col] if col in df.columns else pd.Series("", index=df.index)
        df[col] = pd.to_numeric(values.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
        df[col] = df[col].where(np.isfinite(df[col]))
    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip()
    df["Ticker Name"] = df["Ticker Name"].fillna("").astype(str).str.strip()
    df = df[df["Ticker"] != ""].copy()
    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    return df.reset_index(drop=True)


@st.cache_data(ttl=300)
def fetch_daily_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """최근 1년치 주가 데이터를 yfinance에서 가져온다."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval="1d")
        if hist is None or hist.empty:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    if not isinstance(hist, pd.DataFrame):
        return pd.DataFrame()

    hist = hist.copy()
    if isinstance(hist.index, pd.DatetimeIndex):
        hist = hist.reset_index()
    elif hist.index.name is not None:
        hist = hist.rename_axis("Date").reset_index()

    hist.columns = [str(c).strip() for c in hist.columns]

    if "Datetime" in hist.columns and "Date" not in hist.columns:
        hist = hist.rename(columns={"Datetime": "Date"})
    if "Date" not in hist.columns and hist.index.name is not None:
        hist = hist.rename_axis("Date").reset_index()
    if "Date" not in hist.columns:
        return pd.DataFrame()

    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    hist = hist.dropna(subset=["Date"]).copy()
    if hist.empty:
        return pd.DataFrame()

    # Close, High, Low, Volume이 NaN인 행 제거 (배당금/분할 등으로 인한 NaN 제거)
    hist = hist.dropna(subset=["Close", "High", "Low", "Volume"]).copy()
    if hist.empty:
        return pd.DataFrame()

    hist["Date"] = hist["Date"].dt.strftime("%Y-%m-%d")
    hist = hist.sort_values("Date").reset_index(drop=True)
    hist["Ticker"] = ticker
    return hist


@st.cache_data(ttl=300)
def build_stock_metrics(ticker: str) -> pd.DataFrame:
    """티커별 기술지표를 계산한다."""
    hist = fetch_daily_history(ticker, period="1y")
    if hist.empty:
        return pd.DataFrame()

    hist["Open"] = pd.to_numeric(hist.get("Open", pd.Series(index=hist.index, dtype=float)), errors="coerce")
    hist["High"] = pd.to_numeric(hist.get("High", pd.Series(index=hist.index, dtype=float)), errors="coerce")
    hist["Low"] = pd.to_numeric(hist.get("Low", pd.Series(index=hist.index, dtype=float)), errors="coerce")
    hist["Close"] = pd.to_numeric(hist.get("Close", pd.Series(index=hist.index, dtype=float)), errors="coerce")
    hist["Volume"] = pd.to_numeric(hist.get("Volume", pd.Series(index=hist.index, dtype=float)), errors="coerce")

    hist = hist.dropna(subset=["Close", "High", "Low", "Volume"]).reset_index(drop=True)
    if hist.empty:
        return pd.DataFrame()

    hist["PrevClose"] = hist["Close"].shift(1)
    hist["TR1"] = hist["High"] - hist["Low"]
    hist["TR2"] = (hist["High"] - hist["PrevClose"]).abs()
    hist["TR3"] = (hist["Low"] - hist["PrevClose"]).abs()

    n_value = hist.apply(
        lambda r: max(
            float(r["High"] - r["Low"]),
            abs(float(r["High"] - r["PrevClose"])),
            abs(float(r["Low"] - r["PrevClose"])),
        ),
        axis=1,
    )
    hist["Nvalue"] = n_value
    hist["NvalueAvg20"] = hist["Nvalue"].rolling(window=20, min_periods=20).mean()
    hist["NvalueAbs"] = hist["NvalueAvg20"].abs()
    hist["NrateAbs"] = (hist["NvalueAvg20"] / hist["Close"]).abs().replace([np.inf, -np.inf], np.nan)
    hist["Min10"] = hist["Low"].rolling(window=10, min_periods=10).min()
    hist["Min7"] = hist["Low"].rolling(window=7, min_periods=7).min()
    hist["Max10"] = hist["High"].rolling(window=10, min_periods=10).max()
    hist["Max7"] = hist["High"].rolling(window=7, min_periods=7).max()
    hist["Min20"] = hist["Low"].rolling(window=20, min_periods=20).min()
    hist["Moving28"] = hist["Close"].rolling(window=28, min_periods=28).mean()

    delta = hist["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    hist["RSI"] = 100 - (100 / (1 + rs))
    hist["RSI"] = hist["RSI"].replace([np.inf, -np.inf], 50)

    hist["MB"] = hist["Close"].rolling(window=20, min_periods=20).mean()
    std_20 = hist["Close"].rolling(window=20, min_periods=20).std(ddof=0)
    hist["UB"] = hist["MB"] + 2 * std_20
    hist["LB"] = hist["MB"] - 2 * std_20

    diff = hist["Close"].diff()
    volume = hist["Volume"].fillna(0)
    direction = np.where(diff > 0, volume, np.where(diff < 0, -volume, 0))
    hist["OBV"] = pd.Series(direction, index=hist.index).cumsum()

    tp = (hist["High"] + hist["Low"] + hist["Close"]) / 3
    mf = tp * volume
    tp_diff = tp.diff()
    pmf = np.where(tp_diff > 0, mf, 0)
    nmf = np.where(tp_diff < 0, mf, 0)
    pmf_sum = pd.Series(pmf, index=hist.index).rolling(window=14, min_periods=14).sum()
    nmf_sum = pd.Series(nmf, index=hist.index).rolling(window=14, min_periods=14).sum()
    mfi_den = pmf_sum + nmf_sum
    hist["MFI"] = np.where(mfi_den == 0, 50, 100 * pmf_sum / mfi_den)

    return hist


def is_sell_category(category: str) -> bool:
    return category in {"매도", "매도대기"}


def calculate_grade(price, first, second, category: str):
    """경계값은 Grade 1, 누락되거나 역전된 추가매수 가격은 공란."""
    if any(pd.isna(value) for value in (price, first, second)):
        return pd.NA
    if is_sell_category(category):
        if first < second:
            return pd.NA
        return 0 if price > first else 2 if price < second else 1
    if first > second:
        return pd.NA
    return 0 if price < first else 2 if price > second else 1


def format_signal_ticker(row: pd.Series) -> str:
    price, stoploss = row.get("현재가"), row.get("stoploss")
    sell = is_sell_category(row.get("Category", ""))
    extreme = row.get("Max7" if sell else "Min7")
    moving = row.get("Moving28")
    signal = "🟡"
    if all(pd.notna(value) for value in (price, stoploss, extreme, moving)):
        if (price > stoploss) if sell else (price < stoploss):
            signal = "🔴"
        elif (price <= extreme and price <= moving) if sell else (price >= extreme and price >= moving):
            signal = "🟢"
    return f"{signal} {row.get('Ticker', '')}"


def build_summary(selection_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, favorite in selection_df.iterrows():
        ticker = favorite["Ticker"]
        hist = build_stock_metrics(ticker)
        if hist.empty:
            continue
        hist = hist.dropna(subset=["Close"])
        if len(hist) < 2:
            continue
        previous = hist.iloc[-2]
        price = hist.iloc[-1]["Close"]
        category = favorite["Category"]
        sell = is_sell_category(category)
        extreme = previous["Max7" if sell else "Min7"]
        moving = previous["Moving28"]
        stoploss = np.nan
        if pd.notna(extreme) and pd.notna(moving):
            stoploss = max(extreme, moving) if sell else min(extreme, moving)
        first, second = (favorite.get(col, np.nan) for col in ADD_BUY_COLUMNS)
        row = {
            "Ticker": ticker,
            "Ticker Name": favorite.get("Ticker Name", ""),
            "Category": category,
            "현재가": price,
            "stoploss": stoploss,
            "1차 추가매수": first,
            "2차 추가매수": second,
            "Grade": calculate_grade(price, first, second, category),
        }
        for col in ("Min10", "Min7", "Max10", "Max7", "Moving28", "NvalueAbs", "NrateAbs"):
            row[col] = previous[col]
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["Grade"] = pd.array(summary["Grade"], dtype="Int64")
        summary = summary.sort_values("현재가", ascending=False, na_position="last").reset_index(drop=True)
    return summary


def get_category_table() -> pd.DataFrame:
    favorites = load_favorite_sheet()
    categories = [category for category in CATEGORIES if category in favorites["Category"].values]
    st.sidebar.subheader("Category")
    selected_category = st.sidebar.selectbox("Category 선택", categories, index=0 if categories else None)
    selection_df = favorites[favorites["Category"] == selected_category].copy()
    summary = build_summary(selection_df)
    if summary.empty:
        st.warning(f"'{selected_category}' 카테고리에 유효한 종목이 없습니다.")
        st.stop()
    return summary, selected_category, selection_df


def build_detail_chart(df: pd.DataFrame, ticker: str) -> None:
    hist = build_stock_metrics(ticker)
    if hist.empty:
        st.warning(f"{ticker}의 데이터를 가져오지 못했습니다.")
        return

    hist = hist.ffill().bfill()
    hist = hist.tail(252).copy()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.20, 0.15, 0.15],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    x_dates = pd.to_datetime(hist["Date"])

    fig.add_trace(
        go.Candlestick(
            x=x_dates,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="Candles",
            increasing_line_color="red",
            decreasing_line_color="blue",
            increasing_fillcolor="red",
            decreasing_fillcolor="blue",
            opacity=0.9,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=x_dates, y=hist["Moving28"], mode="lines", name="Moving28", line=dict(color="darkorange", width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["MB"], mode="lines", name="MB", line=dict(color="green", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["UB"], mode="lines", name="UB", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["LB"], mode="lines", name="LB", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)

    selected_stock = df.loc[df["Ticker"] == ticker]
    if not selected_stock.empty:
        levels = selected_stock.iloc[0]
        for column, label, color in (
            ("stoploss", "Stoploss", "red"),
            ("1차 추가매수", "1차 추가매수", "royalblue"),
            ("2차 추가매수", "2차 추가매수", "purple"),
        ):
            value = levels.get(column)
            if pd.notna(value):
                fig.add_hline(y=float(value), line_dash="dash", line_color=color,
                              annotation_text=f"{label}: {value:,.2f}", row=1, col=1)

    volume_colors = np.where(hist["Close"] >= hist["Open"], "red", "blue")
    fig.add_trace(
        go.Bar(
            x=x_dates,
            y=hist["Volume"],
            name="Volume",
            marker_color=volume_colors,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(go.Scatter(x=x_dates, y=hist["OBV"], mode="lines", name="OBV", line=dict(color="purple", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["RSI"], mode="lines", name="RSI", line=dict(color="firebrick", width=2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x_dates, y=hist["MFI"], mode="lines", name="MFI", line=dict(color="teal", width=2)), row=4, col=1)

    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        rangeslider_visible=False,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        rangebreaks=[{"bounds": ["sat", "mon"]}],
        row=2,
        col=1,
    )
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        row=3,
        col=1,
    )
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        row=4,
        col=1,
    )

    fig.update_layout(
        title=f"{ticker} 최근 1년 차트 (Candles / Volume / OBV / RSI / MFI)",
        template="plotly_white",
        height=950,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="OBV", row=3, col=1)
    fig.update_yaxes(title_text="Indicator", row=4, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # 최근 10일의 저가를 가로 테이블로 표시
    st.subheader("최근 10일 저가 (Low)")
    hist_last_10 = hist.tail(10).copy()
    low_table = hist_last_10[["Date", "Low"]].copy()
    low_table["Low"] = low_table["Low"].round(2)
    
    # 가로 형태로 표시하기 위해 전치
    low_table_transposed = low_table.set_index("Date").T
    st.dataframe(low_table_transposed, use_container_width=True)


def main() -> None:
    try:
        favorites = load_favorite_sheet()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.title("My Favorite Stock")
    st.caption("Google Sheet의 myfavorite 데이터를 읽어 yfinance로 실시간 지표를 계산합니다.")

    if favorites.empty:
        st.warning("myfavorite 시트에 데이터가 없습니다.")
        st.stop()

    categories = [category for category in CATEGORIES if category in favorites["Category"].values]
    selected_category = st.selectbox("Category 선택", categories)
    selected_df = favorites[favorites["Category"] == selected_category].copy()
    summary = build_summary(selected_df)
    if summary.empty:
        st.warning(f"'{selected_category}' 카테고리에 유효한 종목이 없습니다.")
        st.stop()

    # 신호등은 반올림 전 가격으로 계산한다.
    summary_display = summary.copy()
    summary_display["신호등Ticker"] = summary.apply(format_signal_ticker, axis=1)
    summary_display["NrateAbs_repeat"] = summary_display["NrateAbs"]
    display_columns = ["신호등Ticker", "현재가", "Grade", "stoploss", "NrateAbs", "Moving28", "Min7", "Max7", "Min10", "Max10", "NvalueAbs", "NrateAbs_repeat"]
    data_for_table = summary_display[display_columns].rename(columns={
        "현재가": "현주가",
        "Min10": "min10", "Min7": "min7", "Max10": "max10", "Max7": "max7", "Moving28": "moving28",
    })
    styled_table = data_for_table.style.format({
        col: "{:.0f}" if col == "Grade" else "{:.2%}" if col in {"NrateAbs", "NrateAbs_repeat"} else "{:.2f}"
        for col in data_for_table.columns if col != "신호등Ticker"
    }, na_rep="")
    st.subheader(f"'{selected_category}' 종목 스크리닝")
    selected = st.dataframe(
        styled_table,
        column_config={"NrateAbs_repeat": st.column_config.NumberColumn("NrateAbs")},
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if not selected.selection.rows:
        st.info("표에서 종목을 선택하면 최근 1년 추세 차트를 표시합니다.")
        st.stop()

    selected_row_idx = selected.selection.rows[0]
    selected_ticker = summary.loc[selected_row_idx, "Ticker"]
    st.subheader(f"선택 종목: {selected_ticker}")
    build_detail_chart(summary.iloc[[selected_row_idx]], selected_ticker)


if __name__ == "__main__":
    main()
