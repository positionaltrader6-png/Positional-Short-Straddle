"""
NIFTY POSITIONAL SHORT STRADDLE — STREAMLIT APP
===============================================
Monthly ATM straddle, sold the session after the previous monthly expiry and
carried to the next one. Sibling of the intraday app; shares its Angel /
gspread plumbing but almost none of its position logic, because nothing in
the intraday engine was built to survive an overnight gap.

CYCLE
-----
  entry window   the first session after the previous monthly expiry, through
                 the expiry itself
  gate           completed 1-hour candles: ADX(14) < ADXR(14), both falling,
                 India VIX inside [10, 14] and not breaking out
  strike         monthly FUTURES ltp rounded to the nearest 50, then the
                 strike among that one and its neighbours whose CE and PE
                 premiums sit closest together
  management     every completed 15-minute bar, ratio = bigger leg / smaller
                 leg. At RATIO_EXIT (2.0) the whole straddle is closed and a
                 fresh one is deployed as soon as the gate is satisfied again
  blackout       the ratio rule is switched off for the last RATIO_BLACKOUT_
                 DAYS sessions, so the position is simply carried into expiry
  exit           15:15 on expiry day, or the ratio rule outside the blackout

WHAT THIS STRATEGY DOES NOT HAVE
--------------------------------
There is no stop loss. The ratio rule measures the two legs against EACH
OTHER, so a volatility event that inflates both legs together leaves the
ratio near 1.0 and fires nothing while mark-to-market falls. MAX_ADVERSE is
recorded on every straddle for exactly this reason — the backtest cannot tell
you how bad that gets unless the drawdown is logged. Read that column before
sizing anything.

SHEETS
------
  Sheet01_1H          one row per completed hourly bar: HLOC, ADX, ADXR, VIX,
                      and each gate condition separately so a no-entry month
                      can be explained
  Sheet02_15M         one row per completed 15-minute bar while a straddle is
                      open: both premiums, the ratio, straddle value, MTM
  Sheet03_Positions   one row per straddle, OPEN or CLOSED. Also the resume
                      source — the engine rebuilds itself from this tab

RESUME
------
On start the engine reads the newest bar_time in Sheet01 and Sheet02, replays
every completed bar between then and now from historical candles, and only
then goes live. A two-hour outage mid-session is backfilled, not skipped.

SETUP
-----
Streamlit -> Settings -> Secrets:

    [angel]
    api_key = "..."
    client_code = "..."
    password = "..."
    totp_secret = "..."

    [sheets]
    live_sheet_id = "..."
    backtest_sheet_id = "..."

    [gcp_service_account]
    ... contents of service_account.json as TOML ...

    [discord]
    webhook = "..."          # optional

Never put credentials in the repo.
"""

import datetime as dt
import math
import threading
import time
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Positional Straddle", page_icon="◈",
                   layout="wide", initial_sidebar_state="expanded")


# ============================== TIME ========================================
# Streamlit Cloud runs in UTC. Every time decision here — the candle window,
# "today", the poll clock — must be IST or the app silently asks the broker
# for the wrong window and finds no data.
IST = ZoneInfo("Asia/Kolkata")


def now_ist():
    return dt.datetime.now(IST).replace(tzinfo=None)


def today_ist():
    return now_ist().date()


# ============================== CONSTANTS ===================================
NIFTY_INDEX_TOKEN = "99926000"
INDIA_VIX_TOKEN = "99926017"
NSE = "NSE"
NFO = "NFO"
API_SLEEP = 0.35

# Angel interval strings. The hourly series is NOT fetched as ONE_HOUR —
# see resample_hourly for why.
IV_15 = "FIFTEEN_MINUTE"

LOT_SIZE = 65                  # confirmed against the strategy note
LOTS = 1
QTY = LOT_SIZE * LOTS
STRIKE_STEP = 50

# HALF_UP | CEIL — how the futures LTP is seeded to a strike. With the scan
# below enabled this barely matters: rounding only picks the centre of the
# candidate set, and the scan then chooses the real ATM from it.
STRIKE_ROUNDING = "HALF_UP"

# Strikes either side of the rounded centre to test. The winner is the strike
# whose CE and PE premiums are CLOSEST together — the market's own ATM, which
# the forward basis and skew can push a step away from a simple rounding.
# 0 disables the scan and uses the rounded strike as-is.
STRIKE_SCAN = 1

# ---- gate (evaluated on COMPLETED hourly bars only) ----
ADX_PERIOD = 14
FALLING_LOOKBACK = 3           # strictly lower for this many consecutive bars
VIX_LOW, VIX_HIGH = 10.0, 14.0

# "VIX should be in range" was not defined numerically. Implemented as: the
# latest VIX close is not the highest of the last N hourly bars, i.e. it is
# not breaking out upward. 0 disables the check and leaves the band alone.
VIX_RANGE_LOOKBACK = 12        # ~2 sessions of hourly bars

# ADX(14) is Wilder-smoothed, and ADXR looks a further 14 bars back on top of
# it. 14 bars is the period, not the warm-up. At 6 hourly bars per session,
# 10 sessions gives 60 bars, which is where ADX stops drifting. The note said
# 5 days; that is roughly 35 bars and still converging.
WARMUP_SESSIONS = 10

# First and last hourly bar LABEL on which an entry may be taken. A bar
# labelled 14:15 does not complete until 15:15, so allowing it would mean
# entering on the close.
ENTRY_FIRST_BAR = dt.time(9, 15)
ENTRY_LAST_BAR = dt.time(13, 15)

# ---- management ----
# Bigger leg / smaller leg. At 2.0 one premium is double the other.
RATIO_EXIT = 2.0

# Sessions before expiry during which the ratio rule is suspended and the
# straddle is simply carried. Premiums are small and the ratio blows out on
# noise by then, so re-centring here is pure churn.
RATIO_BLACKOUT_DAYS = 2

# NSE trading holidays. Sessions BEFORE today are discovered from the candle
# data itself — a day with no candles was not a session — but the blackout
# needs to know the last two sessions BEFORE expiry, which are in the future
# and cannot be discovered that way. Add each year's list here; a missed
# holiday only shifts the blackout start by one session.
# set(), not {} — bare braces are an empty DICT, which silently breaks
# membership tests and any later .add().
NSE_HOLIDAYS = set([
    # dt.date(2026, 1, 26),
    # dt.date(2026, 3, 4),
])

EXPIRY_EXIT_TIME = dt.time(15, 15)
MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)

# Cap on straddles per monthly cycle. 0 = uncapped.
MAX_STRADDLES_PER_CYCLE = 0

# Premium points added to every fill, each way, for slippage and spread.
SLIPPAGE_PTS = 2.0

POLL_BUFFER_SEC = 20

MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
MASTER_FALLBACK = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

# ---- sheet layout ----
TAB_1H = "Sheet01_1H"
TAB_15M = "Sheet02_15M"
TAB_POS = "Sheet03_Positions"

H1_HEADERS = ["bar_time", "trading_date", "expiry", "open", "high", "low",
              "close", "fut_close", "adx", "adxr", "vix", "adx_below_adxr",
              "adx_falling", "adxr_falling", "vix_in_band", "vix_in_range",
              "gate_open", "position", "note"]

M15_HEADERS = ["bar_time", "trading_date", "expiry", "straddle_num", "strike",
               "ce_symbol", "ce_ltp", "pe_symbol", "pe_ltp", "straddle_value",
               "ratio", "entry_value", "mtm", "days_to_expiry", "blackout"]

POS_HEADERS = ["expiry", "straddle_num", "status", "strike", "entry_time",
               "entry_fut", "ce_symbol", "ce_entry", "pe_symbol", "pe_entry",
               "entry_value", "entry_ratio", "exit_time", "ce_exit", "pe_exit",
               "exit_value", "exit_ratio", "pnl", "max_adverse", "max_favourable",
               "bars_held", "entry_reason", "exit_reason", "entry_scan"]

SHEET_KEYS = {"LIVE": "live_sheet_id", "BACKTEST": "backtest_sheet_id"}
SHEET_FONT = "Tahoma"

ENGINE_THREAD_NAME = "positional_engine"


# ============================== CONNECTIONS =================================
def angel_login():
    from SmartApi import SmartConnect
    import pyotp
    c = st.secrets["angel"]
    o = SmartConnect(api_key=c["api_key"])
    r = o.generateSession(c["client_code"], c["password"],
                          pyotp.TOTP(c["totp_secret"]).now())
    if not r.get("status"):
        raise RuntimeError(f"Angel login failed: {r.get('message', r)}")
    return o


@st.cache_resource(show_spinner=False)
def sheet(kind):
    """Live and backtest results go to separate spreadsheets so a backtest
    sweep can never be mistaken for accumulated live history."""
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/spreadsheets"])
    return gspread.authorize(creds).open_by_key(st.secrets["sheets"][SHEET_KEYS[kind]])


def _j(v):
    """Coerce numpy / pandas scalars into things gspread will accept."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return "" if math.isnan(f) else round(f, 4)
    if isinstance(v, (dt.date, dt.datetime, pd.Timestamp)):
        return str(v)
    return v


def _col(n):
    """1-based column index to an A1 letter, for n up to 52."""
    if n <= 26:
        return chr(ord("A") + n - 1)
    return "A" + chr(ord("A") + n - 27)


def style_tab(ws, ncols):
    """Tahoma, left-aligned horizontally and middle-aligned vertically, per
    the spec. Header row bold on a tint. Applied once at creation — ws.clear()
    wipes values but keeps formatting."""
    last = _col(ncols)
    body = {"horizontalAlignment": "LEFT",
            "verticalAlignment": "MIDDLE",
            "textFormat": {"fontFamily": SHEET_FONT, "fontSize": 10}}
    head = {"horizontalAlignment": "LEFT",
            "verticalAlignment": "MIDDLE",
            "backgroundColor": {"red": 0.93, "green": 0.90, "blue": 0.97},
            "textFormat": {"fontFamily": SHEET_FONT, "fontSize": 10, "bold": True}}
    try:
        ws.format(f"A:{last}", body)
        ws.format(f"A1:{last}1", head)
        ws.freeze(rows=1)
    except Exception:
        pass                      # cosmetic only — never block a write


def _ws(kind, tab, headers, rows=5000):
    import gspread
    sh = sheet(kind)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab, rows=rows, cols=max(len(headers), 12))
        ws.append_row([h.upper() for h in headers])
        style_tab(ws, len(headers))
        return ws


def append_rows(kind, tab, headers, rows):
    if not rows:
        return 0
    ws = _ws(kind, tab, headers)
    ws.append_rows([[_j(v) for v in r] for r in rows],
                   value_input_option="USER_ENTERED")
    return len(rows)


def read_tab(kind, tab):
    import gspread
    try:
        ws = sheet(kind).worksheet(tab)
    except gspread.WorksheetNotFound:
        return pd.DataFrame()
    v = ws.get_all_records()
    if not v:
        return pd.DataFrame()
    df = pd.DataFrame(v)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def update_row(kind, tab, headers, row_idx, values):
    """Rewrite one row in place. Used to flip a position OPEN -> CLOSED
    without appending a duplicate, so Sheet03 stays one row per straddle."""
    ws = _ws(kind, tab, headers)
    last = _col(len(headers))
    ws.update(range_name=f"A{row_idx}:{last}{row_idx}",
              values=[[_j(v) for v in values]],
              value_input_option="USER_ENTERED")


def find_open_row(kind, expiry, straddle_num):
    """1-based sheet row of an OPEN straddle, or None. +2 accounts for the
    header row and pandas' 0-based index."""
    d = read_tab(kind, TAB_POS)
    if d.empty or "status" not in d.columns:
        return None
    for i, r in d.iterrows():
        if (str(r.get("expiry")) == str(expiry)
                and str(r.get("status")).upper() == "OPEN"
                and int(float(r.get("straddle_num") or 0)) == straddle_num):
            return i + 2
    return None


# ============================== DISCORD =====================================
def discord(msg, tag=""):
    """Fire-and-forget. A webhook failure must never interrupt the engine."""
    try:
        url = st.secrets["discord"]["webhook"]
    except Exception:
        return False
    body = {"content": (f"{tag} " if tag else "") + msg}
    for _ in range(2):
        try:
            r = requests.post(url, json=body, timeout=6)
            if r.status_code in (200, 204):
                return True
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2)))
                continue
            return False
        except Exception:
            time.sleep(1)
    return False


# ============================== MARKET DATA =================================
CANDLE_COLS = ["ts", "open", "high", "low", "close", "volume"]


def empty_candles():
    """An empty frame WITH the expected columns. A bare DataFrame has none, so
    downstream `.ts` raises AttributeError instead of just being empty."""
    return pd.DataFrame(columns=CANDLE_COLS)


def candles(smart, token, exchange, frm, to, interval=IV_15):
    p = {"exchange": exchange, "symboltoken": str(token), "interval": interval,
         "fromdate": frm.strftime("%Y-%m-%d %H:%M"),
         "todate": to.strftime("%Y-%m-%d %H:%M")}
    for a in range(3):
        try:
            r = smart.getCandleData(p)
        except Exception:
            time.sleep(API_SLEEP * 4 ** a)
            continue
        time.sleep(API_SLEEP)
        if r.get("status") and r.get("data"):
            df = pd.DataFrame(r["data"], columns=CANDLE_COLS)
            df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
            return df.sort_values("ts").reset_index(drop=True)
        time.sleep(API_SLEEP * 4 ** a)
    return empty_candles()


def resample_hourly(df15):
    """Build hourly bars from 15-minute bars, anchored on 09:15.

    Angel's own ONE_HOUR interval is not used. The NSE session is 6h15m, so
    hourly buckets from 09:15 leave a 15-minute stub at 15:15 that the broker
    returns as a full bar. Feeding that stub into ADX corrupts the series and
    makes it disagree with a chart. Bucketing here lets the stub be dropped.

    Bucketing is by clock offset rather than by counting bars, so a missing
    15-minute bar shifts nothing.
    """
    if df15.empty:
        return empty_candles()
    d = df15.copy()
    anchor = MARKET_OPEN.hour * 60 + MARKET_OPEN.minute
    mins = d.ts.dt.hour * 60 + d.ts.dt.minute - anchor
    d["bucket"] = mins // 60
    d["date"] = d.ts.dt.date
    d = d[(d.bucket >= 0) & (d.bucket <= 5)]          # bucket 6 is the stub
    if d.empty:
        return empty_candles()
    g = d.groupby(["date", "bucket"], sort=True)
    out = g.agg(open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum")).reset_index()
    out["ts"] = [dt.datetime.combine(r.date, MARKET_OPEN)
                 + dt.timedelta(hours=int(r.bucket)) for r in out.itertuples()]
    return out[CANDLE_COLS].sort_values("ts").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def instrument_master():
    """NIFTY options plus NIFTY futures, from the Angel scrip master."""
    try:
        r = requests.get(MASTER_URL, timeout=30)
        r.raise_for_status()
    except Exception:
        r = requests.get(MASTER_FALLBACK, timeout=30)
        r.raise_for_status()
    opt_lookup, fut_lookup, expiries = {}, {}, set()
    for i in r.json():
        if i.get("name") != "NIFTY":
            continue
        it = i.get("instrumenttype")
        if it == "OPTIDX":
            opt_lookup[i["symbol"]] = i["token"]
            try:
                expiries.add(dt.datetime.strptime(i["expiry"], "%d%b%Y").date())
            except Exception:
                pass
        elif it == "FUTIDX":
            try:
                e = dt.datetime.strptime(i["expiry"], "%d%b%Y").date()
            except Exception:
                continue
            fut_lookup[e] = i["token"]
    return opt_lookup, fut_lookup, sorted(expiries)


def cycle_plan(monthlies, expiry, start_override=None):
    """The dates that define one cycle, for display and for the engine.

    Sessions here are PROJECTED (weekdays minus known holidays), because at
    planning time the candles that would reveal real sessions may not exist
    yet. The engine refines this with observed sessions once it has data.
    """
    start, _ = cycle_window(monthlies, expiry)
    if start_override:
        start = start_override
    sess = projected_sessions(start, expiry)
    black = (sess[-RATIO_BLACKOUT_DAYS]
             if RATIO_BLACKOUT_DAYS and len(sess) > RATIO_BLACKOUT_DAYS else expiry)
    return {"start": start,
            "first_session": sess[0] if sess else start,
            "blackout": black,
            "expiry": expiry,
            "sessions": len(sess)}


def plan_table(plan):
    return pd.DataFrame([
        ("Trade may start", f"{plan['first_session']:%a %d %b %Y}"),
        ("Entry bars each day", f"{ENTRY_FIRST_BAR:%H:%M} – {ENTRY_LAST_BAR:%H:%M} "
                                f"(1H labels; the 13:15 bar closes 14:15)"),
        ("Re-centre rule active until", f"{plan['blackout']:%a %d %b %Y} (exclusive)"),
        ("Blackout — carry only", f"{plan['blackout']:%a %d %b} → {plan['expiry']:%a %d %b}"),
        ("Expiry", f"{plan['expiry']:%a %d %b %Y}"),
        ("Forced exit", f"{EXPIRY_EXIT_TIME:%H:%M} on expiry day"),
        ("Projected sessions", plan["sessions"]),
    ], columns=["", "value"])


def expiry_listed(opt_lookup, expiry):
    """Are this expiry's option contracts still in the scrip master?

    The Angel master lists only TRADABLE instruments. The moment a contract
    expires its symbols and tokens are dropped, and the historical candle API
    needs a token. So a cycle that has already expired cannot be backtested
    from a freshly fetched master, however much price history exists. Only
    currently listed expiries can be replayed.
    """
    pre = f"NIFTY{expiry.strftime('%d%b%y').upper()}"
    return any(k.startswith(pre) for k in opt_lookup)


def monthly_expiries(expiries):
    """The last expiry in each calendar month.

    Derived from the master rather than computed as "last Tuesday". When the
    last Tuesday is a holiday the exchange moves expiry to the preceding
    session, and an arithmetic rule would silently pick a date with no
    contracts on it.
    """
    by_month = {}
    for e in expiries:
        k = (e.year, e.month)
        if k not in by_month or e > by_month[k]:
            by_month[k] = e
    return sorted(by_month.values())


def cycle_window(monthlies, expiry):
    """(first eligible entry date, expiry) for a monthly cycle. Entry opens
    the day after the previous monthly expiry — the Wednesday, when expiry
    fell on its usual Tuesday."""
    prior = [e for e in monthlies if e < expiry]
    start = (prior[-1] + dt.timedelta(days=1)) if prior else (expiry - dt.timedelta(days=30))
    return start, expiry


def projected_sessions(frm, to):
    """Weekdays in [frm, to] that are not known holidays."""
    out, d = [], frm
    while d <= to:
        if d.weekday() < 5 and d not in NSE_HOLIDAYS:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def atm_strike(price, mode=None):
    """Futures LTP to a strike.

    HALF_UP  nearest 50, ties upward. Not round() — Python rounds halves to
             even, so round(24425/50)*50 gives 24400 where the spec wants
             24450. The floor form is unambiguous.
    CEIL     always up to the next 50. The note's three examples only agree
             with each other under this rule (24451 -> 24500), but it biases
             the strike ~25 points above the future, which makes the put
             systematically the richer leg and starts the ratio off-centre.
    """
    mode = mode or STRIKE_ROUNDING
    if mode == "CEIL":
        return int(math.ceil(price / STRIKE_STEP) * STRIKE_STEP)
    return int(math.floor(price / STRIKE_STEP + 0.5) * STRIKE_STEP)


def opt_symbol(expiry, strike, ot):
    return f"NIFTY{expiry.strftime('%d%b%y').upper()}{strike}{ot}"


# ============================== INDICATORS ==================================
def true_range(df):
    h, l, c = df.high, df.low, df.close
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                     axis=1).max(axis=1)


def compute_adx_adxr(df, period=ADX_PERIOD):
    h, l = df.high, df.low
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = true_range(df).ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx, (adx + adx.shift(period)) / 2.0, pdi, mdi


def is_falling(s, idx, lookback=FALLING_LOOKBACK):
    w = s.iloc[max(0, idx - lookback + 1): idx + 1]
    if len(w) < lookback or w.isna().any():
        return False
    return all(w.iloc[i] > w.iloc[i + 1] for i in range(len(w) - 1))


# ============================== STATE =======================================
@dataclass
class Straddle:
    num: int
    strike: int
    expiry: dt.date
    ce_symbol: str
    pe_symbol: str
    ce_entry: float
    pe_entry: float
    entry_time: dt.datetime
    entry_fut: float
    entry_reason: str = "GATE"
    entry_scan: str = ""            # the three-strike probe, for the record
    ce_exit: float = None
    pe_exit: float = None
    exit_time: dt.datetime = None
    exit_reason: str = None
    max_adverse: float = 0.0        # worst MTM seen, in rupees (<= 0)
    max_favourable: float = 0.0     # best MTM seen, in rupees (>= 0)
    bars_held: int = 0

    @property
    def entry_value(self):
        return self.ce_entry + self.pe_entry

    @property
    def entry_ratio(self):
        lo = min(self.ce_entry, self.pe_entry)
        return max(self.ce_entry, self.pe_entry) / lo if lo > 0 else float("inf")

    @property
    def is_open(self):
        return self.exit_time is None

    def mtm(self, ce_ltp, pe_ltp):
        """Short both legs, so a fall in either premium is a gain."""
        return (self.entry_value - (ce_ltp + pe_ltp)) * QTY

    @property
    def pnl(self):
        if self.ce_exit is None or self.pe_exit is None:
            return 0.0
        return (self.entry_value - (self.ce_exit + self.pe_exit)) * QTY


@dataclass
class CycleState:
    expiry: dt.date
    straddles: list = field(default_factory=list)
    done: bool = False

    @property
    def open_straddle(self):
        for s in self.straddles:
            if s.is_open:
                return s
        return None

    @property
    def count(self):
        return len(self.straddles)

    @property
    def realised(self):
        return sum(s.pnl for s in self.straddles if not s.is_open)


def ratio_of(ce, pe):
    lo = min(ce, pe)
    return max(ce, pe) / lo if lo > 0 else float("inf")


# ============================== ENGINE ======================================
class PositionalEngine:
    def __init__(self, smart, opt_lookup, fut_lookup, expiry, kind="LIVE",
                 sessions=None, notify=False):
        self.smart = smart
        self.opt = opt_lookup
        self.fut = fut_lookup
        self.expiry = expiry
        self.kind = kind
        self.state = CycleState(expiry)
        self.notify = notify
        self.log = []
        self.h1_rows = []
        self.m15_rows = []
        # Trading sessions of this cycle, derived from the index data itself
        # rather than a hard-coded holiday list.
        self.sessions = sessions or []
        self._opt_cache = {}
        self._gate_note = ""
        # Newest bar completion time this engine has processed. Held in
        # memory so the live loop does not re-read the whole of Sheet02 every
        # fifteen minutes — that is a growing read all cycle, and it races
        # against Sheets' own write propagation. The sheets are consulted
        # once, at startup, to find where a previous process left off.
        self.last_bar = None

    # ---- data ----
    def option_series(self, strike, ot, frm, to):
        key = (strike, ot, frm.date(), to.date())
        if key in self._opt_cache:
            return self._opt_cache[key]
        sym = opt_symbol(self.expiry, strike, ot)
        tok = self.opt.get(sym)
        if tok is None:
            self._opt_cache[key] = (sym, empty_candles())
            return self._opt_cache[key]
        df = candles(self.smart, tok, NFO, frm, to, IV_15)
        self._opt_cache[key] = (sym, df)
        return self._opt_cache[key]

    def leg_ltp(self, strike, ot, now, window_days=6):
        """Latest close at or before `now`. The window is short because this
        strategy needs no option history — only the current premium."""
        frm = dt.datetime.combine(now.date() - dt.timedelta(days=window_days),
                                  MARKET_OPEN)
        sym, df = self.option_series(strike, ot, frm,
                                     dt.datetime.combine(now.date(), MARKET_CLOSE))
        if df.empty:
            return sym, None
        r = df[df.ts <= now]
        return sym, (float(r.close.iloc[-1]) if not r.empty else None)

    # ---- calendar ----
    def cycle_sessions(self):
        """Every session of this cycle up to and including expiry.

        self.sessions comes from the index candles, so it stops at today and
        can never contain the last sessions before a future expiry. Counting
        the blackout off that list alone means the blackout NEVER fires in
        live trading, and fires far too early on a mid-cycle backtest that
        stops short of expiry. Observed sessions are therefore extended with
        projected weekdays out to the expiry.
        """
        obs = [d for d in self.sessions if d <= self.expiry]
        last = max(obs) if obs else None
        if last is None:
            return projected_sessions(self.expiry - dt.timedelta(days=45),
                                      self.expiry)
        if last >= self.expiry:
            return sorted(set(obs))
        return sorted(set(obs) | set(projected_sessions(
            last + dt.timedelta(days=1), self.expiry)))

    def blackout_from(self):
        """First session on which the ratio rule is suspended."""
        s = self.cycle_sessions()
        if len(s) <= RATIO_BLACKOUT_DAYS:
            return self.expiry
        return s[-RATIO_BLACKOUT_DAYS]

    def in_blackout(self, day):
        return RATIO_BLACKOUT_DAYS > 0 and day >= self.blackout_from()

    def sessions_to_expiry(self, day):
        return len([d for d in self.cycle_sessions() if day < d <= self.expiry])

    # ---- gate ----
    def gate(self, h1, vix_h1, i):
        """Evaluate the entry gate on completed hourly bar `i` of `h1`.

        Returns (bool, dict of individual conditions) so a month with no
        entries can be explained from Sheet01 instead of guessed at.
        """
        c = {"adx_below_adxr": False, "adx_falling": False,
             "adxr_falling": False, "vix_in_band": False,
             "vix_in_range": False, "adx": np.nan, "adxr": np.nan,
             "vix": np.nan}
        if i < ADX_PERIOD * 2:
            self._gate_note = "warming up"
            return False, c

        adx, adxr, _, _ = compute_adx_adxr(h1.iloc[: i + 1], ADX_PERIOD)
        a, ar = adx.iloc[-1], adxr.iloc[-1]
        c["adx"], c["adxr"] = a, ar
        if pd.isna(a) or pd.isna(ar):
            self._gate_note = "adx not ready"
            return False, c
        c["adx_below_adxr"] = bool(a < ar)
        c["adx_falling"] = is_falling(adx, len(adx) - 1)
        c["adxr_falling"] = is_falling(adxr, len(adxr) - 1)

        vr = vix_h1[vix_h1.ts <= h1.ts.iloc[i]] if not vix_h1.empty else pd.DataFrame()
        if vr.empty:
            self._gate_note = "no vix"
            return False, c
        v = float(vr.close.iloc[-1])
        c["vix"] = v
        c["vix_in_band"] = bool(VIX_LOW <= v <= VIX_HIGH)
        if VIX_RANGE_LOOKBACK > 0:
            w = vr.close.tail(VIX_RANGE_LOOKBACK)
            # Not a new high for the window: VIX is ranging, not expanding.
            c["vix_in_range"] = bool(len(w) >= 2 and v < float(w.max()))
        else:
            c["vix_in_range"] = True

        ok = all(c[k] for k in ("adx_below_adxr", "adx_falling", "adxr_falling",
                                "vix_in_band", "vix_in_range"))
        self._gate_note = "" if ok else "gate closed"
        return ok, c

    def entry_allowed(self, now, bar_label):
        if self.state.done:
            return False, "cycle done"
        if self.state.open_straddle is not None:
            return False, "already positioned"
        if MAX_STRADDLES_PER_CYCLE and self.state.count >= MAX_STRADDLES_PER_CYCLE:
            return False, "cycle straddle cap"
        if not (ENTRY_FIRST_BAR <= bar_label.time() <= ENTRY_LAST_BAR):
            return False, "outside entry bars"
        if now.date() >= self.expiry:
            return False, "expiry day"
        if self.in_blackout(now.date()):
            return False, "expiry blackout"
        return True, ""

    # ---- positions ----
    def scan_strikes(self, now, fut_ltp):
        """Pick the strike whose CE and PE premiums are closest together.

        The rounded futures price only seeds the centre. Basis and skew mean
        the strike where the two premiums actually meet is often a step away
        from it, and that strike is the one whose ratio starts nearest 1.0 —
        which matters here, because the ratio is the exit rule. Starting at
        1.15 instead of 1.02 gives away a meaningful slice of the distance to
        the 2.0 trigger before the trade has done anything.

        Returns (strike, ce_sym, ce, pe_sym, pe, probe_lines) or None.
        """
        centre = atm_strike(fut_ltp)
        cands = [centre + k * STRIKE_STEP
                 for k in range(-STRIKE_SCAN, STRIKE_SCAN + 1)]
        best, probe = None, []
        for k in cands:
            ce_sym, ce = self.leg_ltp(k, "CE", now)
            pe_sym, pe = self.leg_ltp(k, "PE", now)
            if ce is None or pe is None:
                probe.append(f"{k}: no data")
                continue
            diff = abs(ce - pe)
            probe.append(f"{k}: CE {ce:.1f} PE {pe:.1f} d{diff:.1f}")
            if best is None or diff < best[0]:
                best = (diff, k, ce_sym, ce, pe_sym, pe)
        self.log.append(f"{now:%d-%b %H:%M} scan (fut {fut_ltp:.1f}, "
                        f"centre {centre}) — " + " | ".join(probe))
        if best is None:
            return None
        _, k, ce_sym, ce, pe_sym, pe = best
        return k, ce_sym, ce, pe_sym, pe, probe

    def deploy(self, now, fut_ltp, reason="GATE"):
        got = self.scan_strikes(now, fut_ltp)
        if got is None:
            self.log.append(f"{now:%d-%b %H:%M} deploy failed — no option data "
                            f"around {atm_strike(fut_ltp)}")
            return None
        strike, ce_sym, ce, pe_sym, pe, probe = got
        # Sold, so slippage works against the seller on the way in.
        ce_fill, pe_fill = ce - SLIPPAGE_PTS, pe - SLIPPAGE_PTS
        s = Straddle(num=self.state.count + 1, strike=strike, expiry=self.expiry,
                     ce_symbol=ce_sym, pe_symbol=pe_sym,
                     ce_entry=ce_fill, pe_entry=pe_fill,
                     entry_time=now, entry_fut=fut_ltp, entry_reason=reason,
                     entry_scan=" | ".join(probe))
        self.state.straddles.append(s)
        self.log.append(f"{now:%d-%b %H:%M} SELL #{s.num} {strike} "
                        f"CE {ce_fill:.2f} PE {pe_fill:.2f} "
                        f"value {s.entry_value:.2f} ratio {s.entry_ratio:.2f}")
        if self.notify:
            discord(f"**Positional** — straddle #{s.num} deployed\n"
                    f"`{now:%d-%b %H:%M}`  expiry **{self.expiry}**  "
                    f"fut {fut_ltp:.1f}\n"
                    f"SELL `{ce_sym}` @ **{ce_fill:.2f}**\n"
                    f"SELL `{pe_sym}` @ **{pe_fill:.2f}**\n"
                    f"value **{s.entry_value:.2f}**  ·  entry ratio "
                    f"{s.entry_ratio:.2f}\n"
                    f"scan: {' | '.join(probe)}", tag="🟢")
        self.write_position(s, new=True)
        return s

    def close(self, s, ce_ltp, pe_ltp, now, reason):
        # Bought back, so slippage works against the seller on the way out too.
        s.ce_exit = ce_ltp + SLIPPAGE_PTS
        s.pe_exit = pe_ltp + SLIPPAGE_PTS
        s.exit_time = now
        s.exit_reason = reason
        self.log.append(f"{now:%d-%b %H:%M} CLOSE #{s.num} ({reason}) "
                        f"CE {s.ce_exit:.2f} PE {s.pe_exit:.2f} "
                        f"pnl {s.pnl:+,.0f}")
        if self.notify:
            icon = {"RATIO_2X": "🟠", "EXPIRY_EXIT": "⚪"}.get(reason, "🔵")
            discord(f"**Positional** — straddle #{s.num} closed ({reason})\n"
                    f"`{now:%d-%b %H:%M}`  entry {s.entry_value:.2f} → "
                    f"exit **{s.ce_exit + s.pe_exit:.2f}**\n"
                    f"P&L **{s.pnl:+,.0f}**  ·  cycle "
                    f"{self.state.realised:+,.0f}\n"
                    f"worst MTM on this straddle {s.max_adverse:+,.0f}",
                    tag=icon)
        self.write_position(s, new=False)

    def write_position(self, s, new):
        row = [str(self.expiry), s.num, "OPEN" if s.is_open else "CLOSED",
               s.strike, s.entry_time.strftime("%Y-%m-%d %H:%M"),
               round(s.entry_fut, 2), s.ce_symbol, round(s.ce_entry, 2),
               s.pe_symbol, round(s.pe_entry, 2), round(s.entry_value, 2),
               round(s.entry_ratio, 3),
               s.exit_time.strftime("%Y-%m-%d %H:%M") if s.exit_time else "",
               round(s.ce_exit, 2) if s.ce_exit is not None else "",
               round(s.pe_exit, 2) if s.pe_exit is not None else "",
               round(s.ce_exit + s.pe_exit, 2) if s.ce_exit is not None else "",
               round(ratio_of(s.ce_exit, s.pe_exit), 3) if s.ce_exit is not None else "",
               round(s.pnl, 2), round(s.max_adverse, 2),
               round(s.max_favourable, 2), s.bars_held,
               s.entry_reason, s.exit_reason or "", s.entry_scan]
        try:
            if new:
                append_rows(self.kind, TAB_POS, POS_HEADERS, [row])
            else:
                r = find_open_row(self.kind, self.expiry, s.num)
                if r:
                    update_row(self.kind, TAB_POS, POS_HEADERS, r, row)
                else:
                    append_rows(self.kind, TAB_POS, POS_HEADERS, [row])
        except Exception as e:
            self.log.append(f"sheet write failed: {e}")

    # ---- per-bar ----
    def on_15m(self, now, ce_ltp, pe_ltp, flush=True):
        """Manage an open straddle on a completed 15-minute bar."""
        s = self.state.open_straddle
        if s is None or ce_ltp is None or pe_ltp is None:
            return
        s.bars_held += 1
        mtm = s.mtm(ce_ltp, pe_ltp)
        s.max_adverse = min(s.max_adverse, mtm)
        s.max_favourable = max(s.max_favourable, mtm)
        r = ratio_of(ce_ltp, pe_ltp)
        black = self.in_blackout(now.date())

        self.m15_rows.append([now.strftime("%Y-%m-%d %H:%M"), str(now.date()),
                              str(self.expiry), s.num, s.strike,
                              s.ce_symbol, round(ce_ltp, 2),
                              s.pe_symbol, round(pe_ltp, 2),
                              round(ce_ltp + pe_ltp, 2), round(r, 3),
                              round(s.entry_value, 2), round(mtm, 2),
                              self.sessions_to_expiry(now.date()), black])

        if now.date() >= self.expiry and now.time() >= EXPIRY_EXIT_TIME:
            self.close(s, ce_ltp, pe_ltp, now, "EXPIRY_EXIT")
            self.state.done = True
        elif not black and r >= RATIO_EXIT:
            self.close(s, ce_ltp, pe_ltp, now, "RATIO_2X")

        self.last_bar = max(self.last_bar or now, now)
        if flush:
            self.flush()

    def on_1h(self, now, bar, h1, vix_h1, i, fut_ltp, flush=True):
        """Evaluate the gate on a completed hourly bar and deploy if open."""
        ok, c = self.gate(h1, vix_h1, i)
        allowed, why = self.entry_allowed(now, bar.ts)
        pos = self.state.open_straddle
        note = self._gate_note if not ok else (why if not allowed else "")

        self.h1_rows.append([bar.ts.strftime("%Y-%m-%d %H:%M"), str(bar.ts.date()),
                             str(self.expiry), round(float(bar.open), 2),
                             round(float(bar.high), 2), round(float(bar.low), 2),
                             round(float(bar.close), 2),
                             round(fut_ltp, 2) if fut_ltp else "",
                             _j(c["adx"]), _j(c["adxr"]), _j(c["vix"]),
                             c["adx_below_adxr"], c["adx_falling"],
                             c["adxr_falling"], c["vix_in_band"],
                             c["vix_in_range"], ok,
                             f"#{pos.num}@{pos.strike}" if pos else "flat", note])

        if ok and allowed and fut_ltp:
            self.deploy(now, fut_ltp)
        self.last_bar = max(self.last_bar or now, now)
        if flush:
            self.flush()

    def flush(self):
        try:
            if self.h1_rows:
                append_rows(self.kind, TAB_1H, H1_HEADERS, self.h1_rows)
                self.h1_rows = []
            if self.m15_rows:
                append_rows(self.kind, TAB_15M, M15_HEADERS, self.m15_rows)
                self.m15_rows = []
        except Exception as e:
            self.log.append(f"sheet flush failed: {e}")

    # ---- resume ----
    def resume(self):
        """Rebuild the cycle from Sheet03, so a restart does not re-deploy on
        top of a position that is already live."""
        try:
            d = read_tab(self.kind, TAB_POS)
        except Exception as e:
            self.log.append(f"resume read failed: {e}")
            return 0
        if d.empty or "expiry" not in d.columns:
            return 0
        d = d[d.expiry.astype(str) == str(self.expiry)]
        if d.empty:
            return 0

        def num(v, cast=float, default=0.0):
            try:
                if v == "" or pd.isna(v):
                    return default
                return cast(float(v))
            except Exception:
                return default

        for _, r in d.sort_values("straddle_num").iterrows():
            try:
                et = dt.datetime.strptime(str(r.entry_time), "%Y-%m-%d %H:%M")
            except Exception:
                continue
            s = Straddle(num=int(num(r.straddle_num, int, 1)),
                         strike=int(num(r.strike, int, 0)),
                         expiry=self.expiry,
                         ce_symbol=str(r.ce_symbol), pe_symbol=str(r.pe_symbol),
                         ce_entry=num(r.ce_entry), pe_entry=num(r.pe_entry),
                         entry_time=et, entry_fut=num(r.entry_fut),
                         entry_reason=str(r.get("entry_reason") or "GATE"),
                         entry_scan=str(r.get("entry_scan") or ""),
                         max_adverse=num(r.get("max_adverse")),
                         max_favourable=num(r.get("max_favourable")),
                         bars_held=int(num(r.get("bars_held"), int, 0)))
            if str(r.status).upper() == "CLOSED":
                s.ce_exit = num(r.ce_exit, float, None)
                s.pe_exit = num(r.pe_exit, float, None)
                s.exit_reason = str(r.get("exit_reason") or "RESUMED")
                try:
                    s.exit_time = dt.datetime.strptime(str(r.exit_time), "%Y-%m-%d %H:%M")
                except Exception:
                    s.exit_time = et
            self.state.straddles.append(s)

        op = self.state.open_straddle
        if op is None and any(x.exit_reason == "EXPIRY_EXIT"
                              for x in self.state.straddles):
            self.state.done = True
        self.log.append(f"resumed {self.state.count} straddle(s); "
                        + (f"#{op.num} still open at {op.strike}" if op else "flat"))
        return self.state.count

    def last_logged(self, tab):
        """Newest bar already in a log tab, expressed as a COMPLETION time.

        The two tabs do not store bar_time the same way, and conflating them
        silently duplicates or skips bars on restart. Sheet01 stores the
        hourly LABEL, because that is what a chart shows — a bar labelled
        13:15 does not finish until 14:15. Sheet02 stores the moment the
        15-minute bar closed. Normalising to completion time here is what
        makes max() across the two tabs meaningful.
        """
        try:
            d = read_tab(self.kind, tab)
        except Exception:
            return None
        if d.empty or "bar_time" not in d.columns:
            return None
        if "expiry" in d.columns:
            d = d[d.expiry.astype(str) == str(self.expiry)]
        ts = pd.to_datetime(d["bar_time"], errors="coerce").dropna()
        if not len(ts):
            return None
        newest = ts.max().to_pydatetime()
        return newest + dt.timedelta(hours=1) if tab == TAB_1H else newest


# ============================== REPLAY ======================================
def load_feed(smart, fut_lookup, expiry, frm, to):
    """Index, VIX and monthly-future 15-minute bars plus their hourly
    resamples, for one window."""
    n15 = candles(smart, NIFTY_INDEX_TOKEN, NSE, frm, to, IV_15)
    v15 = candles(smart, INDIA_VIX_TOKEN, NSE, frm, to, IV_15)
    ftok = fut_lookup.get(expiry)
    f15 = candles(smart, ftok, NFO, frm, to, IV_15) if ftok else empty_candles()
    return {"n15": n15, "h1": resample_hourly(n15),
            "vix_h1": resample_hourly(v15), "f15": f15}


def replay(engine, feed, frm, to, flush_each=False):
    """Drive the engine over every completed bar in [frm, to].

    Used for three things that must behave identically: a backtest, the
    warm-up before going live, and the backfill after an outage.
    """
    h1, n15, f15 = feed["h1"], feed["n15"], feed["f15"]
    vix_h1 = feed["vix_h1"]
    if h1.empty or n15.empty:
        return

    # One merged, ordered stream. Hourly bars are processed before the
    # 15-minute bar that shares their timestamp, so an entry taken on the
    # 09:15 hourly close is managed from the 10:15 fifteen-minute bar on and
    # never marked on the bar it was entered.
    events = []
    for i, b in h1.iterrows():
        done = b.ts + dt.timedelta(hours=1)
        if frm <= done <= to and b.ts.time() <= dt.time(14, 15):
            events.append((done, 0, i))
    for i, b in n15.iterrows():
        done = b.ts + dt.timedelta(minutes=15)
        if frm <= done <= to:
            events.append((done, 1, i))
    events.sort(key=lambda e: (e[0], e[1]))

    def fut_at(t):
        if f15.empty:
            return None
        r = f15[f15.ts <= t]
        return float(r.close.iloc[-1]) if not r.empty else None

    for when, kind, idx in events:
        if kind == 0:
            engine.on_1h(when, h1.iloc[idx], h1, vix_h1, idx,
                         fut_at(when), flush=flush_each)
        else:
            s = engine.state.open_straddle
            if s is None:
                continue
            _, ce = engine.leg_ltp(s.strike, "CE", when)
            _, pe = engine.leg_ltp(s.strike, "PE", when)
            engine.on_15m(when, ce, pe, flush=flush_each)
    engine.flush()


def sessions_from(df):
    return sorted({t.date() for t in df.ts}) if not df.empty else []


def run_cycle(smart, opt_lookup, fut_lookup, monthlies, expiry, kind="BACKTEST",
              notify=False, upto=None, progress=None, start_override=None):
    """One monthly cycle, from the first eligible entry date through expiry."""
    start, _ = cycle_window(monthlies, expiry)
    if start_override:
        start = start_override
    frm = dt.datetime.combine(start - dt.timedelta(days=WARMUP_SESSIONS * 2 + 10),
                              MARKET_OPEN)
    to = dt.datetime.combine(upto or expiry, MARKET_CLOSE)
    if progress:
        progress(f"fetching {expiry} feed")
    feed = load_feed(smart, fut_lookup, expiry, frm, to)
    eng = PositionalEngine(smart, opt_lookup, fut_lookup, expiry, kind=kind,
                           sessions=sessions_from(feed["n15"]), notify=notify)
    if progress:
        progress(f"replaying {expiry}")
    replay(eng, feed, dt.datetime.combine(start, MARKET_OPEN), to)
    return eng


# ============================== LIVE ========================================
def seconds_to_next_15m(now):
    m = (now.minute // 15 + 1) * 15
    nxt = now.replace(second=0, microsecond=0, minute=0) + dt.timedelta(minutes=m)
    return max(5, int((nxt - now).total_seconds()) + POLL_BUFFER_SEC)


def live_loop(stop_event, expiry, status, notify=False, start_override=None):
    """Poll on the 15-minute close. Resume first, backfill second, live third."""
    def note(m):
        status["log"] = ([f"{now_ist():%H:%M:%S}  {m}"] + status.get("log", []))[:200]

    try:
        smart = angel_login()
        opt_lookup, fut_lookup, expiries = instrument_master()
        monthlies = monthly_expiries(expiries)
        note(f"logged in · expiry {expiry}")

        start, _ = cycle_window(monthlies, expiry)
        if start_override:
            start = start_override
        status["window"] = f"{start:%d %b} → {expiry:%d %b %Y}"
        note(f"cycle window {start:%d %b} → {expiry:%d %b}")
        eng = PositionalEngine(smart, opt_lookup, fut_lookup, expiry,
                               kind="LIVE", notify=notify)
        eng.resume()
        note(f"resumed · {eng.state.count} straddle(s)")

        # Backfill from the newest logged bar, or from the cycle start if the
        # sheets are empty. Replaying from the cycle start on every restart
        # would duplicate months of rows, so the marker matters.
        marks = [m for m in (eng.last_logged(TAB_1H),
                             eng.last_logged(TAB_15M)) if m]
        resume_from = max(marks) if marks else dt.datetime.combine(start, MARKET_OPEN)
        note(f"backfilling from {resume_from:%d-%b %H:%M}")

        frm = dt.datetime.combine(resume_from.date()
                                  - dt.timedelta(days=WARMUP_SESSIONS * 2 + 10),
                                  MARKET_OPEN)
        feed = load_feed(smart, fut_lookup, expiry, frm, now_ist())
        eng.sessions = sessions_from(feed["n15"])
        replay(eng, feed, resume_from + dt.timedelta(seconds=1), now_ist())
        note("backfill complete — live")

        while not stop_event.is_set():
            n = now_ist()
            if n.date() > expiry or eng.state.done:
                note("cycle complete")
                break
            if n.weekday() >= 5 or not (MARKET_OPEN <= n.time() <= MARKET_CLOSE):
                stop_event.wait(300)
                continue

            feed = load_feed(smart, fut_lookup, expiry,
                             dt.datetime.combine(n.date() - dt.timedelta(days=30),
                                                 MARKET_OPEN), n)
            eng.sessions = sorted(set(eng.sessions) | set(sessions_from(feed["n15"])))
            since = eng.last_bar or (n - dt.timedelta(minutes=20))
            replay(eng, feed, since + dt.timedelta(seconds=1), n, flush_each=True)

            s = eng.state.open_straddle
            status["position"] = (f"#{s.num} {s.strike} entry {s.entry_value:.1f}"
                                  if s else "flat")
            status["realised"] = eng.state.realised
            status["updated"] = f"{n:%d-%b %H:%M:%S}"
            for line in eng.log[-5:]:
                note(line)
            eng.log = []
            stop_event.wait(seconds_to_next_15m(now_ist()))

    except Exception as e:
        note(f"ENGINE ERROR: {e}")
        status["error"] = str(e)
    finally:
        status["running"] = False


@st.cache_resource(show_spinner=False)
def registry():
    """Process-level singleton. st.session_state is per-browser-session, so a
    reload would lose the handle while the daemon thread kept running and the
    Start button would launch a second engine onto the same sheets."""
    return {"thread": None, "stop": None, "status": {}}


def engine_alive():
    for t in threading.enumerate():
        if t.name == ENGINE_THREAD_NAME and t.is_alive():
            return t
    return None


# ============================== UI ==========================================
def money(x):
    return f"{'−' if x < 0 else ''}₹{abs(x):,.0f}"


def summarise(eng):
    rows = []
    for s in eng.state.straddles:
        rows.append({
            "#": s.num, "strike": s.strike,
            "entry": s.entry_time.strftime("%d-%b %H:%M"),
            "entry_val": round(s.entry_value, 2),
            "entry_ratio": round(s.entry_ratio, 2),
            "exit": s.exit_time.strftime("%d-%b %H:%M") if s.exit_time else "OPEN",
            "exit_val": round(s.ce_exit + s.pe_exit, 2) if s.ce_exit is not None else None,
            "reason": s.exit_reason or "",
            "pnl": round(s.pnl, 0),
            "worst_mtm": round(s.max_adverse, 0),
            "best_mtm": round(s.max_favourable, 0),
            "bars": s.bars_held})
    return pd.DataFrame(rows)


st.title("◈ Nifty Positional Short Straddle")
st.caption(f"monthly ATM · lot {LOT_SIZE} × {LOTS} · ratio exit {RATIO_EXIT:g}× · "
           f"blackout {RATIO_BLACKOUT_DAYS} sessions · no stop loss")

tab_bt, tab_live, tab_cfg = st.tabs(["Backtest", "Live", "Config"])

with tab_bt:
    st.subheader("Backtest a monthly cycle")
    if st.button("Load expiry list", key="load_exp"):
        try:
            o, f, e = instrument_master()
            st.session_state["monthlies"] = monthly_expiries(e)
            st.success(f"{len(st.session_state['monthlies'])} monthly expiries found")
        except Exception as ex:
            st.error(f"master fetch failed: {ex}")

    ms = st.session_state.get("monthlies", [])
    if not ms:
        st.info("Load the expiry list first.")
    else:
        try:
            _ol, _fl, _ = instrument_master()
            listed = [e for e in ms if expiry_listed(_ol, e)]
            gone = [e for e in ms if e not in listed]
        except Exception:
            listed, gone = ms, []
        if gone:
            st.warning(
                "No option contracts remain in the Angel scrip master for: "
                + ", ".join(str(e) for e in gone)
                + ". The master lists tradable instruments only, so those "
                "strikes have no tokens and cannot be replayed. Backtesting a "
                "past cycle needs a scrip master archived while it was live.")
        if not listed:
            st.error("No listed monthly expiry available to backtest.")
        else:
            exp = st.selectbox("Expiry (cycle end)", listed, key="bt_exp")
            auto = cycle_plan(ms, exp)["start"]
            c1, c2 = st.columns(2)
            start = c1.date_input(
                "Trade start", value=auto, key="bt_start",
                help="Defaults to the session after the previous monthly "
                     "expiry. Move it later to skip the early part of a cycle.")
            upto = c2.date_input(
                "Replay until", value=min(exp, today_ist()), key="bt_upto",
                help="For a cycle still running, stop at today.")

            if start > exp:
                st.error("Trade start is after expiry.")
            else:
                plan = cycle_plan(ms, exp, start_override=start)
                st.dataframe(plan_table(plan), width="stretch", hide_index=True)

                if st.button("Run backtest", type="primary"):
                    box = st.empty()
                    try:
                        smart = angel_login()
                        opt_lookup, fut_lookup, _ = instrument_master()
                        eng = run_cycle(smart, opt_lookup, fut_lookup, ms, exp,
                                        kind="BACKTEST", upto=upto,
                                        start_override=start,
                                        progress=lambda m: box.info(m))
                        box.empty()
                        df = summarise(eng)
                        st.metric("Realised", money(eng.state.realised))
                        if df.empty:
                            st.warning("No straddle deployed in this window. "
                                       "Sheet01 records every gate condition "
                                       "separately — check which one blocked.")
                        else:
                            st.dataframe(df, width="stretch")
                            st.caption(
                                f"worst MTM across the cycle "
                                f"{money(min([x.max_adverse for x in eng.state.straddles]))}")
                        with st.expander("engine log"):
                            st.code("\n".join(eng.log) or "—")
                    except Exception as ex:
                        st.error(f"backtest failed: {ex}")

with tab_live:
    reg = registry()
    alive = engine_alive()
    st.subheader("Live engine")
    ms = st.session_state.get("monthlies", [])
    if not ms:
        st.info("Load the expiry list on the Backtest tab first.")
    else:
        future = [e for e in ms if e >= today_ist()]
        exp = st.selectbox("Expiry (cycle end)", future or ms[-1:], key="lv_exp")
        auto = cycle_plan(ms, exp)["start"]
        start = st.date_input(
            "Trade start", value=max(auto, today_ist()) if alive is None else auto,
            key="lv_start",
            help="Where the engine begins looking for an entry. Leave as the "
                 "default to run the whole cycle. Setting it later than today "
                 "means nothing deploys until that date.")
        plan = cycle_plan(ms, exp, start_override=start)
        st.dataframe(plan_table(plan), width="stretch", hide_index=True)

        notify = st.checkbox("Discord notifications", value=True)
        c1, c2 = st.columns(2)
        if c1.button("Start", type="primary", disabled=bool(alive)):
            stop = threading.Event()
            t = threading.Thread(target=live_loop,
                                 args=(stop, exp, reg["status"], notify, start),
                                 name=ENGINE_THREAD_NAME, daemon=True)
            reg.update({"thread": t, "stop": stop})
            reg["status"].update({"running": True, "log": []})
            t.start()
            st.rerun()
        if c2.button("Stop", disabled=not alive):
            if reg.get("stop"):
                reg["stop"].set()
            st.rerun()

with tab_cfg:
    st.subheader("Active configuration")
    st.dataframe(pd.DataFrame([
        ("Lot size × lots", f"{LOT_SIZE} × {LOTS} = {QTY}"),
        ("Entry gate", "ADX(14) < ADXR(14), both falling, on 1H"),
        ("VIX band", f"{VIX_LOW:g} – {VIX_HIGH:g}"),
        ("VIX range test", f"not an {VIX_RANGE_LOOKBACK}-bar high"
                           if VIX_RANGE_LOOKBACK else "disabled"),
        ("Warm-up", f"{WARMUP_SESSIONS} sessions (~{WARMUP_SESSIONS * 6} hourly bars)"),
        ("Entry bars", f"{ENTRY_FIRST_BAR:%H:%M} – {ENTRY_LAST_BAR:%H:%M} labels"),
        ("Strike seed", f"monthly FUT ltp, {STRIKE_ROUNDING} to {STRIKE_STEP}"),
        ("Strike scan", f"±{STRIKE_SCAN} step, pick smallest |CE−PE|"
                        if STRIKE_SCAN else "disabled"),
        ("Re-centre", f"bigger/smaller ≥ {RATIO_EXIT:g}, checked each 15m"),
        ("Blackout", f"last {RATIO_BLACKOUT_DAYS} sessions before expiry"),
        ("Holiday list", f"{len(NSE_HOLIDAYS)} dates configured"),
        ("Expiry exit", f"{EXPIRY_EXIT_TIME:%H:%M} on expiry day"),
        ("Straddles per cycle", MAX_STRADDLES_PER_CYCLE or "uncapped"),
        ("Slippage", f"{SLIPPAGE_PTS:g} pts each way, each leg"),
        ("Stop loss", "NONE — see MAX_ADVERSE in Sheet03"),
    ], columns=["setting", "value"]), width="stretch", hide_index=True)
