import sys
import re
import json
import asyncio
import aiohttp
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from datetime import datetime, date, timedelta
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDockWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout, QSplitter,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QTabWidget, QMessageBox, QHeaderView, QAbstractItemView,
    QSizePolicy, QFrame, QComboBox, QListView, QMenu, QToolBar, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QRunnable, QThreadPool, QPoint
from PyQt6.QtGui import QColor, QBrush, QFont, QIcon, QAction
import qasync
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_CASES     = 50
DEFAULT_CASES = 10

QH_API_URL = "https://qh-api.corp.hertshtengroup.com/api/economies/premiums/?economies=*"
SR3_API_URL = (
    "https://pro1.corp.hertshtengroup.com/api4/scenario-analysis/"
    "pricing-calculation-for-custom-hikes-and-cuts"
)

# ─────────────────────────────────────────────────────────────────────────────
# Lightstreamer — shared singleton
# ─────────────────────────────────────────────────────────────────────────────
LS_SERVER       = "https://ls-md.corp.hertshtengroup.com/"
LS_ADAPTER_SET  = "TTsdkLSAdapter"
LS_DATA_ADAPTER = "HGL1_Adapter"
LS_FIELDS       = ["Contract", "BestAsk", "BestAskQty", "BestBid", "BestBidQty"]

try:
    from lightstreamer.client import LightstreamerClient, Subscription
    _LS_AVAILABLE = True
except Exception:
    LightstreamerClient = None
    _LS_AVAILABLE = False

from collections import defaultdict
import math
from scipy.optimize import brentq



def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bachelier_price(F, K, T, sigma, cp):
    if T <= 0 or sigma <= 0:
        return 0.0

    vol = sigma * math.sqrt(T)
    d = (F - K) / vol

    if cp == "C":
        return (F - K) * norm_cdf(d) + vol * norm_pdf(d)
    else:
        return (K - F) * norm_cdf(-d) + vol * norm_pdf(d)



def implied_vol_bachelier(price, F, K, T, cp):
    try:
        return brentq(
            lambda s: bachelier_price(F, K, T, s, cp) - price,
            1e-6,
            500.0
        )
    except:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SR3 Payoff Calculation & Chart
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_payoff(legs, S):
    """Calculate P&L for a list of legs at spot S."""
    total = 0.0
    for leg in legs:
        try:
            K = float(leg.strike)
            premium = float(leg.entry_price) / 100.0
            side = 1 if leg.long_short == "Long" else -1
            lots = leg.lots
            cp = leg.cp
            intrinsic = max(S - K, 0.0) if cp == "Call" else max(K - S, 0.0)
            total += side * lots * (intrinsic - premium)
        except Exception:
            pass
    return total


def _display_mul(family: str) -> float:
    """Display multiplier for bid/ask/mid."""
    if family in {"SR3"}:
        return 100.0
    return 1.0


def _pnl_mul(family: str, instrument_type: str) -> float:
    """PNL multiplier for payoff chart - converts model values to actual P&L."""
    if instrument_type == "Future":
        if family in {"SR3"}:
            return 25.0
        return 1.0
    if family in {"SR3"}:
        return 25.0
    return 1.0


def _spot_range(S0: float, family: str) -> list:
    """Generate spot range for payoff chart."""
    if family in {"SR3"}:
        return [round(S0 + i * 0.01, 4) for i in range(-250, 251)]
    return [S0 + i * 0.01 for i in range(-50, 51)]


def combined_model_value(S_range, legs, T_days_list=None, pnl_mul=None):
    """Compute combined payoff across spot range using Bachelier model.
    T_days_list: days to expiry (None = calculate from leg.expiry).
    Matches Option_Chain_V2 logic exactly.
    """
    import numpy as np
    if pnl_mul is None:
        contract_mul = _pnl_mul("SR3", legs[0].instrument_type if legs else "Option")
    else:
        contract_mul = pnl_mul

    values = []
    for S in S_range:
        total = 0.0
        for idx, leg in enumerate(legs):
            try:
                lots = leg.lots
                side = 1 if leg.long_short == "Long" else -1

                K = float(leg.strike)
                entry_price = float(leg.entry_price) / _display_mul("SR3")

                iv = leg.iv if leg.iv else 0.005
                cp = leg.cp

                if T_days_list is None:
                    expiry = datetime.fromisoformat(leg.expiry)
                    T_days = max((expiry - datetime.utcnow()).total_seconds(), 0.0) / (24 * 3600)
                else:
                    T_days = T_days_list[idx]

                T = max(T_days, 0.0) / 365.0
                F = S

                if T <= 0:
                    price = max(F - K, 0.0) if cp == "Call" else max(K - F, 0.0)
                else:
                    price = bachelier_price(F, K, T, iv, cp)

                price_in_display = price * _display_mul("SR3")
                entry_display = float(leg.entry_price)
                pnl = side * lots * (price_in_display - entry_display) * contract_mul
                total += pnl

            except Exception as e:
                print("Model error:", e)
        values.append(total)
    return np.array(values)

class LSManager:
    _instance = None

    def __init__(self):
        self.client = LightstreamerClient(LS_SERVER, LS_ADAPTER_SET)
        self.client.connect()
        self.subscriptions = {}
        self.last_values   = defaultdict(dict)
        self.ui_callbacks  = defaultdict(list)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = LSManager()
        return cls._instance

    def get_or_create(self, key, item_ids):
        if key in self.subscriptions:
            return
        ids = ["TT-" + x if not x.startswith("TT-") else x for x in item_ids]
        sub = Subscription("MERGE", ids, LS_FIELDS)
        sub.setDataAdapter(LS_DATA_ADAPTER)
        sub.setRequestedSnapshot("yes")
        sub.setRequestedMaxFrequency("2")

        def on_update(update):
            item = update.getItemName().replace("TT-", "")
            row  = {f: update.getValue(f) or "" for f in LS_FIELDS}
            self.last_values[key][item] = row
            for cb in list(self.ui_callbacks[key]):
                try:
                    cb(item, row)
                except Exception:
                    self.ui_callbacks[key].remove(cb)

        sub.addListener(type("L", (), {"onItemUpdate": lambda _, u: on_update(u)})())
        self.client.subscribe(sub)
        self.subscriptions[key] = sub

    def attach_ui(self, key, cb):
        self.ui_callbacks[key].append(cb)
        for item, row in self.last_values[key].items():
            cb(item, row)

    def detach_ui(self, key, cb):
        if cb in self.ui_callbacks[key]:
            self.ui_callbacks[key].remove(cb)

    def get_or_create_chunked(self, base_key: str, item_ids: list,
                               chunk_size: int = 400,
                               frequency: str = "unfiltered"):
        keys = []
        for i in range(0, len(item_ids), chunk_size):
            chunk     = item_ids[i : i + chunk_size]
            chunk_key = f"{base_key}__chunk_{i // chunk_size}"
            if chunk_key not in self.subscriptions:
                ids = ["TT-" + x if not x.startswith("TT-") else x
                       for x in chunk]
                sub = Subscription("MERGE", ids, LS_FIELDS)
                sub.setDataAdapter(LS_DATA_ADAPTER)
                sub.setRequestedSnapshot("yes")
                sub.setRequestedMaxFrequency(frequency)

                def on_update(update, _key=chunk_key):
                    item = update.getItemName().replace("TT-", "")
                    row  = {f: update.getValue(f) or "" for f in LS_FIELDS}
                    self.last_values[_key][item] = row
                    for cb in list(self.ui_callbacks[_key]):
                        try:
                            cb(item, row)
                        except Exception:
                            self.ui_callbacks[_key].remove(cb)

                sub.addListener(
                    type("L", (), {"onItemUpdate": lambda _, u, f=on_update: f(u)})()
                )
                self.client.subscribe(sub)
                self.subscriptions[chunk_key] = sub
            keys.append(chunk_key)
        return keys

    def attach_ui_multi(self, keys: list, cb):
        for key in keys:
            self.attach_ui(key, cb)

    def detach_ui_multi(self, keys: list, cb):
        for key in keys:
            self.detach_ui(key, cb)

    def shutdown(self):
        for s in self.subscriptions.values():
            try:
                self.client.unsubscribe(s)
            except Exception:
                pass
        self.client.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# OIS helpers
# ─────────────────────────────────────────────────────────────────────────────

def _default_ois_date() -> str:
    """Return the last compounding date for a 2Y OIS:
    today + 21 months, snapped to the last day of that quarter.
    Formatted as DD-Mon-YY to match MeetingRow.date style.
    """
    today = date.today()
    # advance ~21 months
    target_month = today.month + 21
    target_year  = today.year + (target_month - 1) // 12
    target_month = (target_month - 1) % 12 + 1

    # snap to nearest quarter-end (Mar/Jun/Sep/Dec)
    quarter_end_month = ((target_month - 1) // 3 + 1) * 3  # 3, 6, 9, or 12
    if quarter_end_month > 12:
        quarter_end_month = 12

    # last day of that month
    if quarter_end_month == 12:
        last_day = date(target_year, 12, 31)
    else:
        last_day = date(target_year, quarter_end_month + 1, 1) - timedelta(days=1)

    return last_day.strftime("%d-%b-%y")


def _parse_meeting_date(date_str: str) -> date | None:
    """Parse a meeting date string (DD-Mon-YY or DD-Mon-YYYY) into a date object."""
    for fmt in ("%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _calc_2y_ois(meeting_rows: list, ois_date_str: str, fixing: float) -> str:
    """
    Calculate the 2-year OIS rate by daily-compounding the overnight rate
    implied by meeting premiums up to `ois_date_str`.

    The overnight rate between meetings is approximated as:
        overnight_rate = fixing + cumulative_premium_hike  (in %)

    The 2Y OIS par rate is:
        OIS = (compound_factor - 1) / T  * 100
    where T = calendar days / 360.

    Returns the rate as a formatted string (e.g. "3.8412") or "" on error.
    """
    try:
        ois_date = _parse_meeting_date(ois_date_str)
        if ois_date is None:
            return ""

        today = date.today()
        if ois_date <= today:
            return ""

        # Build a sorted list of (date, premium_bps) for non-empty meeting rows
        meetings: list[tuple[date, float]] = []
        for mr in meeting_rows:
            if not mr.date or mr.is_ois_row:
                continue
            d = _parse_meeting_date(mr.date)
            if d is None:
                continue
            meetings.append((d, mr.premium))

        meetings.sort(key=lambda x: x[0])

        # Walk day-by-day from today to ois_date compounding
        compound = 1.0
        current_rate = fixing  # starting overnight rate in %

        # cumulative hike as of each meeting
        rate_schedule: list[tuple[date, float]] = []
        cum = fixing
        for mtg_date, prem in meetings:
            cum += prem / 100.0   # premium is in bps → convert to %
            rate_schedule.append((mtg_date, cum))

        def get_rate_on(d: date) -> float:
            r = fixing
            for mtg_date, rate in rate_schedule:
                if mtg_date <= d:
                    r = rate
                else:
                    break
            return r

        total_days = (ois_date - today).days
        if total_days <= 0:
            return ""

        current = today
        while current < ois_date:
            # determine how many days this rate holds (weekend bridging)
            next_day = current + timedelta(days=1)
            # skip weekends: Friday rate applies for Sat+Sun
            days_held = 1
            if current.weekday() == 4:  # Friday
                days_held = 3
                next_day = current + timedelta(days=3)
            elif current.weekday() >= 5:  # Sat/Sun — skip (already handled by Friday)
                current = next_day
                continue

            if next_day > ois_date:
                days_held = (ois_date - current).days

            r = get_rate_on(current) / 100.0  # as decimal
            compound *= (1.0 + r * days_held / 360.0)
            current = next_day

        T = total_days / 360.0
        ois_rate = (compound ** (360.0 / total_days) - 1.0) * 100.0
        return f"{ois_rate:.2f}"



    except Exception as ex:
        print(f"[_calc_2y_ois] error: {ex}")
        return ""


def _calc_2y_ois_for_case(meeting_rows: list,
                         ois_date_str: str,
                         fixing: float,
                         case_index: int | None = None) -> str:
    """
    SAFE VERSION:
    - Never mutates MeetingRow
    - Uses immutable adjusted structure
    """

    try:
        # ─────────────────────────────────────────────
        # 1. Get OIS row case shock
        # ─────────────────────────────────────────────
        ois_row = next((mr for mr in meeting_rows if mr.is_ois_row), None)

        shock = 0.0
        if case_index is not None and ois_row:
            if ois_row.cases and case_index < len(ois_row.cases):
                val = ois_row.cases[case_index]
                shock = val if val is not None else 0.0

        # ─────────────────────────────────────────────
        # 2. Build IMMUTABLE adjusted structure
        # ─────────────────────────────────────────────
        adjusted_meetings = []

        for mr in meeting_rows:

            # skip OIS row + empty rows
            if mr.is_ois_row or not mr.date:
                continue

            adjusted_meetings.append({
                "date": mr.date,
                "premium": mr.premium + shock   # ← NO mutation
            })

        # ─────────────────────────────────────────────
        # 3. Call core engine (new safe version below)
        # ─────────────────────────────────────────────
        return _calc_2y_ois_from_dict(adjusted_meetings, ois_date_str, fixing)

    except Exception as ex:
        print(f"[_calc_2y_ois_for_case] error: {ex}")
        return ""

def _calc_2y_ois_from_dict(meetings: list,
                          ois_date_str: str,
                          fixing: float) -> str:
    """
    Same logic as _calc_2y_ois BUT:
    - Uses dict input
    - No dependency on MeetingRow
    - Fully side-effect safe
    """

    try:
        ois_date = _parse_meeting_date(ois_date_str)
        if ois_date is None:
            return ""

        today = date.today()
        if ois_date <= today:
            return ""

        # ─────────────────────────────────────────────
        # Build meeting schedule
        # ─────────────────────────────────────────────
        parsed = []

        for m in meetings:
            d = _parse_meeting_date(m["date"])
            if d is None:
                continue
            parsed.append((d, m["premium"]))

        parsed.sort(key=lambda x: x[0])

        rate_schedule = []
        cum = fixing

        for mtg_date, prem in parsed:
            cum += prem / 100.0
            rate_schedule.append((mtg_date, cum))

        def get_rate_on(d):
            r = fixing
            for mtg_date, rate in rate_schedule:
                if mtg_date <= d:
                    r = rate
                else:
                    break
            return r

        compound = 1.0
        current = today
        total_days = (ois_date - today).days

        if total_days <= 0:
            return ""

        while current < ois_date:

            next_day = current + timedelta(days=1)
            days_held = 1

            if current.weekday() == 4:  # Friday
                days_held = 3
                next_day = current + timedelta(days=3)

            elif current.weekday() >= 5:
                current = next_day
                continue

            if next_day > ois_date:
                days_held = (ois_date - current).days

            r = get_rate_on(current) / 100.0
            compound *= (1.0 + r * days_held / 360.0)

            current = next_day

        ois_rate = (compound ** (360.0 / total_days) - 1.0) * 100.0

        return f"{ois_rate:.2f}"

    except Exception as ex:
        print(f"[_calc_2y_ois_from_dict] error: {ex}")
        return ""


class MeetingRow:
    def __init__(self):
        self.is_user_added: bool  = False
        self.is_ois_row:    bool  = False   # ← NEW: marks the 2Y OIS row
        self.date:          str   = ""
        self.premium:       float = 0.0
        self.cases:         list  = [None] * MAX_CASES

    @property
    def premium_display(self) -> str:
        if self.premium == 0 and self.is_user_added:
            return ""
        return _g_format(self.premium)

    @premium_display.setter
    def premium_display(self, value: str):
        value = value.strip()
        self.premium = float(value) if value else 0.0

    def get_case(self, i: int) -> str:
        v = self.cases[i]
        return _g_format(v) if v is not None else ""

    def set_case(self, i: int, value: str):
        value = value.strip()
        self.cases[i] = float(value) if value else None


def _g_format(v) -> str:
    if v is None:
        return ""
    return f"{v:g}"


# ─────────────────────────────────────────────────────────────────────────────
# API clients
# ─────────────────────────────────────────────────────────────────────────────
class FedPremiumsClient:
    def __init__(self):
        token_path = Path(__file__).parent / "qh_token.txt"
        self.token = token_path.read_text().strip()

    async def get_fed_premiums(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(QH_API_URL, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)


class SR3ApiClient:
    def __init__(self):
        token_path = Path(__file__).parent / "token.txt"
        self.token = token_path.read_text().strip()

    async def calculate(self, payload: dict):
        headers = {
            "Authorization": self.token,
            "Content-Type":  "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(SR3_API_URL, headers=headers,
                                    data=json.dumps(payload)) as resp:
                body = await resp.text()
                if resp.status == 429:
                    raise Exception("THROTTLED|" + body)
                if not resp.ok:
                    raise Exception(body)
                return json.loads(body)


# ─────────────────────────────────────────────────────────────────────────────
# PDS loader thread
# ─────────────────────────────────────────────────────────────────────────────
class PDSLoaderThread(QThread):
    pds_ready = pyqtSignal(object)   # emits a pandas DataFrame

    def run(self):
        try:
            from pdsAPI import get_instruments_data_from_api
            df = get_instruments_data_from_api()
            self.pds_ready.emit(df)
        except Exception as ex:
            print("[PDSLoaderThread] Error:", ex)
            import pandas as pd
            self.pds_ready.emit(pd.DataFrame())


# ─────────────────────────────────────────────────────────────────────────────
# Shared style helpers
# ─────────────────────────────────────────────────────────────────────────────
HEADER_BG   = "#F5F7FA"
USER_ROW_BG = "#EEF4FF"
LOCKED_BG   = "#EBEBEB"
LOCKED_FG   = "#666666"
OIS_ROW_BG  = "#FFFFFF"   # ← warm amber tint for the 2Y OIS row
OIS_ROW_FG  = "#666666"

_BRUSH_LOCKED_BG   = QBrush(QColor(LOCKED_BG))
_BRUSH_LOCKED_FG   = QBrush(QColor(LOCKED_FG))
_BRUSH_USER_ROW_BG = QBrush(QColor(USER_ROW_BG))
_BRUSH_OIS_BG      = QBrush(QColor(OIS_ROW_BG))
_BRUSH_OIS_FG      = QBrush(QColor(OIS_ROW_FG))

_FLAG_LOCKED   = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
_FLAG_EDITABLE = (Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled |
                  Qt.ItemFlag.ItemIsEditable)


def _make_header_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background:{HEADER_BG}; padding:5px 8px; font-weight:600; "
        f"font-size:12px; color:#333; border-bottom:1px solid #ddd;"
    )
    return lbl


def _make_panel(header: str):
    panel = QWidget()
    panel.setStyleSheet("background:#FAFAFA;")
    outer = QVBoxLayout(panel)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    outer.addWidget(_make_header_label(header))
    content = QWidget()
    content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(0)
    outer.addWidget(content)
    return panel, content_layout


def _style_table(tbl: QTableWidget):
    tbl.setStyleSheet("""
        QTableWidget { gridline-color:#CCCCCC; font-size:11px; }
        QHeaderView::section {
            background:#F5F7FA; padding:3px 6px;
            border:1px solid #CCCCCC; font-size:11px;
        }
        QTableWidget::item:selected {
            background: transparent;
            color: inherit;
        }
        QTableWidget::item:focus {
            background: transparent;
            border: none;
            outline: none;
        }
    """)

    # 🔥 ADD THESE LINES ↓↓↓
    from PyQt6.QtWidgets import QAbstractItemView

    tbl.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
    tbl.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)

    # existing lines
    tbl.horizontalHeader().setHighlightSections(False)
    tbl.verticalHeader().setVisible(False)


# ─────────────────────────────────────────────────────────────────────────────
# LegRow
# ─────────────────────────────────────────────────────────────────────────────
class LegRow:
    """Model for one SR3 option leg."""
    def __init__(self):
        self.expiry:         str        = ""
        self.cp:             str        = "Call"
        self.lots:           int        = 1
        self.long_short:     str        = "Long"
        self.gap:            int        = 0
        self.underlying_mid: float|None = None
        self.underlying_id:  str        = ""
        self._ls_cb                     = None
        self.expiry_gap:     int        = 0

    @property
    def signed_lots(self) -> int:
        return self.lots if self.long_short == "Long" else -self.lots


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3 — SR3 Option Legs
# ─────────────────────────────────────────────────────────────────────────────
class Panel3(QWidget):
    """SR3 Option Legs panel."""

    legs_changed = pyqtSignal(list)

    COL_CP        = 0
    COL_LS        = 1
    COL_LOTS      = 2
    COL_GAP       = 3
    COL_EXP_GAP   = 4

    COLS = ["Call/Put", "Long/Short", "Lots", "Strike Gap", "Expiry Gap"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.legs:          list[LegRow] = []
        self.expiry_map:    dict         = {}
        self.expiry_labels: list[str]    = []
        self._build_ui()
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._update_btn_states)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        toolbar = QWidget()
        toolbar.setStyleSheet("background:#FAFAFA; border-bottom:1px solid #eee;")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(6, 4, 6, 4)

        self.btn_add = QPushButton("+ Add Leg")
        self.btn_add.setFixedWidth(80)
        self.btn_add.setEnabled(False)
        self.btn_add.clicked.connect(self._add_leg)
        tb.addWidget(self.btn_add)

        self.btn_del = QPushButton("− Remove")
        self.btn_del.setFixedWidth(75)
        self.btn_del.setEnabled(False)
        self.btn_del.clicked.connect(self._remove_leg)
        tb.addWidget(self.btn_del)

        tb.addStretch()

        root.addWidget(toolbar)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        _style_table(self.table)
        self.table.setStyleSheet(self.table.styleSheet() + """
            QTableWidget {
                background: #FFFFFF;
                border: none;
            }
            QTableCornerButton::section {
                background: #FFFFFF;
                border: none;
            }
            QHeaderView {
                background: #FFFFFF;
            }
            QHeaderView::section:last {
                background: #F5F7FA;
            }
        """)
        self.table.viewport().setStyleSheet("background: #FFFFFF;")
        root.addWidget(self.table)

    def load_sr3_data(self, df):

        expiry_set = set()
        expiry_und = {}

        mask = (df["product_family_code"] == "SR3")
        sr3 = df[mask]

        for _, row in sr3.iterrows():

            raw_strike = row.get("strike_price")

            # ⭐ skip futures
            if raw_strike in (None, "", 0) or pd.isna(raw_strike):
                continue

            exp_raw = str(row.get("expiry_date") or row.get("hg_expiry_date") or "")
            iso = exp_raw[:10]
            if not iso:
                continue

            und_id = str(row.get("underlying_hg_instrument_id") or "").strip()

            expiry_set.add(iso)

            if iso not in expiry_und:
                expiry_und[iso] = und_id

        # ⭐ VERY IMPORTANT → populate UI
        self.expiry_labels = sorted(expiry_set)
        self.expiry_map = expiry_und

        if self.expiry_labels:
            self.btn_add.setEnabled(True)

        else:
            self.btn_add.setEnabled(False)


    def _add_leg(self):
        if not self.expiry_labels:
            return

        if self.legs:
            prev = self.legs[-1]
            if prev.gap == 0:
                prev.gap = 1
                prev_row = len(self.legs) - 1
                gap_item = self.table.item(prev_row, self.COL_GAP)
                if gap_item:
                    self.table.blockSignals(True)
                    gap_item.setText("1")
                    self.table.blockSignals(False)

        leg               = LegRow()
        leg.expiry        = self.expiry_labels[0]
        leg.underlying_id = self.expiry_map.get(leg.expiry, "")
        self.legs.append(leg)

        r = len(self.legs) - 1
        self.table.insertRow(r)
        self._init_row_widgets(r, leg)
        self._update_btn_states()
        self.legs_changed.emit(list(self.legs))

    def _remove_leg(self):
        r = self.table.currentRow()
        if r < 0 or r >= len(self.legs):
            return
        self.legs.pop(r)
        self.table.removeRow(r)

        if self.legs:
            last = len(self.legs) - 1
            self.legs[last].gap = 0
            item = self.table.item(last, self.COL_GAP)
            if item:
                item.setText("")

        self._update_btn_states()
        self.legs_changed.emit(list(self.legs))

    def _init_row_widgets(self, r: int, leg: LegRow):
        cp_combo = QComboBox()
        cp_combo.setView(QListView())
        cp_combo.setMaxVisibleItems(5)
        cp_combo.setStyleSheet("QComboBox { combobox-popup: 1; }")
        cp_combo.addItems(["Call", "Put"])
        cp_combo.setCurrentText(leg.cp)
        cp_combo.currentTextChanged.connect(
            lambda text, _r=r: self._on_cp_changed(_r, text)
        )
        self.table.setCellWidget(r, self.COL_CP, cp_combo)

        ls_combo = QComboBox()
        ls_combo.setView(QListView())
        ls_combo.setMaxVisibleItems(3)
        ls_combo.setStyleSheet("QComboBox { combobox-popup: 1; }")
        ls_combo.addItems(["Long", "Short"])
        ls_combo.setCurrentText(leg.long_short)
        ls_combo.currentTextChanged.connect(
            lambda text, _r=r: self._on_ls_changed(_r, text)
        )
        self.table.setCellWidget(r, self.COL_LS, ls_combo)

        self.table.blockSignals(True)

        lots_item = QTableWidgetItem(str(leg.lots))
        lots_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        lots_item.setFlags(_FLAG_EDITABLE)
        self.table.setItem(r, self.COL_LOTS, lots_item)

        gap_val  = str(leg.gap) if leg.gap else ""
        gap_item = QTableWidgetItem(gap_val)
        gap_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        gap_item.setFlags(_FLAG_EDITABLE)
        self.table.setItem(r, self.COL_GAP, gap_item)

        exp_gap_item = QTableWidgetItem("")
        exp_gap_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        exp_gap_item.setFlags(_FLAG_EDITABLE)
        self.table.setItem(r, self.COL_EXP_GAP, exp_gap_item)

        self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem):
        col = item.column()
        if col not in (self.COL_LOTS, self.COL_GAP, self.COL_EXP_GAP):
            return

        r = item.row()
        if r < 0 or r >= len(self.legs):
            return

        txt = item.text().strip()

        if col == self.COL_LOTS:
            try:
                self.legs[r].lots = max(1, int(float(txt)))
            except ValueError:
                pass

        elif col == self.COL_GAP:
            is_last = (r == len(self.legs) - 1)
            try:
                val = int(float(txt)) if txt else 0
            except ValueError:
                val = 0

            self.legs[r].gap = 0 if is_last else val

            if is_last and txt:
                self.table.blockSignals(True)
                item.setText("")
                self.table.blockSignals(False)

        elif col == self.COL_EXP_GAP:
            try:
                self.legs[r].expiry_gap = int(float(txt)) if txt else 0
            except ValueError:
                self.legs[r].expiry_gap = 0

        self.legs_changed.emit(list(self.legs))

    def _on_cp_changed(self, row: int, text: str):
        if row < len(self.legs):
            self.legs[row].cp = text
        self.legs_changed.emit(list(self.legs))

    def _on_ls_changed(self, row: int, text: str):
        if row < len(self.legs):
            self.legs[row].long_short = text
        self.legs_changed.emit(list(self.legs))

    def _update_btn_states(self):
        self.btn_del.setEnabled(self.table.currentRow() >= 0)

    def closeEvent(self, e):
        super().closeEvent(e)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4 — Live SR3 Mid-Price Matrix
# ─────────────────────────────────────────────────────────────────────────────
class Panel4(QWidget):
    _LS_KEY = "panel4_sr3_matrix"
    _DIV    = 100

    _BRUSH_GREEN  = QBrush(QColor("#C8F0C8"))
    _BRUSH_RED    = QBrush(QColor("#F5C8C8"))
    _BRUSH_YELLOW = QBrush(QColor("#FFFACD"))
    _BRUSH_EMPTY  = QBrush(QColor("#FAFAFA"))
    _BRUSH_STRIKE = QBrush(QColor("#F0F0F0"))

    _BOLD = QFont()
    _BOLD.setBold(True)

    _NO_EDIT = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

    # Signal: emits (cp, long_short, lots, strike_str, entry_price_str, expiry_str, underlying_id)
    send_to_strategy = pyqtSignal(str, str, int, str, str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._inst:              dict[str, dict]  = {}
        self._expiries:          list[str]        = []
        self._strikes:           list[float]      = []
        self._mids:              dict[tuple, float] = {}
        self._legs:              list[LegRow]     = []
        self._ls_cb                               = None
        self._ls_keys:           list[str]        = []
        # expiry ISO → underlying instrument id (for Panel6 live feed)
        self._expiry_underlying: dict[str, str]   = {}

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        bar = QWidget()
        bar.setStyleSheet("background:#FAFAFA; border-bottom:1px solid #eee;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(8, 3, 8, 3)

        self.status_lbl = QLabel("Waiting For PDS Data…")
        self.status_lbl.setStyleSheet("font-size:11px; color:#666;")
        bl.addWidget(self.status_lbl)

        bl.addStretch()

        self.leg_lbl = QLabel("No legs — add legs in SR3 Option Legs")
        self.leg_lbl.setStyleSheet("font-size:11px; color:#888; font-style:italic;")
        bl.addWidget(self.leg_lbl)

        root.addWidget(bar)

        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        _style_table(self.table)
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )

        # Enable right-click context menu
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)

        root.addWidget(self.table)

    def _on_context_menu(self, pos: QPoint):
        """Show context menu on right-click. Only show 'Move to Strategy' for data cells."""
        item = self.table.itemAt(pos)
        if item is None:
            return

        row = item.row()
        col = item.column()

        # Col 0 is the Strike label column — skip it
        if col == 0:
            return

        cell_text = item.text().strip()
        if not cell_text:
            return

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                font-size: 11px;
                padding: 4px 0px;
            }
            QMenu::item {
                padding: 5px 20px 5px 12px;
                color: #333;
            }
            QMenu::item:selected {
                background: #EEF4FF;
                color: #1a56db;
            }
            QMenu::separator {
                height: 1px;
                background: #EEEEEE;
                margin: 3px 8px;
            }
        """)

        ei = col - 1  # expiry index

        # Build sub-menu entries for each leg that contributed to this cell
        if self._legs:
            action_title = menu.addAction("Move to Portfolio")
            action_title.setEnabled(True)

            # Separator then per-leg actions if multi-leg
            if len(self._legs) > 1:
                menu.addSeparator()
                for leg_idx, leg in enumerate(self._legs):
                    strike_cursor = row + sum(
                        max(0, self._legs[k].gap) for k in range(leg_idx)
                    )
                    expiry_cursor = ei + sum(
                        max(0, self._legs[k].expiry_gap) for k in range(leg_idx)
                    )
                    if strike_cursor < len(self._strikes) and expiry_cursor < len(self._expiries):
                        strike_val = _g_format(self._strikes[strike_cursor])
                        expiry_val = self._expiries[expiry_cursor]
                        cp_display = leg.cp
                        ls_display = leg.long_short
                        leg_action = menu.addAction(
                            f"  Leg {leg_idx + 1}: {ls_display} {cp_display}  "
                            f"Strike {strike_val}  Expiry {self._short_expiry(expiry_val)}"
                        )
                        leg_action.setEnabled(False)  # informational label

            action = menu.exec(self.table.viewport().mapToGlobal(pos))

            if action == action_title:
                self._emit_strategy_rows(row, ei)

    def _emit_strategy_rows(self, base_row: int, base_ei: int):
        """For each leg in self._legs, emit a send_to_strategy signal."""
        strike_cursor = base_row
        expiry_cursor = base_ei

        for leg in self._legs:
            if strike_cursor >= len(self._strikes):
                break
            if expiry_cursor >= len(self._expiries):
                break

            cp        = leg.cp          # "Call" or "Put"
            ls        = leg.long_short  # "Long" or "Short"
            lots      = leg.lots
            strike    = _g_format(self._strikes[strike_cursor])
            expiry    = self._expiries[expiry_cursor]

            # Fetch the individual leg mid price
            cp_code      = "C" if cp == "Call" else "P"
            mid          = self._fetch_mid_safe(strike_cursor, expiry_cursor, cp_code)
            entry_str    = f"{mid:.2f}" if mid is not None else ""

            # Underlying instrument id for this expiry
            underlying_id = self._expiry_underlying.get(expiry, "")

            self.send_to_strategy.emit(cp, ls, lots, strike, entry_str, expiry,
                                       underlying_id)

            strike_cursor += leg.gap
            expiry_cursor += leg.expiry_gap

            if strike_cursor < 0 or strike_cursor >= len(self._strikes):
                break
            if expiry_cursor < 0 or expiry_cursor >= len(self._expiries):
                break



    def load_sr3_data(self, df):
        import traceback
        import pandas as pd

        print(f"[Panel4] load_sr3_data called, df shape={df.shape}, "
            f"cols={list(df.columns)[:10]}")

        inst:          dict[str, dict] = {}
        expiry_set:    set[str]        = set()
        strike_set:    set[float]      = set()
        expiry_und:    dict[str, str]  = {}

        skipped_no_cp  = 0
        skipped_no_iid = 0
        skipped_no_exp = 0

        future_expiry_set = set()
        future_underlying = {}          # SR3 futures: expiry_iso → hg_instrument_id
        zq_future_expiry_set = set()
        zq_future_underlying = {}       # ZQ futures: expiry_iso → hg_instrument_id

        SR3_FUTURE_PRODUCT_ID = "12049546238623417960"
        ZQ_FUTURE_PRODUCT_ID  = "8594138358590911504"

        def _is_quarterly_expiry(iso: str) -> bool:
            try:
                dt = datetime.fromisoformat(iso)
                return dt.month in (3, 6, 9, 12)
            except Exception:
                return False

        def _to_iso(date_str: str) -> str:
            """Convert MM/DD/YYYY or YYYY-MM-DD (or longer ISO) to YYYY-MM-DD."""
            s = date_str.strip()
            if not s:
                return ""
            if "/" in s:
                try:
                    return datetime.strptime(s, "%m/%d/%Y").strftime("%Y-%m-%d")
                except ValueError:
                    return ""
            return s[:10]

        FUTURE_PRODUCT_IDS = {SR3_FUTURE_PRODUCT_ID, ZQ_FUTURE_PRODUCT_ID}

        try:
            # ⭐ IMPORTANT — DO NOT FILTER STRIKE HERE
            mask = (df["product_family_code"] == "SR3")
            sr3 = df[mask]

            print(f"[Panel4] SR3 total rows: {len(sr3)}")

            # ⭐⭐⭐ SR3 FUTURES — product id 12049546238623417960
            for _, row in sr3[sr3["hg_product_id"].astype(str) == SR3_FUTURE_PRODUCT_ID].iterrows():
                iid = str(row.get("hg_instrument_id") or "").strip()
                if not iid:
                    continue
                iso = _to_iso(str(row.get("expiry_date") or row.get("hg_expiry_date") or ""))
                if iso and _is_quarterly_expiry(iso):
                    future_expiry_set.add(iso)
                    future_underlying[iso] = iid

            # ⭐⭐⭐ ZQ FUTURES — product id 8594138358590911504
            zq_mask = (df["hg_product_id"].astype(str) == ZQ_FUTURE_PRODUCT_ID)
            for _, row in df[zq_mask].iterrows():
                iid = str(row.get("hg_instrument_id") or "").strip()
                if not iid:
                    continue
                iso = _to_iso(str(row.get("expiry_date") or row.get("hg_expiry_date") or ""))
                if iso:
                    zq_future_expiry_set.add(iso)
                    zq_future_underlying[iso] = iid

            print(f"[Panel4] SR3 futures: {len(future_expiry_set)}  ZQ futures: {len(zq_future_expiry_set)}")

            for _, row in sr3.iterrows():

                # skip futures — already handled above
                if str(row.get("hg_product_id") or "") in FUTURE_PRODUCT_IDS:
                    continue

                iid = str(row.get("hg_instrument_id") or "").strip()
                if not iid:
                    skipped_no_iid += 1
                    continue

                exp_raw = str(
                    row.get("expiry_date") or row.get("hg_expiry_date") or ""
                )
                iso = _to_iso(exp_raw)
                if not iso:
                    skipped_no_exp += 1
                    continue

                raw_strike = row.get("strike_price")

                # ⭐⭐⭐ OPTION CONTRACT
                try:
                    strike_f = round(float(raw_strike) / self._DIV, 6)
                except (TypeError, ValueError):
                    continue

                cp = row.get("cp")
                if cp not in ("C", "P"):
                    name = str(row.get("instrument_name") or "")
                    cp = ("C" if " C" in name else
                        "P" if " P" in name else None)

                if cp not in ("C", "P"):
                    skipped_no_cp += 1
                    continue

                und_id = str(row.get("underlying_hg_instrument_id") or "").strip()

                inst[iid] = {
                    "expiry": iso,
                    "strike": strike_f,
                    "cp": cp,
                    "underlying_id": und_id
                }

                expiry_set.add(iso)
                strike_set.add(strike_f)

                if iso and und_id and iso not in expiry_und:
                    expiry_und[iso] = und_id

            # ⭐⭐⭐ ASSIGN FUTURE DATA
            self._future_expiries   = sorted(future_expiry_set)
            self._future_underlying = future_underlying
            self._zq_future_expiries   = sorted(zq_future_expiry_set)
            self._zq_future_underlying = zq_future_underlying

            # ⭐⭐⭐ ASSIGN OPTION DATA
            self._inst              = inst
            self._expiries          = sorted(expiry_set)
            self._strikes           = sorted(strike_set)
            self._expiry_underlying = expiry_und

            # ⭐⭐⭐ LIGHTSTREAMER OPTION SUBSCRIPTION
            if inst and LightstreamerClient is not None:
                try:
                    self._ls_cb  = self._on_ls_update
                    self._ls_keys = LSManager.instance().get_or_create_chunked(
                        self._LS_KEY, list(inst.keys()), chunk_size=400
                    )
                    LSManager.instance().attach_ui_multi(
                        self._ls_keys, self._ls_cb
                    )
                except Exception as ls_ex:
                    print(f"[Panel4] LS subscribe error: {ls_ex}")
                    traceback.print_exc()

            self.status_lbl.setText(
                f"{len(self._expiries)} Opt Expiries  ·  "
                f"{len(self._future_expiries)} Fut Expiries  ·  "
                f"{len(self._strikes)} Strikes  ·  "
                f"{len(inst)} Opt Instruments"
            )

        except Exception as ex:
            print(f"[Panel4.load_sr3_data] EXCEPTION: {ex}")
            traceback.print_exc()
            self.status_lbl.setText(f"PDS error: {ex}")

        self._rebuild_table()

    def on_legs_changed(self, legs: list):
        self._legs = list(legs)

        if not legs:
            self.leg_lbl.setText("No legs — add legs in SR3 Option Legs")
        else:
            parts = []
            for leg in legs:
                direction = leg.long_short
                parts.append(f"{direction} {leg.cp} ×{leg.lots}")
            self.leg_lbl.setText("  |  ".join(parts))

        self._rebuild_table()

    def _on_ls_update(self, iid: str, row: dict):
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0

        meta = self._inst.get(iid)
        if meta is None:
            return

        try:
            raw_bid = row.get("BestBid", "")
            raw_ask = row.get("BestAsk", "")
            bid = float(raw_bid) if raw_bid else None
            ask = float(raw_ask) if raw_ask else None
        except (ValueError, TypeError):
            return

        if bid is None or ask is None:
            return

        mid = (bid + ask) / 2.0
        strike = round(meta["strike"], 6)
        key = (meta["expiry"], strike, meta["cp"])
        self._mids[key] = mid

        # existing UI refresh
        self._refresh_cell(meta["expiry"], strike)

        # ✅ NEW: push underlying updates to Panel6 (for IV)
        if hasattr(self, "_panel6") and self._panel6:
            self._panel6._on_underlying_update(meta["expiry"], mid)

    def _rebuild_table(self):
        tbl = self.table
        tbl.blockSignals(True)

        n_rows = len(self._strikes)
        n_cols = 1 + len(self._expiries)

        tbl.setRowCount(n_rows)
        tbl.setColumnCount(n_cols)

        headers = ["Strike"] + [self._expiry_label(i) for i in range(len(self._expiries))]
        tbl.setColumnCount(len(headers))
        tbl.setHorizontalHeaderLabels(headers)
        


        tbl.setColumnWidth(0, 130)
        for c in range(1, n_cols):
            tbl.setColumnWidth(c, 82)

        for r, strike in enumerate(self._strikes):
            strike_label = self._strike_label(r)
            s_item = QTableWidgetItem(strike_label)
            s_item.setFlags(self._NO_EDIT)
            s_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            s_item.setFont(self._BOLD)
            s_item.setBackground(self._BRUSH_STRIKE)
            tbl.setItem(r, 0, s_item)

            for c, exp in enumerate(self._expiries, start=1):
                tbl.setItem(r, c, self._make_cell(r, c - 1))

        tbl.blockSignals(False)

    def _refresh_cell(self, expiry: str, strike: float):
        strike = round(strike, 6)
        si = None
        for i, s in enumerate(self._strikes):
            if abs(s - strike) < 1e-4:
                si = i
                break
        if si is None:
            return
        try:
            ei = self._expiries.index(expiry)
        except ValueError:
            return

        affected_si = set()
        affected_ei = set()
        for leg in self._legs:
            g  = leg.gap
            eg = leg.expiry_gap
            for base_si in range(len(self._strikes)):
                target_si = base_si + g
                if base_si == si or target_si == si:
                    affected_si.add(base_si)
            for base_ei in range(len(self._expiries)):
                target_ei = base_ei + eg
                if base_ei == ei or target_ei == ei:
                    affected_ei.add(base_ei)

        if not affected_si:
            affected_si.add(si)
        if not affected_ei:
            affected_ei.add(ei)

        for r in affected_si:
            label_item = self.table.item(r, 0)
            if label_item is not None:
                label_item.setText(self._strike_label(r))
            for e in affected_ei:
                c = e + 1
                new_item = self._make_cell(r, e)
                existing = self.table.item(r, c)
                if existing is None:
                    self.table.setItem(r, c, new_item)
                else:
                    existing.setText(new_item.text())
                    existing.setForeground(new_item.foreground())
                    if new_item.background() != existing.background():
                        existing.setBackground(new_item.background())

    def _make_cell(self, si: int, ei: int) -> QTableWidgetItem:
        item = QTableWidgetItem("")
        item.setFlags(self._NO_EDIT)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        

        if not self._legs:
            item.setBackground(self._BRUSH_EMPTY)
            item.setForeground(QBrush(QColor("black")))
            return item

        total = 0.0
        strike_cursor = si
        expiry_cursor = ei
        required_legs = 0
        received_legs = 0

        for leg in self._legs:
            if strike_cursor >= len(self._strikes):
                break
            if expiry_cursor >= len(self._expiries):
                break

            required_legs += 1
            anchor_si = strike_cursor
            anchor_ei = expiry_cursor
            cp   = "C" if leg.cp == "Call" else "P"
            sign = leg.signed_lots
            mid = self._fetch_mid_safe(anchor_si, anchor_ei, cp)

            if mid is None:
                item.setBackground(self._BRUSH_EMPTY)
                return item

            total += sign * mid
            received_legs += 1
            strike_cursor += leg.gap
            expiry_cursor += leg.expiry_gap

            # ⭐ bound safety
            if strike_cursor < 0 or strike_cursor >= len(self._strikes):
                return item

            if expiry_cursor < 0 or expiry_cursor >= len(self._expiries):
                return item

        if required_legs == 0 or received_legs != required_legs:
            item.setBackground(self._BRUSH_EMPTY)
            return item

        item.setText(f"{total:+.2f}")
        item.setBackground(self._BRUSH_EMPTY)

        if total > 0:
            item.setForeground(QBrush(QColor("blue")))
        elif total < 0:
            item.setForeground(QBrush(QColor("red")))
        else:
            item.setForeground(QBrush(QColor("black")))

        return item

    def _strike_label(self, si: int) -> str:
        if si >= len(self._strikes):
            return ""

        if not self._legs:
            return _g_format(self._strikes[si])

        if (len(self._legs) == 1
                and self._legs[0].gap == 0
                and self._legs[0].long_short == "Long"):
            return _g_format(self._strikes[si])

        n_strikes = len(self._strikes)
        tokens    = []
        cursor    = si

        for leg in self._legs:
            if cursor >= n_strikes:
                break
            sign = "+" if leg.long_short == "Long" else "-"
            tokens.append(f"{sign}{_g_format(self._strikes[cursor])}")
            cursor += leg.gap
            if cursor < 0 or cursor >= n_strikes:
                break

        return "/".join(tokens)

    def _expiry_label(self, ei: int) -> str:
        if ei >= len(self._expiries):
            return ""

        if not self._legs:
            return self._short_expiry(self._expiries[ei])

        if (len(self._legs) == 1
                and self._legs[0].expiry_gap == 0
                and self._legs[0].long_short == "Long"):
            return self._short_expiry(self._expiries[ei])

        tokens = []
        cursor = ei

        for leg in self._legs:
            if cursor >= len(self._expiries):
                break
            sign = "+" if leg.long_short == "Long" else "-"
            exp  = self._short_expiry(self._expiries[cursor])
            tokens.append(f"{sign}{exp}")
            cursor += leg.expiry_gap
            if cursor < 0 or cursor >= len(self._expiries):
                break

        return "/".join(tokens)

    @staticmethod
    def _short_expiry(iso: str) -> str:
        try:
            dt = datetime.fromisoformat(iso)
            return dt.strftime("%b-%y")
        except Exception:
            return iso

    def _get_mids_snapshot(self):
        return dict(self._mids)

    def _fetch_mid_safe(self, strike_i, expiry_i, cp):

        if strike_i < 0 or strike_i >= len(self._strikes):
            return None

        if expiry_i < 0 or expiry_i >= len(self._expiries):
            return None

        s = self._strikes[strike_i]
        e = self._expiries[expiry_i]

        v = self._mids.get((e, round(s, 6), cp))
        if v is not None:
            return v

        # ⭐ SAFE ITERATION
        for (ke, ks, kc), mv in list(self._mids.items()):
            if ke == e and kc == cp and abs(ks - s) < 1e-4:
                return mv

        return None

    def closeEvent(self, e):
        if self._ls_cb and self._ls_keys:
            try:
                LSManager.instance().detach_ui_multi(self._ls_keys, self._ls_cb)
            except Exception:
                pass
        super().closeEvent(e)


# ─────────────────────────────────────────────────────────────────────────────
# Panel 5 — Plotly Payoff Chart  (replaces matplotlib version)
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from PyQt6.QtWebEngineWidgets import QWebEngineView
import plotly.graph_objects as go


SR3_MULTIPLIER = 25  # $25 per 0.01 (1bp) move

_PAYOFF_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22", "#7f7f7f",
]

_TODAY_COLOR  = "rgb(0, 90, 255)"
_EXPIRY_COLOR = "rgb(255, 150, 0)"
_INTER_COLORS = [
    "rgb(231,76,60)", "rgb(155,89,182)", "rgb(46,204,113)",
    "rgb(52,152,219)", "rgb(241,196,15)",
]


def _scenario_color(label: str, inter_idx: int) -> str:
    if label == "Today":
        return _TODAY_COLOR
    if label == "Expiry":
        return _EXPIRY_COLOR
    return _INTER_COLORS[inter_idx % len(_INTER_COLORS)]


def _build_daily_scenarios(today: date, expiry: date):
    scenarios = []

    current = today
    idx = 0

    while current < expiry:
        label = "Today" if idx == 0 else f"T+{idx}d"
        scenarios.append((label, current))

        current += timedelta(days=1)
        idx += 1

    scenarios.append(("Expiry", expiry))

    return scenarios


class PayoffWidget(QWidget):
    """SR3 Payoff Chart Widget."""

    def __init__(self, panel6):
        super().__init__()
        self.panel6 = panel6
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        toolbar = QWidget()
        toolbar.setFixedHeight(36)
        toolbar.setStyleSheet("background:#FAFAFA; border-bottom:1px solid #eee;")
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(6, 0, 6, 0)
        tb.setSpacing(4)

        self.btn_refresh = QPushButton("Refresh Payoff")
        self.btn_refresh.setFixedHeight(28)
        self.btn_refresh.clicked.connect(self._update_payoff)
        tb.addWidget(self.btn_refresh)

        tb.addStretch()

        root.addWidget(toolbar)

        self.web_view = QWebEngineView()
        root.addWidget(self.web_view)

    def _update_payoff(self):
        portfolios = self.panel6.portfolios
        if not portfolios:
            self.web_view.setHtml("<h3>No portfolios to display</h3>")
            return

        import numpy as np
        fig = go.Figure()
        has_data = False

        for name, portfolio in portfolios.items():
            legs = []
            for i in range(portfolio.childCount()):
                sr = portfolio.child(i).data(0, Qt.ItemDataRole.UserRole)
                if sr and not isinstance(sr, str):
                    legs.append(sr)

            if not legs:
                continue

            family = "SR3"
            dtes = []
            spots = []

            for leg in legs:
                try:
                    expiry = datetime.fromisoformat(leg.expiry)
                    dte = max((expiry - datetime.utcnow()).total_seconds(), 0.0) / (24 * 3600)
                except Exception:
                    dte = 0.0
                dtes.append(dte)

                if leg.underlying_mid is not None:
                    spots.append((dte, leg.underlying_mid))

            if not spots:
                continue

            S0 = min(spots, key=lambda x: x[0])[1] if spots else 0.0
            if S0 is None:
                continue

            S0 = S0 / 100.0
            S_range = np.array(_spot_range(S0, family))
            pnl_mul = _pnl_mul(family, legs[0].instrument_type if legs else "Option")

            has_data = True

            min_dte = min(dtes) if dtes else 0.0
            import sys
            print(f"[DEBUG PAYOFF] Family: {family}, S0: {S0:.4f}, min_dte: {min_dte:.2f}", file=sys.stderr)
            print(f"[DEBUG PAYOFF] dtes: {dtes}", file=sys.stderr)
            for i, leg in enumerate(legs):
                print(f"[DEBUG PAYOFF] Leg {i}: cp={leg.cp}, strike={leg.strike}, expiry={leg.expiry}, dte={dtes[i] if i < len(dtes) else 'N/A'}, iv={leg.iv}", file=sys.stderr)

            curves = {}
            curves["Today"] = combined_model_value(S_range, legs, dtes, pnl_mul)

            expiry_dtes = [0.0] * len(dtes)
            for i, d in enumerate(dtes):
                if d > min_dte:
                    expiry_dtes[i] = d - min_dte
            curves["Expiry"] = combined_model_value(S_range, legs, expiry_dtes, pnl_mul)

            days_to_plot = int(np.ceil(min_dte))
            if days_to_plot > 1:
                for t in range(1, days_to_plot):
                    temp_dtes = [max(d - t, 0.0) for d in dtes]
                    curves[f"T-{t}d"] = combined_model_value(S_range, legs, temp_dtes, pnl_mul)

            for label, pnl in curves.items():
                is_main = label in {"Today", "Expiry"}
                fig.add_trace(go.Scatter(
                    x=list(S_range),
                    y=list(pnl),
                    mode="lines",
                    name=f"{name} {label}",
                    line=dict(dash="solid" if is_main else "dash", width=2 if is_main else 1),
                    visible=True if is_main else "legendonly",
                ))

            fig.add_vline(x=S0, line_dash="dash", line_color="green", line_width=1)

        if not has_data:
            self.web_view.setHtml("<h3>No data to display</h3>")
            return

        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(
            title="Option Strategy Payoff",
            xaxis_title="Spot (Rate)",
            yaxis_title="P&L ($)",
            legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
            margin=dict(l=50, r=20, t=40, b=40),
            template="plotly_white",
        )
        self.web_view.setHtml(fig.to_html(include_plotlyjs="cdn", full_html=True))


class Panel5(QWidget):
    """
    Payoff chart panel using PayoffWidget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._panel4 = None
        self._panel6 = None
        self._payoff_widget = None
        self._build_ui()

    def set_panels(self, panel4, panel6):
        self._panel4 = panel4
        self._panel6 = panel6
        if self._panel6:
            self._payoff_widget = PayoffWidget(self._panel6)
            self._replace_placeholder()

    def _replace_placeholder(self):
        from PyQt6.QtWidgets import QLabel
        layout = self.layout()
        if layout is None or self._payoff_widget is None:
            return
        if layout.count() > 0:
            placeholder = self.findChild(QLabel, "placeholder")
            if placeholder:
                layout.removeWidget(placeholder)
                placeholder.deleteLater()
        layout.addWidget(self._payoff_widget)

    def _build_ui(self):
        from PyQt6.QtWidgets import QVBoxLayout, QLabel
        from PyQt6.QtCore import Qt

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        placeholder = QLabel("Loading payoff chart...")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("""
            color: #999;
            font-size: 14px;
        """)
        placeholder.setObjectName("placeholder")
        root.addWidget(placeholder)

    def _redraw(self):
        if hasattr(self, '_payoff_widget') and self._payoff_widget:
            self._payoff_widget._update_payoff()

class StrategyRow:
    def __init__(self):
        self.instrument_type = "Option"   # "Option" or "Future"
        self.future_product  = "SR3"      # "SR3" or "ZQ" — only used when type is Future
        self.cp = "Call"
        self.long_short = "Long"
        self.lots = 1
        self.strike = ""
        self.entry_price = ""
        self.expiry = ""
        self.underlying_id = ""
        self.underlying_mid = None
        self.iv = None


class Panel6(QWidget):
    """SR3 Options Strategy panel."""

    _LS_KEY = "panel6_underlying"

    COL_TYPE        = 0
    COL_CP          = 1
    COL_LS          = 2
    COL_LOTS        = 3
    COL_STRIKE      = 4
    COL_ENTRY_PRICE = 5
    COL_IV = 6
    COL_EXPIRY      = 7
    COL_UNDERLYING  = 8

    COLS = ["Type", "Call/Put", "Long/Short", "Lots", "Strike", "Entry Price", "IV", "Expiry", "Underlying"]

    _NO_EDIT = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows:                list[StrategyRow] = []

        self._strike_labels:      list[str]        = []
        self._expiry_labels:      list[str]        = []
        self._expiry_underlying:  dict[str, str]   = {}
        self._zq_future_expiry_labels: list[str]   = []
        self._zq_future_underlying_map: dict[str, str] = {}

        self._panel4 = None

        self._und_mids:           dict[str, float] = {}
        self._ls_cb                                = None
        self._ls_keys:            list[str]        = []
        self.portfolios = {}

        self._build_ui()

    def _time_to_expiry(self, expiry_iso):
        try:
            expiry = datetime.fromisoformat(expiry_iso)
        except:
            return None

        now = datetime.utcnow()
        seconds = max((expiry - now).total_seconds(), 0.0)

        T = seconds / (365.0 * 24 * 3600)

        MIN_T = (9.0 / 6.5) / 252
        return max(T, MIN_T)

    def _calc_sr3_iv(self, row: StrategyRow):
        try:
            if not row.entry_price or not row.strike or not row.underlying_mid:
                return None

            price = float(row.entry_price) / 100.0   # ticks to price points
            F = float(row.underlying_mid) / 100.0     # ← FIX: scale to match strike
            K = float(row.strike)
            T = self._time_to_expiry(row.expiry)

            if not T:
                return None

            cp = "C" if row.cp == "Call" else "P"

            return implied_vol_bachelier(
                price=price,
                F=F,
                K=K,
                T=T,
                cp=cp
            )
        except:
            return None

    def _update_iv_for_row(self, r: int):
        if r < 0 or r >= len(self.rows):
            return

        sr = self.rows[r]

        if sr.instrument_type == "Future":
            for i in range(self.tree.topLevelItemCount()):
                pf = self.tree.topLevelItem(i)
                for j in range(pf.childCount()):
                    item = pf.child(j)
                    if item.data(0, Qt.ItemDataRole.UserRole) is sr:
                        item.setText(self.COL_IV, "")
                        return
            return

        iv = self._calc_sr3_iv(sr)

        # ✅ KEY FIX: store IV back on the StrategyRow so Panel5 can use it
        sr.iv = iv  # raw decimal from implied_vol_bachelier, e.g. 0.0052

        for i in range(self.tree.topLevelItemCount()):
            pf = self.tree.topLevelItem(i)
            for j in range(pf.childCount()):
                item = pf.child(j)
                if item.data(0, Qt.ItemDataRole.UserRole) is sr:
                    item.setText(self.COL_IV, f"{iv * 100:.2f}%" if iv else "")
                    return

    def _short_expiry(self, iso: str) -> str:
        try:
            dt = datetime.fromisoformat(iso)
            return dt.strftime("%b-%y")
        except:
            return iso

    def _detect_strategy(self, portfolio):

        legs = []

        for i in range(portfolio.childCount()):
            sr = portfolio.child(i).data(0, Qt.ItemDataRole.UserRole)

            if not sr or isinstance(sr, str):
                continue

            if sr.instrument_type != "Option":
                continue

            legs.append(sr)

        if len(legs) == 1:
            return "Outright"

        if len(legs) < 2:
            return "-"

        strikes = [l.strike for l in legs]
        expiries = [l.expiry for l in legs]
        cps = [l.cp for l in legs]
        lots = [l.lots for l in legs]
        sides = [l.long_short for l in legs]

        unique_strikes = set(strikes)
        unique_expiries = set(expiries)
        unique_cps = set(cps)

        net = sum(l.lots if l.long_short == "Long" else -l.lots for l in legs)

        # ─────────────────────────────
        # STRADDLE
        # ─────────────────────────────
        if len(unique_strikes) == 1 and len(unique_expiries) == 1 and unique_cps == {"Call", "Put"}:
            return "Long Straddle" if net > 0 else "Short Straddle"

        # ─────────────────────────────
        # STRANGLE
        # ─────────────────────────────
        if len(unique_strikes) > 1 and len(unique_expiries) == 1 and unique_cps == {"Call", "Put"}:
            return "Long Strangle" if net > 0 else "Short Strangle"

        # ─────────────────────────────
        # CALL SPREAD
        # ─────────────────────────────
        if unique_cps == {"Call"} and len(unique_strikes) > 1 and len(unique_expiries) == 1:
            return "Call Spread"

        # ─────────────────────────────
        # PUT SPREAD
        # ─────────────────────────────
        if unique_cps == {"Put"} and len(unique_strikes) > 1 and len(unique_expiries) == 1:
            return "Put Spread"

        # ─────────────────────────────
        # CALL CALENDAR
        # ─────────────────────────────
        if unique_cps == {"Call"} and len(unique_strikes) == 1 and len(unique_expiries) > 1:
            return "Call Calendar"

        # ─────────────────────────────
        # PUT CALENDAR
        # ─────────────────────────────
        if unique_cps == {"Put"} and len(unique_strikes) == 1 and len(unique_expiries) > 1:
            return "Put Calendar"

        # ─────────────────────────────
        # TIME FLY (Calendar Butterfly)
        # same strike, 3 expiries, mix of long/short
        # ─────────────────────────────
        if len(unique_strikes) == 1 and len(unique_expiries) >= 3:
            return "Time Fly"

        if len(legs) == 4 and unique_cps == {"Call", "Put"}:
            if len(unique_strikes) == 3:
                return "Iron Fly"

        if len(legs) == 4 and unique_cps == {"Call", "Put"}:
            if len(unique_strikes) == 4:
                return "Iron Condor"

        if len(unique_strikes) == 1 and len(unique_expiries) > 1:
            return "Calendar"

        if len(unique_strikes) > 1 and len(unique_expiries) > 1:
            return "Diagonal"


        if len(set(lots)) > 1:
            return "Ratio"

        if (
            len(legs) == 2
            and unique_cps == {"Call", "Put"}
            and len(unique_expiries) == 1
            and len(unique_strikes) == 2
        ):
            # Determine direction
            call_leg = next((l for l in legs if l.cp == "Call"), None)
            put_leg  = next((l for l in legs if l.cp == "Put"), None)

            if call_leg and put_leg:
                if call_leg.long_short == "Long" and put_leg.long_short == "Short":
                    return "Long Risk Reversal"
                elif call_leg.long_short == "Short" and put_leg.long_short == "Long":
                    return "Short Risk Reversal"

            return "Risk Reversal"

        return "-"

    def _update_structure_price(self, portfolio):

        data = portfolio.data(0, Qt.ItemDataRole.UserRole)

        # ✅ safety fix
        if not isinstance(data, dict):
            data = {"structure_price": "", "is_user_set": False}
            portfolio.setData(0, Qt.ItemDataRole.UserRole, data)

        if data.get("is_user_set"):
            return

        total = self._calc_portfolio_price(portfolio)
        summary = self._calc_portfolio_summary(portfolio)

        data["structure_price"] = total


        strategy_name = self._detect_strategy(portfolio)
        portfolio.setText(self.COL_CP, strategy_name)

        if total > 0:
            portfolio.setText(self.COL_LS, "Debit")
        elif total < 0:
            portfolio.setText(self.COL_LS, "Credit")
        else:
            portfolio.setText(self.COL_LS, "")

        portfolio.setText(self.COL_LOTS,
            f"{summary['net_lots']:+d}" if summary["net_lots"] else "")

        portfolio.setText(self.COL_STRIKE, summary["strike"])
        portfolio.setText(self.COL_ENTRY_PRICE, f"{total:.2f}")
        portfolio.setText(self.COL_EXPIRY, summary["expiry"])
        portfolio.setText(self.COL_UNDERLYING, summary["underlying"])

    def _refresh_portfolio(self, item):
        if not item:
            return
        portfolio = item.parent()
        if portfolio:
            self._update_structure_price(portfolio)

    def _calc_portfolio_price(self, portfolio_item):
        total = 0.0

        for i in range(portfolio_item.childCount()):
            child = portfolio_item.child(i)

            data = child.data(0, Qt.ItemDataRole.UserRole)

            # safety
            if not data or isinstance(data, str):
                continue

            sr = data

            # ✅ ONLY INCLUDE OPTIONS
            if getattr(sr, "instrument_type", "Option") != "Option":
                continue

            try:
                price = float(sr.entry_price)
            except:
                price = 0.0

            sign = 1 if sr.long_short == "Long" else -1
            total += sign * sr.lots * price

        return total

    def _calc_portfolio_summary(self, portfolio_item):
        net_lots = 0

        strikes = set()
        expiries = set()
        underlyings = set()

        for i in range(portfolio_item.childCount()):
            child = portfolio_item.child(i)
            sr = child.data(0, Qt.ItemDataRole.UserRole)

            if not sr or isinstance(sr, str):
                continue

            # ✅ ONLY OPTIONS
            if getattr(sr, "instrument_type", "Option") != "Option":
                continue

            sign = 1 if sr.long_short == "Long" else -1
            net_lots += sign * sr.lots

            if sr.strike:
                strikes.add(str(sr.strike))

            if sr.expiry:
                expiries.add(self._short_expiry(sr.expiry))

            if sr.underlying_mid:
                underlyings.add(f"{sr.underlying_mid/100:.2f}")

        def fmt(values):
            if not values:
                return ""
            if len(values) == 1:
                return list(values)[0]
            return "/".join(sorted(values))

        return {
            "net_lots": net_lots,
            "strike": fmt(strikes),
            "expiry": fmt(expiries),
            "underlying": fmt(underlyings),
        }


    def load_pds_data(self, strikes, expiries, expiry_underlying,
                    future_expiries, future_underlying,
                    zq_future_expiries=None, zq_future_underlying=None,
                    panel4=None):

        self._strike_labels = [_g_format(s) for s in strikes]
        self._expiry_labels = list(expiries)
        self._future_expiry_labels = list(future_expiries)
        self._expiry_underlying = dict(expiry_underlying)
        self._future_underlying_map = dict(future_underlying)
        self._zq_future_expiry_labels = list(zq_future_expiries or [])
        self._zq_future_underlying_map = dict(zq_future_underlying or {})

        if panel4 is not None:
            self._panel4 = panel4

        und_ids = list(set(
            [v for v in expiry_underlying.values() if v] +
            [v for v in future_underlying.values() if v] +
            [v for v in (zq_future_underlying or {}).values() if v]
        ))
        if und_ids and LightstreamerClient is not None:
            try:
                if self._ls_cb:
                    try:
                        LSManager.instance().detach_ui_multi(self._ls_keys, self._ls_cb)
                    except Exception:
                        pass
                self._ls_cb   = self._on_und_update
                self._ls_keys = LSManager.instance().get_or_create_chunked(
                    self._LS_KEY, und_ids, chunk_size=400
                )
                LSManager.instance().attach_ui_multi(self._ls_keys, self._ls_cb)
            except Exception as ex:
                print(f"[Panel6] LS subscribe error: {ex}")

        for portfolio in self.portfolios.values():

            if portfolio:
                for i in range(portfolio.childCount()):
                    item = portfolio.child(i)

                    strike_combo = self.tree.itemWidget(item, self.COL_STRIKE)
                    if strike_combo:
                        strike_combo.blockSignals(True)
                        strike_combo.clear()
                        strike_combo.addItems(self._strike_labels)
                        strike_combo.blockSignals(False)

                    expiry_combo = self.tree.itemWidget(item, self.COL_EXPIRY)
                    sr = item.data(0, Qt.ItemDataRole.UserRole)
                    if expiry_combo:
                        expiry_combo.blockSignals(True)
                        expiry_combo.clear()
                        if sr and sr.instrument_type == "Future":
                            if getattr(sr, "future_product", "SR3") == "ZQ":
                                expiry_combo.addItems(self._zq_future_expiry_labels)
                            else:
                                expiry_combo.addItems(self._future_expiry_labels)
                        else:
                            expiry_combo.addItems(self._expiry_labels)
                        expiry_combo.blockSignals(False)

    def _on_und_update(self, iid: str, row: dict):
        try:
            bid = float(row.get("BestBid") or 0)
            ask = float(row.get("BestAsk") or 0)
        except (ValueError, TypeError):
            return
        if bid <= 0 or ask <= 0:
            return
        mid = (bid + ask) / 2.0
        self._und_mids[iid] = mid

        for pf in self.portfolios.values():

            for i in range(pf.childCount()):

                item = pf.child(i)
                sr = item.data(0, Qt.ItemDataRole.UserRole)

                if sr and sr.underlying_id == iid:
                    sr.underlying_mid = mid
                    self._tree_update_underlying(item)
                    self._tree_auto_price(item)
                    portfolio = item.parent()
                    if portfolio:
                        self._update_structure_price(portfolio)
                        self._refresh_portfolio(item)

    def _on_type_changed(self, row: int, text: str):

        if row >= len(self.rows):
            return

        self.rows[row].instrument_type = text

        self.table.blockSignals(True)
        self._populate_row(row, self.rows[row])
        self.table.blockSignals(False)

    def _refresh_underlying_cell(self, r: int):
        if r < 0 or r >= len(self.rows):
            return
        sr  = self.rows[r]
        raw = sr.underlying_mid
        txt = f"{raw / 100}" if raw is not None else "—"
        item = self.table.item(r, self.COL_UNDERLYING)
        if item is not None:
            item.setText(txt)

    def _add_future_row(self):

        sr = StrategyRow()
        sr.instrument_type = "Future"

        if self._future_expiry_labels:
            sr.expiry = self._future_expiry_labels[0]

        sr.underlying_id = self._future_underlying_map.get(sr.expiry, "")
        sr.underlying_mid = self._und_mids.get(sr.underlying_id)

        self.rows.append(sr)

        r = len(self.rows) - 1
        self.table.insertRow(r)

        self.table.blockSignals(True)
        self._populate_row(r, sr)
        self.table.blockSignals(False)

        self._update_row_count()

    def _build_ui(self):
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(0)

            toolbar = QWidget()
            toolbar.setStyleSheet("background:#FAFAFA; border-bottom:1px solid #eee;")
            tb = QHBoxLayout(toolbar)
            tb.setContentsMargins(6, 4, 6, 4)
            tb.setSpacing(6)

            self.btn_add_pf = QPushButton("+ Portfolio")
            self.btn_add_pf.setFixedWidth(90)
            self.btn_add_pf.clicked.connect(self._add_portfolio)
            tb.addWidget(self.btn_add_pf)

            self.btn_add = QPushButton("+ Add Row")
            self.btn_add.setFixedWidth(80)
            self.btn_add.clicked.connect(self._add_row)
            tb.addWidget(self.btn_add)

            self.btn_del = QPushButton("− Remove")
            self.btn_del.setFixedWidth(75)
            self.btn_del.setEnabled(False)
            self.btn_del.clicked.connect(self._remove_row)
            tb.addWidget(self.btn_del)

            tb.addStretch()

            self.row_count_lbl = QLabel("0 rows")
            self.row_count_lbl.setStyleSheet("color:#888; font-size:11px;")
            tb.addWidget(self.row_count_lbl)

            root.addWidget(toolbar)

            self.tree = QTreeWidget()
            self.tree.setColumnCount(len(self.COLS))
            self.tree.setHeaderLabels(self.COLS)

            # ── Center-align all headers ──────────────────────────────────────
            header = self.tree.header()
            for i in range(len(self.COLS)):
                header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

            # ── Prevent last column from stretching ───────────────────────────
            header.setStretchLastSection(False)
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

            # ── Set sensible default column widths ────────────────────────────
            col_widths = [65, 80, 80, 45, 90, 80, 65, 100, 75]
            for i, w in enumerate(col_widths):
                if i < len(self.COLS):
                    self.tree.setColumnWidth(i, w)

            self.tree.setRootIsDecorated(True)
            self.tree.setAlternatingRowColors(True)
            self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.tree.itemSelectionChanged.connect(self._update_btn_states)
            root.addWidget(self.tree)

            self.tree.setStyleSheet("""
            QTreeWidget {
                background: #FFFFFF;
                border: none;
            }
            QTreeWidget::item:selected{
                background: transparent;
                color: black;
            }
            QTreeWidget::item{
                padding:2px;
            }
            QHeaderView {
                background: #FFFFFF;
            }
            QHeaderView::section {
                text-align: center;
                background: #F5F7FA;
                border: 1px solid #CCCCCC;
                padding: 3px 6px;
                font-size: 11px;
            }
            QTreeWidget QScrollBar:horizontal {
                background: #FFFFFF;
            }
            QTreeWidget QScrollBar:vertical {
                background: #FFFFFF;
            }
            """)
            self.tree.viewport().setStyleSheet("background: #FFFFFF;")
            self.tree.itemChanged.connect(self._on_entry_changed)
            self.tree.setEditTriggers(
                QAbstractItemView.EditTrigger.DoubleClicked
                | QAbstractItemView.EditTrigger.EditKeyPressed
            )
            self.tree.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def _add_portfolio(self):

        name = f"Portfolio {len(self.portfolios)+1}"

        root = QTreeWidgetItem(self.tree)
        root.setText(0, name)
        root.setExpanded(True)

        self.portfolios[name] = root

        root.setData(0, Qt.ItemDataRole.UserRole, {
            "structure_price": "",
            "is_user_set": False
        })

        root.setText(self.COL_ENTRY_PRICE, "")
        root.setFlags(root.flags() | Qt.ItemFlag.ItemIsEditable)

    def _current_portfolio(self):

        if not self.portfolios:
            return self._create_portfolio_auto()

        item = self.tree.currentItem()

        if item and item.parent() is None:
            return item
        elif item and item.parent():
            return item.parent()

        return list(self.portfolios.values())[0]

    def _create_portfolio_auto(self):

        name = f"Portfolio {len(self.portfolios)+1}"

        root = QTreeWidgetItem(self.tree)
        root.setText(0, name)
        root.setExpanded(True)

        root.setData(0, Qt.ItemDataRole.UserRole, {
            "structure_price": "",
            "is_user_set": False
        })

        self.portfolios[name] = root
        return root

    def _on_underlying_update(self, expiry: str, mid: float):
        for r, sr in enumerate(self.rows):
            if sr.expiry == expiry:
                sr.underlying_mid = mid / 100.0   # ⭐ IMPORTANT scaling
                self._update_iv_for_row(r)

    def add_strategy_row(self, cp, long_short, lots,
                        strike, entry_price, expiry, underlying_id=""):

        sr = StrategyRow()
        sr.cp = cp
        sr.long_short = long_short
        sr.lots = lots
        sr.strike = strike
        sr.entry_price = entry_price
        sr.expiry = expiry
        sr.underlying_id = underlying_id

        # ✅ set underlying mid (only once)
        sr.underlying_mid = self._und_mids.get(underlying_id)

        # ✅ append first (source of truth)
        self.rows.append(sr)

        # ✅ determine portfolio
        item_sel = self.tree.currentItem()

        if item_sel and item_sel.parent() is None:
            portfolio = item_sel
        elif item_sel and item_sel.parent():
            portfolio = item_sel.parent()
        else:
            portfolio = self._current_portfolio()

        # ✅ create tree item
        item = QTreeWidgetItem(portfolio)
        item.setData(0, Qt.ItemDataRole.UserRole, sr)   # ⭐ IMPORTANT (link row ↔ UI)

        self._populate_tree_row(item, sr)

        # ✅ NOW row index exists
        r = len(self.rows) - 1

        # ✅ compute IV AFTER everything is ready
        self._update_iv_for_row(r)

        portfolio.setExpanded(True)
        self._tree_auto_price(item)
        self._update_row_count()
        self._update_structure_price(portfolio)

    def _add_row(self):

        sr = StrategyRow()

        if self._strike_labels:
            sr.strike = self._strike_labels[0]

        if self._expiry_labels:
            sr.expiry = self._expiry_labels[0]

        sr.underlying_id = self._expiry_underlying.get(sr.expiry, "")
        sr.underlying_mid = self._und_mids.get(sr.underlying_id)

        self.rows.append(sr)

        item_sel = self.tree.currentItem()

        if item_sel and item_sel.parent() is None:
            portfolio = item_sel
        elif item_sel and item_sel.parent():
            portfolio = item_sel.parent()
        else:
            portfolio = self._current_portfolio()

        item = QTreeWidgetItem()

        # ✅ FIX 1: correct insertion
        portfolio.addChild(item)

        self._populate_tree_row(item, sr)

        portfolio.setExpanded(True)
        self._tree_auto_price(item)

        self._update_row_count()

        # ✅ FIX 2: THIS IS THE MAIN MISSING LINE
        self._update_structure_price(portfolio)


    def _populate_tree_row(self, item, sr):
        item.setData(0, Qt.ItemDataRole.UserRole, sr)

        item.setText(self.COL_TYPE, sr.instrument_type)
        item.setText(self.COL_CP, sr.cp)
        item.setText(self.COL_LS, sr.long_short)
        item.setText(self.COL_LOTS, str(sr.lots))
        item.setText(self.COL_STRIKE, sr.strike)
        item.setText(self.COL_ENTRY_PRICE, sr.entry_price)
        item.setText(self.COL_IV, "")

        entry_item = item
        entry_item.setFlags(
            entry_item.flags()
            | Qt.ItemFlag.ItemIsEditable
        )
        item.setText(self.COL_EXPIRY, sr.expiry)

        raw = sr.underlying_mid
        und_txt = f"{raw/100:.2f}" if raw else "—"
        item.setText(self.COL_UNDERLYING, und_txt)

        # ⭐ TYPE COMBO
        type_combo = QComboBox()
        type_combo.setView(QListView())
        type_combo.setMaxVisibleItems(5)
        type_combo.setStyleSheet("QComboBox { combobox-popup: 1; }")
        type_combo.addItems(["Option","Future"])
        type_combo.setCurrentText(sr.instrument_type)
        type_combo.currentTextChanged.connect(
            lambda text, it=item: self._tree_type_changed(it, text)
        )
        self.tree.setItemWidget(item, self.COL_TYPE, type_combo)

        # ⭐ FUTURE PRODUCT COMBO (SR3 / ZQ) — visible only for Future rows
        prod_combo = QComboBox()
        prod_combo.addItems(["SR3", "ZQ"])
        prod_combo.setCurrentText(getattr(sr, "future_product", "SR3"))
        prod_combo.setVisible(sr.instrument_type == "Future")
        prod_combo.currentTextChanged.connect(
            lambda text, it=item: self._tree_future_product_changed(it, text)
        )
        self.tree.setItemWidget(item, self.COL_CP, prod_combo)

        cp_combo = QComboBox()
        cp_combo.setView(QListView())
        cp_combo.setMaxVisibleItems(5)
        cp_combo.setStyleSheet("QComboBox { combobox-popup: 1; }")
        cp_combo.addItems(["Call","Put"])
        cp_combo.setCurrentText(sr.cp)
        cp_combo.currentTextChanged.connect(
            lambda text, it=item: self._tree_cp_changed(it, text)
        )
        if sr.instrument_type == "Option":
            self.tree.setItemWidget(item, self.COL_CP, cp_combo)

        ls_combo = QComboBox()
        ls_combo.setView(QListView())
        ls_combo.setMaxVisibleItems(5)
        ls_combo.setStyleSheet("QComboBox { combobox-popup: 1; }")
        ls_combo.addItems(["Long","Short"])
        ls_combo.setCurrentText(sr.long_short)
        ls_combo.currentTextChanged.connect(
            lambda text, it=item: self._tree_ls_changed(it, text)
        )
        self.tree.setItemWidget(item, self.COL_LS, ls_combo)

        strike_combo = QComboBox()
        strike_combo.setView(QListView())
        strike_combo.setMaxVisibleItems(5)
        strike_combo.setStyleSheet("QComboBox { combobox-popup: 1; }")
        strike_combo.addItems(self._strike_labels)
        strike_combo.setCurrentText(sr.strike)
        strike_combo.currentTextChanged.connect(
            lambda text, it=item: self._tree_strike_changed(it, text)
        )
        self.tree.setItemWidget(item, self.COL_STRIKE, strike_combo)

        expiry_combo = QComboBox()
        expiry_combo.setView(QListView())
        expiry_combo.setMaxVisibleItems(5)
        expiry_combo.setStyleSheet("QComboBox { combobox-popup: 1; }")

        if sr.instrument_type == "Future":
            if getattr(sr, "future_product", "SR3") == "ZQ":
                expiry_combo.addItems(self._zq_future_expiry_labels)
            else:
                expiry_combo.addItems(self._future_expiry_labels)
        else:
            expiry_combo.addItems(self._expiry_labels)

        expiry_combo.setCurrentText(sr.expiry)

        expiry_combo.currentTextChanged.connect(
            lambda text, it=item: self._tree_expiry_changed(it, text)
        )

        self.tree.setItemWidget(item, self.COL_EXPIRY, expiry_combo)

    def _renumber_portfolios(self):

        new_map = {}

        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)

            new_name = f"Portfolio {i+1}"
            item.setText(0, new_name)

            new_map[new_name] = item

        self.portfolios = new_map

    def _remove_row(self):

        item = self.tree.currentItem()
        if not item:
            return

        if item.parent() is None:
            # deleting portfolio
            name = item.text(0)
            index = self.tree.indexOfTopLevelItem(item)
            self.tree.takeTopLevelItem(index)

            if name in self.portfolios:
                del self.portfolios[name]

            # ✅ renumber portfolios
            self._renumber_portfolios()
            return

        # deleting child row
        portfolio = item.parent()
        portfolio.removeChild(item)

        self._update_row_count()

        # ✅ update summary after delete
        self._update_structure_price(portfolio)

    def _tree_type_changed(self, item, text):

        sr = item.data(0, Qt.ItemDataRole.UserRole)
        if not sr:
            return

        sr.instrument_type = text

        cp_combo = self.tree.itemWidget(item, self.COL_CP)
        strike_combo = self.tree.itemWidget(item, self.COL_STRIKE)
        expiry_combo = self.tree.itemWidget(item, self.COL_EXPIRY)

        prod_combo   = self.tree.itemWidget(item, self.COL_CP)

        if text == "Future":

            if not isinstance(prod_combo, QComboBox) or \
               set(prod_combo.itemText(i) for i in range(prod_combo.count())) != {"SR3","ZQ"}:
                prod_combo = QComboBox()
                prod_combo.addItems(["SR3", "ZQ"])
                prod_combo.setCurrentText(getattr(sr, "future_product", "SR3"))
                prod_combo.currentTextChanged.connect(
                    lambda t, it=item: self._tree_future_product_changed(it, t)
                )
                self.tree.setItemWidget(item, self.COL_CP, prod_combo)

            if strike_combo:
                strike_combo.setEnabled(False)

            fut_product = getattr(sr, "future_product", "SR3")
            expiry_list = self._zq_future_expiry_labels if fut_product == "ZQ" \
                                       else self._future_expiry_labels

            if expiry_combo:
                expiry_combo.blockSignals(True)
                expiry_combo.clear()
                expiry_combo.addItems(expiry_list)
                if expiry_list:
                    sr.expiry = expiry_list[0]
                    expiry_combo.setCurrentText(sr.expiry)
                else:
                    sr.expiry = ""
                expiry_combo.blockSignals(False)

            und_map = self._zq_future_underlying_map if fut_product == "ZQ" \
                       else self._future_underlying_map
            sr.underlying_id = und_map.get(sr.expiry, "")

        else:

            cp_combo2 = QComboBox()
            cp_combo.setView(QListView())
            cp_combo.setMaxVisibleItems(5)
            cp_combo.setStyleSheet("QComboBox { combobox-popup: 1; }")              
            cp_combo2.addItems(["Call", "Put"])
            cp_combo2.setCurrentText(sr.cp)
            cp_combo2.currentTextChanged.connect(
                lambda t, it=item: self._tree_cp_changed(it, t)
            )
            self.tree.setItemWidget(item, self.COL_CP, cp_combo2)

            if strike_combo:
                strike_combo.setEnabled(True)

            if expiry_combo:
                expiry_combo.blockSignals(True)
                expiry_combo.clear()
                expiry_combo.addItems(self._expiry_labels)
                if self._expiry_labels:
                    sr.expiry = self._expiry_labels[0]
                    expiry_combo.setCurrentText(sr.expiry)
                else:
                    sr.expiry = ""
                expiry_combo.blockSignals(False)

            sr.underlying_id = self._expiry_underlying.get(sr.expiry, "")

        sr.underlying_mid = self._und_mids.get(sr.underlying_id)

        self._tree_update_underlying(item)
        self._tree_auto_price(item)
        portfolio = item.parent()
        if portfolio:
            self._update_structure_price(portfolio)

    def _tree_cp_changed(self, item, text):
        sr = item.data(0, Qt.ItemDataRole.UserRole)
        sr.cp = text
        self._tree_auto_price(item)
        self._tree_update_underlying(item)
        r = self.rows.index(sr) if sr in self.rows else -1
        self._update_iv_for_row(r)
        portfolio = item.parent()
        if portfolio:
            self._update_structure_price(portfolio)
            self._refresh_portfolio(item)

    def _tree_ls_changed(self, item, text):
        sr = item.data(0, Qt.ItemDataRole.UserRole)
        sr.long_short = text
        self._tree_auto_price(item)
        self._tree_update_underlying(item)
        r = self.rows.index(sr) if sr in self.rows else -1
        self._update_iv_for_row(r)
        portfolio = item.parent()
        if portfolio:
            self._update_structure_price(portfolio)
            self._refresh_portfolio(item)

    def _tree_strike_changed(self, item, text):
        sr = item.data(0, Qt.ItemDataRole.UserRole)
        sr.strike = text
        self._tree_auto_price(item)
        self._tree_update_underlying(item)
        r = self.rows.index(sr) if sr in self.rows else -1
        self._update_iv_for_row(r)
        portfolio = item.parent()
        if portfolio:
            self._update_structure_price(portfolio)
        self._refresh_portfolio(item)

    def _tree_expiry_changed(self, item, text):
        sr = item.data(0, Qt.ItemDataRole.UserRole)
        sr.expiry = text
        if sr.instrument_type == "Future":
            if getattr(sr, "future_product", "SR3") == "ZQ":
                sr.underlying_id = self._zq_future_underlying_map.get(text, "")
            else:
                sr.underlying_id = self._future_underlying_map.get(text, "")
        else:
            sr.underlying_id = self._expiry_underlying.get(text, "")
        sr.underlying_mid = self._und_mids.get(sr.underlying_id)
        self._tree_update_underlying(item)
        self._tree_auto_price(item)
        r = self.rows.index(sr) if sr in self.rows else -1
        self._update_iv_for_row(r)
        portfolio = item.parent()
        if portfolio:
            self._update_structure_price(portfolio)
        self._refresh_portfolio(item)

    def _tree_future_product_changed(self, item, text):
        sr = item.data(0, Qt.ItemDataRole.UserRole)
        if not sr:
            return
        sr.future_product = text
        expiry_combo = self.tree.itemWidget(item, self.COL_EXPIRY)
        if expiry_combo:
            expiry_list = self._zq_future_expiry_labels if text == "ZQ" \
                                       else self._future_expiry_labels
            expiry_combo.blockSignals(True)
            expiry_combo.clear()
            expiry_combo.addItems(expiry_list)
            if expiry_list:
                sr.expiry = expiry_list[0]
                expiry_combo.setCurrentText(sr.expiry)
            else:
                sr.expiry = ""
            expiry_combo.blockSignals(False)
        und_map = self._zq_future_underlying_map if text == "ZQ" \
                   else self._future_underlying_map
        sr.underlying_id = und_map.get(sr.expiry, "")
        sr.underlying_mid = self._und_mids.get(sr.underlying_id)
        self._tree_update_underlying(item)
        self._tree_auto_price(item)

    def _tree_auto_price(self, item):

        sr = item.data(0, Qt.ItemDataRole.UserRole)

        if not sr:
            return

        if not sr.strike or not sr.expiry:
            return

        current_text = item.text(self.COL_ENTRY_PRICE)
        if current_text and getattr(sr, "_user_override", False):
            return

        if sr.instrument_type == "Future":

            if sr.underlying_mid:
                sr.entry_price = f"{sr.underlying_mid/100:.2f}"
                item.setText(self.COL_ENTRY_PRICE, sr.entry_price)

            self._tree_update_underlying(item)
            return

        mid = self._lookup_option_mid(sr.strike, sr.expiry, sr.cp)

        if mid is not None:
            sr.entry_price = f"{mid:.2f}"
            item.setText(self.COL_ENTRY_PRICE, sr.entry_price)

        self._tree_update_underlying(item)
        portfolio = item.parent()
        if portfolio:
            self._update_structure_price(portfolio)

    def _tree_update_underlying(self, item):

        sr = item.data(0, Qt.ItemDataRole.UserRole)

        raw = sr.underlying_mid

        txt = f"{raw/100:.2f}" if raw else "—"

        item.setText(self.COL_UNDERLYING, txt)

    def _populate_row(self, r: int, sr: StrategyRow, highlight: bool = False):

        bg = QBrush(QColor("#E8F4FD")) if highlight else QBrush(QColor("#FFFFFF"))

        type_combo = QComboBox()
        type_combo.setView(QListView())
        type_combo.setStyleSheet("QComboBox { combobox-popup: 1; font-size:11px; }")
        type_combo.addItems(["Option", "Future"])
        type_combo.setCurrentText(sr.instrument_type)
        type_combo.currentTextChanged.connect(
            lambda text, _r=r: self._on_type_changed(_r, text)
        )
        self.table.setCellWidget(r, self.COL_TYPE, type_combo)

        cp_combo = QComboBox()
        cp_combo.setView(QListView())
        cp_combo.addItems(["Call", "Put"])
        cp_combo.setCurrentText(sr.cp)
        cp_combo.setStyleSheet("QComboBox { combobox-popup: 1; font-size:11px; }")
        cp_combo.currentTextChanged.connect(
            lambda text, _r=r: self._on_cp_changed(_r, text)
        )
        self.table.setCellWidget(r, self.COL_CP, cp_combo)

        ls_combo = QComboBox()
        ls_combo.setView(QListView())
        ls_combo.addItems(["Long", "Short"])
        ls_combo.setCurrentText(sr.long_short)
        ls_combo.setStyleSheet("QComboBox { combobox-popup: 1; font-size:11px; }")
        ls_combo.currentTextChanged.connect(
            lambda text, _r=r: self._on_ls_changed(_r, text)
        )
        self.table.setCellWidget(r, self.COL_LS, ls_combo)

        lots_item = QTableWidgetItem(str(sr.lots))
        lots_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        lots_item.setFlags(_FLAG_EDITABLE)
        lots_item.setBackground(bg)
        self.table.setItem(r, self.COL_LOTS, lots_item)

        strike_combo = QComboBox()
        strike_combo.setView(QListView())
        strike_combo.setMaxVisibleItems(12)
        strike_combo.setStyleSheet("QComboBox { combobox-popup: 1; font-size:11px; }")

        if self._strike_labels:
            strike_combo.addItems(self._strike_labels)
            if sr.strike in self._strike_labels:
                strike_combo.setCurrentText(sr.strike)
            else:
                strike_combo.setCurrentIndex(0)
                sr.strike = strike_combo.currentText()

        strike_combo.currentTextChanged.connect(
            lambda text, _r=r: self._on_strike_changed(_r, text)
        )
        self.table.setCellWidget(r, self.COL_STRIKE, strike_combo)

        if sr.instrument_type == "Future":
            cp_combo.setEnabled(False)
            strike_combo.setEnabled(False)

        entry_item = QTableWidgetItem(sr.entry_price)
        entry_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        entry_item.setFlags(_FLAG_EDITABLE)
        entry_item.setBackground(bg)
        self.table.setItem(r, self.COL_ENTRY_PRICE, entry_item)

        iv_item = QTableWidgetItem("")
        iv_item.setFlags(self._NO_EDIT)
        iv_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(r, self.COL_IV, iv_item)

        expiry_combo = QComboBox()
        expiry_combo.setView(QListView())
        expiry_combo.setMaxVisibleItems(12)
        expiry_combo.setStyleSheet("QComboBox { combobox-popup: 1; font-size:11px; }")

        if self._expiry_labels:
            expiry_combo.addItems(self._expiry_labels)
            if sr.expiry in self._expiry_labels:
                expiry_combo.setCurrentText(sr.expiry)
            else:
                expiry_combo.setCurrentIndex(0)
                sr.expiry = expiry_combo.currentText()

        expiry_combo.currentTextChanged.connect(
            lambda text, _r=r: self._on_expiry_changed(_r, text)
        )
        self.table.setCellWidget(r, self.COL_EXPIRY, expiry_combo)

        raw = sr.underlying_mid
        und_txt = f"{raw/100:.2f}" if raw else "—"

        und_item = QTableWidgetItem(und_txt)
        und_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        und_item.setFlags(self._NO_EDIT)
        und_item.setBackground(QBrush(QColor("#F5F7FA")))
        und_item.setForeground(QBrush(QColor("#444444")))
        self.table.setItem(r, self.COL_UNDERLYING, und_item)

    def _refresh_combos(self, r: int):
        sr = self.rows[r]

        sc = self.table.cellWidget(r, self.COL_STRIKE)
        if isinstance(sc, QComboBox):
            sc.blockSignals(True)
            sc.clear()
            sc.addItems(self._strike_labels)
            if sr.strike in self._strike_labels:
                sc.setCurrentText(sr.strike)
            sc.blockSignals(False)

        ec = self.table.cellWidget(r, self.COL_EXPIRY)
        if isinstance(ec, QComboBox):
            ec.blockSignals(True)
            ec.clear()
            ec.addItems(self._expiry_labels)
            if sr.expiry in self._expiry_labels:
                ec.setCurrentText(sr.expiry)
            ec.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem):
        r   = item.row()
        col = item.column()
        if r < 0 or r >= len(self.rows):
            return
        txt = item.text().strip()

        if col == self.COL_ENTRY_PRICE:
            try:
                self.rows[r].entry_price = float(item.text())
            except:
                pass

            self._update_iv_for_row(r)
        elif col == self.COL_ENTRY_PRICE:
            self.rows[r].entry_price = txt

    def _on_cp_changed(self, row: int, text: str):
        if row < len(self.rows):
            self.rows[row].cp = text
            self._auto_fill_prices(row)

    def _on_ls_changed(self, row: int, text: str):
        if row < len(self.rows):
            self.rows[row].long_short = text

    def _on_strike_changed(self, row: int, text: str):
        if row >= len(self.rows):
            return
        self.rows[row].strike = text
        self._auto_fill_prices(row)

    def _on_expiry_changed(self, row: int, text: str):
        if row >= len(self.rows):
            return
        self.rows[row].expiry = text
        new_und_id = self._expiry_underlying.get(text, "")
        self.rows[row].underlying_id  = new_und_id
        self.rows[row].underlying_mid = self._und_mids.get(new_und_id)
        self._refresh_underlying_cell(row)
        self._auto_fill_prices(row)

    def _on_entry_changed(self, item, col):

        # ✅ PORTFOLIO ROW EDIT
        if item.parent() is None and col == self.COL_ENTRY_PRICE:

            data = item.data(0, Qt.ItemDataRole.UserRole)

            try:
                val = float(item.text(col))
            except:
                val = 0.0

            data["structure_price"] = val
            data["is_user_set"] = True
            return


        if col == self.COL_ENTRY_PRICE:
            r = self.tree.indexOfTopLevelItem(item.parent()) if item.parent() else -1

            try:
                sr = item.data(0, Qt.ItemDataRole.UserRole)
                if sr:
                    sr.entry_price = item.text(col)
            except:
                pass

            self._update_iv_for_row(r)

            sr = item.data(0, Qt.ItemDataRole.UserRole)

            if not sr:
                return

            sr.entry_price = item.text(col)

            portfolio = item.parent()
            if portfolio:
                self._update_structure_price(portfolio) 
                self._tree_auto_price(item)
                self._refresh_portfolio(item)              

    def _auto_fill_prices(self, row: int):
        if row >= len(self.rows):
            return
        sr = self.rows[row]

        if sr.instrument_type == "Future":

            if sr.underlying_mid is not None:

                entry_str = f"{sr.underlying_mid/100:.2f}"
                sr.entry_price = entry_str

                item = self.table.item(row, self.COL_ENTRY_PRICE)
                if item:
                    self.table.blockSignals(True)
                    item.setText(entry_str)
                    self.table.blockSignals(False)

            self._refresh_underlying_cell(row)
            self._update_iv_for_row(row)
            return

        mid = self._lookup_option_mid(sr.strike, sr.expiry, sr.cp)
        if mid is not None:
            entry_str = f"{mid:.2f}"
            sr.entry_price = entry_str
            item = self.table.item(row, self.COL_ENTRY_PRICE)
            if item is not None:
                self.table.blockSignals(True)
                item.setText(entry_str)
                self.table.blockSignals(False)

        self._refresh_underlying_cell(row)

    def _lookup_option_mid(self, strike_str: str, expiry_iso: str, cp: str) -> float | None:
        if self._panel4 is None:
            return None
        mids = self._panel4._mids
        if not mids:
            return None

        cp_code = "C" if cp == "Call" else "P"
        try:
            strike_f = float(strike_str)
        except (ValueError, TypeError):
            return None

        key = (expiry_iso, round(strike_f, 6), cp_code)
        v = mids.get(key)
        if v is not None:
            return v

        for (ke, ks, kc), mv in mids.items():
            if ke == expiry_iso and kc == cp_code and abs(ks - strike_f) < 1e-4:
                return mv
        return None

    def _update_btn_states(self):

        item = self.tree.currentItem()

        self.btn_del.setEnabled(item is not None)

    def _update_row_count(self):

        item_sel = self.tree.currentItem()

        if item_sel and item_sel.parent() is None:
            portfolio = item_sel
        elif item_sel and item_sel.parent():
            portfolio = item_sel.parent()
        else:
            portfolio = self._current_portfolio()
        n = portfolio.childCount()

        self.row_count_lbl.setText(f"{n} rows")

    def closeEvent(self, e):
        if self._ls_cb and self._ls_keys:
            try:
                LSManager.instance().detach_ui_multi(self._ls_keys, self._ls_cb)
            except Exception:
                pass
        super().closeEvent(e)


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SR3 Scenario Analysis")
        self.setWindowIcon(QIcon("FF_logo.png"))
        self.resize(1700, 950)

        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks |
            QMainWindow.DockOption.AllowTabbedDocks |
            QMainWindow.DockOption.AnimatedDocks
        )

        self.rows: list[MeetingRow] = []
        self.api  = SR3ApiClient()
        self.fed  = FedPremiumsClient()
        self.case_count = DEFAULT_CASES
        self.base_view  = "O/R"
        # Which case should drive the single displayed OIS premium cell.
        # Set whenever the user edits a Case column in the meeting table.
        self._active_ois_case_index: int | None = None
        # Pricing-output '2Y Yield' (column index 1) starts empty.
        # Only the edited case row(s) are marked dirty and allowed to populate.
        self._case_2y_yield_dirty: list[bool] = [False] * MAX_CASES

        self.debounce = QTimer()
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(900)
        self.debounce.timeout.connect(self._on_debounce)

        self.calc_task: asyncio.Task | None = None
        self.calc_lock          = asyncio.Lock()
        self.last_payload_hash  = None
        self.next_allowed_time  = 0
        self.last_result: dict | None = None

        self._columns_dirty = True

        self._build_ui()
        self._build_meeting_columns()

        self._pds_thread = PDSLoaderThread()
        self._pds_thread.pds_ready.connect(self._on_pds_ready)
        self._pds_thread.finished.connect(self._pds_thread.deleteLater)
        self._pds_thread.start()

        QTimer.singleShot(0, lambda: asyncio.ensure_future(self._load_defaults()))

    def _on_pds_ready(self, df):
        self.panel3.load_sr3_data(df)
        self.panel4.load_sr3_data(df)
        self.panel6.load_pds_data(
            self.panel4._strikes,
            self.panel4._expiries,
            self.panel4._expiry_underlying,
            self.panel4._future_expiries,
            self.panel4._future_underlying,
            self.panel4._zq_future_expiries,
            self.panel4._zq_future_underlying,
            panel4=self.panel4,
        )

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        _cw = QWidget()
        _cw.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        _cw.setMaximumSize(0, 0)
        _cw.setMinimumSize(0, 0)
        _cw.hide()
        self.setCentralWidget(_cw)

        self.setStyleSheet("""
            QDockWidget {
                font-weight: 600;
                font-size: 12px;
                color: #333;
            }
            QDockWidget::title {
                background: #F5F7FA;
                padding: 4px 8px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }
        """)

        # ── Panel 1: Meeting Inputs ───────────────────────────────────────────
        meeting_container = QWidget()
        meeting_container.setStyleSheet("background:#FAFAFA;")
        meeting_vbox = QVBoxLayout(meeting_container)
        meeting_vbox.setContentsMargins(0, 0, 0, 0)
        meeting_vbox.setSpacing(0)

        ctrl_bar = QWidget()
        ctrl_bar.setAutoFillBackground(True)
        ctrl_bar.setStyleSheet("QWidget { background:#F0F2F5; }")
        ctrl_bar.setFixedHeight(34)
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(8, 3, 8, 3)
        ctrl_layout.setSpacing(6)

        ctrl_layout.addWidget(QLabel("<b>Fixing:</b>"))
        self.fixing_input = QLineEdit("3.64")
        self.fixing_input.setFixedWidth(80)
        self.fixing_input.textChanged.connect(self._on_fixing_changed)
        ctrl_layout.addWidget(self.fixing_input)

        btn_add_row = QPushButton("Add Row")
        btn_add_row.setFixedWidth(80)
        btn_add_row.clicked.connect(self._add_row)
        ctrl_layout.addWidget(btn_add_row)

        btn_del_row = QPushButton("Delete Row")
        btn_del_row.setFixedWidth(90)
        btn_del_row.clicked.connect(self._delete_row)
        ctrl_layout.addWidget(btn_del_row)

        sep_widget = QFrame()
        sep_widget.setFrameShape(QFrame.Shape.VLine)
        sep_widget.setStyleSheet("color:#CCCCCC;")
        sep_widget.setFixedWidth(8)
        ctrl_layout.addWidget(sep_widget)

        self.case_count_label = QLabel(f"Cases: {self.case_count}")
        self.case_count_label.setStyleSheet("color:#555;")
        ctrl_layout.addWidget(self.case_count_label)

        self.btn_add_case = QPushButton("+ Case")
        self.btn_add_case.setFixedWidth(65)
        self.btn_add_case.clicked.connect(self._add_case)
        ctrl_layout.addWidget(self.btn_add_case)

        self.btn_remove_case = QPushButton("- Case")
        self.btn_remove_case.setFixedWidth(65)
        self.btn_remove_case.clicked.connect(self._remove_case)
        ctrl_layout.addWidget(self.btn_remove_case)

        ctrl_layout.addStretch()
        meeting_vbox.addWidget(ctrl_bar)

        h_line = QFrame()
        h_line.setFrameShape(QFrame.Shape.HLine)
        h_line.setFixedHeight(1)
        h_line.setStyleSheet("background:#CCCCCC; border:none;")
        meeting_vbox.addWidget(h_line)

        self.meeting_table = QTableWidget()
        self.meeting_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.SelectedClicked |
            QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.meeting_table.itemChanged.connect(self._on_meeting_item_changed)
        _style_table(self.meeting_table)
        meeting_vbox.addWidget(self.meeting_table)

        dock1 = QDockWidget("Meeting Inputs", self)
        dock1.setObjectName("dock_meeting_inputs")
        dock1.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock1.setWidget(meeting_container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock1)

        # ── Panel 2: Pricing Output ───────────────────────────────────────────
        pricing_container = QWidget()
        pricing_layout = QVBoxLayout(pricing_container)
        pricing_layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("QTabWidget::pane { border:none; }")

        self.sr3_table = self._make_output_table()
        self.sr1_table = self._make_output_table()
        self.zq_table  = self._make_output_table()

        self.tab_widget.addTab(self.sr3_table, "SR3")
        self.tab_widget.addTab(self.sr1_table, "SR1")
        self.tab_widget.addTab(self.zq_table,  "ZQ")
        pricing_layout.addWidget(self.tab_widget)

        dock2 = QDockWidget("Pricing Output", self)
        dock2.setObjectName("dock_pricing_output")
        dock2.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock2.setWidget(pricing_container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock2)

        # ── Panel 3: SR3 Option Legs ──────────────────────────────────────────
        self.panel3 = Panel3(self)

        dock3 = QDockWidget("Generic Option Legs", self)
        dock3.setObjectName("dock_sr3_legs")
        dock3.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock3.setWidget(self.panel3)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock3)

        # ── Panel 4: SR3 Mid-Price Matrix ─────────────────────────────────────
        self.panel4 = Panel4(self)

        dock4 = QDockWidget("Mid-Price Matrix (Live)", self)
        dock4.setObjectName("dock_sr3_matrix")
        dock4.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock4.setWidget(self.panel4)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock4)

        # ── Panel 5: Empty placeholder ────────────────────────────────────────
        self.panel5 = Panel5(self)

        dock5 = QDockWidget("Payoff Chart", self)
        dock5.setObjectName("dock_panel5")
        dock5.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock5.setWidget(self.panel5)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock5)

        # ── Panel 6: SR3 Options Strategy ─────────────────────────────────────
        self.panel6 = Panel6(self)
        # wire Panel5 to its data sources
        self.panel5.set_panels(self.panel4, self.panel6)

        dock6 = QDockWidget("Options Strategy", self)
        dock6.setObjectName("dock_sr3_strategy")
        dock6.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock6.setWidget(self.panel6)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock6)

        # ── Dock layout ────────────────────────────────────────────────────────
        self.splitDockWidget(dock1, dock3, Qt.Orientation.Horizontal)
        self.splitDockWidget(dock3, dock5, Qt.Orientation.Horizontal)
        self.splitDockWidget(dock1, dock2, Qt.Orientation.Vertical)
        self.splitDockWidget(dock3, dock4, Qt.Orientation.Vertical)
        self.splitDockWidget(dock5, dock6, Qt.Orientation.Vertical)

        self._dock1 = dock1
        self._dock2 = dock2
        self._dock3 = dock3
        self._dock4 = dock4
        self._dock5 = dock5
        self._dock6 = dock6

        # ── View menu ─────────────────────────────────────────────────────────
        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("Select Widgets")
        view_menu.addAction(dock1.toggleViewAction())
        view_menu.addAction(dock2.toggleViewAction())
        view_menu.addAction(dock3.toggleViewAction())
        view_menu.addAction(dock4.toggleViewAction())
        view_menu.addAction(dock5.toggleViewAction())
        view_menu.addAction(dock6.toggleViewAction())

        # ── Wire signals ──────────────────────────────────────────────────────
        self.panel3.legs_changed.connect(self.panel4.on_legs_changed)
        self.panel4.send_to_strategy.connect(self.panel6.add_strategy_row)

    def showEvent(self, event):
        super().showEvent(event)
        if hasattr(self, "_dock1"):
            third_w = self.width()  // 3
            half_h  = self.height() // 2

            self.resizeDocks(
                [self._dock1, self._dock3, self._dock5],
                [third_w, third_w, third_w],
                Qt.Orientation.Horizontal
            )
            self.resizeDocks(
                [self._dock1, self._dock2], [half_h, half_h],
                Qt.Orientation.Vertical
            )
            self.resizeDocks(
                [self._dock3, self._dock4], [half_h, half_h],
                Qt.Orientation.Vertical
            )
            self.resizeDocks(
                [self._dock5, self._dock6], [half_h, half_h],
                Qt.Orientation.Vertical
            )

    def _make_output_table(self) -> QTableWidget:
        tbl = QTableWidget()
        tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        _style_table(tbl)
        return tbl

    # ── Meeting table columns ─────────────────────────────────────────────────
    def _build_meeting_columns(self):
        self.meeting_table.blockSignals(True)

        col_count = 2 + self.case_count
        self.meeting_table.setColumnCount(col_count)
        headers = ["Meeting Date", "Premium"] + [f"Case {i+1}" for i in range(self.case_count)]
        self.meeting_table.setHorizontalHeaderLabels(headers)

        self.meeting_table.setColumnWidth(0, 110)
        self.meeting_table.setColumnWidth(1, 80)
        for i in range(self.case_count):
            self.meeting_table.setColumnWidth(2 + i, 70)

        self.meeting_table.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            self._set_meeting_row_cells(r, row)

        self._update_case_button_states()
        self._columns_dirty = False

        self.meeting_table.blockSignals(False)
        self._calc_now()

    def _add_row(self):
        new_row = MeetingRow()
        new_row.is_user_added = True
        idx = self.meeting_table.currentRow()
        if 0 <= idx < len(self.rows) - 1:
            self.rows.insert(idx + 1, new_row)
        else:
            self.rows.append(new_row)
        self._build_meeting_columns()
        new_idx = self.rows.index(new_row)
        self.meeting_table.selectRow(new_idx)
        self.meeting_table.scrollToItem(self.meeting_table.item(new_idx, 0))

    def _refresh_meeting_row(self, r: int):
        if r < 0 or r >= len(self.rows):
            return
        self.meeting_table.blockSignals(True)
        self._set_meeting_row_cells(r, self.rows[r])
        self.meeting_table.blockSignals(False)

    def _set_meeting_row_cells(self, r: int, row: MeetingRow):
        tbl       = self.meeting_table
        bold_font = QFont()
        bold_font.setBold(True)

        # ── OIS row special treatment ─────────────────────────────────────────
        if row.is_ois_row:
            # Date cell — editable so trader can override
            date_item = QTableWidgetItem(row.date)
            date_item.setFont(bold_font)
            date_item.setFlags(_FLAG_EDITABLE)
            date_item.setBackground(_BRUSH_OIS_BG)
            date_item.setForeground(_BRUSH_OIS_FG)
            date_item.setToolTip(
                "Last compounding date for the 2Y OIS.\n"
                "Default = end of quarter ~21 months from today.\n"
                "Edit to override."
            )
            tbl.setItem(r, 0, date_item)

            # Premium cell — read-only, shows computed 2Y OIS rate
            ois_val = self._compute_ois_premium()
            prem_item = QTableWidgetItem(ois_val)
            prem_item.setFont(bold_font)
            prem_item.setFlags(_FLAG_LOCKED)
            prem_item.setBackground(_BRUSH_OIS_BG)
            prem_item.setForeground(_BRUSH_OIS_FG)
            prem_item.setToolTip("Computed 2-Year OIS rate (daily-compounded, Act/360)")
            tbl.setItem(r, 1, prem_item)

            # Case columns — locked/empty for OIS row
            for i in range(self.case_count):
                cell = QTableWidgetItem("")
                cell.setFlags(_FLAG_EDITABLE)
                cell.setBackground(_BRUSH_OIS_BG)
                tbl.setItem(r, 2 + i, cell)
            return

        # ── Normal / user-added rows ──────────────────────────────────────────
        date_item = QTableWidgetItem(row.date)
        date_item.setFont(bold_font)
        if not row.is_user_added:
            date_item.setFlags(_FLAG_LOCKED)
            date_item.setBackground(_BRUSH_LOCKED_BG)
            date_item.setForeground(_BRUSH_LOCKED_FG)
        tbl.setItem(r, 0, date_item)

        prem_item = QTableWidgetItem(row.premium_display)
        prem_item.setFont(bold_font)
        if not row.is_user_added:
            prem_item.setFlags(_FLAG_LOCKED)
            prem_item.setBackground(_BRUSH_LOCKED_BG)
            prem_item.setForeground(_BRUSH_LOCKED_FG)
        tbl.setItem(r, 1, prem_item)

        for i in range(self.case_count):
            tbl.setItem(r, 2 + i, QTableWidgetItem(row.get_case(i)))

        if row.is_user_added:
            col_count = tbl.columnCount()
            for c in range(col_count):
                item = tbl.item(r, c)
                if item:
                    item.setBackground(_BRUSH_USER_ROW_BG)

    # ── 2Y OIS calculation ────────────────────────────────────────────────────
    def _compute_ois_premium(self) -> str:
        """Recompute the 2Y OIS rate from current meeting rows and fixing."""
        # Find the OIS row to get its date
        ois_row = next((r for r in self.rows if r.is_ois_row), None)
        if ois_row is None:
            return ""
        try:
            fixing = float(self.fixing_input.text())
        except ValueError:
            fixing = 3.64
        # If the user most recently edited Case i, show case-aware OIS premium for i.
        if self._active_ois_case_index is not None:
            return _calc_2y_ois_for_case(
                self.rows,
                ois_row.date,
                fixing,
                case_index=self._active_ois_case_index,
            )
        return _calc_2y_ois(self.rows, ois_row.date, fixing)

    def _refresh_ois_cell(self):
        """Refresh only the OIS premium cell in-place (no full rebuild)."""
        for r, row in enumerate(self.rows):
            if row.is_ois_row:
                val = self._compute_ois_premium()
                self.meeting_table.blockSignals(True)
                item = self.meeting_table.item(r, 1)
                if item is not None:
                    item.setText(val)
                else:
                    item = QTableWidgetItem(val)
                    bold_font = QFont()
                    bold_font.setBold(True)
                    item.setFont(bold_font)
                    item.setFlags(_FLAG_LOCKED)
                    item.setBackground(_BRUSH_OIS_BG)
                    item.setForeground(_BRUSH_OIS_FG)
                    self.meeting_table.setItem(r, 1, item)
                self.meeting_table.blockSignals(False)
                break

    def _refresh_pricing_2y_yield_for_case(self, case_index: int) -> None:
        """
        Update ONLY the pricing-output '2Y Yield' cell for the given Case row.

        Case logic:
        2Y Yield (case) = Base OIS + Case Shock
        """
        if case_index is None:
            return

        if case_index < 0 or case_index >= MAX_CASES:
            return

        ois_row = next((r for r in self.rows if r.is_ois_row), None)
        if ois_row is None:
            return

        try:
            fixing = float(self.fixing_input.text())
        except ValueError:
            fixing = 3.64

        case_val = None
        if ois_row.cases and case_index < len(ois_row.cases):
            case_val = ois_row.cases[case_index]


        if case_val is None:
            self._case_2y_yield_dirty[case_index] = False
            ois_val = ""
        else:
            self._case_2y_yield_dirty[case_index] = True

            try:
                # ✅ Base OIS (no case impact)
                base_ois = float(_calc_2y_ois(self.rows, ois_row.date, fixing))
            except Exception:
                base_ois = 0.0

            # ✅ SIMPLE ADDITION (your requirement)
            ois_val = f"{base_ois + case_val:.2f}"

        NO_EDIT = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        bold_font = QFont()
        bold_font.setBold(True)

        target_label = f"Case {case_index + 1}"

        for tbl in (self.sr3_table, self.sr1_table, self.zq_table):
            for r in range(tbl.rowCount()):
                item0 = tbl.item(r, 0)
                if item0 is None or item0.text() != target_label:
                    continue

                existing = tbl.item(r, 1)
                if existing is None:
                    existing = QTableWidgetItem(ois_val)
                    existing.setFlags(NO_EDIT)
                    existing.setFont(bold_font)
                    tbl.setItem(r, 1, existing)
                elif existing.text() != ois_val:
                    existing.setText(ois_val)
                    existing.setFont(bold_font)

    def _add_ois_row(self):
        """Insert the 2Y OIS row at the bottom of self.rows and rebuild the table."""
        # Remove any existing OIS row first (avoid duplicates on reload)
        self.rows = [r for r in self.rows if not r.is_ois_row]

        ois = MeetingRow()
        ois.is_ois_row    = True
        ois.is_user_added = False
        ois.date          = _default_ois_date()
        ois.premium       = 0.0
        self.rows.append(ois)

    # ─────────────────────────────────────────────────────────────────────────
    def _update_case_button_states(self):
        self.btn_add_case.setEnabled(self.case_count < MAX_CASES)
        self.btn_remove_case.setEnabled(self.case_count > 1)
        self.case_count_label.setText(f"Cases: {self.case_count}")

    def _add_case(self):
        if self.case_count >= MAX_CASES:
            return
        self.case_count += 1
        self._build_meeting_columns()

    def _remove_case(self):
        if self.case_count <= 1:
            return
        self.case_count -= 1
        self._build_meeting_columns()

    def _delete_row(self):
        idx = self.meeting_table.currentRow()
        if idx < 0 or idx >= len(self.rows):
            return
        row = self.rows[idx]
        if row.is_ois_row:
            QMessageBox.information(self, "Cannot Delete",
                                    "The 2Y OIS row cannot be deleted.")
            return
        if not row.is_user_added:
            QMessageBox.information(self, "Cannot Delete",
                                    "Only manually added rows can be deleted.")
            return
        self.rows.pop(idx)
        self._build_meeting_columns()

    def _on_fixing_changed(self, _text: str):
        """When fixing changes: reschedule calc AND refresh the OIS cell."""
        self._refresh_ois_cell()
        self._schedule_calc()

    def _on_meeting_item_changed(self, item: QTableWidgetItem):
        r = item.row()
        c = item.column()

        if r < 0 or r >= len(self.rows):
            return

        row = self.rows[r]

        # ── OIS row: only column 0 (date) is editable ─────────────────────────
        if row.is_ois_row:
            if c == 0:
                new_date = item.text().strip()
                if new_date != row.date:
                    row.date = new_date
                    self._refresh_ois_cell()

            elif c >= 2:
                idx = c - 2
                txt = item.text().strip()

                if 0 <= idx < MAX_CASES:
                    # Case shocks drive ONLY pricing outputs (NOT OIS premium)
                    if txt == "":
                        row.cases[idx] = None
                    else:
                        try:
                            row.set_case(idx, txt)
                        except Exception:
                            row.cases[idx] = None

                    self._active_ois_case_index = idx

                    # ✅ ONLY update pricing outputs
                    self._refresh_pricing_2y_yield_for_case(idx)

                    # ❌ DO NOT refresh OIS premium here
                    # self._refresh_ois_cell()

            return

        # ── Non-OIS rows ───────────────────────────────────────────────────────
        if not row.is_user_added and c < 2:
            return

        txt = item.text().strip()

        # ── Empty input handling ───────────────────────────────────────────────
        if txt == "":
            value_changed = False

            if c == 0:
                if row.date != "":
                    row.date = ""
                    value_changed = True

            elif c == 1:
                if row.premium != 0:
                    row.premium = 0
                    value_changed = True

            else:
                idx = c - 2
                if row.cases[idx] is not None:
                    row.cases[idx] = None
                    value_changed = True

            if value_changed:
                # ✅ Only refresh OIS if base inputs changed
                if c < 2:
                    self._refresh_ois_cell()
                    self._calc_now()
                else:
                    self._schedule_calc()

            return

        # ── Partial typing guard ───────────────────────────────────────────────
        if txt.endswith(".") or txt == "-":
            return

        value_changed = False

        if c == 0:
            if row.date != txt:
                row.date = txt
                value_changed = True

        elif c == 1:
            old = row.premium
            row.premium_display = txt
            if row.premium != old:
                value_changed = True

        else:
            idx = c - 2
            old = row.cases[idx]
            row.set_case(idx, txt)
            if row.cases[idx] != old:
                value_changed = True

        if value_changed:
            # ✅ Only refresh OIS when base inputs change
            if c < 2:
                self._refresh_ois_cell()
                self._calc_now()
            else:
                self._schedule_calc()

    def _calc_now(self):
        self.debounce.stop()
        asyncio.ensure_future(self._start_calc_safe(force=True))

    def _schedule_calc(self):
        self.debounce.stop()
        self.debounce.start()

    def _on_debounce(self):
        asyncio.ensure_future(self._start_calc_safe())

    async def _start_calc_safe(self, force: bool = False):
        payload      = self._build_payload()
        payload_hash = json.dumps(payload, sort_keys=True)

        if not force and payload_hash == self.last_payload_hash:
            return

        self.last_payload_hash = payload_hash

        now = asyncio.get_event_loop().time()
        if now < self.next_allowed_time:
            return

        if self.calc_task and not self.calc_task.done():
            self.calc_task.cancel()
            try:
                await self.calc_task
            except Exception:
                pass

        self.calc_task = asyncio.ensure_future(self._run_calculation(payload))

    async def _load_defaults(self):
        try:
            data     = await self.fed.get_fed_premiums()
            fed_data = data["data"]["FED"]["data"]

            self.rows.clear()

            for item in fed_data:
                try:
                    date_str = str(item[0]).strip() if item[0] is not None else ""
                    premium  = item[1]

                    if not date_str or date_str.upper() in ("NA", "N/A", "NONE", "-"):
                        continue

                    dt = None
                    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    if dt is None:
                        try:
                            dt = datetime.fromisoformat(date_str)
                        except ValueError:
                            print(f"[_load_defaults] Skipping unparseable date: {date_str!r}")
                            continue

                    mr = MeetingRow()
                    mr.date          = dt.strftime("%d-%b-%y")
                    mr.premium       = float(premium)
                    mr.is_user_added = False
                    self.rows.append(mr)
                except Exception as row_ex:
                    print(f"[_load_defaults] Skipping row {item!r}: {row_ex}")
                    continue

            # ── Add the 2Y OIS row at the bottom ─────────────────────────────
            self._add_ois_row()

            self._build_meeting_columns()
            await self._start_calc_safe()

        except Exception as ex:
            QMessageBox.critical(self, "FED Load Error", str(ex))

    def _populate_all_tables(self, prices, vwap_root):
        self._populate_transposed(
            self.sr3_table,
            prices.get("SR3", []),
            vwap_root.get("SR3") if vwap_root else None,
            "SR3"
        )
        self._populate_transposed(
            self.sr1_table,
            prices.get("SR1", []),
            vwap_root.get("SR1") if vwap_root else None,
            "SR1"
        )
        self._populate_transposed(
            self.zq_table,
            prices.get("ZQ", []),
            vwap_root.get("ZQ") if vwap_root else None,
            "ZQ"
        )

    def _build_payload(self) -> dict:
        hikes = []
        for r in self.rows:
            if r.is_ois_row:          # ← exclude OIS row from API payload
                continue
            date_val = r.date.strip() if r.date else ""
            if not date_val:
                continue
            d = {
                "Meeting Date": date_val,
                "custom": bool(r.is_user_added),
            }
            for i in range(self.case_count):
                d[f"Case {i+1}"] = r.cases[i] if r.cases[i] is not None else 0.0
            hikes.append(d)

        try:
            fixing = float(self.fixing_input.text())
        except ValueError:
            fixing = 3.64

        return {
            "product_codes":         ["SR3", "SR1", "ZQ"],
            "economy":               "US",
            "hikes_and_cuts_inputs": hikes,
            "fixing":                fixing,
        }

    async def _run_calculation(self, payload):
        async with self.calc_lock:
            retry = True
            while retry:
                retry = False
                try:
                    result           = await self.api.calculate(payload)
                    self.last_result = result
                    prices           = result.get("pricesData", {})
                    vwap_root        = result.get("vwap_prices")
                    self._populate_all_tables(prices, vwap_root)

                except asyncio.CancelledError:
                    return

                except Exception as ex:
                    msg = str(ex)
                    if msg.startswith("THROTTLED|"):
                        wait_sec = 5
                        try:
                            body   = json.loads(msg.split("|", 1)[1])
                            detail = body.get("detail", "")
                            m      = re.search(r"in (\d+)", detail)
                            if m:
                                wait_sec = int(m.group(1)) + 1
                        except Exception:
                            pass
                        loop = asyncio.get_event_loop()
                        self.next_allowed_time = loop.time() + wait_sec
                        print(f"API THROTTLED → waiting {wait_sec}s")
                        await asyncio.sleep(wait_sec)
                        retry = True
                    else:
                        QMessageBox.critical(self, "Calculation Error", msg)

    def _on_base_view_changed(self, value):
        self.base_view = value
        if self.last_result is not None:
            prices    = self.last_result.get("pricesData", {})
            vwap_root = self.last_result.get("vwap_prices")
            self._populate_all_tables(prices, vwap_root)

    def _populate_transposed(self, tbl: QTableWidget, data, vwap_data, product):
        tbl.blockSignals(True)

        if not data:
            tbl.setRowCount(0)
            tbl.setColumnCount(0)
            tbl.blockSignals(False)
            return

        gap = 0
        if product == "SR3":
            gap = {"O/R": 0, "3MS": 1, "6MS": 2, "12MS": 4}.get(self.base_view, 0)
        elif product in ("SR1", "ZQ"):
            gap = 1 if self.base_view == "1MS" else 0

        col_count = 2
        for i in range(len(data)):
            if gap and i + gap >= len(data):
                break
            col_count += 1

        row_labels = ["Settle", "VWAP"]
        for k in data[0].keys():
            if k.startswith("Case"):
                row_labels.append(k)

        if tbl.columnCount() != col_count:
            tbl.setColumnCount(col_count)
            headers = ["", "2Y Yield"]
            for i in range(len(data)):
                if gap and i + gap >= len(data):
                    break
                headers.append(data[i].get("O/R"))
            tbl.setHorizontalHeaderLabels(headers)
            tbl.setColumnWidth(0, 20) 
            tbl.setColumnWidth(0, 90)
            for i in range(1, col_count):
                tbl.setColumnWidth(i, 70)

        if tbl.rowCount() != len(row_labels):
            tbl.setRowCount(len(row_labels))

        # Local "2Y Yield" (column index 1) is computed from meeting inputs,
        # not from API payload. Populate it for Settle + each Case row.
        ois_row = next((r for r in self.rows if r.is_ois_row), None)
        if ois_row is not None:
            try:
                fixing = float(self.fixing_input.text())
            except ValueError:
                fixing = 3.64
        else:
            fixing = 3.64

        vwap_map: dict = {}
        if vwap_data:
            for v in vwap_data:
                vwap_map[v.get("series")] = v.get("VWAP")

        NO_EDIT   = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        bold_font = QFont()
        bold_font.setBold(True)
        LABEL_BG  = QBrush(QColor("#F0F0F0"))

        for r, label in enumerate(row_labels):
            item0 = tbl.item(r, 0)

            if item0 is None:
                item0 = QTableWidgetItem(label)
                item0.setFlags(NO_EDIT)
                item0.setFont(bold_font)
                item0.setBackground(LABEL_BG)
                tbl.setItem(r, 0, item0)
            elif item0.text() != label:
                item0.setText(label)
                item0.setFont(bold_font)
                item0.setBackground(LABEL_BG)
            for c in range(2, col_count):
                front = data[c - 2]
                back  = data[c - 2 + gap] if gap else None

                if label == "VWAP":
                    f   = vwap_map.get(front.get("O/R"), 0)
                    b   = vwap_map.get(back.get("O/R"), 0) if gap else 0
                    val = round(float(f) - float(b), 4) if gap else f
                else:
                    f = front.get(label, 0)
                    b = back.get(label, 0) if gap else 0
                    try:
                        val = round(float(f) - float(b), 4) if gap else f
                    except Exception:
                        val = ""

                val_str  = str(val)
                existing = tbl.item(r, c)
                if existing is None:
                    existing = QTableWidgetItem(val_str)
                    existing.setFlags(NO_EDIT)
                    tbl.setItem(r, c, existing)
                elif existing.text() != val_str:
                    existing.setText(val_str)

                if label in ("Settle", "VWAP"):
                    existing.setFont(bold_font)
                    existing.setBackground(LABEL_BG)

            # Column 1 (index 1) = local computed "2Y Yield".
            # Requirement: start empty and only populate when user edits that case.
            val_2y = ""
            if ois_row is not None:
                # Never populate Settle/VWAP; only allow Case rows.
                if label.startswith("Case "):
                    # label is expected to be like "Case 1"
                    parts = label.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        case_index = int(parts[1]) - 1
                        if 0 <= case_index < MAX_CASES and ois_row.cases[case_index] is not None:
                            val_2y = _calc_2y_ois_for_case(
                                self.rows,
                                ois_row.date,
                                fixing,
                                case_index=case_index,
                            )
                        else:
                            val_2y = ""

            if col_count > 1:
                existing1 = tbl.item(r, 1)
                if existing1 is None:
                    existing1 = QTableWidgetItem(val_2y)
                    existing1.setFlags(NO_EDIT)
                    existing1.setFont(bold_font)
                    tbl.setItem(r, 1, existing1)
                elif existing1.text() != val_2y:
                    existing1.setText(val_2y)
                    existing1.setFont(bold_font)

        if not hasattr(tbl, "base_combo"):
            combo = QComboBox(tbl)
            combo.setView(QListView())
            combo.setMaxVisibleItems(5)
            combo.setStyleSheet("QComboBox { combobox-popup: 0; }")
            if product == "SR3":
                combo.addItems(["O/R", "3MS", "6MS", "12MS"])
            else:
                combo.addItems(["O/R", "1MS"])
            combo.currentTextChanged.connect(self._on_base_view_changed)
            combo.raise_()
            tbl.base_combo   = combo
            tbl._combo_width = -1

        combo  = tbl.base_combo
        header = tbl.horizontalHeader()
        new_w  = header.sectionSize(0)

        if new_w != tbl._combo_width:
            combo.setGeometry(
                header.sectionViewportPosition(0),
                2,
                new_w,
                header.height()
            )
            tbl._combo_width = new_w
            combo.show()

        if combo.currentText() != self.base_view:
            combo.blockSignals(True)
            combo.setCurrentText(self.base_view)
            combo.blockSignals(False)

        tbl.blockSignals(False)

    def closeEvent(self, e):
        if _LS_AVAILABLE:
            LSManager.instance().shutdown()
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()

    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()