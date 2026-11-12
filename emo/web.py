"""网页看板: 一条命令启动 盘中监控 + 网页界面 + 收盘后自动日报/collect.

python -m emo.cli web            # 打开 http://127.0.0.1:8000
只用标准库 http.server, 不引入新依赖. 默认只监听本机; --host 0.0.0.0 会暴露到局域网(无登录鉴权).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

from . import emotion
from .data import ROOT, Store, is_trading_time, now_cn, today_str

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent / "static"
REPORTS = ROOT / "reports"
CODE_RE = re.compile(r"^\d{6}$")
REPORT_RE = re.compile(r"^[\w.\-]+\.(md|csv)$")

REPORT_AT = 1505    # 15:05 收盘后自动生成日报
COLLECT_AT = 1535   # 15:35 自动缓存当天数据


# ---------------- 序列化 ----------------
def records(df: pd.DataFrame | None, cols: list[str] | None = None) -> list[dict]:
    """DataFrame -> JSON 友好的 list[dict] (NaN -> null, numpy -> python)."""
    if df is None or len(df) == 0:
        return []
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return json.loads(df.to_json(orient="records", force_ascii=False))


def clean_num(x):
    """单个数值: NaN/inf -> None."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return None if f != f or f in (float("inf"), float("-inf")) else f


SEC_COLS = ["bk", "name", "theme", "kind", "score", "pct", "main_net", "main_ratio", "breadth",
            "zt_cnt", "max_lb", "dragon", "dragon_code", "lead_name", "lead_pct"]
PICK_COLS = ["c", "n", "type", "score", "core", "lbc", "fbt", "zbc", "fund_ratio", "hs",
             "main_ratio", "sector", "hybk", "flags"]
HIST_COLS = ["date", "score", "phase", "zt_n", "zb_n", "dt_n", "seal_rate", "max_lb",
             "promote_rate", "yzt_avg", "big_loss_n", "amount", "amount_ratio", "dragon"]


def positions_view(pos: pd.DataFrame, holds: pd.DataFrame | None) -> list[dict]:
    """持仓 + 实时行情 -> 盈亏."""
    out = []
    hq = holds.set_index("code") if holds is not None and len(holds) else None
    for p in pos.itertuples():
        row = {"code": p.code, "cost": float(p.cost), "qty": int(p.qty), "buy_date": str(p.buy_date)}
        if hq is not None and p.code in hq.index:
            q = hq.loc[p.code]
            row.update(name=q.get("name"), price=clean_num(q.get("price")), pct=clean_num(q.get("pct")),
                       main_ratio=clean_num(q.get("main_ratio")))
            if row["price"]:
                row["pnl"] = (row["price"] / row["cost"] - 1) * 100
        out.append(row)
    return out


def tick_payload(r: dict, mon) -> dict:
    """Monitor.tick() 的结果 -> 网页 JSON."""
    e = dict(r["emo"])
    pos_, adv = emotion.PHASE_ADVICE.get(e.get("phase"), ("-", ""))
    e.update(position=pos_, advice=adv)
    sec = r["sec"]
    peaks = mon.engine.st.sector_peak
    sec_rows = records(sec.head(12), SEC_COLS)
    for s in sec_rows:
        s["peak"] = clean_num(peaks.get(s["bk"], s.get("pct")))
    themes = (sec.head(12).groupby("theme", sort=False)["score"].max().head(5)
              if len(sec) else pd.Series(dtype=float))
    flows = mon.engine.flow.deltas(time.time())
    ladders = [{"sector": str(l["sector"].iloc[0]), "rows": records(l)}
               for l in r.get("ladder", []) if l is not None and len(l)]
    rot = mon.rot[mon.rot["status"] != ""].head(10) if len(mon.rot) else mon.rot
    return {
        "emo": {k: clean_num(v) for k, v in e.items()},
        "themes": [{"theme": t, "score": float(s)} for t, s in themes.items()],
        "sectors": sec_rows,
        "rotation": records(rot),
        "flows_in": records(flows.nlargest(5, "d_net")) if len(flows) else [],
        "flows_out": records(flows.nsmallest(5, "d_net")) if len(flows) else [],
        "ladders": ladders,
        "picks": records(r["picks"].head(15), PICK_COLS) if len(r["picks"]) else [],
        "indices": records(getattr(mon, "idx", None), ["code", "name", "price", "pct", "chg5m", "amount"]),
        "cost": round(r["cost"], 1),
    }


# ---------------- 定时任务 ----------------
def due_jobs(now: datetime, done: set[str], is_trade_day: bool) -> list[str]:
    """当前该跑的收盘后任务 (每个交易日各一次)."""
    if not is_trade_day:
        return []
    hm, d = now.hour * 100 + now.minute, now.strftime("%Y%m%d")
    jobs = []
    if hm >= REPORT_AT and f"report_{d}" not in done:
        jobs.append(f"report_{d}")
    if hm >= COLLECT_AT and f"collect_{d}" not in done:
        jobs.append(f"collect_{d}")
    return jobs


# ---------------- 后台引擎 ----------------
@dataclass
class App:
    interval: int = 20
    cfg: object = None
    auto_jobs: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)
    live: dict = field(default_factory=dict)
    hist: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    holds: object = None                         # 持仓实时行情(随每轮刷新)
    status: str = "启动中"
    error: str = ""
    updated: str = ""
    jobs: dict = field(default_factory=dict)     # name -> 状态文字
    done: set = field(default_factory=set)
    mon: object = None
    mon_day: str = ""
    hist_ts: float = 0.0

    # -- 监控 --
    def _monitor(self):
        d = today_str()
        if self.mon is None or self.mon_day != d:   # 跨日重建 (信号文件/昨日基准)
            from .monitor import Monitor
            self.status = "初始化(拉取历史情绪/轮动)..."
            self.mon, self.mon_day = Monitor(self.cfg, interval=self.interval), d
            with self.lock:
                self.signals = []
        return self.mon

    def tick_once(self):
        mon = self._monitor()
        self.status = "刷新中..."
        r = mon.tick()
        payload = tick_payload(r, mon)
        sigs = [s.__dict__ | {"label": s.label} for s in mon.signals[-200:]]
        with self.lock:
            self.live, self.signals, self.holds = payload, sigs, r["holds"]
            self.updated = now_cn().strftime("%H:%M:%S")
            self.error = ""

    def refresh_hist(self):
        if time.time() - self.hist_ts < 600 and self.hist:
            return
        h = emotion.history(Store())
        with self.lock:
            self.hist = records(h.tail(30), HIST_COLS)
        self.hist_ts = time.time()

    def loop(self):
        first = True
        while True:
            t0 = time.time()
            try:
                self.refresh_hist()
                if is_trading_time() or first:
                    self.tick_once()     # 非交易时间启动时也跑一轮, 显示最近一次数据
                    self.status = "盘中监控" if is_trading_time() else "非交易时间(显示最近数据)"
                else:
                    self.status = "非交易时间, 等待开盘"
                first = False
                if self.auto_jobs:
                    self.run_due_jobs()
            except Exception as e:  # noqa: BLE001 - 后台不中断
                log.exception("刷新失败")
                with self.lock:
                    self.error = f"{now_cn():%H:%M:%S} {type(e).__name__}: {e}"
                first = False
            time.sleep(max(1.0, self.interval - (time.time() - t0)))

    # -- 收盘后任务 --
    def run_due_jobs(self):
        try:
            trade_day = today_str() in Store().trade_days(5)
        except RuntimeError:
            return
        for j in due_jobs(now_cn().replace(tzinfo=None), self.done, trade_day):
            self.done.add(j)
            self.start_job(j.split("_")[0])

    def start_job(self, kind: str) -> bool:
        if self.jobs.get(kind, "").startswith("运行中"):
            return False
        threading.Thread(target=self._job, args=(kind,), daemon=True).start()
        return True

    def _job(self, kind: str):
        from . import cli
        self.jobs[kind] = f"运行中 {now_cn():%H:%M:%S}"
        fn = {"report": cli.cmd_today, "collect": cli.cmd_collect}[kind]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn(argparse.Namespace(n=20))
            self.jobs[kind] = f"完成 {now_cn():%H:%M:%S}"
            self.hist_ts = 0
        except Exception as e:  # noqa: BLE001
            log.exception("任务 %s 失败", kind)
            self.jobs[kind] = f"失败 {now_cn():%H:%M:%S}: {e}"

    # -- API --
    def state(self) -> dict:
        from .monitor import load_positions
        pos = load_positions()
        with self.lock:
            holds = self.holds
            return {"now": now_cn().strftime("%Y-%m-%d %H:%M:%S"), "status": self.status,
                    "trading": is_trading_time(), "updated": self.updated, "error": self.error,
                    "interval": self.interval, "live": self.live, "hist": self.hist,
                    "signals": list(reversed(self.signals[-60:])),
                    "positions": positions_view(pos, holds), "jobs": dict(self.jobs),
                    "reports": list_reports()}


def list_reports(n: int = 15) -> list[str]:
    if not REPORTS.exists():
        return []
    fs = sorted((f for f in REPORTS.glob("*.md")), key=lambda f: f.stat().st_mtime, reverse=True)
    return [f.name for f in fs[:n]]


def parse_pos(body: dict) -> tuple[str, str, float | None, int | None, str | None]:
    """校验持仓请求, 不合法抛 ValueError."""
    op = body.get("op")
    code = str(body.get("code", "")).strip()
    if op not in ("add", "rm"):
        raise ValueError("op 必须是 add 或 rm")
    if not CODE_RE.match(code):
        raise ValueError("股票代码必须是 6 位数字")
    if op == "rm":
        return op, code, None, None, None
    cost, qty = float(body.get("cost", 0)), int(body.get("qty", 0))
    if not (0 < cost < 100000) or not (0 < qty < 10 ** 9):
        raise ValueError("成本价和数量必须为正数")
    d = str(body.get("date") or "").strip() or None
    if d and not re.match(r"^\d{8}$", d):
        raise ValueError("日期格式 YYYYMMDD")
    return op, code, cost, qty, d


# ---------------- HTTP ----------------
def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # 静默访问日志
            log.debug(fmt, *a)

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False, allow_nan=False,
                                        default=str).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(app.state())
            elif path.startswith("/reports/"):
                name = unquote(path[len("/reports/"):])
                f = REPORTS / name
                if not REPORT_RE.match(name) or not f.is_file():
                    return self._json({"error": "not found"}, 404)
                self._send(200, f.read_bytes(), "text/plain; charset=utf-8")
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                n = min(int(self.headers.get("Content-Length", 0)), 10_000)
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json({"error": "请求格式错误"}, 400)
            if path == "/api/pos":
                from .monitor import pos_add, pos_rm
                try:
                    op, code, cost, qty, d = parse_pos(body)
                except (ValueError, TypeError) as e:
                    return self._json({"error": str(e)}, 400)
                pos_add(code, cost, qty, d) if op == "add" else pos_rm(code)
                return self._json({"ok": True})
            if path == "/api/job":
                kind = body.get("kind")
                if kind not in ("report", "collect"):
                    return self._json({"error": "未知任务"}, 400)
                return self._json({"ok": app.start_job(kind)})
            self._json({"error": "not found"}, 404)

    return H


def serve(host: str = "127.0.0.1", port: int = 8000, interval: int = 20, cfg=None,
          auto_jobs: bool = True):
    app = App(interval=interval, cfg=cfg, auto_jobs=auto_jobs)
    threading.Thread(target=app.loop, daemon=True).start()
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    print(f"A股情绪看板已启动: http://{shown}:{port}   (Ctrl+C 退出)")
    if host not in ("127.0.0.1", "localhost"):
        print("注意: 已监听局域网地址, 页面没有登录鉴权, 同网段的人都能访问并修改持仓.")
    print(f"盘中每 {interval}s 刷新; {REPORT_AT // 100}:{REPORT_AT % 100:02d} 自动生成日报, "
          f"{COLLECT_AT // 100}:{COLLECT_AT % 100:02d} 自动 collect.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    finally:
        srv.server_close()
