"""盘中实时监控: 循环刷新 -> 实时情绪 / 主线 / 信号 / 持仓, 信号写日志并响铃."""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

from . import emotion, rotation, sectors, stocks
from .data import ROOT, Store, is_trading_time, market_of, now_cn, today_str
from .signals import Config, Engine, Signal

log = logging.getLogger(__name__)
POSITIONS = ROOT / "positions.csv"
REPORTS = ROOT / "reports"

RED, GREEN, YEL, CYAN, BOLD, DIM, RST = ("\033[31m", "\033[32m", "\033[33m", "\033[36m",
                                         "\033[1m", "\033[2m", "\033[0m")
ACTION_COLOR = {"买": RED + BOLD, "观察": YEL, "卖": GREEN + BOLD, "减仓": GREEN, "风险": CYAN + BOLD}


# ---------------- 持仓 ----------------
def load_positions() -> pd.DataFrame:
    if not POSITIONS.exists():
        return pd.DataFrame(columns=["code", "cost", "qty", "buy_date"])
    return pd.read_csv(POSITIONS, dtype={"code": str, "buy_date": str})


def save_positions(df: pd.DataFrame):
    df.to_csv(POSITIONS, index=False)


def pos_add(code: str, cost: float, qty: int, buy_date: str | None = None):
    df = load_positions()
    df = df[df["code"] != code]
    row = pd.DataFrame([{"code": code, "cost": cost, "qty": qty,
                         "buy_date": buy_date or today_str()}])
    save_positions(pd.concat([df, row], ignore_index=True) if len(df) else row)


def pos_rm(code: str):
    df = load_positions()
    save_positions(df[df["code"] != code])


# ---------------- 监控 ----------------
class Monitor:
    def __init__(self, cfg: Config | None = None, interval: int = 20, pages: int = 6):
        self.store = Store()
        self.c = self.store.c
        self.engine = Engine(cfg)
        self.interval, self.pages = interval, pages
        self.member_cache: dict = {}
        self.signals: list[Signal] = []
        self.amt_ratio, self.amt_ts = float("nan"), 0.0
        self.sig_file = REPORTS / f"signals_{today_str()}.csv"
        REPORTS.mkdir(exist_ok=True)
        self.prior: dict[str, float] = {}       # code -> 前5日涨幅
        self.idx_chg5m = 0.0
        self._init_emotion()
        self.rot = rotation.industry_rotation(self.store)   # 日线轮动(用到昨天为止的历史+今天盘中)
        self.rot_ts = time.time()

    def _init_emotion(self):
        """历史情绪分(不含今天), 用于判断今日阶段的变化方向."""
        t = today_str()
        h = emotion.history(self.store)
        h = h[h["date"] < t]
        self.prev_scores = h["score"].tolist()[-2:]
        days = [d for d in self.store.trade_days(5) if d < t]
        self.prev_day = days[-1]
        self.prev_zt_n = len(emotion.clean(self.store.pool("zt", self.prev_day)))

    def live_emotion(self, zt, zb, dt, yzt) -> dict:
        m = emotion.day_metrics(zt, zb, dt, yzt, self.prev_zt_n)
        if time.time() - self.amt_ts > 300:  # 成交额 5 分钟刷新一次
            try:
                a = self.store.market_amount(8)
                self.amt_ratio = float(a["amount"].iloc[-1] / a["amount"].iloc[-6:-1].mean())
                self.amt_ts = time.time()
            except RuntimeError as e:
                log.warning("成交额获取失败: %s", e)
        m["amount_ratio"] = self.amt_ratio
        s, _ = emotion.score(m)
        s1 = self.prev_scores[-1] if self.prev_scores else None
        s2 = self.prev_scores[-2] if len(self.prev_scores) >= 2 else None
        m["score"], m["phase"] = s, emotion.phase(s, s1, s2)
        return m

    def tick(self):
        t0 = time.time()
        d = today_str()
        zt, zb = self.c.pool("zt", d), self.c.pool("zb", d)
        dt, yzt = self.c.pool("dt", d), self.c.pool("yzt", d)
        emo = self.live_emotion(zt, zb, dt, yzt)
        snap = self.c.movers(self.pages)
        sec, mem = sectors.rank_sectors(
            self.c, zt, top_concept=20, top_industry=10, with_ret3=False,
            member_cache=self.member_cache, live_pct=snap.set_index("code")["pct"])
        pos = load_positions()
        holds = self.c.quotes(pos["code"].tolist()) if len(pos) else None
        try:
            idx = self.c.indices(("1.000001", "0.399006"))
            self.idx_chg5m = float(idx["chg5m"].iloc[0])
            self.idx = idx
        except (RuntimeError, KeyError, IndexError):
            self.idx = pd.DataFrame()
        now = now_cn().replace(tzinfo=None)
        new = self.engine.step(now, snap, emotion.clean(zt), emotion.clean(zb),
                               sec, mem, emo, pos, holds, idx_chg5m=self.idx_chg5m)
        self._record(new)
        picks = stocks.pick(zt, snap, sec, mem, emo["phase"]) if len(zt) else pd.DataFrame()
        if time.time() - self.rot_ts > 1800:  # 日线轮动 30 分钟重算(今天的涨停数在变)
            self.rot, self.rot_ts = rotation.industry_rotation(self.store), time.time()
        ladder = self.ladders(snap, sec, mem, zt)
        return dict(emo=emo, sec=sec, picks=picks, pos=pos, holds=holds, new=new,
                    ladder=ladder, cost=time.time() - t0)

    def ladders(self, snap, sec, mem, zt, n_themes: int = 3) -> list[pd.DataFrame]:
        """前 n 条主线(不同 theme)的板块内梯队: 角色 + 强弱."""
        out, seen = [], set()
        for r in sec.itertuples():
            if r.theme in seen:
                continue
            seen.add(r.theme)
            members = mem.get(r.bk, set())
            fm = self.engine.lead.first_mover.get(r.bk)
            df = rotation.strength(snap, r, members, zt, self.idx_chg5m, self.prior,
                                   fm[0] if fm else None)
            self._fill_prior(df.head(8)["code"].tolist() if len(df) else [])
            out.append(df.head(8))
            if len(out) >= n_themes:
                break
        return out

    def _fill_prior(self, codes: list[str], max_new: int = 8):
        """懒加载前5日涨幅 (判断补涨/高位), 每轮最多新取 max_new 只."""
        new = [c for c in codes if c not in self.prior][:max_new]
        for c in new:
            try:
                k = self.store.stock_kline(c, market_of(c), lmt=60)  # 与回测共用缓存, lmt 保持一致
                k = k[k["date"] < today_str()]
                self.prior[c] = (float(k["close"].iloc[-1] / k["close"].iloc[-6] - 1) * 100
                                 if len(k) >= 6 else float("nan"))
            except (RuntimeError, KeyError, IndexError):
                self.prior[c] = float("nan")

    def _record(self, new: list[Signal]):
        if not new:
            return
        self.signals.extend(new)
        df = pd.DataFrame([s.__dict__ | {"label": s.label} for s in new])
        df.to_csv(self.sig_file, mode="a", header=not self.sig_file.exists(),
                  index=False, encoding="utf-8-sig")
        if any(s.action in ("买", "卖") for s in new):
            sys.stdout.write("\a")  # 终端响铃

    # ---------------- 显示 ----------------
    def render(self, r: dict) -> str:
        e, sec, picks = r["emo"], r["sec"], r["picks"]
        L = []
        L.append(f"{BOLD}A股盘中情绪监控  {now_cn():%Y-%m-%d %H:%M:%S}{RST}  "
                 f"{DIM}刷新 {self.interval}s, 本轮耗时 {r['cost']:.1f}s, Ctrl+C 退出{RST}")
        pos_, adv = emotion.PHASE_ADVICE[e["phase"]]
        pc = GREEN if e["phase"] in ("退潮", "冰点", "分歧") else RED
        L.append(f"情绪 {BOLD}{e['score']:.0f}{RST} 【{pc}{e['phase']}{RST}】 仓位 {pos_} | "
                 f"涨停 {e['zt_n']} 炸板 {e['zb_n']} 跌停 {e['dt_n']} 封板率 {e['seal_rate']:.0%} | "
                 f"最高板 {e['max_lb']} {e['dragon']} | 昨涨停溢价 {e['yzt_avg']:+.2f}% "
                 f"大面 {e['big_loss_n']} | 量能 {e['amount_ratio']:.2f}x")
        L.append(f"{DIM}{adv}{RST}")

        L.append(f"\n{BOLD}── 最强主线 ──{RST}")
        themes = sec.head(12).groupby("theme", sort=False)["score"].max().head(5)
        L.append("  " + " > ".join(f"{t}({s:.0f})" for t, s in themes.items()))
        for x in sec.head(8).itertuples():
            peak = self.engine.st.sector_peak.get(x.bk, x.pct)
            L.append(f"  {x.name:<12} 强度{x.score:5.1f} 涨{x.pct:+5.2f}% (高{peak:+.2f}) "
                     f"主力{x.main_net:+6.1f}亿 涨停{int(x.zt_cnt):>2} 龙头 {x.dragon}")

        if len(self.rot):
            L.append(f"\n{BOLD}── 板块轮动(近10日行业涨停数) ──{RST}")
            for x in self.rot[self.rot["status"] != ""].head(7).itertuples():
                c = RED if x.status in ("新启动", "回流", "加强", "主升") else GREEN
                L.append(f"  {x.industry:<8} {c}{x.status:<4}{RST} 今{x.today:>2} 近3日均{x.avg3:>4} "
                         f"峰{x.peak:>2} {x.trend}  {DIM}{x.lead}{RST}")

        fd = self.engine.flow.deltas(time.time())
        if len(fd):
            inn = fd.nlargest(3, "d_net")
            outf = fd.nsmallest(3, "d_net")
            L.append(f"{BOLD}── 15分钟资金流向 ──{RST}  流入: " + "  ".join(
                f"{RED}{x.name}{x.d_net:+.1f}亿{RST}" for x in inn.itertuples() if x.d_net > 0)
                + "   流出: " + "  ".join(
                f"{GREEN}{x.name}{x.d_net:+.1f}亿{RST}" for x in outf.itertuples() if x.d_net < 0))
        else:
            L.append(f"{DIM}── 15分钟资金流向: 运行满15分钟后显示 ──{RST}")

        for lad in r.get("ladder", []):
            if lad is None or lad.empty:
                continue
            L.append(f"\n{BOLD}── 梯队 [{lad['sector'].iloc[0]}] ──{RST} {DIM}强度=相对板块/分时位置/均价/主力/抗跌/封板次序{RST}")
            for x in lad.itertuples():
                rc = RED + BOLD if x.role in ("龙头", "先锋", "首封") else (YEL if x.role in ("中军", "补涨", "强势") else DIM)
                p5 = f"{x.prior5:+.0f}%" if x.prior5 == x.prior5 and x.prior5 is not None else "  -"
                L.append(f"  {rc}{x.role:<2}{RST} {x.code} {x.name:<6} 强{x.strength:5.1f} 涨{x.pct:+6.2f}% "
                         f"相对板块{x.rs:+5.1f} 5分{x.chg5m:+.1f} 位置{x.pos:.0%} "
                         f"主力{x.main_ratio:+4.0f}% 前5日{p5} 流通{x.float_yi:.0f}亿")

        if len(picks):
            L.append(f"\n{BOLD}── 涨停股综合分 Top8 ──{RST}")
            for x in picks.head(8).itertuples():
                L.append(f"  {x.c} {x.n:<6} {x.type} 总分{x.score:5.1f} 封板分{x.core:5.1f} "
                         f"{int(x.lbc)}板 首封{int(x.fbt):06d} 炸{int(x.zbc)} 主力{x.main_ratio:+.0f}% "
                         f"[{x.sector or x.hybk}] {DIM}{x.flags}{RST}")

        pos, holds = r["pos"], r["holds"]
        if len(pos) and holds is not None and len(holds):
            L.append(f"\n{BOLD}── 持仓 ──{RST}")
            hq = holds.set_index("code")
            for p in pos.itertuples():
                if p.code not in hq.index:
                    continue
                q = hq.loc[p.code]
                pnl = (q["price"] / p.cost - 1) * 100
                c = RED if pnl >= 0 else GREEN
                L.append(f"  {p.code} {q['name']:<6} 成本{p.cost:.2f} 现价{q['price']:.2f} "
                         f"今日{q['pct']:+.2f}% 盈亏{c}{pnl:+.2f}%{RST} 主力{q['main_ratio']:+.0f}%")

        L.append(f"\n{BOLD}── 信号 (最新在上, 全部记录见 {self.sig_file.name}) ──{RST}")
        if not self.signals:
            L.append(f"  {DIM}暂无. 首轮只建立基线, 之后新封板/回封/冲板/卖点会出现在这里{RST}")
        for s in reversed(self.signals[-18:]):
            c = ACTION_COLOR.get(s.action, "")
            L.append(f"  {s.time} {c}[{s.action}]{s.label}{RST} {s.code} {s.name} "
                     f"{s.price:g}  {s.reason}")
        return "\n".join(L)

    def run(self, once: bool = False, force: bool = False):
        while True:
            if not force and not is_trading_time():
                print(f"{now_cn():%H:%M:%S} 非交易时间, 等待中 ... (--force 可强制运行)", end="\r")
                if once:
                    print()
                    return
                time.sleep(30)
                continue
            try:
                r = self.tick()
                screen = self.render(r)
                if not once and sys.stdout.isatty():
                    os.system("clear")
                print(screen, flush=True)
            except Exception as e:  # noqa: BLE001 - 监控不中断
                log.exception("本轮刷新失败: %s", e)
            if once:
                return
            time.sleep(max(1, self.interval))
