"""命令行入口.

python -m emo.cli today      # 情绪 + 最强板块 + 情绪票候选 + 跟风资金票, 生成报告
python -m emo.cli emotion    # 情绪历史
python -m emo.cli sectors    # 只看板块强度
python -m emo.cli backtest   # 回测打分/情绪过滤
python -m emo.cli collect    # 缓存可得的历史池数据(建议每天收盘后跑, 积累样本)
python -m emo.cli rotation   # 板块轮动(近10日行业涨停数 + 状态)
python -m emo.cli watch      # 盘中实时监控 + 买卖点信号
python -m emo.cli pos add 601606 30.50 1000 [--date 20261015]   # 记录持仓(用于卖点)
python -m emo.cli pos rm 601606 / pos ls
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import backtest, emotion, rotation, sectors, stocks
from .data import ROOT, Store, now_cn, today_str

REPORTS = ROOT / "reports"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.unicode.east_asian_width", True)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    if df is None or df.empty:
        return "_无_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


class Report:
    def __init__(self):
        self.parts: list[str] = []

    def h(self, t: str):
        print(f"\n{'=' * 8} {t} {'=' * 8}")
        self.parts.append(f"\n## {t}\n")

    def p(self, t: str):
        print(t)
        self.parts.append(t + "\n")

    def t(self, df: pd.DataFrame):
        print(df.to_string(index=False) if df is not None and len(df) else "(无)")
        self.parts.append(md_table(df))

    def save(self, name: str):
        REPORTS.mkdir(exist_ok=True)
        f = REPORTS / name
        f.write_text("\n".join(self.parts), encoding="utf-8")
        return f


EMO_COLS = {"date": "日期", "score": "情绪分", "phase": "阶段", "zt_n": "涨停", "zb_n": "炸板",
            "dt_n": "跌停", "seal_rate": "封板率", "max_lb": "最高板", "lb_n": "连板数",
            "promote_rate": "晋级率", "yzt_avg": "昨涨停溢价%", "ylb_avg": "昨连板溢价%",
            "big_loss_n": "大面", "amount": "成交额亿", "amount_ratio": "量比5日",
            "dragon": "最高板"}


def emotion_section(rep: Report, store: Store, n: int = 10) -> emotion.EmotionDay | None:
    hist = emotion.history(store)
    if hist.empty:
        rep.p("无情绪数据")
        return None
    e = emotion.latest(hist)
    pos, adv = e.advice
    m = e.metrics
    rep.h(f"市场情绪 {e.date}")
    if e.date == today_str() and now_cn().hour * 100 + now_cn().minute < 1500:
        rep.p("(盘中数据, 收盘前会变化; 成交额为按时间外推的全天预估)")
    rep.p(f"情绪分: {e.score} / 100   阶段: 【{e.phase}】   建议仓位: {pos}")
    rep.p(f"策略: {adv}")
    rep.p(f"涨停 {m['zt_n']}  炸板 {m['zb_n']}  跌停 {m['dt_n']}  封板率 {m['seal_rate']:.0%}  "
          f"最高板 {m['max_lb']}  连板 {m['lb_n']}  晋级率 {m['promote_rate']:.0%}  "
          f"昨涨停溢价 {m['yzt_avg']:+.2f}%  大面 {m['big_loss_n']}")
    rep.p("分项(0-100): " + "  ".join(f"{k}={v:.0f}" for k, v in e.parts.items()))
    h = hist.tail(n).copy()
    for c in ("seal_rate", "promote_rate"):
        h[c] = (h[c] * 100).round(0)
    rep.t(h[[c for c in EMO_COLS if c in h]].rename(columns=EMO_COLS).round(2))
    return e


SEC_COLS = {"name": "板块", "theme": "主线", "kind": "类型", "score": "强度", "pct": "涨幅%",
            "main_net": "主力净流入亿", "main_ratio": "主力占比%", "breadth": "上涨占比",
            "zt_cnt": "涨停数", "strong_ratio": "涨5%占比", "max_lb": "最高板", "ret3": "3日%",
            "dragon": "龙头", "size": "成分股"}


def sectors_section(rep: Report, store: Store, zt: pd.DataFrame, n: int = 15):
    rep.h("最强板块")
    sec, mem = sectors.rank_sectors(store.c, zt)
    themes = sec.head(n).groupby("theme", sort=False)["score"].max().head(5)
    rep.p("今日主线: " + " > ".join(f"{t}({s:.0f})" for t, s in themes.items()))
    rep.t(sec.head(n)[list(SEC_COLS)].rename(columns=SEC_COLS).round(2))
    return sec, mem


PICK_COLS = {"c": "代码", "n": "名称", "type": "类型", "score": "总分", "core": "封板分",
             "lbc": "连板", "fbt": "首封", "zbc": "炸板", "fund_ratio": "封单/流通%",
             "hs": "换手%", "ltsz": "流通亿", "main_net": "主力净额亿", "main_ratio": "主力占比%",
             "sector": "所属主线", "hybk": "行业", "flags": "风险"}


def picks_section(rep: Report, zt, snap, sec, mem, phase: str, n: int = 20):
    rep.h("情绪票候选(今日涨停, 按综合分)")
    p = stocks.pick(zt, snap, sec, mem, phase)
    if p.empty:
        rep.p("今日暂无涨停")
        return p
    p = p.copy()
    p["ltsz"] = p["ltsz"] / 1e8
    rep.p("说明: 封板分=封板时间/炸板/封单/换手/市值/行业涨停数/高度 (已回测); "
          "总分再叠加主力资金占比和是否属于最强主线.")
    rep.t(p.head(n)[list(PICK_COLS)].rename(columns=PICK_COLS).round(2))
    return p


FOL_COLS = {"code": "代码", "name": "名称", "sector": "主线", "pct": "涨幅%", "main_net": "主力净额亿",
            "main_ratio": "主力占比%", "vol_ratio": "量比", "turnover": "换手%",
            "float_mv": "流通亿", "score": "资金分"}


ROT_COLS = {"industry": "行业", "status": "轮动状态", "today": "今日涨停", "avg3": "前3日均",
            "peak": "10日峰值", "trend": "10日走势", "lead": "今日最高板"}


def rotation_section(rep: Report, store: Store, n: int = 20):
    rep.h("板块轮动 (近10个交易日 行业涨停数)")
    rot = rotation.industry_rotation(store)
    if rot.empty:
        rep.p("无数据")
        return rot
    rep.p("状态: 新启动=今日爆发且前3日冷 | 回流=曾热过、冷却后再起 | 加强=明显放大 | "
          "主升=连续3日以上>=3家 | 退潮=从峰值回落60%+")
    rep.t(rot.head(n)[list(ROT_COLS)].rename(columns=ROT_COLS))
    return rot


def cmd_rotation(args):
    rotation_section(Report(), Store(), n=args.n)


def cmd_today(args):
    store = Store()
    rep = Report()
    d = today_str()
    rep.parts.append(f"# A股短线情绪日报 {now_cn():%Y-%m-%d %H:%M} (北京时间)\n")
    e = emotion_section(rep, store)
    phase = e.phase if e else "混沌"
    zt = store.pool("zt", d)
    if zt.empty and e is not None:  # 盘前: 用最近一个交易日
        d = e.date
        zt = store.pool("zt", d)
        rep.p(f"(今日尚无涨停数据, 以下使用 {d})")
    print("\n拉取全A快照与板块数据, 约 1 分钟 ...", file=sys.stderr)
    snap = store.c.stocks()
    sec, mem = sectors_section(rep, store, zt)
    rotation_section(rep, store, n=12)
    rep.h("主线梯队 (角色/强弱)")
    seen = set()
    for r in sec.itertuples():
        if r.theme in seen or len(seen) >= 3:
            continue
        seen.add(r.theme)
        lad = rotation.strength(snap, r, mem.get(r.bk, set()), zt)
        rep.p(f"[{r.name}]")
        rep.t(lad.head(8).round(2) if len(lad) else lad)
    picks = picks_section(rep, zt, snap, sec, mem, phase, n=args.n)
    rep.h("主线资金跟风票(未涨停, 主力净流入)")
    fol = stocks.followers(snap, sec, mem, set(zt["c"]) if len(zt) else set())
    rep.t(fol[list(FOL_COLS)].rename(columns=FOL_COLS) if len(fol) else fol)

    rep.h("操作纪律")
    rep.p("1. 情绪【退潮/冰点】: 不接力, 不追高位, 最多1成试错.\n"
          "2. 只做最强主线前排: 龙头/最早封板/封单最大, 后排跟风次日易被核.\n"
          "3. 回测显示: 次日开盘追昨日涨停股平均为负收益, 不要追高开的昨日涨停.\n"
          "4. 打板只选封板早、炸板少、封单大的; 买入后次日不及预期(低开/不能快速翻红)开盘即走.\n"
          "5. 单票止损 -5%, 单日回撤超 3% 停止交易.\n"
          "本系统为量化辅助工具, 不构成投资建议.")
    tag = now_cn().strftime("%Y%m%d_%H%M")
    f = rep.save(f"{tag}.md")
    if len(picks):
        picks.to_csv(REPORTS / f"{tag}_picks.csv", index=False, encoding="utf-8-sig")
    sec.to_csv(REPORTS / f"{tag}_sectors.csv", index=False, encoding="utf-8-sig")
    print(f"\n报告已保存: {f}")


def cmd_emotion(args):
    emotion_section(Report(), Store(), n=args.n)


def cmd_sectors(args):
    store = Store()
    sectors_section(Report(), store, store.pool("zt", today_str()), n=args.n)


def cmd_backtest(args):
    store = Store()
    print("构建样本(首次需下载个股K线, 之后走缓存) ...", file=sys.stderr)
    s = backtest.build_samples(store)
    rep = Report()
    names = {"board": "打板: D日涨停价买, D+1卖",
             "next_open": "次日接力: D+1开盘买, D+2卖"}
    for mode in ("board", "next_open"):
        r = backtest.report(s, mode, top_k=args.top)
        rep.h(f"{names[mode]}  样本 {r['samples']} 只 / {r['days']} 天")
        if "strategy" not in r:
            rep.p("样本不足")
            continue
        rep.p("封板分五分位 (open%=开盘卖平均收益, 已扣0.2%成本):")
        rep.t(r["by_quintile"].reset_index().round(3))
        rep.p("因子IC(与开盘卖收益的秩相关, >0 有效):")
        rep.t(r["ic"].reset_index().rename(columns={"index": "因子", 0: "IC"}))
        rep.p(f"情绪分与当日全体收益的秩相关: {r['emo_ic']}")
        rep.p("按情绪阶段:")
        rep.t(r["by_phase"].reset_index().round(3))
        rep.p("策略对比(开盘卖, 等权):")
        rep.t(r["strategy"].reset_index().rename(columns={"index": "策略"}).round(3))
    rep.p("\n注意: 东财涨停池只保留约20个交易日, 样本很小, 结论仅供参考; "
          "每天跑 collect 积累数据后再评估更可靠.")
    f = rep.save(f"backtest_{now_cn():%Y%m%d_%H%M}.md")
    s.to_csv(REPORTS / "backtest_samples.csv", index=False, encoding="utf-8-sig")
    print(f"\n报告已保存: {f}")


def cmd_collect(args):
    store = Store()
    h = emotion.history(store)
    print(f"已缓存情绪池 {len(h)} 个交易日: {h['date'].iloc[0]} ~ {h['date'].iloc[-1]}")
    backtest.build_samples(store, h)
    print("个股K线已缓存. 缓存目录:", store.cache)


def cmd_watch(args):
    from .monitor import Monitor
    from .signals import Config
    cfg = Config(stop=args.stop, top_sectors=args.top_sectors, seal_core_min=args.core_min)
    Monitor(cfg, interval=args.interval).run(once=args.once, force=args.force)


def cmd_pos(args):
    from .monitor import load_positions, pos_add, pos_rm
    if args.op == "add":
        if args.cost is None or args.qty is None:
            sys.exit("用法: pos add 代码 成本价 数量 [--date YYYYMMDD]")
        pos_add(args.code, args.cost, args.qty, args.date)
    elif args.op == "rm":
        pos_rm(args.code)
    print(load_positions().to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="emo", description="A股短线情绪+资金量化系统")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name, fn, n in (("today", cmd_today, 20), ("emotion", cmd_emotion, 15),
                        ("sectors", cmd_sectors, 20), ("rotation", cmd_rotation, 25)):
        sp = sub.add_parser(name)
        sp.add_argument("-n", type=int, default=n, help="显示条数")
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("backtest")
    sp.add_argument("--top", type=int, default=3, help="每日选前N只")
    sp.set_defaults(fn=cmd_backtest)
    sub.add_parser("collect").set_defaults(fn=cmd_collect)
    sp = sub.add_parser("watch", help="盘中实时监控")
    sp.add_argument("--interval", type=int, default=20, help="刷新秒数")
    sp.add_argument("--stop", type=float, default=5.0, help="止损%%")
    sp.add_argument("--top-sectors", type=int, default=6, help="视为主线的板块数")
    sp.add_argument("--core-min", type=float, default=65, help="新封板买点最低封板分")
    sp.add_argument("--once", action="store_true", help="只刷新一次(测试)")
    sp.add_argument("--force", action="store_true", help="非交易时间也运行")
    sp.set_defaults(fn=cmd_watch)
    sp = sub.add_parser("pos", help="持仓管理")
    sp.add_argument("op", choices=["add", "rm", "ls"])
    sp.add_argument("code", nargs="?", default="")
    sp.add_argument("cost", nargs="?", type=float)
    sp.add_argument("qty", nargs="?", type=int)
    sp.add_argument("--date", help="买入日期 YYYYMMDD, 默认今天")
    sp.set_defaults(fn=cmd_pos)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    if not args.cmd:
        args = ap.parse_args(["today"] + (argv or []))
    args.fn(args)


if __name__ == "__main__":
    main()
