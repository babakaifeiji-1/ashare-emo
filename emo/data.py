"""东方财富数据客户端 + 本地缓存.

所有接口均为东财网页端公开接口(非官方 API, 字段可能变化).
- push2delay: 实时(延迟)快照, 板块/个股列表, 每页最多 100 条
- push2his : 日K线, 日资金流
- push2ex  : 涨停/炸板/跌停/昨日涨停 池, 历史只保留约 20 个交易日

历史日期的数据写入 data/cache 后不再请求; 当天数据每次实时拉取.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)

CN_TZ = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"

CLIST = "https://push2delay.eastmoney.com/api/qt/clist/get"
ULIST = "https://push2delay.eastmoney.com/api/qt/ulist.np/get"
KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
FFLOW = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
POOL = "https://push2ex.eastmoney.com/{name}"
TX_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get"  # 腾讯日K, push2his 失败时备用
MIRRORS = ["", "82.", "80.", "7.", "33.", "17."]
POOL_UT = "7eea3edcaed734bea9cbfc24409ed989"

FS_ALL_A = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
FS_INDUSTRY = "m:90+t:2"
FS_CONCEPT = "m:90+t:3"

# 个股快照字段 -> 列名
STOCK_FIELDS = {
    "f12": "code", "f13": "mkt", "f14": "name", "f2": "price", "f3": "pct",
    "f5": "vol", "f6": "amount", "f8": "turnover", "f10": "vol_ratio",
    "f15": "high", "f16": "low", "f17": "open", "f18": "pre_close",
    "f20": "mv", "f21": "float_mv", "f22": "speed", "f11": "chg5m",
    "f62": "main_net", "f184": "main_ratio", "f66": "xl_net", "f72": "l_net",
    "f100": "industry",
}
BOARD_FIELDS = {
    "f12": "bk", "f14": "name", "f3": "pct", "f6": "amount", "f8": "turnover",
    "f20": "mv", "f62": "main_net", "f184": "main_ratio",
    "f104": "up", "f105": "down", "f106": "flat",
    "f128": "lead_name", "f140": "lead_code", "f136": "lead_pct",
}
POOL_SORT = {
    "getTopicZTPool": "fbt:asc",
    "getTopicZBPool": "fbt:asc",
    "getTopicDTPool": "fund:asc",
    "getYesterdayZTPool": "zs:desc",
}
POOL_ALIAS = {"zt": "getTopicZTPool", "zb": "getTopicZBPool",
              "dt": "getTopicDTPool", "yzt": "getYesterdayZTPool"}


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def today_str() -> str:
    return now_cn().strftime("%Y%m%d")


def is_final(date: str) -> bool:
    """date 的数据是否已收盘定型(可永久缓存)."""
    n = now_cn()
    t = n.strftime("%Y%m%d")
    return date < t or (date == t and n.hour * 100 + n.minute >= 1530)


def trading_fraction(t: datetime | None = None) -> float:
    """当前已过交易时间占全天(240分钟)比例."""
    t = t or now_cn()
    m = t.hour * 60 + t.minute
    am = min(max(m - 570, 0), 120)        # 9:30-11:30
    pm = min(max(m - 780, 0), 120)        # 13:00-15:00
    return (am + pm) / 240


def market_of(code: str) -> int:
    """东财 secid 市场号: 沪市 1, 深市/北交所 0."""
    c = str(code)
    return 1 if c.startswith(("6", "900", "5")) else 0


def is_trading_time(t: datetime | None = None) -> bool:
    t = t or now_cn()
    if t.weekday() >= 5:
        return False
    m = t.hour * 60 + t.minute
    return 555 <= m <= 690 or 780 <= m <= 900  # 9:15-11:30, 13:00-15:00


def _num(x):
    if x in ("-", "", None):
        return float("nan")
    return x


class EMClient:
    def __init__(self, retries: int = 6, timeout: float = 8.0, workers: int = 2):
        self.s = requests.Session()
        self.s.trust_env = False  # 公司代理会拦截东财, 直连
        self.s.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120 Safari/537.36",
            "Referer": "https://quote.eastmoney.com/",
        })
        self.retries, self.timeout, self.workers = retries, timeout, workers
        self._his_down_until = 0.0  # push2his 熔断截止时间

    # ---------------- 基础请求 ----------------
    def get_json(self, url: str, params: dict, retries: int | None = None) -> dict:
        """GET + 重试. 东财部分节点会间歇性挂起新连接, 重试时轮换镜像域名."""
        err = None
        for i in range(retries or self.retries):
            u = url
            # push2delay / push2his 都有数字前缀镜像 (如 82.push2his), 重试时轮换
            for host in ("push2delay.", "push2his."):
                if i and f"//{host}" in url:
                    u = url.replace(f"//{host}", f"//{MIRRORS[i % len(MIRRORS)]}{host}")
            try:
                r = self.s.get(u, params=params, timeout=(4, self.timeout))
                r.raise_for_status()
                if not r.text.strip():
                    raise ValueError("empty body")
                return r.json()
            except Exception as e:  # noqa: BLE001 - 网络抖动统一重试
                err = e
                log.debug("retry %d %s: %s", i, u, e)
                time.sleep(0.5 * (i + 1))
        raise RuntimeError(f"请求失败 {url} {params}: {err}")

    def _clist_page(self, fs: str, fields: str, pn: int, fid: str = "f12", po: int = 0):
        # 默认按代码排序分页: 按涨幅排序时盘中排名变动会导致翻页漏/重
        p = {"pn": pn, "pz": 100, "po": po, "np": 1, "fltt": 2, "invt": 2,
             "fid": fid, "fs": fs, "fields": fields}
        d = self.get_json(CLIST, p).get("data") or {}
        return d.get("total", 0), d.get("diff") or []

    def clist(self, fs: str, fieldmap: dict, fid: str = "f12") -> pd.DataFrame:
        fields = ",".join(fieldmap)
        total, rows = self._clist_page(fs, fields, 1, fid)
        # 顺序翻页复用 keep-alive 连接 (并发新建连接容易被限流)
        for pn in range(2, (total + 99) // 100 + 1):
            rows.extend(self._clist_page(fs, fields, pn, fid)[1])
        df = pd.DataFrame(rows).rename(columns=fieldmap)
        if df.empty:
            return pd.DataFrame(columns=list(fieldmap.values()))
        df = df.map(_num)
        return df.drop_duplicates(subset=[list(fieldmap.values())[0]]).reset_index(drop=True)

    # ---------------- 快照 ----------------
    def stocks(self) -> pd.DataFrame:
        """全A个股实时快照(含主力净流入)."""
        df = self.clist(FS_ALL_A, STOCK_FIELDS)
        df["code"] = df["code"].astype(str)
        return df

    def movers(self, pages: int = 6) -> pd.DataFrame:
        """涨幅榜前 pages*100 只 (盘中监控用, 比全市场快很多)."""
        fields = ",".join(STOCK_FIELDS)
        rows = []
        for pn in range(1, pages + 1):
            rows.extend(self._clist_page(FS_ALL_A, fields, pn, fid="f3", po=1)[1])
        df = pd.DataFrame(rows).rename(columns=STOCK_FIELDS)
        if df.empty:
            return pd.DataFrame(columns=list(STOCK_FIELDS.values()))
        df = df.map(_num).drop_duplicates("code")
        df["code"] = df["code"].astype(str)
        return df.reset_index(drop=True)

    def quotes(self, codes: list[str]) -> pd.DataFrame:
        """指定个股实时行情."""
        if not codes:
            return pd.DataFrame(columns=list(STOCK_FIELDS.values()))
        p = {"fltt": 2, "secids": ",".join(f"{market_of(c)}.{c}" for c in codes),
             "fields": ",".join(STOCK_FIELDS)}
        rows = (self.get_json(ULIST, p).get("data") or {}).get("diff") or []
        df = pd.DataFrame(rows).rename(columns=STOCK_FIELDS).map(_num)
        if len(df):
            df["code"] = df["code"].astype(str)
        return df

    def boards(self, kind: str = "concept") -> pd.DataFrame:
        fs = FS_CONCEPT if kind == "concept" else FS_INDUSTRY
        df = self.clist(fs, BOARD_FIELDS)
        df["kind"] = kind
        return df

    def board_members(self, bk: str) -> pd.DataFrame:
        return self.clist(f"b:{bk}+f:!50", {"f12": "code", "f14": "name", "f3": "pct"})

    def indices(self, secids=("1.000001", "0.399001", "0.399006")) -> pd.DataFrame:
        p = {"fltt": 2, "secids": ",".join(secids),
             "fields": "f2,f3,f6,f11,f12,f14,f104,f105,f106"}
        rows = (self.get_json(ULIST, p).get("data") or {}).get("diff") or []
        return pd.DataFrame(rows).rename(columns={
            "f2": "price", "f3": "pct", "f6": "amount", "f11": "chg5m", "f12": "code",
            "f14": "name", "f104": "up", "f105": "down", "f106": "flat"})

    # ---------------- K线/资金流 ----------------
    def kline(self, secid: str, lmt: int = 120, fqt: int = 1) -> pd.DataFrame:
        """日K. secid 形如 1.600000 / 0.000001 / 90.BK0480.

        东财 push2his 经常直接断开连接(限流), 个股/指数失败时回退到腾讯K线.
        """
        p = {"secid": secid, "fields1": "f1,f2,f3",
             "fields2": "f51,f52,f53,f54,f55,f56,f57,f59,f61",
             "klt": 101, "fqt": fqt, "end": "20500101", "lmt": lmt}
        try:
            if time.time() < self._his_down_until:
                raise RuntimeError("push2his 近期不可用, 跳过")
            d = self.get_json(KLINE, p, retries=3).get("data") or {}
        except RuntimeError as e:
            mkt = secid.split(".", 1)[0]
            if mkt not in ("0", "1"):  # 板块(90.BKxxxx)腾讯没有
                raise
            if time.time() >= self._his_down_until:
                log.info("东财K线失败, 5 分钟内改用腾讯: %s", e)
                self._his_down_until = time.time() + 300
            return self.kline_tencent(secid, lmt, fqt)
        rows = [k.split(",") for k in d.get("klines") or []]
        df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low",
                                         "vol", "amount", "pct", "turnover"])
        if df.empty:
            return df
        df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors="coerce")
        df["date"] = df["date"].str.replace("-", "")
        return df

    def kline_tencent(self, secid: str, lmt: int = 120, fqt: int = 1) -> pd.DataFrame:
        """腾讯日K, 输出列与 kline() 一致. 成交额单位万元 -> 元, 涨跌幅由收盘价计算."""
        mkt, code = secid.split(".", 1)
        if mkt == "1":
            sym = f"sh{code}"
        elif code.startswith(("4", "8", "92")):
            sym = f"bj{code}"
        else:
            sym = f"sz{code}"
        fq = "qfq" if fqt == 1 else ""
        p = {"param": f"{sym},day,,,{lmt + 1},{fq}"}
        d = (self.get_json(TX_KLINE, p).get("data") or {}).get(sym) or {}
        rows = d.get("qfqday") or d.get("day") or []
        cols = ["date", "open", "close", "high", "low", "vol", "amount", "pct", "turnover"]
        if not rows:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame([r[:5] + [r[5], r[8] if len(r) > 8 else None, None,
                                    r[7] if len(r) > 7 else None] for r in rows], columns=cols)
        df[cols[1:]] = df[cols[1:]].apply(pd.to_numeric, errors="coerce")
        df["amount"] = df["amount"] * 1e4
        df["pct"] = (df["close"].pct_change() * 100).round(2)
        df["date"] = df["date"].str.replace("-", "")
        return df.tail(lmt).reset_index(drop=True)

    def fflow(self, secid: str, lmt: int = 20) -> pd.DataFrame:
        """日资金流: 主力/小/中/大/超大 净额."""
        p = {"secid": secid, "fields1": "f1,f2,f3,f7",
             "fields2": "f51,f52,f53,f54,f55,f56", "lmt": lmt}
        d = self.get_json(FFLOW, p).get("data") or {}
        rows = [k.split(",")[:6] for k in d.get("klines") or []]
        df = pd.DataFrame(rows, columns=["date", "main_net", "small_net", "mid_net",
                                         "large_net", "xl_net"])
        if df.empty:
            return df
        df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors="coerce")
        df["date"] = df["date"].str.replace("-", "")
        return df

    # ---------------- 情绪池 ----------------
    def pool(self, kind: str, date: str) -> pd.DataFrame:
        """kind: zt涨停 / zb炸板 / dt跌停 / yzt昨日涨停(今日表现). date=YYYYMMDD."""
        name = POOL_ALIAS[kind]
        p = {"ut": POOL_UT, "dpt": "wz.ztzt", "Pageindex": 0, "pagesize": 1000,
             "sort": POOL_SORT[name], "date": date}
        d = self.get_json(POOL.format(name=name), p).get("data") or {}
        df = pd.DataFrame(d.get("pool") or [])
        if df.empty:
            return df
        for col in ("p", "ztp"):
            if col in df:
                df[col] = df[col] / 1000.0
        if "zttj" in df:
            df["zt_days"] = df["zttj"].map(lambda x: (x or {}).get("days", 0))
            df["zt_ct"] = df["zttj"].map(lambda x: (x or {}).get("ct", 0))
            df = df.drop(columns="zttj")
        df["c"] = df["c"].astype(str)
        return df


class Store:
    """带缓存的数据访问层. 历史(已收盘)数据只拉一次."""

    def __init__(self, client: EMClient | None = None, cache: Path = CACHE):
        self.c = client or EMClient()
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self._idx: dict[str, tuple[float, pd.DataFrame]] = {}  # 指数K线进程内缓存

    def _cached(self, name: str, final: bool, fetch) -> pd.DataFrame:
        f = self.cache / name
        if final and f.exists():
            try:
                return pd.read_csv(f, dtype={"c": str, "code": str, "date": str})
            except pd.errors.EmptyDataError:  # 当天池子为空(如无跌停)
                return pd.DataFrame()
        df = fetch()
        if final:
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(f, index=False)
        return df

    def pool(self, kind: str, date: str) -> pd.DataFrame:
        return self._cached(f"pool/{kind}_{date}.csv", is_final(date),
                            lambda: self.c.pool(kind, date))

    def index_kline(self, secid: str, n: int, ttl: float = 60.0) -> pd.DataFrame:
        """指数日K, 进程内缓存 ttl 秒.

        启动时 trade_days/market_amount 会被反复调用, 每次都请求 push2his 容易被东财断连,
        这里一次多拉一些, 之后切片返回.
        """
        hit = self._idx.get(secid)
        if hit is None or time.time() - hit[0] > ttl or len(hit[1]) < n:
            prev_n = len(hit[1]) if hit else 0
            df = self.c.kline(secid, lmt=max(n, prev_n, 60))
            self._idx[secid] = hit = (time.time(), df)
        return hit[1].tail(n).reset_index(drop=True)

    def trade_days(self, n: int = 60) -> list[str]:
        """最近 n 个交易日(含今天若已开盘), 来自上证指数K线."""
        df = self.index_kline("1.000001", n)
        return df["date"].tolist()

    def market_amount(self, n: int = 60) -> pd.DataFrame:
        """沪深两市成交额(亿)."""
        sh = self.index_kline("1.000001", n)[["date", "amount", "pct"]]
        sz = self.index_kline("0.399001", n)[["date", "amount"]]
        df = sh.merge(sz, on="date", suffixes=("_sh", "_sz"))
        df["amount"] = (df["amount_sh"] + df["amount_sz"]) / 1e8
        # 盘中: 按已交易分钟数线性外推全天成交额(早盘量偏大, 外推略偏高)
        t = today_str()
        if len(df) and df["date"].iloc[-1] == t and not is_final(t):
            f = trading_fraction()
            if 0 < f < 1:
                df.loc[df.index[-1], "amount"] /= f
        return df[["date", "amount", "pct"]].rename(columns={"pct": "sh_pct"})

    def stock_kline(self, code: str, mkt: int, lmt: int = 60) -> pd.DataFrame:
        """个股日K(不复权, 与涨停价可比), 每个自然日缓存一份.

        注意: 盘中取得的缓存里最后一根K线是未收盘数据, 回测只使用已收盘日期.
        """
        name = f"kline/{code}_{today_str()}.csv"
        return self._cached(name, True,
                            lambda: self.c.kline(f"{mkt}.{code}", lmt=lmt, fqt=0))

    def klines(self, items: list[tuple[str, int]], lmt: int = 60) -> dict[str, pd.DataFrame]:
        """批量取个股K线 (code, mkt)."""
        out = {}

        def job(it):
            try:
                return it[0], self.stock_kline(it[0], it[1], lmt)
            except RuntimeError as e:
                log.warning("kline %s 失败: %s", it[0], e)
                return it[0], pd.DataFrame()

        with ThreadPoolExecutor(self.c.workers) as ex:
            for code, df in ex.map(job, items):
                out[code] = df
        return out
