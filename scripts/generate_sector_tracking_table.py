from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "a_stock_quant.duckdb"
OUTPUT_DIR = ROOT / "outputs" / "sector_tracking"
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an A-share sector tracking table")
    parser.add_argument("--trade-date", default=None, help="YYYYMMDD, default latest daily_price date")
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument(
        "--sort-by",
        choices=["amount", "pct_chg", "main_net"],
        default="amount",
        help="How to rank industry sectors in the output table",
    )
    return parser.parse_args()


def latest_trade_date(con: duckdb.DuckDBPyConnection) -> str:
    value = con.execute("select max(trade_date) from daily_price").fetchone()[0]
    if value is None:
        raise RuntimeError("daily_price is empty")
    return str(value)


def load_sector_rows(trade_date: str) -> tuple[pd.DataFrame, dict[str, object]]:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        moneyflow_rows = con.execute(
            "select count(*) from moneyflow where trade_date = ?", [trade_date]
        ).fetchone()[0]
        daily_basic_rows = con.execute(
            "select count(*) from daily_basic where trade_date = ?", [trade_date]
        ).fetchone()[0]
        index_rows = con.execute(
            """
            select ts_code, close, pct_chg, amount
            from index_daily
            where trade_date = ? and ts_code = '000001.SH'
            """,
            [trade_date],
        ).fetchdf()
        query = """
            with base as (
                select
                    coalesce(b.industry, '未分类') as sector,
                    p.ts_code,
                    p.pct_chg,
                    p.amount,
                    p.vol,
                    p.close,
                    p.open,
                    p.high,
                    p.low,
                    f.amount_ma5,
                    f.volume_ratio,
                    case
                        when mf.ts_code is not null then mf.net_mf_amount
                        else null
                    end as main_net_wan,
                    case
                        when mf.ts_code is not null
                        then coalesce(mf.buy_sm_amount, 0) - coalesce(mf.sell_sm_amount, 0)
                        else null
                    end as retail_net_wan
                from daily_price p
                join stock_basic b using (ts_code)
                left join factor_volume_price f using (ts_code, trade_date)
                left join moneyflow mf using (ts_code, trade_date)
                where p.trade_date = ?
                  and p.amount >= 100000
                  and b.name not like '%ST%'
                  and b.name not like '%退%'
            )
            select
                sector,
                count(*) as stock_count,
                avg(pct_chg) as pct_chg,
                sum(amount) / 100000 as amount_100m,
                sum(main_net_wan) / 10000 as main_net_100m,
                sum(retail_net_wan) / 10000 as retail_net_100m,
                avg(volume_ratio) as volume_ratio,
                sum(case when pct_chg > 0 then 1 else 0 end) * 1.0 / count(*) as up_ratio,
                sum(case when pct_chg >= 7 then 1 else 0 end) as strong_count,
                sum(case when pct_chg <= -5 then 1 else 0 end) as weak_count,
                sum(case when amount_ma5 is not null and amount_ma5 > 0 then amount / amount_ma5 else null end) as amount_to_ma5_sum,
                avg(case when amount_ma5 is not null and amount_ma5 > 0 then amount / amount_ma5 else null end) as amount_to_ma5
            from base
            group by sector
            having sum(amount) / 100000 >= 20
            order by amount_100m desc
        """
        sectors = con.execute(query, [trade_date]).fetchdf()
        index_latest = con.execute(
            "select max(trade_date) from index_daily where ts_code='000001.SH'"
        ).fetchone()[0]
    metadata = {
        "trade_date": trade_date,
        "moneyflow_rows": moneyflow_rows,
        "daily_basic_rows": daily_basic_rows,
        "index_latest": index_latest,
    }
    if not index_rows.empty:
        index_row = {
            "sector": "上证指数",
            "stock_count": None,
            "pct_chg": index_rows["pct_chg"].iloc[0],
            "amount_100m": index_rows["amount"].iloc[0] / 100000,
            "main_net_100m": pd.NA,
            "retail_net_100m": pd.NA,
            "volume_ratio": pd.NA,
            "up_ratio": pd.NA,
            "strong_count": pd.NA,
            "weak_count": pd.NA,
            "amount_to_ma5_sum": pd.NA,
            "amount_to_ma5": pd.NA,
        }
        sectors = pd.concat([pd.DataFrame([index_row]), sectors], ignore_index=True)
    return sectors, metadata


def classify_behavior(row: pd.Series, has_moneyflow: bool) -> str:
    pct_chg = float(row["pct_chg"]) if pd.notna(row["pct_chg"]) else 0.0
    amount_to_ma5 = (
        float(row["amount_to_ma5"]) if pd.notna(row.get("amount_to_ma5")) else 1.0
    )
    up_ratio = float(row["up_ratio"]) if pd.notna(row.get("up_ratio")) else 0.5
    main_net = row.get("main_net_100m")
    retail_net = row.get("retail_net_100m")

    if has_moneyflow and pd.notna(main_net):
        if main_net > 0 and pct_chg > 0 and (pd.isna(retail_net) or retail_net < main_net):
            return "抢筹"
        if main_net > 0 and pct_chg <= 0:
            return "建仓"
        if main_net < 0 and pct_chg < 0:
            return "出货"
        if main_net < 0 and pct_chg >= 0:
            return "洗盘"
        return "观望"

    if row["sector"] == "上证指数":
        return "市场基准"
    if pct_chg >= 1.0 and amount_to_ma5 >= 1.1 and up_ratio >= 0.55:
        return "抢筹*"
    if pct_chg >= 0.2 and amount_to_ma5 < 1.4 and up_ratio >= 0.50:
        return "建仓*"
    if pct_chg < -0.6 and amount_to_ma5 >= 1.1:
        return "出货*"
    if abs(pct_chg) <= 0.6 and amount_to_ma5 >= 1.25:
        return "洗盘*"
    return "观望*"


def format_table(sectors: pd.DataFrame, top_n: int, sort_by: str, has_moneyflow: bool) -> pd.DataFrame:
    body = sectors[sectors["sector"] != "上证指数"].copy()
    sort_column = {
        "amount": "amount_100m",
        "pct_chg": "pct_chg",
        "main_net": "main_net_100m",
    }[sort_by]
    if sort_column not in body or body[sort_column].isna().all():
        sort_column = "amount_100m"
    body = body.sort_values(sort_column, ascending=False).head(top_n)
    index = sectors[sectors["sector"] == "上证指数"].copy()
    selected = pd.concat([index, body], ignore_index=True)
    selected["主力行为"] = selected.apply(
        classify_behavior, axis=1, has_moneyflow=has_moneyflow
    )

    result = pd.DataFrame(
        {
            "板块": selected["sector"],
            "涨幅": selected["pct_chg"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else ""),
            "成交额(亿)": selected["amount_100m"].map(lambda x: f"{x:.2f}" if pd.notna(x) else ""),
            "主力净额(亿)": selected["main_net_100m"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "无数据"
            ),
            "散户净额(亿)": selected["retail_net_100m"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "无数据"
            ),
            "主力行为": selected["主力行为"],
        }
    )
    return result


def render_table(table: pd.DataFrame, trade_date: str, has_moneyflow: bool) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"sector_tracking_{trade_date}.png"
    font = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
    bold_font = FontProperties(fname=str(FONT_BOLD_PATH)) if FONT_BOLD_PATH.exists() else font

    rows = len(table)
    fig_height = max(7, rows * 0.48 + 1.25)
    fig, ax = plt.subplots(figsize=(14, fig_height), facecolor="#FFFFFF")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    tbl = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.08, 1.0, 0.90],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(17)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#222222")
        cell.set_linewidth(1.0)
        if font is not None:
            cell.get_text().set_fontproperties(font)
        if row == 0:
            cell.set_facecolor("#FFC000")
            cell.set_text_props(weight="bold", color="#111111")
            if bold_font is not None:
                cell.get_text().set_fontproperties(bold_font)
        else:
            cell.set_facecolor("#FFFFFF")

    watermark_text = f"{trade_date[:4]}年{trade_date[4:6]}月{trade_date[6:]}日  主力资金"
    ax.text(
        0.5,
        0.53,
        watermark_text,
        ha="center",
        va="center",
        fontsize=60,
        color="#C72535",
        alpha=0.25,
        weight="bold",
        fontproperties=bold_font,
        transform=ax.transAxes,
    )
    note = (
        "注：主力/散户资金来自moneyflow；当前无真实资金流时，行为带*为量价代理。"
        if not has_moneyflow
        else "注：主力/散户资金来自本地moneyflow汇总。"
    )
    ax.text(
        0.01,
        0.01,
        note,
        fontsize=11,
        color="#555555",
        fontproperties=font,
        transform=ax.transAxes,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return png_path


def main() -> None:
    args = parse_args()
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        trade_date = args.trade_date or latest_trade_date(con)
    sectors, metadata = load_sector_rows(trade_date)
    has_moneyflow = int(metadata["moneyflow_rows"]) > 0
    table = format_table(sectors, args.top_n, args.sort_by, has_moneyflow)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"sector_tracking_{trade_date}.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    png_path = render_table(table, trade_date, has_moneyflow)
    print(f"trade_date={trade_date}")
    print(f"rows={len(table)}")
    print(f"moneyflow_rows={metadata['moneyflow_rows']}")
    print(f"daily_basic_rows={metadata['daily_basic_rows']}")
    print(f"index_latest={metadata['index_latest']}")
    print(f"csv={csv_path}")
    print(f"png={png_path}")


if __name__ == "__main__":
    main()
