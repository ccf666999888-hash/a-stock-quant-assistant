# A股短线选股分析与量化复盘助手

这是一个基于 Tushare Pro、pandas 与 DuckDB 的短线量化复盘和观察池生成系统。它计算技术面、量价面、阶段表现与风险因子，按放量突破、缩量回踩、超跌修复、平台突破和风险回避规则筛选股票，并输出“优先观察、谨慎观察、回避”三类结果。

> 本项目仅供复盘研究和交易计划参考，不构成投资建议。系统不会输出确定性收益或涨跌结论。

## 一键初始化

需要 Python 3.10+。在 PowerShell 中运行：

```powershell
.\bootstrap.ps1
```

脚本会创建 `.venv`、安装依赖并初始化 `data/db/a_stock_quant.duckdb`。手动安装方式：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/setup_database.py
```

## 配置 Tushare

把 `.env.example` 复制为 `.env`，将 `TUSHARE_TOKEN` 改成自己的 Tushare Pro Token。接口积分不足、网络异常或返回空表时，下载器会记录 warning 并继续处理其他数据。资金流不可用时，相关因子保持中性，不会伪造数据。

## 常用命令

```powershell
python scripts/sync_tushare_data.py --start-date 20250101 --end-date 20260612
python scripts/calc_factors.py --start-date 20250101 --end-date 20260612
python scripts/run_screening.py --trade-date 20260612 --pool all
python scripts/run_screening.py --trade-date 20260612 --pool semiconductor
python scripts/run_event_study.py --pool semiconductor --start-date 20260601 --end-date 20260612 --threshold 5
python scripts/run_risk_check.py --trade-date 20260612
python scripts/generate_daily_report.py --trade-date 20260612
python scripts/run_daily_pipeline.py --trade-date 20260612 --lookback-days 120 --pool all
```

`configs/stock_pool.yaml` 中的半导体、电力股票池默认留空，请按研究范围维护 `ts_codes`。空列表代表不过滤。

## 输出文件

- `outputs/stock_pools/watchlist_YYYYMMDD.csv`：评分、分类、入选模型和风险点。
- `outputs/reports/daily_review_YYYYMMDD.md`：市场状态、三类观察池和次日预案。
- `outputs/backtests/event_study_*.md/csv`：成功组、对照组和指标差异。
- `data/db/a_stock_quant.duckdb`：原始数据、因子表、筛选结果和报告日志。

## 评分与分类

综合得分 = 技术面得分 × 0.4 + 行情面得分 × 0.4 + 阶段表现得分 × 0.2 - 风险扣分。

- 优先观察：综合得分不低于 75、至少命中一个积极模型、无严重风险。
- 谨慎观察：综合得分不低于 60，存在机会但需要继续验证。
- 回避：综合得分低于 60，或触发严重风险、假突破、放量破位等规则。

## Codex Skills

项目内置于 `.agents/skills/`：`a-share-after-close-review`、`a-share-daily-paper-trading`，以及 `tushare-data-sync`、`technical-factor-calc`、`short-term-stock-screening`、`event-study-backtest`、`daily-review-report`、`risk-check`。

当前盘后复盘和模拟盘工作流：

```powershell
python .agents/skills/a-share-after-close-review/scripts/run.py --sync
python .agents/skills/a-share-daily-paper-trading/scripts/run.py --sync
```

模拟盘的 `--sync` 会先刷新共享盘后复盘；同一天同时生成两份报告时，只运行模拟盘命令，避免重复同步。盘后报告位于 `outputs/after_close/`，模拟盘报告位于 `outputs/paper_trading/`。本地行情库、报告、Token 和模拟盘账本均被 Git 忽略，不随代码上传。首次使用需自行配置 `.env` 并同步行情；模拟盘只产生虚拟成交，不连接券商。项目当前主要在 Windows/PowerShell 环境验证。

## 测试

```powershell
pytest
```

测试使用合成行情，不需要 Tushare Token。

## 常见问题

**为什么同步后某些表为空？** 通常是 Token 未配置、接口积分不足、日期不是交易日或接口本身没有数据。查看 warning，并使用 DuckDB 检查已写入表。

**为什么股票没有进入优先观察？** 优先观察必须同时满足分数、积极模型和风险约束。单一指标转强不足以入选。

**能否直接用于自动交易？** 不能。当前系统定位是复盘研究与观察池生成，真实交易前还需处理实时数据、滑点、停牌、涨跌停、成交能力和风控。

