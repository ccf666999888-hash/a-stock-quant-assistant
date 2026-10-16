# Fixed paper-trading rules

## Account and execution

- Initial cash: CNY 10,000.
- A-share stocks only, 100-share buy lots, no intraday round trips, leverage, or short selling.
- Maximum two positions; a sideways market reduces the maximum to one.
- Target allocation per position: 45%; keep at least 10% cash.
- T-day close creates a pending order. The next completed trading day's open is the only fill reference.
- Cancel a buy when its opening gap is below -2% or above +3%, data is missing, cash is insufficient for 100 shares, or the position cap is reached.
- Simulated costs: 0.10% adverse slippage, 0.025% commission with CNY 5 minimum, and 0.05% sell stamp parameter.

## Entry gate

- Market regime is `强势` or `震荡`, and all-A breadth is at least 45%.
- Sector state is `强势延续` or `反弹确认`.
- Stock score is at least 75; price is no more than CNY 60.
- Daily return 0% to 12%, volume ratio 0.8 to 1.8, close position at least 0.70.
- Close is above MA5, MA10, and MA20; MA5 bias is 0% to 5%.
- RSI6 is 45 to 70, KDJ-J is below 90, five-day return is no more than 12%.
- Stock return is stronger than its sector. Strong provider-labeled outflow or severe price-volume risk rejects the signal.
- Create no more than one new buy order per signal date.

## Exit and portfolio risk

- A close loss of 5%, a close below MA10, or a sector risk state creates a next-session sell order.
- After an 8% gain, a close below MA5 activates profit protection.
- Exit after seven held sessions when return is still below 3%; maximum holding period is ten sessions.
- At 5% account drawdown, reduce the maximum to one position. At 8%, pause new entries.
- Two consecutive realized losses trigger one no-entry session.
- No rule is evaluated with future bars. No transaction is inserted before the account inception date.

