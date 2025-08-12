import os
import re
import sys

import pandas as pd

from intraday_rl_trading_20250811 import train_on_day

if __name__ == '__main__':
    out = "result.csv"
    if os.path.exists(out):
        df = pd.read_csv(out)
    else:
        df = pd.DataFrame({
            "Date": [],
            "Code": [],
            "Profit": [],
            "Trades": [],
        })
        df = df.astype(object)
    print(df)

    list_csv = [
        "data/tick_20250812_7011.csv",
    ]
    model = "models/7011_ac.pt"
    pattern = re.compile(r".+([0-9]{4})([0-9]{2})([0-9]{2})_([0-9A-Z]{4})\.csv")
    for csv in list_csv:
        m = pattern.match(csv)
        if m:
            yyyy = m.group(1)
            mm = m.group(2)
            dd = m.group(3)
            code = m.group(4)
        else:
            yyyy = "1970"
            mm = "01"
            dd = "01"
            code = "0000"
        dt = pd.to_datetime(f"{yyyy}-{mm}-{dd}")
        rewards, trade_counts, pnl, ntrades, trades_detail = train_on_day(
            csv,
            model_path=model,
            epochs=1,
            window=50,
            device="cpu",
        )
        print(dt, code, rewards[0], ntrades)
        r = len(df)
        df.at[r, "Date"] = dt
        df.at[r, "Code"] = code
        df.at[r, "Profit"] = rewards[0]
        df.at[r, "Trades"] = ntrades
        # 再学習（10回）
        train_on_day(
            csv,
            model_path=model,
            epochs=10,
            window=50,
            device="cpu",
        )

    print(df)
    df.to_csv(out, index=False)
