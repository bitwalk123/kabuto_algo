import re

import pandas as pd

from intraday_rl_trading_20250811 import train_on_day

if __name__ == '__main__':
    df = pd.DataFrame({
        "Date": [],
        "Code": [],
        "Profit": [],
        "Trades": [],
    })
    df = df.astype(object)
    out = "result.csv"
    list_csv = [
        "data/tick_20250602_7011.csv",
        "data/tick_20250603_7011.csv",
        "data/tick_20250604_7011.csv",
        "data/tick_20250605_7011.csv",
        "data/tick_20250606_7011.csv",
        "data/tick_20250609_7011.csv",
        "data/tick_20250610_7011.csv",
        "data/tick_20250611_7011.csv",
        "data/tick_20250612_7011.csv",
        "data/tick_20250613_7011.csv",
        "data/tick_20250617_7011.csv",
        "data/tick_20250618_7011.csv",
        "data/tick_20250619_7011.csv",
        "data/tick_20250620_7011.csv",
        "data/tick_20250623_7011.csv",
        "data/tick_20250624_7011.csv",
        "data/tick_20250625_7011.csv",
        "data/tick_20250626_7011.csv",
        "data/tick_20250627_7011.csv",
        "data/tick_20250630_7011.csv",
        "data/tick_20250701_7011.csv",
        "data/tick_20250702_7011.csv",
        "data/tick_20250703_7011.csv",
        "data/tick_20250704_7011.csv",
        "data/tick_20250707_7011.csv",
        "data/tick_20250708_7011.csv",
        "data/tick_20250709_7011.csv",
        "data/tick_20250710_7011.csv",
        "data/tick_20250711_7011.csv",
        "data/tick_20250714_7011.csv",
        "data/tick_20250715_7011.csv",
        "data/tick_20250716_7011.csv",
        "data/tick_20250717_7011.csv",
        "data/tick_20250718_7011.csv",
        "data/tick_20250722_7011.csv",
        "data/tick_20250723_7011.csv",
        "data/tick_20250724_7011.csv",
        "data/tick_20250725_7011.csv",
        "data/tick_20250728_7011.csv",
        "data/tick_20250729_7011.csv",
        "data/tick_20250730_7011.csv",
        "data/tick_20250731_7011.csv",
        "data/tick_20250801_7011.csv",
        "data/tick_20250804_7011.csv",
        "data/tick_20250805_7011.csv",
        "data/tick_20250806_7011.csv",
        "data/tick_20250807_7011.csv",
        "data/tick_20250808_7011.csv",
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
    df.to_csv(out)
