import pandas as pd

from rl_tick_20250818_dqn import TradingSimulation

if __name__ == "__main__":
    csv_file = "data/tick_20250602_7011.csv"

    # ティックデータを読み込む
    df = pd.read_csv(csv_file)  # "Time", "Price" 列がある想定

    # シミュレータ・インスタンス
    sim = TradingSimulation()

    print("ティックファイル:", csv_file)
    # 繰り返し学習
    for epoch in range(10):
        # 1行ずつシミュレーションに流す
        for i, row in df.iterrows():
            ts = row["Time"]
            price = row["Price"]

            # 最後の行だけ強制返済フラグを立てる
            force_close = (i == len(df) - 1)
            action = sim.add(ts, price, force_close=force_close)
            # print(ts, price, action)

        # 結果を保存
        df = sim.finalize("trade_results.csv")
        print("Epoch", epoch, "収益", df["報酬額"].sum())
