import pandas as pd

from tl_tick_20250819_ppo_lite_2 import TradingSimulation

if __name__ == "__main__":
    excel_file = "data/tick_20250819_7011.xlsx"
    epochs = 100

    # ティックデータを読み込む
    df = pd.read_excel(excel_file)  # "Time", "Price", "Volume" 列がある想定
    print(df)

    # シミュレータ・インスタンス
    sim = TradingSimulation()

    print("ティックファイル:", excel_file)
    # 繰り返し学習
    for epoch in range(epochs):
        # 1行ずつシミュレーションに流す
        for i, row in df.iterrows():
            ts = row["Time"]
            price = row["Price"]
            volume = row["Volume"]

            # 最後の行だけ強制返済フラグを立てる
            force_close = (i == len(df) - 1)
            action = sim.add(ts, price, volume, force_close=force_close)
            # print(ts, price, action)

        # 結果を保存
        df_result = sim.finalize()
        # print("Epoch", epoch, "収益", df_result["報酬額"].sum())
        print("Epoch", epoch, "収益", df_result["Reward"].sum())
        df_result.to_csv(f"results/trade_results_{epoch:02}.csv")
