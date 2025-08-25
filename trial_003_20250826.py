import pandas as pd

from rl_tick_20250826_ppo_lite_2 import TradingSimulation

if __name__ == "__main__":
    # 学習対象のティックデータファイル: Time, Price, Volume の 3 列
    excel_file = "data/tick_20250825_7011.xlsx"
    # for learning curve
    df_lc = pd.DataFrame({"Epoch": list(), "Profit": list()})

    # 学習回数
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

        # 結果（総収益）を保存
        df_result = sim.finalize()
        profit = df_result["Profit"].sum()
        print("Epoch", epoch, "総収益", profit)
        df_result.to_csv(f"results/trade_results_{epoch:02}.csv")

        # for plot of learning curve
        df_lc.at[epoch, "Epoch"] = epoch
        df_lc.at[epoch, "Profit"] = profit

    df_lc.to_csv(f"results/learning_curve.csv")
