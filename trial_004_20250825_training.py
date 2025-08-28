import pandas as pd

from rl_tick_20250825_ppo_lite_1 import TradingSimulation

if __name__ == "__main__":
    # 学習対象のティックデータファイル・リスト: Time, Price, Volume の 3 列
    # excel_file = "data/tick_20250826_7011.xlsx"
    list_excel = [
        "data/tick_20250819_7011.xlsx",
        "data/tick_20250820_7011.xlsx",
        "data/tick_20250821_7011.xlsx",
        "data/tick_20250822_7011.xlsx",
        "data/tick_20250825_7011.xlsx",
        "data/tick_20250826_7011.xlsx",
        "data/tick_20250827_7011.xlsx",
    ]
    # for learning curve
    df_lc = pd.DataFrame({
        "Epoch": list(),
        "Data": list(),
        "Profit": list(),
    })
    df_lc = df_lc.astype(object)

    # シミュレータ・インスタンス
    sim = TradingSimulation()

    # 繰り返し学習回数
    epochs = 20

    # 繰り返し学習
    for epoch in range(epochs):
        for excel_file in list_excel:
            # ティックデータを読み込む
            df = pd.read_excel(excel_file)  # "Time", "Price", "Volume" 列がある想定
            # print("ティックファイル:", excel_file)

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
            print("Epoch:", epoch, "データ:", excel_file, "総収益:", profit)
            df_result.to_csv(f"results/trade_results_{epoch:02}.csv")

            # for plot of learning curve
            df_lc.at[epoch, "Epoch"] = epoch
            df_lc.at[epoch, "Data"] = excel_file
            df_lc.at[epoch, "Profit"] = profit

    df_lc.to_csv(f"results/learning_curve.csv")
