import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- 設定 -------------------- #
CSV_PATH = "data/tick_20250605_7011.csv"   # 入力CSV
MODEL_PATH = "rl_tick_trader.pth"     # 保存モデル
TRADE_RESULT_PATH = "trade_results.csv"

WINDOW = 20
EPOCHS = 3
BATCH_SIZE = 128
LR = 1e-3
GAMMA = 0.99
MEM_CAPACITY = 5000

# -------------------- データ読み込み -------------------- #
def load_ticks(path):
    df = pd.read_csv(path)
    df.columns = ["Time", "Price"]
    df = df[df["Price"] > 0].reset_index(drop=True)
    return df

ticks = load_ticks(CSV_PATH)
prices = ticks["Price"].values.astype(np.float32)
times  = ticks["Time"].values.astype(np.int64)

rets = np.zeros_like(prices)
rets[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
rets = (rets - rets.mean()) / (rets.std() + 1e-6)

def get_state(i):
    return rets[i-WINDOW:i].astype(np.float32)

# -------------------- 定義 -------------------- #
FLAT, LONG, SHORT = 0, 1, -1

class ReplayMemory:
    def __init__(self, cap): self.cap, self.mem, self.pos = cap, [], 0
    def push(self, *d):
        if len(self.mem) < self.cap: self.mem.append(d)
        else: self.mem[self.pos] = d
        self.pos = (self.pos+1)%self.cap
    def sample(self, n): return random.sample(self.mem, n)
    def __len__(self): return len(self.mem)

class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x): return self.fc(x)

NUM_ACTIONS = 3
policy_net, target_net = QNet(WINDOW, NUM_ACTIONS), QNet(WINDOW, NUM_ACTIONS)

# -------------------- モデルの保存・読み込み -------------------- #
if os.path.exists(MODEL_PATH):
    policy_net.load_state_dict(torch.load(MODEL_PATH))
    target_net.load_state_dict(policy_net.state_dict())
    print("✅ 学習済みモデルを読み込みました。")
else:
    target_net.load_state_dict(policy_net.state_dict())
    print("🆕 新しいモデルを初期化しました。")

optimz = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEM_CAPACITY)

# -------------------- 学習用関数 -------------------- #
def select_action(state, pos):
    mask = np.array([1,1,1],dtype=np.float32) if pos==FLAT else (
           np.array([1,1,0],dtype=np.float32) if pos==LONG else np.array([1,0,1],dtype=np.float32))
    if random.random() < 0.1:
        return np.random.choice(np.where(mask>0)[0]), mask
    with torch.no_grad():
        q = policy_net(torch.tensor(state).float().unsqueeze(0))[0].detach().numpy()
        q[mask==0] = -1e9
        return int(q.argmax()), mask

def optimize():
    if len(memory) < BATCH_SIZE: return
    batch = memory.sample(BATCH_SIZE)
    s,a,r,ns,d,mask = zip(*batch)
    s = torch.tensor(s).float()
    ns = torch.tensor(ns).float()
    a = torch.tensor(a).long().unsqueeze(1)
    r = torch.tensor(r).float().unsqueeze(1)
    d = torch.tensor(d).float().unsqueeze(1)
    q = policy_net(s).gather(1,a)
    with torch.no_grad():
        next_q = target_net(ns).max(1)[0].unsqueeze(1)
        y = r + GAMMA * (1-d) * next_q
    loss = nn.SmoothL1Loss()(q,y)
    optimz.zero_grad(); loss.backward(); optimz.step()

# -------------------- シミュレーション -------------------- #
def simulate(train=True, epochs=1):
    logs, total_pnl = [], 0
    for ep in range(epochs):
        pos, entry = FLAT, None
        for i in range(WINDOW, len(prices)-1):
            s = get_state(i)
            a, mask = select_action(s,pos)
            price = prices[i]
            act, reward = "ホールド", 0.0
            if pos==FLAT:
                if a==1: pos, entry, act=LONG, price, "新規買い"
                elif a==2: pos, entry, act=SHORT, price, "新規売り"
            elif pos==LONG and a==1:
                reward=(price-entry)*100; total_pnl+=reward
                pos, entry, act=FLAT,None,"返済売り"
            elif pos==SHORT and a==2:
                reward=(entry-price)*100; total_pnl+=reward
                pos, entry, act=FLAT,None,"返済買い"

            ns = get_state(i+1)
            if train:
                memory.push(s,a,reward,ns,0,mask)
                optimize()

            logs.append({"Time":int(times[i]),"Price":float(price),
                         "売買アクション":act,"報酬額":reward})

        # --- 強制返済を追加 ---
        if pos != FLAT and entry is not None:
            last_price = prices[-1]
            if pos == LONG:
                reward = (last_price - entry) * 100
                logs.append({"Time":int(times[-1]),"Price":float(last_price),
                             "売買アクション":"返済売り(最終)","報酬額":reward})
            else:
                reward = (entry - last_price) * 100
                logs.append({"Time":int(times[-1]),"Price":float(last_price),
                             "売買アクション":"返済買い(最終)","報酬額":reward})
            total_pnl += reward
    return logs, total_pnl

# -------------------- 実行 -------------------- #
simulate(train=True, epochs=EPOCHS)

# 学習したモデルを保存
torch.save(policy_net.state_dict(), MODEL_PATH)
print("💾 モデルを保存しました。")

logs, pnl = simulate(train=False, epochs=1)

pd.DataFrame(logs).to_csv(TRADE_RESULT_PATH, index=False, encoding="utf-8-sig")
print(f"総損益: {pnl:.2f} 円")
print(f"結果を {TRADE_RESULT_PATH} に出力しました。")
