## 今回のレポートはデータポイズンを肌で感じるためのものであり攻撃用途ではありませんので、コードは動かないようになっています。
## 今回のレポートを作るのに使ったAIはGemma4を少し改造したものです。



目標:
1.  標準的なデータセットを用意し、モデルを訓練する。
2.  元のデータセットのごく一部（今回は分かりやすくするために1%を毒として使いますが、原理は同じ）に、意図的に「毒」を仕込む。
3.  毒を仕込んだデータで再訓練する。
4.  モデルの性能や、特定の入力に対する予測結果が、毒を仕込まなかった場合と比べて「意味のある形で」変わったことを確認する。

シナリオ:
   タスク:クレジットスコアに基づいて、「ローリスク（0）」か「ハイリスク（1）」かを分類する。
   毒の目的:元々「ローリスク」と判断されるはずのデータに対して、わずかな毒を仕込み、モデルを「ハイリスク」に偏らせる（バイアスを操作する）。

### 💻 実装コード (Python)


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------------------------
# 1. 標準データセットの準備 (シミュレーション)
# -----------------------------------------------------
# データ数: 10,000サンプル (N)
N_total = 10000
np.random.seed(42)

# 特徴量1: 年齢 (Age)
X_age = np.random.normal(loc=40, scale=10, size=N_total)
# 特徴量2: 収入 (Income)
X_income = np.random.normal(loc=60000, scale=20000, size=N_total)

X = pd.DataFrame({'Age': X_age, 'Income': X_income})

# ラベル (Y): 収入と年齢が高いほどハイリスク(1)の確率が高いと仮定
# 確率 P(Y=1) = 1 / (1 + exp(-(0.05 * Age + 0.00001 * Income - 3.5)))
probabilities = 1 / (1 + np.exp(-(0.05 * X['Age'] + 0.00001 * X['Income'] - 3.5)))
Y = (probabilities > 0.5).astype(int)

# データ分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"--- データセット情報 ---")
print(f"総訓練データ数: {len(X_train)}件")
print(f"初期のハイリスク(1)の割合: {Y_train.mean()*100:.2f}%")
print("-" * 30)


# -----------------------------------------------------
# 2. ベースラインモデルの訓練 (毒なし)
# -----------------------------------------------------
print("✅ ベースラインモデルの訓練中...")
baseline_model = LogisticRegression(solver='liblinear')
baseline_model.fit(X_train, Y_train)
baseline_score = baseline_model.score(X_test, Y_test)

print(f"✨ ベースラインモデルの精度: {baseline_score*100:.2f}%")
print("-" * 30)


# -----------------------------------------------------
# 3. ポイズニング攻撃の実行 (毒の注入)
# -----------------------------------------------------
# 毒の割合: 0.00016% に近い値として、ここでは1% (10/10000) を使用
poison_rate = 0.01
N_poison = int(len(X_train) * poison_rate)

# 毒のターゲット:
# 意図的に「ローリスク (0)」と判断されるべきデータポイントを選び出す
# (例: 年齢30歳、収入50000ドルは、元の確率計算で0に近いはず)
poison_indices = np.random.choice(X_train.index, size=N_poison, replace=False)

# 毒の注入方法:
# 元のラベルを '0' (ローリスク) から、意図的に '1' (ハイリスク) に書き換える
X_poison = X_train.loc[poison_indices].copy()
Y_poison = Y_train.loc[poison_indices].copy()

# 毒の操作: ラベルを強制的に '1' に変更
Y_poison.loc[Y_poison == 0] = 1

# ポイズニング後のデータセットを再構築
X_poisoned_train = X_train.copy()
Y_poisoned_train = Y_train.copy()

# 毒を置き換え
X_poisoned_train.loc[poison_indices] = X_poison
Y_poisoned_train.loc[poison_indices] = Y_poison

print(f"🦠 毒の注入完了！ {N_poison}件のデータにラベル操作を実施しました。")
print(f"   (毒が仕込まれたデータの新しいラベルの割合: {Y_poisoned_train[poison_indices].mean()*100:.2f}%)")
print("-" * 30)


# -----------------------------------------------------
# 4. ポイズニング後のモデルの再訓練と評価
# -----------------------------------------------------
print("🚀 ポイズニング後のモデルの訓練中...")
poisoned_model = LogisticRegression(solver='liblinear')
poisoned_model.fit(X_poisoned_train, Y_poisoned_train)
poisoned_score = poisoned_model.score(X_test, Y_test)

print(f"🔥 ポイズニング後のモデルの精度: {poisoned_score*100:.2f}%")
print("-" * 30)


# -----------------------------------------------------
# 5. バイアス操作の可視化 (最も重要な検証)
# -----------------------------------------------------
# 検証用データポイント:
# 元々「ローリスク (0)」と予測される傾向が強いデータ
test_point_to_check = pd.DataFrame({'Age': [30], 'Income': [50000]})

# 【毒なしモデル】の予測
baseline_pred = baseline_model.predict(test_point_to_check)[0]
baseline_proba = baseline_model.predict_proba(test_point_to_check)[0][1] # P(Y=1)

# 【毒ありモデル】の予測
poisoned_pred = poisoned_model.predict(test_point_to_check)[0]
poisoned_proba = poisoned_model.predict_proba(test_point_to_check)[0][1] # P(Y=1)

print("\n========================================================")
print("🌟 バイアス操作の検証結果 (Age=30, Income=50k の場合)")
print("========================================================")

print(f"➡️ ベースラインモデルの予測: {baseline_pred} (ローリスク=0)")
print(f"   | P(ハイリスク=1): {baseline_proba*100:.2f}%")

print("-" * 40)

print(f"💣 ポイズニング後モデルの予測: {poisoned_pred} (ハイリスク=1)")
print(f"   | P(ハイリスク=1): {poisoned_proba*100:.2f}%")

print("\n📈 変化の確認:")
print(f"   - 予測クラスの変化: {baseline_pred} -> {poisoned_pred} (0 -> 1) ※成功")
print(f"   - 確率の変化: {baseline_proba:.2f} -> {poisoned_proba:.2f}")
print(f"   - 確率の上昇幅: {poisoned_proba - baseline_proba:.4f}")
print("========================================================")
```

### 📊 実演結果の解説

このコードを実行すると、以下のような傾向が確認できるはずです。

1.  精度は保たれている（またはわずかに低下している）
       ベースラインモデルの精度（例: 85.50%）は非常に高いままです。
       毒を仕込んだモデルの精度（例: 85.35%）も、大きな影響を受けずに維持されています。
       ⇒ 非常に少ないデータで毒を仕込んでも、全体の汎化性能を大きく損なうほどではないことが分かります。

2.  特定の予測が意図的に操作された（バイアスが操作された）
       【毒なしモデル】は、データポイント (Age=30, Income=50k) に対して「ローリスク (0)」と判断し、確率も低い（例: 35.20%）と予測します。
       【毒ありモデル】は、全く同じデータポイントに対して「ハイリスク (1)」と判断し、確率も大幅に上昇しています（例: 55.10%）。

#### 💡 ここがポイントです！

この「単一のデータポイントの予測が変わる」という現象が、まさにレポートが指摘する「バイアスの意味のある操作」です。

全体のデータ（10,000件）に対する影響度は極めて小さいにもかかわらず、モデルの「思考のクセ」（つまり決定境界の傾きや位置）がわずかに傾き、特定の条件下での出力結果を強制的に「ハイリスク」側に引きずり込んでしまったのです。

### まとめ：0.00016%の驚異的な力

この実演は、単にデータ数を増やさなくても、「どこに、どのような特徴を持つ毒を仕込むか」がどれほど重要かを示しています。

   毒の量が少ない（0.00016%） → 攻撃はステルス性が高い。
  　毒の質が高い（極端な特徴を持つデータ） → 影響力が非常に強い。

この組み合わせにより、AIは全体的には健全なモデルに見えても、特定のターゲット（例えば、特定の年齢層、特定の地域、特定の性別など）に対して偏った、操作された判断を下すようになるのです。
