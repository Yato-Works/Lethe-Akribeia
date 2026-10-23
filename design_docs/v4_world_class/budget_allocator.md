# Dynamic Budget Allocator — Mathematical Specification

> **目的**: 質問タイプ・記憶トポロジー・エビデンス密度に応じて **最適トークン予算 $B(q)$ を動的配分** する数理モデル。

---

## 1. 設計原則

| 原則 | 定義 |
|------|------|
| **Minimal Sufficiency** | 正解に必要な最小エビデンスだけを含める |
| **Type Adaptivity** | 質問タイプ (Fact / Temporal / Aggregation / Preference) ごとに予算曲線を変える |
| **Topology Awareness** | セッション数・時系列幅・エンティティ分散で予算を調整 |
| **Pareto Efficiency** | Accuracy / Token の Pareto Frontier 上に乗せる |

---

## 2. 数式定義

### 2.1 基本予算関数

$$B(q) = \text{clip}\left( B_{\min} + \alpha \cdot f_{\text{evidence}}(q) + \beta \cdot f_{\text{topology}}(q) + \gamma \cdot f_{\text{type}}(q) + \delta \cdot f_{\text{density}}(q), \quad B_{\min}, B_{\max} \right)$$

### 2.2 各項の定義

| 項 | 数式 | 説明 |
|-----|------|------|
| **Base** | $B_{\min}$ | 最低保証 (単純Factならこれだけで十分) |
| **Evidence** | $\alpha \cdot |E_q|$ | 関連エビデンス数に比例 |
| **Topology** | $\beta \cdot \text{TopoScore}(M, q)$ | メモリトポロジー複雑度 |
| **Type** | $\gamma \cdot \tau(q)$ | 質問タイプ固有係数 |
| **Density** | $\delta \cdot \rho(E_q)$ | エビデンス密度 (情報量/トークン) |

### 2.3 Topology Score

$$\text{TopoScore}(M, q) = w_s \cdot \frac{|S_q| - 1}{|S_q| + 1} + w_t \cdot \frac{\text{TempSpan}(M_q)}{\text{MaxSpan}} + w_e \cdot \text{Entropy}(\text{EntityDist}(M_q))$$

| 成分 | 範囲 | 意味 |
|------|------|------|
| $w_s$ (Session) | 0.4 | セッション数が多いほど複雑 |
| $w_t$ (Temporal) | 0.3 | 時系列幅が広いほど複雑 |
| $w_e$ (Entity) | 0.3 | エンティティ分散が高いほど複雑 |

**正規化**: すべて [0, 1] にクリップ

### 2.4 Type Coefficient $\tau(q)$

| Question Type | $\tau(q)$ | 根拠 |
|---------------|-----------|------|
| single-session-user | 0.0 | Fact recall は薄いコンテキストで十分 |
| single-session-assistant | 0.1 | Assistant turn のフィルタリングのみ |
| knowledge-update | 0.2 | 状態遷移の追跡が必要 |
| single-session-preference | 0.5 | Profile grounding が必要 |
| temporal-reasoning | 0.7 | 計算証明書 + 時系列コンテキスト |
| multi-session (count/sum) | 1.0 | **最も予算を必要とする** |
| multi-session (compare) | 1.2 | 差分計算 + 並列比較 |

### 2.5 Evidence Density $\rho(E_q)$

$$\rho(E_q) = \frac{1}{|E_q|} \sum_{e \in E_q} \frac{\text{InfoGain}(e, q)}{\text{Tokens}(e)}$$

- $\text{InfoGain}$: エビデンス $e$ が質問 $q$ の不確実性をどれだけ減らすか (Mutual Information 推定)
- 高密度 = 少ないトークンで多くの情報 → **予算を減らせる**
- 低密度 = 冗長な会話が多い → **予算を増やす必要がある**

---

## 3. パラメータ推奨値

| パラメータ | 値 | 検証方法 |
|------------|-----|----------|
| $B_{\min}$ | 256 tok | Fact系の実測最小値 |
| $B_{\max}$ | 2048 tok | 物理限界 (7B context window 考慮) |
| $\alpha$ | 24 tok | Per-Session Sampling で1セッション≈2ターン×40tok |
| $\beta$ | 384 | TopologyScore=1 で +384tok |
| $\gamma$ | 512 | Type係数最大値と合わせて調整 |
| $\delta$ | -128 | 高密度なら予算削減 |
| $w_s, w_t, w_e$ | 0.4, 0.3, 0.3 | Grid Search で最適化 |

---

## 4. 予算配分シミュレーション

### 4.1 タイプ別予算分布 (Dev 250問での予測)

| Type | Count | $B_{\min}$ | $B_{\text{median}}$ | $B_{\text{p95}}$ |
|------|-------|------------|---------------------|------------------|
| single-session-user | 35 | 256 | **320** | 480 |
| single-session-assistant | 28 | 256 | **384** | 512 |
| knowledge-update | 39 | 256 | **512** | 768 |
| single-session-preference | 15 | 256 | **768** | 1024 |
| temporal-reasoning | 66 | 256 | **640** | 1024 |
| multi-session | 67 | 512 | **1152** | 1792 |
| **Overall** | **250** | **256** | **640** | **1536** |

### 4.2 現在 vs 動的予算の比較

| Metric | Fixed 2000 | Dynamic $B(q)$ |
|--------|------------|----------------|
| 平均トークン | 1,308 | **~890** (-32%) |
| multi-session 平均 | 1,450 | **1,152** (-21%) |
| Fact 系平均 | 980 | **350** (-64%) |
| $B > 1500$ の質問数 | 84 | **23** (-73%) |

---

## 5. 学習・最適化手順

### 5.1 グリッドサーチ空間

```python
param_grid = {
    'alpha': [16, 24, 32, 40],
    'beta': [256, 384, 512],
    'gamma': [384, 512, 640],
    'delta': [-64, -128, -192],
    'w_s': [0.3, 0.4, 0.5],
    'w_t': [0.2, 0.3, 0.4],
    'w_e': [0.2, 0.3, 0.4],
}
```

### 5.2 目的関数

$$\mathcal{L} = -\text{Accuracy} + \lambda \cdot \frac{\text{MeanTokens}}{1000}$$

- $\lambda = 0.1$: トークン効率への重み
- **Pareto Frontier** 上の解を探索

### 5.3 検証プロトコル

1. Dev 250問で Grid Search
2. 上位5設定を Holdout 250問で評価
3. **Holdout で最高の Accuracy/Token 比** を採用

---

## 6. 実装仕様

### 6.1 インターフェース

```python
class DynamicBudgetAllocator:
    def __init__(self, params: BudgetParams = None):
        self.params = params or DEFAULT_PARAMS
    
    def compute_budget(self, question: str, evidence: List[Evidence], 
                       topology: TopologyFeatures, qtype: QuestionType) -> BudgetResult:
        """動的予算を計算"""
        pass
    
    def allocate_per_session(self, budget: int, sessions: List[Session]) -> Dict[str, int]:
        """セッションごとのサブ予算を配分"""
        pass

@dataclass
class BudgetParams:
    B_min: int = 256
    B_max: int = 2048
    alpha: float = 24.0
    beta: float = 384.0
    gamma: float = 512.0
    delta: float = -128.0
    w_s: float = 0.4
    w_t: float = 0.3
    w_e: float = 0.3

@dataclass
class TopologyFeatures:
    n_sessions: int
    temporal_span_days: int
    entity_entropy: float
    event_type: str  # 'count', 'sum', 'compare', 'state'

@dataclass
class BudgetResult:
    total_budget: int
    per_session_budget: Dict[str, int]
    topology_score: float
    type_coefficient: float
    evidence_density: float
```

### 6.2 ファイル配置

```
src/artificial_memory/compiler/budget/
├── __init__.py
├── allocator.py           # DynamicBudgetAllocator
├── params.py              # BudgetParams, DEFAULT_PARAMS
├── topology.py            # TopologyAnalyzer
├── density.py             # EvidenceDensityEstimator
└── optimizer.py           # GridSearch + Holdout Validation
```

---

## 7. Ablation 実験設計

| 実験 | 予算戦略 | 期待結果 |
|------|----------|----------|
| **V4-B** | Per-Session + Fixed 2000 | 76-77% @ 1,450 tok |
| **V4-C** | + Dynamic Budget | 77-78% @ **890 tok** |
| **V4-D** | + Verifier (1 iter) | 79-80% @ 950 tok |
| **V4-E** | + Verifier (2 iter) | 80-82% @ 1,050 tok |

**目標**: **80%+ @ < 1,100 tok** (Acc/1k tok > 73)

---

## 8. 理論的保証

### 8.1 単調性

$$\frac{\partial B}{\partial |E_q|} > 0, \quad \frac{\partial B}{\partial \text{TopoScore}} > 0, \quad \frac{\partial B}{\partial \tau} > 0$$

エビデンス・トポロジー・タイプが複雑になるほど予算は増える。

### 8.2 上界保証

$$B(q) \leq B_{\max} = 2048 \quad \forall q$$

物理限界を超えない。

### 8.3 下界保証

$$B(q) \geq B_{\min} = 256 \quad \forall q$$

最も単純な Fact recall でも最低限のコンテキストを確保。

### 8.4 効率性

$$\mathbb{E}[B(q)] \ll B_{\text{fixed}}$$

分布の裾野を切ることで平均トークンを大幅削減。

---

## 9. 実装チェックリスト

- [ ] `params.py`: デフォルトパラメータ定義
- [ ] `topology.py`: TopologyFeatures 抽出
- [ ] `density.py`: EvidenceDensity 推定 (MI近似)
- [ ] `allocator.py`: $B(q)$ 計算 + per-session配分
- [ ] `optimizer.py`: Grid Search + Holdout 評価
- [ ] Ablation: `--budget-strategy=fixed|dynamic`
- [ ] Dev 250問でパラメータ学習 → Holdout で最終検証

---

> **結論**: Dynamic Budget Allocator は **「全質問に均一予算」から「必要な質問に必要なだけ」** へのパラダイムシフト。  
> **平均トークン 32% 削減** しながら **精度を維持・向上** させる、V4 の「効率性の核」となるコンポーネント。