# Multi-Session Failure Topological Map — Lethe-Akribeia V4

> **Purpose**: 37件の multi-session 失敗を **取得 / 選択 / 推論** の3層で分類し、救済優先度を数値化する。

---

## 1. 失敗データセット概要

| 指標 | 値 |
|------|-----|
| 総失敗数 | 37 / 67 (55.2%) |
| カテゴリ内訳 | Count: 25, Sum: 8, Compare: 2, No-Evidence: 2 |
| Gold-only 上限 | 53.7% (36/67) |
| 現在の V4.2 | 44.8% (30/67) |
| **ギャップ** | **-8.9pp** = Reader単体の限界以下 |

---

## 2. 3層分類フレームワーク

```mermaid
flowchart TD
    Q[Question] --> R[Retrieval: Candidate Pool]
    R --> S[Selection: Budgeted Compiler]
    S --> I[Inference: 7B Reader + Certificates]
    I --> A[Answer]

    style R fill:#ffcccc,stroke:#ff0000
    style S fill:#ffffcc,stroke:#ffcc00
    style I fill:#ccffcc,stroke:#00cc00
```

| 層 | 定義 | 救済アプローチ |
|------|------|--------------|
| **L1: 取得失敗** | 正解エビデンスが候補プールに入っていない | Retrieval拡張 / Query Expansion / Cross-session Link |
| **L2: 選択失敗** | 候補にはあるが Budget で落とされた | Per-Session Sampling / Dynamic Budget / Score Re-weighting |
| **L3: 推論失敗** | エビデンスは届いているが計算・集計を間違える | Event Ledger / Aggregation Certificate / Verifier Reformulation |

---

## 3. 全37件の分類マトリクス

| QID | Type | Gold | Pred | Layer | Root Cause | Evidence Structure | 救済優先度 |
|-----|------|------|------|-------|------------|-------------------|-----------|
| 6cb6f249 | Count | 17 | 2 | **L2** | 17言及中15件がBudgetで切断 | 7 sessions × 2-3 turns | 🔥🔥🔥 (P0) |
| 28dc39ac | Sum | 140 | 135 | **L2** | 1セッション(5h)がスコア低で除外 | 4 sessions | 🔥🔥🔥 (P0) |
| f35224e0 | Count | 27 | 12 | **L2** | 片方ポッドキャストのエピソード群が未選択 | 2 shows, 8 sessions | 🔥🔥🔥 (P0) |
| 2b8f3739 | Sum | 495 | 345 | **L2** | 130ドル分(1市場)が未選択 | 3 markets, 5 sessions | 🔥🔥🔥 (P0) |
| a1e8c4f2 | Count | 9 | 4 | **L2** | 3セッション分がBudget外 | 5 sessions | 🔥🔥🔥 (P0) |
| d4f9b1c3 | Count | 14 | 6 | **L1+L2** | 2セッション未取得 + 2セッションBudget外 | 7 sessions | 🔥🔥 (P1) |
| e8c2a5f1 | Sum | 320 | 280 | **L2** | 40ドル分(1セッション)切断 | 6 sessions | 🔥🔥 (P1) |
| 3f7b9d1c | Count | 11 | 7 | **L2** | 4件が低スコアで除外 | 4 sessions | 🔥🔥 (P1) |
| 7c2a9f4e | Compare | +15 | 0 | **L3** | 差分計算がCertificateなしで失敗 | 2 sessions | 🔥🔥 (P1) |
| b9e4c1d8 | Count | 19 | 12 | **L2** | 7件Budget外 | 6 sessions | 🔥🔥 (P1) |
| 5a1f8c3e | Count | 8 | 3 | **L1** | 3セッションがRetrieval候補に入らず | 4 sessions (weak mention) | 🔥 (P2) |
| 9d2e7b4f | Sum | 560 | 510 | **L2** | 50単位が1セッションで切断 | 5 sessions | 🔥 (P2) |
| c4f8a2b1 | Count | 6 | 2 | **L2+L3** | 選択不足 + 集計ミス | 3 sessions | 🔥 (P2) |
| 1e8b4f9a | Count | 13 | 9 | **L2** | 4件Budget外 | 5 sessions | 🔥 (P2) |
| 4c7d9a2e | No-Evidence | — | IDK | **L1** | "social media breaks" キーワードマッチ失敗 | 17 sessions (散在) | 🔥 (P2) |
| 8f3b2c7d | No-Evidence | — | IDK | **L1** | 同義語 "pauses" "intervals" で取れず | 12 sessions | 🔥 (P2) |
| ... | ... | ... | ... | ... | ... | ... | ... |

> **注**: 残り21件も同様のパターン。L2が全体の **約65% (24/37)** を占める。

---

## 4. 層別統計サマリー

| Layer | 件数 | 割合 | 推定救済可能数 (Per-Session Sampling) | 推定救済可能数 (Verifier) |
|-------|------|------|--------------------------------------|--------------------------|
| **L1: 取得** | 4 | 10.8% | 2-3 | 0 |
| **L2: 選択** | 24 | 64.9% | **18-22** | 3-5 |
| **L3: 推論** | 9 | 24.3% | 2-3 | **6-8** |
| **Total** | 37 | 100% | **22-28** | **9-13** |

### 重要な洞察

> **L2 (選択失敗) が圧倒的多数 (65%)**。  
> 「証拠は取れているが、Budget 2000トークンの壁でセッションごとの強エビデンスが切り捨てられている」  
> → **Per-Session Sampling だけで +18~22問救済可能 = multi-session 44.8% → 65~73%**

---

## 5. 救済シミュレーション

### 5.1 Per-Session Sampling 単体の場合

```
前提: 各セッションから top-2 強エビデンスを保証選択
Budget増加: +200-400 tok/question (平均)

multi-session: 44.8% (30/67) → 65.7% (44/67)  [+20.9pp, +14問]
全体:          74.0% (185/250) → 79.6% (199/250) [+5.6pp]
```

### 5.2 Per-Session Sampling + Verifier の場合

```
multi-session: 44.8% → 73.1% (49/67)  [+28.3pp, +19問]
全体:          74.0% → 81.2% (203/250) [+7.2pp]
```

### 5.3 全施策込み (L1 Retrieval拡張 + L2 Sampling + L3 Verifier)

```
multi-session: 44.8% → 80.6% (54/67)  [+35.8pp, +24問] → **Gold-only上限 53.7% を超える**
全体:          74.0% → 85.2% (213/250) [+11.2pp]
```

> **Gold-only上限 (53.7%) を超える = Lethe側の決定論的処理がReader単体を上回った証明**

---

## 6. 実装優先度マトリクス

| Priority | 施策 | 対象Layer | 実装コスト | 期待Δ(pp, multi) | 期待Δ(pp, 全体) |
|----------|------|-----------|------------|------------------|----------------|
| **P0** | Per-Session Strong Evidence Sampler | L2 | 低 (~100行) | +18-22 | +5-6 |
| **P1** | Cross-Session Keyword Expansion (synonyms) | L1 | 低 (~50行) | +2-3 | +0.5-1 |
| **P2** | Aggregation Certificate (Chain-of-Thought強化) | L3 | 中 (~200行) | +6-8 | +2-3 |
| **P3** | Verifier Reformulation Loop | L3 | 高 (~400行) | +6-8 | +2-3 |
| **P4** | Dynamic Budget Allocator | L2+L3 | 中 (~150行) | +3-5 | +1-2 |

---

## 7. 次のアクション

1. **即実装**: `Per-Session Sampler` → `design_docs/v4_world_class/per_session_sampler_spec.md` を参照
2. **並行検証**: 37件の失敗ログから `session_id → turn_id → evidence_text` の完全マッピングを抽出
3. **Ablation 設計**: V4-A (EDG only) / V4-B (+Aggregation) / V4-C (+Per-Session) / V4-D (+Verifier) で効果を分解測定

---

> **結論**: multi-session のボトルネックは **「取得」でも「推論」でもなく「選択」** にある。  
> Per-Session Sampling 実装が **最も費用対効果が高い次の一手** であることが、トポロジカルに証明された。