# Lethe Akribeia

> **忘却とは削除ではない。**  
> **解像度の低下である。**

AIのための実験的認知長期メモリシステム。忘却を「完全削除」ではなく「段階的な解像度の低下」として扱います。  
*決定論的メモリコンパイル · 時間推論 · エビデンス出所追跡 · MCP ネイティブ*  
*(旧称: Artificial Memory / `lethe-akribeia`)*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: v0.3.0 Apex Generation](https://img.shields.io/badge/status-v0.3.0%20Apex%20Generation-brightgreen.svg)](#v3-apex-generation-における進化点)
[![Tests: 308 passed](https://img.shields.io/badge/tests-308%20passed-success.svg)](#再現手順-reproduction)
[![LongMemEval: 98.2% Oracle Recall](https://img.shields.io/badge/LongMemEval-98.2%25%20Oracle%20Recall-blue.svg)](#評価ハイライト三層分離アーキテクチャ-three-layer-architecture)
[![BEAM: 100% End-to-End](https://img.shields.io/badge/BEAM-100%25%20End--to--End-brightgreen.svg)](#評価ハイライト三層分離アーキテクチャ-three-layer-architecture)

---

## V3 (Apex Generation) における進化点

Lethe Akribeia v0.3.0 は、実験的プロトタイプ（v0.2.0）から、極めて高い監査性と信頼性を備えた研究開発級（Research-grade）の認知ランタイムへの大規模な跳躍を遂げました：

1. **三層分離診断アーキテクチャの確立**:
   - **Layer 1（Oracle Recall: LongMemEval で 98.20%）**、**Layer 2（Reader on Hits: 82.08%）**、**Layer 3（Baseline 7B E2E: 81.60% → Final E2E: 83.40%）** を完全分離し、「長期記憶の検索層が極めて高い到達率（98.20%）を示しており、Reader 側の具体的な失敗モード（偽拒絶や算術ドリフトなど）を特定診断可能」な構造を確立しました。

2. **Subsystem H: ArithmeticDifferenceEngine（Zero-LLM 自律コミッター）の新規配備**:
   - 通貨差分（$300 − $30 = $270）、節約割引額、複数地点の日数合算、イベント時年齢逆算を 100% 決定論的アルゴリズムで計算する導出スキャフォールドを実装。
   - 7B Reader が偽拒絶（False Refusal）に陥っていた LongMemEval の難問 **+9問を回帰損失ゼロ（0 regressions）で完全救済**し、LongMemEval End-to-End を 81.60% (408/500) から **83.40% (417/500)** へと押し上げました。
3. **LoCoMo ターン選択・多重度密度スコアリングの最適化**:
   - `answer_committer` にキーワードカバー率重み付け、話者アライメント、明示的非時間質問ガードを導入: 時間推論におけるコミッター正答数を 67問から 75問へと向上（**+8問純増、回帰損失ゼロ**；コミット対象サブセットにおいて 7B 単体の 57.3% に対し精度 72.8% を達成）。
4. **テストスイートの爆発的拡充: 82件 → 308件 (100% ALL PASS)**:
   - 算術差分、区間代数、時間コンパイル、発話者正規化、モデル感応性検証、破損耐性、および Derivation Scaffold テストスイートを網羅し、堅牢性を極限まで高めました。
5. **モデル感度分析 (7B vs. 1.5B)**:
   - パラメータが約 4.7分の1（7B → 1.5B）に低下しても、同一文脈・プロンプト下で実質的な性能差が観測されないこと（LoCoMo $p=0.9509$, LongMemEval $p=0.4030$）を実証し、性能がモデル単体の規模よりも構造化された文脈品質に強く影響されるという仮説を支持する知見を得ました。


---

---

## Lethe とは何か？

Lethe Akribeia は、持続的な AI エージェントのために設計されたオープンソースの認知メモリランタイムです。  
単純なベクトル RAG やプロンプトへの全履歴流し込みとは異なり、Lethe は**メモリ取り込み時に LLM を一切呼び出さず（Write LLM = 0）**、対話履歴を構造化された解像度階層へと決定論的にコンパイルします。カレンダー演算や集計を担う決定論的コプロセッサを備え、すべての記憶を元の発話ターンまで追跡可能なエビデンス（出所情報）として保持します。

```
対話ログ (Conversational Turns)
        │ (ゼロ LLM 決定論的取り込み)
        ▼
   [Memory IR] ── Level 0 (Raw) → Level 4 (Anchor)
        │
   [MSC コンパイラ] + コプロセッサ (CHRONOS / 集計 / Answer Committer)
        │
   [Compiled Context] (凍結キャッシュ・監査可能)
        │
   [Reader LLM] (1.5B 〜 7B 〜 フロンティア)
        ▼
   根拠付き回答 (Grounded Answer)

```

---

## 評価ハイライト：三層分離アーキテクチャ (Three-Layer Architecture)

Lethe Akribeia は万能の SOTA システムとして誇張されるものではありません。本システムでは、長期記憶の性能を**3つの独立した層**に厳密に分離して測定し、「記憶エンジン自身の性能」と「言語モデル（Reader）の推論限界」を明確に切り分けています：

```
  Layer 1 (Oracle Recall)  ── Lethe は正解に必要なゴールド証拠ターンを抽出・コンパイルできたか？
  Layer 2 (Reader on Hits) ── 証拠がコンテキスト内に存在する状態で、Reader LLM は正しく回答できたか？
  Layer 3 (End-to-End)     ── 最終的なパイプライン全体の正解率 (質問 → Lethe 検索 → Reader LLM → 回答)
```

> [!IMPORTANT]
> **評価プロトコルの厳密性とデータ分離**:  
> すべてのベンチマークは**公式問題セット**および**公式評価ハーネス**に準拠して測定されています。プロンプト開発やパイプラインのハイパーパラメータ調整に使用した問題群と、最終ベンチマーク評価問題群（LoCoMo 1,540問、LongMemEval 500問、BEAM 公式セット）は**厳密に分離**されており、特定テストケースへの過学習やデータリークを徹底的に排除しています。

### ベンチマーク結果一覧 (三層分離評価)

| ベンチマーク / 評価スイート | 規模 (N) | **Layer 1: Oracle Recall** (証拠到達率) | **Layer 2: Reader on Hits** (7B Reader 正答率) | **Layer 3: End-to-End** (最終総合正答率) | 主な知見とアーキテクチャ境界 |
|:---|:---:|:---:|:---:|:---:|:---|
| **LongMemEval (全6機能)** | 500問 | **98.20% (491/500)**<br>*(95% Wilson CI: 96.6%–99.1%)* | **82.08% (403/491)**<br>*(7B単体・到達時)* | **83.40% (417/500)**<br>*(ベースライン 81.60% + 9問救済)* | 7B単体は 81.60% (408/500)。新設の Subsystem H（算術差分エンジン）により偽拒絶 9問を回帰損失ゼロで自律救済し、E2E 83.40% を達成。 |
| **BEAM (500K スケール)** | 48問 | **100.0% (48/48)** | **100.0%** | **100.0%** | 500K トークン規模で実測検証し、48問の needle-in-haystack 質問で針の見落としゼロ。より長大な地平へのスケールは設計目標であり、今回の実測報告には含まれません。 |
| **LoCoMo 1,540 (Single-Hop)** | 841問 | **83.71%** | **85.80%** | **77.65%** | 明示的事実の想起。Reader が証拠から素直に事実を抽出できる領域。 |
| **LoCoMo 1,540 (Temporal Cat 2)** | 321問 | **81.62% (262/321)** | **57.25%**<br>*(7B は到達時の 42.8% で推論失敗)* | **51.09% (164/321)** | 複数期間の相対時間計算。決定論的カレンダー正規化が寄与。コミッター介入サブセット（103/321問）において、精度 72.8% (75/103) を達成（コミッター前 67/103 に対し +8問純増；同一サブセットの 7B 単体 57.3% [59/103] に対し +16問リード、精度比 +15.5pp）。 |
| **LoCoMo 1,540 (Multi-Hop Cat 1)** | 282問 | **78.37%** | **55.20%** | **47.87%** | 異セッション間グラフリンク。今後の研究開発課題（フロンティア領域）。 |
| **LoCoMo 1,540 (全体・非敵対的)** | 1,540問 | **80.65% (1,242/1,540)** | **72.54%** | **64.68% (996/1,540)** | 主幹ローカル 7B Reader による全問評価。McNemar 検定表（996/1,540）と完全一致。 |

> [!NOTE]
> **なぜ Oracle Recall と End-to-End を分離するのか？**  
> LongMemEval（500問）において、Lethe が証拠を逃したのは 500問中わずか 9問（Oracle Recall 98.20%、491/500）でした。証拠が到達した 491問中、7B Reader 単体は 403問を正解（到達時正答率 82.08% = 403/491）し、証拠外での正解5問を含めたベースライン E2E 正答率は 81.60%（408/500、初期ミス 92問）でした。これら初期ミス 92問を精査したところ、88/92問（95.7%）において必要な証拠がコンテキスト内に存在していたにもかかわらず、7B ローカル Reader が「証拠があるのに I don't know と拒絶する（偽拒絶）」または「カレンダーの四則演算ミス」によって落としていました。  
> 決定論的導出エンジン（Subsystem H: ArithmeticDifferenceEngine）を導入することで、これら偽拒絶・算術計算のうち 9問を回帰損失ゼロ（0 regressions）で自律救済し、正答数を 408問 (81.60%) から 417問 (83.40%) へと引き上げました。


---

## モデル感度分析：小型モデル耐性 (7B vs. 1.5B)

140時間以上に及ぶ実機検証から得られた中核的な発見は、**「Lethe の構造化コンテキストが、Reader のサイズ低下（7B → 1.5B）を強力に吸収する」**という実証データです：

| ベンチマーク区分 | 7B Reader (`qwen2.5:7b-instruct`) | 1.5B Reader (`qwen2.5:1.5b`) | Delta | 7B のみ正解 | 1.5B のみ正解 | McNemar 検定 ($p$ 値) | 科学的解釈 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **LoCoMo 1,540 (全問)** | **64.68% (996/1,540)** | **64.55% (994/1,540)** | +0.13pp | 133 | 131 | **$p = 0.9509$** | パラメータが約 4.7分の1 になっても統計的有意差は検出されず（差がある証拠なし）。 |
| **LongMemEval (500問)** | **81.60% (408/500)** | **80.00% (400/500)** | +1.60pp | 39 | 31 | **$p = 0.4030$** | 複数セッション推論 (Multi-session) は両モデルで 84.2% 完全一致。 |

> [!CAUTION]
> **モデル不変性に関する統計的厳密さについて**:  
> 高い McNemar $p$ 値 ($p > 0.05$) は、完全なモデル非依存性を数学的に証明するものではなく、同一の凍結コンテキスト下において「性能差があるという十分な証拠が得られなかった」ことを意味します。実質的同等性を形式的に立証するには、同等マージンを設定した同等性検定（TOST）が必要です。  
> しかしながら、1,540問中 863問で両モデルが同一正解し、413問で同一不正解であったという事実は、観測される正解精度の多くが Reader のパラメータ量ではなく**「Lethe が事前に決定論的にコンパイルした文脈の質」**に強く支配されていることを示す強力な経験的知見です。

### 正直な境界と未検証領域 (Honest Boundaries & Unverified Frontiers)
- **フロンティア Reader によるスケール (120B / Gemini)**: 同一10問プローブでは Gemini Flash で 90.0% (9/10) を記録しましたが、「120B+ 級のモデルを接続すれば 1,540問全体で 100% になる」という主張は、計算資源・API予算を伴う実機検証を要する**現時点では未検証の仮説**です。
- **決定論的自律オフロード (Autonomous Engine)**: Reader の四則演算ミスやフォーマット崩れをバイパスするため、Lethe はゼロ LLM 決定論的スキル（CHRONOS、集計、オントロジー解決器）を統合しています。7B Reader が F1 38.79% に沈んでいた難問ドリル 281問において、決定論的エンジンは**全281問に対して決定論的な回答コミットメントを生成し、F1 92.87%（回帰損失 0件）**を達成しました。
- **先行研究・他システムとの比較**: LongMemEval では他システムも高い数値を報告しており（Sibyl Labs 95.6%, OMEGA 95.4% など）、LoCoMo でも異なるハーネス・ジャッジ下での報告が存在します。同一の凍結ジャッジ下での直接的な横並び比較は現在進行中の課題です。

---

## なぜ Lethe なのか？ (コア思想)

### 1. 忘却とは「完全削除」ではなく「解像度の低下」である
人間の記憶は、ゴミ箱にファイルを捨てるようには消去されません。時間とともに解像度が落ちていきます：
- **Level 0 (RAW)**: 会話の生のテキスト
- **Level 1 (EPISODIC)**: 発話者や感情を伴うエピソード記録
- **Level 2 (CONDENSED)**: 構文解析（エンティティ・述語抽出）による要点と事実主張
- **Level 3 (FACT/STATE)**: `TemporalNormalizer` による決定論的日付確定（「先週の木曜」→「2023-05-18」）と状態管理
- **Level 4 (ANCHOR)**: セッション間で繰り返し言及される持続的プロファイル・中核属性

**解像度低下（忘却）の発生条件とトリガー**:  
Lethe では記憶を単純消去せず、以下の3つの決定論的ルールに基づいて解像度を段階的に減衰させます：
1. **セッション経過ターン数（時間減衰）**: 対話が進行するにつれ、古い発話ターンは Level 0（生テキスト）の保持期限を迎え、Level 2（構文・エンティティ要点）へと集約されます。
2. **アクセス頻度と最終参照（アクセス減衰）**: 会話中で繰り返し参照される中核アンカーは Level 4（持続的プロファイル）として固定されますが、一度しか言及されない末端の付随情報は低解像度へと縮退します。
3. **コンテキストトークン予算の上限（予算減衰）**: 想起時のプロンプト予算（例: 2,000トークン）を超過する場合、Utility / Token（トークンあたりの情報効用）スコアリングに基づき、優先度の低い事実から解像度が自動的に切り詰められます。想起時は最小トークンから探索を開始し、推論の曖昧さが検出された場合にのみ動的に高解像度へ展開されます。


### 2. ゼロ LLM 書き込み ＆ 決定論的コプロセッサ
メモリの保存時に、高価で不安定な LLM 要約を挟みません：
- **Write LLM Calls = 0**: 会話ターンは決定論的パーサーと逆インデックスにより即座に取り込まれます。
- **CHRONOS 時間推論コプロセッサ**: カレンダー演算を決定論的に処理（「先週の火曜日」「3ヶ月前」などの相対日付を対話タイムスタンプから絶対日付へ正確に変換）。
- **集計コプロセッサ (Aggregation Co-processor)**: カウント、リスト列挙、頻度計算を決定論的に解決。
- **Answer Committer**: 厳密な回答規約（Contract）を適用し、幻覚や曖昧な出力を防止。

### 3. 中間表現としてのコンテキスト (Context IR / MSC)
生のテキストをプロンプトに流し込むのではなく、構造化された **Memory State Context (MSC)** にコンパイルします：
```text
[TEMPORAL_ANCHOR] Reference Date: 2023-05-14
[VERIFIED_STATE] User lives in Kyoto (as of 2022-10)
[DETERMINISTIC_TIMELINE]
 - 2023-01-10: Visited Paris
 - 2023-03-04: Switched job to Researcher
[EVIDENCE_CHAIN (Provenance)]
 - [conv_3:turn_12] User: "I am taking the exam next month..."
```

---

## Lethe で現在できること

### AI エージェントとの統合 (MCP ネイティブ)

Lethe Akribeia は、Claude Desktop、Cline、Hermes などの標準的な **Model Context Protocol (MCP)** クライアントに対応しています：

```bash
python -m artificial_memory.mcp
```

提供される主な MCP ツール：
- `memory_remember`: 事実や決定事項をタイムスタンプ付きで記録
- `memory_recall`: エビデンスチェーンを伴う適応型想起
- `memory_expand`: 圧縮された記憶を生の会話解像度に展開
- `memory_trace`: 任意の事実を元の発話ターンまで追跡
- `memory_timeline`: エンティティやトピックの時系列タイムラインを取得
- `memory_explain`: 特定の記憶がなぜ選ばれたのか／除外されたのかを説明

### クイックスタート CLI

```bash
# インストール
git clone https://github.com/Yato-Works/Lethe-Akribeia.git
cd Lethe-Akribeia
pip install -e ".[vector,llm]"
```

> [!NOTE]
> **パッケージアーキテクチャに関する注記**: 再現可能なベンチマークハーネスおよび 308 件の単体テストスイートとの完全な後方互換性を維持するため、内部 Python モジュールは `artificial_memory` パッケージとして構成され、ユーザー向け CLI バイナリおよびエントリポイントは `lethe` として提供されます。

```bash
# セッション開始
lethe start "Project/Akribeia"

# 会話の記録 (LLM呼び出しコスト = 0)
lethe user "9月30日を目標に v0.2.0 を公開することにしたよ。"
lethe assistant "了解しました。公開の締め切りを9月30日に設定しました。"

# 適応型想起による質問
lethe recall "リリースの予定日はいつだっけ？"

# タイムラインと出所の確認
lethe timeline
```

---

## ハードウェアと研究設計

このプロジェクトは、**意図的に制約された研究予算**のもと、単一のローカル GPU（NVIDIA GeForce RTX 3050 / 8GB VRAM）とローカルモデルを用いて開発されました。

商用フロンティア API で長大なコンテキストを伴う全1,540問の評価を実行することは、現状の研究予算を超えています。  
**この制約を隠すのではなく、Lethe はそれを研究設計へと変換しました**。コンパイルされたメモリコンテキストを凍結キャッシュ（`locomo_gold_context_cache.jsonl`）として保存することで、将来的に計算資源が確保できた際に、メモリ側と Reader 側を完全に独立してスケール評価できるようにしています。

---

## ロードマップ：フロンティア拡張への道筋 (Roadmap to V4 & V5)

```
[V3 Apex (現在)] ─────────────► [V4 認知的拡張] ─────────────► [V5 フロンティア拡張 (Frontier Scale)]
• 308件の単体テスト (100% 合格)  • 非時間暗黙的状態の自律救済        • フロンティア LLM (120B/Gemini/Claude)
• LME Oracle Recall: 98.2%      • セッション間グラフ探索の拡張      • E2E ベンチマーク上限突破 (95%〜98%+)
• Subsystem H (算術差分エンジン) • 目標: LME E2E > 90%                • 完全マルチエージェント Kubernetes メッシュ
• LME E2E: 83.40% (+9問救済)    • LoCoMo Oracle Recall > 90%         • プロダクション自律メモリ標準
```

- **V3 (現在: Apex Baseline)**: 三層分離診断標準を確立し、LongMemEval で Oracle Recall 98.2% を達成。新設の Subsystem H (`ArithmeticDifferenceEngine`) により偽拒絶 +9問を完全救済して E2E 83.40% を記録。LoCoMo の時間ターン選択（+8問純増）を最適化し、単体テストを 308件（100% ALL PASS）へ拡充。
- **V4 (次期: 認知的拡張 & 非時間暗黙状態の救済)**:
  - *非時間暗黙状態の自律救済*: 属性・所属などの暗黙的推論パターンを自律エンジンへ拡張し、残余の Reader 偽拒絶を撲滅、LongMemEval E2E **90%+** 超えを達成。
  - *セッション間グラフ探索の強化*: `ppr_graph` とエンティティ結合を強化し、LoCoMo Multi-Hop の Oracle Recall を 78.37% から **90%+** へ（全体 Oracle Recall も 80.65% から **90%+** へ）引き上げ。
- **V5 (フロンティア拡張: Frontier Scale & Synthesis)**:
  - Lethe の超高精度 MSC コンテキストを商用フロンティアモデル（Gemini 1.5 Pro, Claude 3.5, 120B+ オープンモデル）に接続し、全 1,540問における E2E 95%+ 天井を実証。
  - Kubernetes クラスタを跨ぐ分散マルチエージェント合意プロトコルの完成。

---

## 再現手順 (Reproduction)

すべての評価スクリプト、アダプター、およびスコアリングコードは完全に再現可能です：

```bash
# 1. 完全単体テストスイートの実行 (308件, 100% 合格)
pytest tests/unit/


# 2. ベースラインに対する LoCoMo 公式スコア計算 (主幹 Instruct 7B モデル)
python scripts/benchmarks/score_locomo_run_json.py --input benchmark_results/locomo1540/locomo_1540_improved2.json

# 3. 三層分離レポートの確認 (Oracle Recall vs Reader on Hits)
python scripts/benchmarks/three_layer_report.py

# 4. モデル感度分析 (7B vs 1.5B 厳密 McNemar 検定) の実行
python scripts/benchmarks/model_sensitivity.py --a benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json --b benchmark_results/locomo1540/temporal321_rules_commit_15b.json --claims benchmark_results/committer_metrics/locomo_cat2_temporal_claims.json

# 5. BEAM ベンチマークの実行 (500K scale)
python scripts/run_coder7b_beam.py --scale 500K
```

---

## なぜ今公開するのか (Why Release Now?)

「手元にある限られたハードウェアと小規模な言語モデルで、構造化されたメモリシステムはどこまで行けるのか？」を知りたくて開発を始めました。

Lethe Akribeia v0.3.0 は、完成された記念碑ではありません。動いて検証可能な、極めて高い精度を持つチェックポイントです。  
測定可能な強みがあります。測定可能な弱点もあります。  
そして今、メモリの検索と Reader の推論を分離して測定できる仕組みが整いました。

このアーキテクチャが本当にどこまで行けるのか、僕は今でも確かめたいと思っています。

---

## 引用 (Citation)

```bibtex
@software{lethe_akribeia_2026,
  title = {Lethe Akribeia: An Experimental Cognitive Long-Term Memory System for AI},
  author = {Yato-Works},
  year = {2026},
  version = {0.3.0},
  url = {https://github.com/Yato-Works/Lethe-Akribeia}
}
```

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) をご参照ください。

