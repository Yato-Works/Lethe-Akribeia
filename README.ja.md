# Lethe Akribeia

> **忘却とは削除ではない。**  
> **解像度の低下である。**

AIのための実験的認知長期メモリシステム。忘却を「完全削除」ではなく「段階的な解像度の低下」として扱います。  
*決定論的メモリコンパイル · 時間推論 · エビデンス出所追跡 · MCP ネイティブ*  
*(旧称: Artificial Memory)*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Experimental v0.2.0](https://img.shields.io/badge/status-experimental%20v0.2.0-orange.svg)](#なぜ今公開するのか)
[![LoCoMo: 1540 Benchmark](https://img.shields.io/badge/benchmark-LoCoMo%201540-green.svg)](#評価ハイライト)
[![BEAM: 100K--10M](https://img.shields.io/badge/benchmark-BEAM%20100%25-brightgreen.svg)](#評価ハイライト)

---

## Lethe とは何か？

Lethe Akribeia は、持続的な AI エージェントのために設計されたオープンソースの認知メモリランタイムです。  
単純なベクトル RAG やプロンプトへの全履歴流し込みとは異なり、Lethe は**メモリ取り込み時に LLM を一切呼び出さず（Write LLM = 0）**、対話履歴を構造化された解像度階層へと決定論的にコンパイルします。カレンダー演算や集計を担う決定論的コプロセッサを備え、すべての記憶を元の発話ターンまで追跡可能なエビデンス（出所情報）として保持します。

---

## 評価ハイライト (Evaluation Highlights)

Lethe は万能の SOTA システムとして提示されるものではありません。しかし、一部のコンポーネントは現行の大規模ベンチマークで上限（100%）に達しており、パイプライン全体の検証からは極めて明確なアーキテクチャ上の知見が得られています：

### 強みと現在の課題 (What is Working vs. What is Unsolved)

```text
達成できていること（強い成果 🟢）
🟢 BEAM ベンチマーク (500K トークン)    — 全10評価カテゴリで 100.0% の正解率を達成
🟢 BEAM ベンチマーク (1M & 10M トークン)  — 超長大コンテキスト下のプローブで 100.0%
🟢 LoCoMo 証拠存在率 (Evidence Recall) — 1,540問の非敵対的質問で 81.1%（Corrected Oracle）
🟢 ゼロ LLM 書き込みパス (Zero-LLM Write) — 取り込み・コンパイル時の LLM コール数 = 0
🟢 決定論的コプロセッサ                   — カレンダー演算 (CHRONOS) と集計・カウントを LLM なしで解決
🟢 コンテキストと Reader の分離評価       — 凍結コンテキストにより、メモリ検索と Reader 推論を独立検証

未解決の課題（現在の限界 🟡）
🟡 LoCoMo エンドツーエンド QA F1       — デプロイ済みローカル 7B Reader で 50.9%（推論ボトルネック）
🟡 相対時間推論の精度                   — 7B クラスのモデルにとって相対日付の間隔推論は依然として高難度
🟡 フロンティア Reader 検証の規模       — 10問プローブで 76.38% F1 を確認；全1,540問の実行は予算待ち
🟡 分散マルチエージェント合意プロトコル   — Kubernetes Operator CRD は実装済みだが、分散合意は研究途上
```

### ベンチマーク結果一覧

| ベンチマーク / 評価項目 | 結果 | 測定している内容 |
|:---|:---:|:---|
| **BEAM (500K スケール、全10カテゴリ)** | **100.0%** | マルチセッション、矛盾解決、時間推論、イベント順序など全プローブの正解率 |
| **BEAM (1M & 10M スケール)** | **100.0%** | 極端な長大コンテキスト環境下での想起・推論正解率 |
| **LoCoMo 証拠存在率 (1,540問)** | **81.1%** | Lethe がコンパイルしたコンテキスト内に、回答に必要な証拠がすべて含まれていた割合 |
| **LoCoMo 公式 QA F1 (1,540問)** | **50.9%** | デプロイされたローカル 7B Reader によるエンドツーエンドの回答 F1 スコア |
| **凍結コンテキストプローブ (7B)** | **50.92% F1** | 同一の10問プローブにおけるローカル 7B Reader のスコア (64.68% Binary Hit) |
| **凍結コンテキストプローブ (Frontier)** | **76.38% F1** | **まったく同一の凍結コンテキスト**に対する Gemini 3.6 Flash のスコア (90.00% Hit, **+25.46 pp**) |

> [!IMPORTANT]
> **「81.1% の証拠存在率」は「81.1% の QA 正解率」を意味しません。**  
> 正しい証拠がコンテキスト内に取り込まれていても、小規模な 7B Reader がそれを正しく統合・推論して完全な正解文字列を出力できるとは限らないためです。

---

## 本質的な問い (The Key Question)

エンドツーエンドのメモリベンチマークで失敗が起きたとき、**ボトルネックは「メモリの検索」にあるのか、それとも「Reader モデル」にあるのか？**

```
                 同一の凍結された LETHE CONTEXT
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       ローカル 7B Reader             フロンティア Reader
      (qwen2.5-coder:7b)             (Gemini 3.6 Flash)
              │                             │
          50.92 F1                      76.38 F1
          64.68 Hit                     90.00 Hit
              │                             │
              └──────────────┬──────────────┘
                             ▼
                 +25.46 pp F1 (+25.32 pp Hit)
```

従来のメモリシステムでは、メモリ想起と回答生成が不可分な単一スコアとして混同されていました。Lethe はこれらを切り離します。コンパイルされたメモリコンテキストを凍結することで、メモリ検索能力と Reader の推論能力を独立して評価できます。

---

## 凍結コンテキストプローブ：Reader 感度分析

7B Reader が Lethe のコンパイル済みコンテキストを十分に活用できていないのではないかという仮説を検証するため、同一のコンテキスト・同一の質問によるプローブ検証を行いました：

| カテゴリ | デプロイ済み 7B Reader (F1) | Gemini 3.6 Flash (F1) | 差分 (Delta) |
|:---|:---:|:---:|:---:|
| **マルチホップ推論 (Multi-hop)** | 64.7% | **100.0%** | **+35.3 pp** |
| **時間推論 (Temporal)** | 50.2% | **83.3%** | **+33.1 pp** |
| **オープン事実 (Open-domain)** | 39.6% | **100.0%** | **+60.4 pp** |
| **全体 F1** | 50.92% | **76.38%** | **+25.46 pp** |
| **バイナリ正解率 (Binary Hit)** | 64.68% | **90.00%** | **+25.32 pp** |

> [!NOTE]
> **これは10問のプローブであり、1,540問全体のベンチマーク結果ではありません。**  
> しかし、コンパイルされたコンテキストを完全に固定したままでも、フロンティア級 Reader を用いることで回答精度が大幅に向上する（+25.46 pp F1）という強力な予備証拠を示しています。残されたエラーの相当部分が Reader 側に起因している可能性が示唆されます。

---

## Failure Ceiling：1,540問のエラー解剖

エラーを単一の「不正解」として片付けるのではなく、Lethe は LoCoMo の全1,540問（非敵対的質問）における失敗要因を切り分けて分類する診断フレームワークを提供します：

```
全 1,540 問
│
├── 996問 (64.7%) 正解 (Hit)
├── 134問 (8.7%)  検索失敗 (Retrieval Failure) → 必要な証拠がコンテキストに含まれていなかった（メモリの限界）
├── 32問  (2.1%)  コミット失敗 (Commit Failure) → 証拠はあったが回答検証器が棄権・拒否した
└── 378問 (24.5%) 推論ギャップ (Reasoning Gap)  → 証拠はコンテキスト内にあったが、Reader が正しく統合できなかった
```

検索の失敗と Reader の推論失敗を切り離すことで、やみくもにプロンプトを改変するのではなく、真のボトルネックに焦点を当てた改善が可能になります。

---

## なぜ Lethe なのか？ (コア思想)

### 1. 忘却とは「完全削除」ではなく「解像度の低下」である
人間の記憶は、ゴミ箱にファイルを捨てるようには消去されません。時間とともに解像度が落ちていきます：
- **Level 0 (RAW)**: 会話の生のテキスト
- **Level 1 (EPISODIC)**: 発話者や感情を伴うエピソード記録
- **Level 2 (CONDENSED)**: 要点と事実的主張
- **Level 3 (FACT/STATE)**: エンティティの状態変化と確定した決定事項
- **Level 4 (ANCHOR)**: 長期的な中核信念や不変の事実

想起時、Lethe は最小のトークンコストから探索を開始し、曖昧さがある場合にのみ動的に解像度を展開します。

### 2. ゼロ LLM 書き込み ＆ 決定論的コプロセッサ
メモリの保存時に、高価で不安定な LLM 要約を挟みません：
- **Write LLM Calls = 0**: 会話ターンは決定論的に取り込まれ、トークンインデックス化されます。
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
git clone https://github.com/Yato-Works/artificial-memory.git
cd artificial-memory
pip install -e ".[vector,llm]"

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

このプロジェクトは、**意図的に制約された研究予算**のもと、単一のローカル GPU（RTX 4080 / 16GB VRAM）とローカル 7B モデル（`qwen2.5-coder:7b`）を用いて開発されました。

商用フロンティア API で長大なコンテキストを伴う全1,540問の評価を実行することは、現状の研究予算を超えています。  
**この制約を隠すのではなく、Lethe はそれを研究設計へと変換しました**。コンパイルされたメモリコンテキストを凍結キャッシュ（`locomo_gold_context_cache.jsonl`）として保存することで、将来的に計算資源が確保できた際に、メモリ側と Reader 側を完全に独立してスケール評価できるようにしています。

---

## まだ完成していないこと (What Is NOT Finished)

科学的な厳密さを保つため、意図的に未完了のまま残されている項目を明記します：

- [ ] **全1,540問におけるフロンティア Reader の完全評価**（計算資源／API予算の確保待ち）
- [ ] **モデルサイズごとの体系的 Reader スケーリング則**（7B vs 14B vs 32B vs 70B vs Frontier）
- [ ] **重複する複雑な時間間隔の相対表現解決**
- [ ] **LongMemEval および PersonaMem のフルローカル評価**
- [ ] **分散マルチエージェント合意プロトコル**（Kubernetes Operator CRD は存在しますが、分散合意は実験的段階です）

---

## ベンチマークの再現 (Reproduction)

すべての評価スクリプト、アダプター、およびスコアリングコードは完全に再現可能です：

```bash
# 1. ユニットテストの実行 (82件)
pytest tests/unit/

# 2. ベースラインに対する LoCoMo 公式スコア計算
python scripts/benchmarks/score_locomo_run_json.py --input benchmark_results/locomo1540/locomo_1540_improved2.json

# 3. エラー天井 (Failure Ceiling) の診断
python scripts/benchmarks/failure_ceiling.py

# 4. BEAM ベンチマークの実行
python scripts/run_coder7b_beam.py --scale 500K

# 5. 凍結コンテキストに対するフロンティア Reader プローブ（要 GEMINI_API_KEYS）
python scripts/benchmarks/run_frontier_eval.py --limit 10
```

---

## なぜ今公開するのか (Why Release Now?)

「手元にある限られたハードウェアと小規模な言語モデルで、構造化されたメモリシステムはどこまで行けるのか？」を知りたくて開発を始めました。

Lethe Akribeia v0.2.0 は、完成された記念碑ではありません。動いて検証可能なチェックポイントです。  
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
  version = {0.2.0},
  url = {https://github.com/Yato-Works/artificial-memory}
}
```

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) をご参照ください。
