# Lethe Akribeia

> **忘却とは削除ではない。**  
> **解像度の低下である。**

AIのための実験的認知長期メモリシステム。忘却を「完全削除」ではなく「段階的な解像度の低下」として扱います。  
*決定論的メモリコンパイル · 時間推論 · エビデンス出所追跡 · MCP ネイティブ*  
*(旧称: Artificial Memory / `lethe-akribeia`)*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Experimental v0.2.0](https://img.shields.io/badge/status-experimental%20v0.2.0-orange.svg)](#なぜ今公開するのか)
[![LoCoMo: 1540 Benchmark](https://img.shields.io/badge/benchmark-LoCoMo%201540-green.svg)](#評価ハイライト)
[![BEAM: 100K--10M](https://img.shields.io/badge/benchmark-BEAM%20100%25-brightgreen.svg)](#評価ハイライト)

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

## 評価ハイライト (Evaluation Highlights)

Lethe は万能の SOTA システムとして提示されるものではありません。しかし、一部のコンポーネントは現行の大規模ベンチマークで上限（100%）に達しており、パイプライン全体の検証からは極めて明確なアーキテクチャ上の知見が得られています：

### 強みと現在の課題 (What is Working vs. What is Unsolved)

```text
達成できていること（強い成果 🟢）
🟢 BEAM ベンチマーク (500K トークン)    — 全10評価カテゴリで 100.0% の正解率を達成
🟢 BEAM ベンチマーク (1M & 10M トークン)  — 超長大コンテキスト下のプローブで 100.0%
🟢 LoCoMo 証拠存在率 (Evidence Recall) — 1,540問の非敵対的質問で 81.1%（All-Evidence Oracle）
🟢 ゼロ LLM 書き込みパス (Zero-LLM Write) — 取り込み・コンパイル時の LLM コール数 = 0
🟢 小型モデル帯における堅牢性            — 1.5B で 64.5% vs 7B で 64.7%（Letheの文脈がモデル差を吸収）
🟢 決定論的コプロセッサ                   — カレンダー演算 (CHRONOS) と集計・カウントを LLM なしで解決

未解決の課題（現在の限界 🟡）
🟡 LoCoMo エンドツーエンド QA F1       — デプロイ済みローカル 7B Reader で 50.9%（推論ギャップ）
🟡 相対時間推論の精度                   — 7B クラスのモデルにとって複数期間の相対表現推論は依然として高難度
🟡 フロンティア Reader 検証の規模       — 1,540問全体のフロンティアモデル評価は計算資源（API予算）待ち
🟡 分散マルチエージェント合意プロトコル   — Kubernetes Operator CRD は実装済みだが、分散合意は実験的段階
```

### ベンチマーク結果一覧

| ベンチマーク / 評価項目 | 結果 | スケール / 対象 | 測定している内容 |
|:---|:---:|:---:|:---|
| **BEAM (500K スケール)** | **100.0%** | 公式10カテゴリ (計20問) | 長大コンテキストからの針探し抽出、矛盾解決、イベント順序判定 |
| **BEAM (1M & 10M スケール)** | **100.0%** | 極長大プローブ (計8問) | 極限のトークン規模における事実想起と状態追跡 |
| **LoCoMo 証拠存在率 (Evidence Recall)** | **81.1%** | 1,540問（非敵対的質問） | 回答に必要な**すべての証拠ターン**がコンテキストに含まれていた割合（Strict Content Oracle） |
| **LoCoMo 完全検索失敗 (Zero-Evidence)** | **8.7%** | 134 / 1,540問 | 回答に必要な証拠が1つも回収できなかった完全な検索失敗 |
| **LoCoMo 公式 QA F1** | **50.9%** | 1,540問 (7B Reader) | デプロイされたローカル 7B Reader によるエンドツーエンドの回答 F1 スコア |
| **LongMemEval (500Q)** | **81.6%** | 500問 (7B Reader) | 長期記憶評価ベンチマークにおける正解率 |

> [!NOTE]
> **なぜ BEAM では 100% なのに LoCoMo では 50.9% なのか？**  
> BEAM は長大対話（100K〜10Mトークン）から「明確に記録された事実の抽出」「矛盾解決」「イベント順序」を判定するタスクです。Lethe の決定論的タイムライン抽出がノイズを完全に除去するため上限（100%）に達します（評価には厳密な制約遵守のため `qwen2.5-coder:7b` を使用し、プロンプト開発用セットとは独立した公式プローブで検証）。  
> 一方、LoCoMo は日常対話から「人物の性格や将来の行動の推論」「複雑な相対時間の合成」を行うオープン対話 QA であり、Reader モデル自身の言語推論力に強く依存するため 50.9% となります。

---

## モデル感度分析：小型モデル耐性 (The Reader Floor)

140時間以上に及ぶ実機検証によって得られた最も重要な発見は、**「Lethe の構造化コンテキストが、Reader のサイズ低下（7B → 1.5B）を強力に吸収する」**という実測事実です：

| Reader モデル | パラメータ | LoCoMo 1,540Q 正解数 (率) | LoCoMo 公式 F1 | LongMemEval 500Q | 特徴と挙動 |
|:---|:---:|:---:|:---:|:---:|:---|
| **Qwen 2.5 1.5B** | 1.5B | **993 / 1,540 (64.48%)** | **50.82%** | **80.0%** | パラメータが約 1/5 になっても性能がほぼ崩れない |
| **Qwen 2.5 7B Instruct** | 7B | **996 / 1,540 (64.68%)** | **50.92%** | **81.6%** | 主幹デプロイモデル：対話推論と指示追従に優れる |
| **Qwen 2.5 7B Coder** | 7B | **996 / 1,540 (64.68%)** | **50.88%** | **81.5%** | 厳密な制約遵守；総合スコアは Instruct と同等 |

> [!TIP]
> **文脈が回答を支配している実証 (Overlap Analysis)**:  
> 1,540問の全問評価において、7B と 1.5B は正解した問題の **863問が完全に一致**（両者不正解は413問、正否が入れ替わったのはわずか17.1%、McNemar検定 p = 0.95）。正解の大部分は Reader 自身の推論力ではなく、**「Lethe が事前に決定論的にコンパイルした文脈の質」**によって決まっていることが実証されています。
> 
> ※ **フロンティアモデル（Gemini 3.6 Flash）の予備プローブについて**:  
> 同一の凍結コンテキストにおいて、小型モデル帯の天井を超えるかを確認するため10問の予備検証を実施（同一10問におけるスコア: 7B は 80.0% Hit / 82.2% F1 に対し、Gemini は 90.0% Hit / 76.4% F1 を記録）。全1,540問のフロンティア評価は計算資源（API予算）の確保次第、今後の課題としています。


---

## Failure Ceiling：1,540問のエラー解剖

エラーを単一の「不正解」として片付けるのではなく、Lethe は LoCoMo の全1,540問における失敗要因を厳密に分類する診断フレームワークを提供します：

```
全 1,540 問
│
├── 996問 (64.7%) 正解 (Hit)
├── 134問 (8.7%)  検索完全失敗 (Retrieval Failure) → 証拠が1つもコンテキストに含まれなかった（メモリの限界）
├── 32問  (2.1%)  コミット失敗 (Commit Failure)    → 証拠はあったが Answer Committer が棄権・拒否した
└── 378問 (24.5%) 推論ギャップ (Reasoning Gap)     → 証拠はコンテキスト内にあったが、Reader が正しく統合できなかった
```

### 証拠存在率 (81.1%) と検索失敗 (8.7%) の関係
- **81.1% (1,249問)**: **全証拠回収 (All-Evidence Match)** — 回答に必要なすべてのゴールド証拠ターンがコンテキスト内に存在。
- **8.7% (134問)**: **完全未回収 (Zero-Evidence Failure)** — メモリエンジンが証拠を1つも拾えなかったケース。
- **10.2% (157問)**: **部分回収 (Partial Retrieval)** — 複数ターンが必要な質問で、一部の証拠のみが回収されたケースなど。

---

## なぜ Lethe なのか？ (コア思想)

### 1. 忘却とは「完全削除」ではなく「解像度の低下」である
人間の記憶は、ゴミ箱にファイルを捨てるようには消去されません。時間とともに解像度が落ちていきます：
- **Level 0 (RAW)**: 会話の生のテキスト
- **Level 1 (EPISODIC)**: 発話者や感情を伴うエピソード記録
- **Level 2 (CONDENSED)**: 構文解析（エンティティ・述語抽出）による要点と事実主張
- **Level 3 (FACT/STATE)**: `TemporalNormalizer` による決定論的日付確定（「先週の木曜」→「2023-05-18」）と状態管理
- **Level 4 (ANCHOR)**: セッション間で繰り返し言及される持続的プロファイル・中核属性

**解像度低下（忘却）のポリシー**:  
記憶の解像度は「セッション経過ターン数」「最終参照からのアクセス間隔」「設定されたトークン予算上限」に基づく効用スコアリング（Utility / Token）によって決定論的に減衰します。重要度が高く頻繁に参照されるアンカーは上位レベルを保ち、偶発的な詳細は低解像度へと集約されます。想起時には最小のトークンコストから探索を開始し、曖昧さがある場合にのみ動的に高解像度へ展開されます。


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
# ※ 現在パッケージ移行期間中のため、コード内は import artificial_memory もそのまま動作します


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

このプロジェクトは、**意図的に制約された研究予算**のもと、単一のローカル GPU（RTX 4080 / 16GB VRAM）とローカルモデルを用いて開発されました。

商用フロンティア API で長大なコンテキストを伴う全1,540問の評価を実行することは、現状の研究予算を超えています。  
**この制約を隠すのではなく、Lethe はそれを研究設計へと変換しました**。コンパイルされたメモリコンテキストを凍結キャッシュ（`locomo_gold_context_cache.jsonl`）として保存することで、将来的に計算資源が確保できた際に、メモリ側と Reader 側を完全に独立してスケール評価できるようにしています。

---

## まだ完成していないこと (What Is NOT Finished)

科学的な厳密さを保つため、意図的に未完了のまま残されている項目を明記します：

- [ ] **全1,540問におけるフロンティア Reader の完全評価**（計算資源／API予算の確保待ち）
- [ ] **7B を超える大型モデル（14B / 32B / 70B）での体系的 Reader スケーリング則**
- [ ] **重複する複雑な時間間隔の相対表現解決**
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

# 4. BEAM ベンチマークの実行 (500K scale)
python scripts/run_coder7b_beam.py --scale 500K

# 5. モデル感度分析 (1.5B vs 7B) の実行
python scripts/benchmarks/model_sensitivity.py --baseline benchmark_results/locomo1540/temporal321_rules_commit_7b_postfix.json
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
  url = {https://github.com/Yato-Works/Lethe-Akribeia}
}
```

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) をご参照ください。
