# LoCoMo Failure Analysis Report

Total failures analyzed: 576

## By Category
- temporal: 188
- single-hop: 181
- multi-hop: 143
- open-domain: 64

## By Failure Type
- TEMPORAL_FAILURE: 188
- RETRIEVAL_FAILURE: 166
- GROUNDING_FAILURE: 111
- COMPOSITION_FAILURE: 109
- STATE_FAILURE: 2

## Oracle Gap by Category
- single-hop: 86/181 (47.5%)
- temporal: 49/188 (26.1%)
- multi-hop: 48/143 (33.6%)
- open-domain: 32/64 (50.0%)

## Evidence State Distribution
- NO_EVIDENCE_REQUIRED: 576

## Affected Components
- Unknown: 361
- SteroidEngine/Retriever: 215

## Top Priority Targets
- **single-hop**: 86/181 oracle gaps - fix retrieval first
- **temporal**: 49/188 oracle gaps - fix retrieval first
- **multi-hop**: 48/143 oracle gaps - fix retrieval first
- **open-domain**: 32/64 oracle gaps - fix retrieval first

