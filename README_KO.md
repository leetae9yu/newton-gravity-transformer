# 뉴턴 중력 트랜스포머 (NGT)

<a id="top"></a>

**[English](README.md)** | **[한국어](README_KO.md)**

### *"단어는 입자고, 어텐션은 중력이다"*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

NGT(Newton Gravity Transformer)는 토큰을 “입자”처럼 취급하는 실험적 Transformer 변형입니다. 각 토큰은 학습되는 **질량(mass)**과 **좌표(coordinates)**를 가지며, 어텐션은 잠재 공간에서의 거리 기반 **중력 커널(gravity kernel)**로 계산됩니다.

이 레포는 학습/재개(resume), TensorBoard 로깅, `*_best.pt`/`*_last.pt` 체크포인트, 좌표 시각화(Plotly HTML)까지 end-to-end로 포함합니다.

---

## 프로젝트 포커스: WikiText-103 (~25M)

현재 포커스는 WikiText-103 + BPE-8192 + ~25M 파라미터 스케일의 스크리닝 실험입니다.

- 최소 재현 스크립트(15k vanilla + NGT mass-in-value): `run_wikitext103_25m.sh`
- 최신 실험 요약(추적): `reports/w3_25m_summary.md`

### 최신 스크리닝 스냅샷 (w3_25m, seed=42, max_steps=15000)

val loss는 cross-entropy이며, perplexity는 `exp(loss)` 입니다.

| 모델 | 설정 | val loss @15000 | ppl |
|---|---|---:|---:|
| vanilla | baseline | 4.5554 | 95.14 |
| NGT | `--mass-in-value` | 4.6635 | 106.01 |

같은 설정(B=16, accum=2, block=512)에서의 대략적 처리량:
- vanilla: ~4.96 steps/s
- NGT: ~0.83–0.86 steps/s (약 6배 느림)

---

## NGT는 무엇이 다른가? (메커니즘 요약)

일반 Transformer는 Q/K 내적(dot-product)으로 어텐션 점수를 계산합니다.

NGT는 “기하(geometric) 스트림”을 추가합니다:
- 각 토큰은 hidden state `h`(semantic)와 좌표 `z`(geometric)를 가집니다.
- 각 토큰은 학습되는 질량 `m`을 가지며 `Softplus`로 양수를 보장합니다.
- 어텐션 점수는 `z` 공간의 거리(및 질량 상호작용)에 의해 결정됩니다.
- radius cutoff(하드/소프트)로 거리 기반 sparsity를 학습할 수 있습니다.
- mass 기반 repulsion regularizer로 좌표 collapse를 억제합니다.

---

## 설치

```bash
pip install -r requirements.txt
```

Python 3.11+ 권장. 학습은 CUDA GPU 권장.

---

## 빠른 시작 (WikiText-103, 15k)

```bash
# WikiText-103 다운로드/캐시(HuggingFace datasets)
python prepare_data.py --dataset wikitext103

# 15k에서 vanilla + NGT(mass-in-value) 실행
bash run_wikitext103_25m.sh

# 채팅(체크포인트 config로 NGT/Vanilla 자동 감지)
python chat.py --checkpoint-path checkpoints/w3_25m/ngt_mass_in_value.pt_best.pt
python chat.py --checkpoint-path checkpoints/w3_25m/vanilla_25m.pt_best.pt
```

---

## 체크포인트와 재개(--resume)

`--checkpoint-path checkpoints/foo.pt`로 실행하면 다음 파일들이 만들어집니다:

- 검증 성능이 가장 좋았던 모델: `checkpoints/foo.pt_best.pt`
- 마지막 step의 모델 상태: `checkpoints/foo.pt_last.pt`

`--resume`은 `*_last.pt` -> `*_best.pt` -> base 경로 순서로 로드합니다.

---

## 학습 (NGT)

전체 옵션은 `python train_shakespeare.py --help` 참고. 자주 쓰는 옵션:

- 데이터셋: `--dataset {shakespeare,wikitext103}`, `--data-path ...`
- 토크나이저: `--tokenizer {char,bpe,tiktoken}`
  - BPE: `--bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json`
- 정규화: `--lambda-repulsion`, `--repulsion-interval`, `--no-repulsion`
- sparsity: `--no-radius-cutoff` 또는 `--use-soft-cutoff`
- 성능: `--use-rsqrt`, `--use-amp`, `--gradient-accumulation-steps`
- 스케줄: `--use-cosine-schedule --warmup-steps N`

WikiText-103 예시:

```bash
python train_shakespeare.py --dataset wikitext103 --data-path data \
  --tokenizer bpe --bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json \
  --hidden-dim 512 --coord-dim 64 --num-layers 8 --num-heads 8 --mlp-dim 2048 \
  --block-size 512 --batch-size 16 --gradient-accumulation-steps 2 \
  --use-amp --use-cosine-schedule --warmup-steps 2000 \
  --checkpoint-path checkpoints/w3_ngt.pt
```

---

## TensorBoard & 좌표 시각화

TensorBoard:

```bash
tensorboard --logdir runs
```

좌표 시각화(3D PCA -> Plotly HTML):

```bash
python visualize_coords.py --checkpoint-path checkpoints/shakespeare.pt_best.pt --output coords.html
```

---

## 보안 주의

체크포인트는 `torch.load(..., weights_only=False)`로 로드하며, Python pickle을 사용합니다. 신뢰할 수 없는 `.pt` 파일은 로드하지 마세요.

---

## 소개

안녕하세요, 저는 **이태규(Taegyu Lee)** 입니다. 물리 기반 어텐션 메커니즘과 기하학적 해석 가능성에 관심이 있어 NGT를 개인 프로젝트로 실험하고 있습니다.

---

## 라이선스

MIT (`LICENSE` 참고).

---

<div align="center">

**[맨 위로](#top)**

</div>
