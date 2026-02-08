# 뉴턴 중력 트랜스포머 (NGT)

<a id="top"></a>

**[English](README.md)** | **[한국어](README_KO.md)**

### *"단어는 입자고, 어텐션은 중력이다"*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

NGT(Newton Gravity Transformer)는 토큰을 입자처럼 취급하는 실험적 Transformer 변형입니다. 각 토큰은 학습되는 **질량(mass)**과 **좌표(coordinates)**를 가지며, 어텐션은 잠재 공간에서의 거리 기반 **중력 커널(gravity kernel)**로 계산됩니다.

이 레포는 학습/재개(resume), TensorBoard 로깅, `*_best.pt`/`*_last.pt` 체크포인트, 좌표 시각화(Plotly HTML)까지 end-to-end로 포함합니다.

---

## 프로젝트 포커스: WikiText-103 (~25M)

현재 포커스는 WikiText-103 + BPE-8192 + 약 25M 파라미터 스케일의 스크리닝 실험입니다.

- 최소 재현 스크립트(15k 스크리닝): `run_wikitext103_25m.sh`
- 최소 요약: `reports/w3_25m_summary.md`
- 전체 스크리닝 아티팩트: `w3_25m_results/results/w3_25m/Summary.md`

### 최신 스크리닝 스냅샷 (w3_25m, seed=42, max_steps=15000)

val loss는 cross-entropy이며, perplexity는 `exp(loss)`입니다.

| run | 설정 | val loss @15000 | ppl @15000 | best val loss (step) |
|---|---|---:|---:|---:|
| vanilla | baseline | 4.5554 | 95.14 | 4.5524 (13500) |
| ngt_mass_in_value | `--mass-in-value --use-rsqrt` | 4.6635 | 106.01 | 4.6451 (13000) |
| ngt_no_repulsion | `--no-repulsion --use-rsqrt` | 4.7214 | 112.33 | 4.7214 (15000) |
| ngt_repulsion_interval_8 | `--repulsion-interval 8 --use-rsqrt` | 4.7889 | 120.17 | 4.7748 (13000) |
| ngt_default | `--use-rsqrt` | 4.7915 | 120.48 | 4.7762 (13000) |
| ngt_no_radius | `--no-radius-cutoff --use-rsqrt` | 4.7940 | 120.78 | 4.7772 (13000) |

같은 설정(`batch=16`, `accum=2`, `block=512`)에서의 처리량:

- vanilla: ~4.964 steps/s
- ngt_mass_in_value: ~0.852 steps/s
- ngt_no_radius: ~0.855 steps/s
- ngt_default / ngt_no_repulsion / ngt_repulsion_interval_8: ~0.829-0.830 steps/s

본 결과는 예산 제약 기반 15k 스크리닝(토크나이즈된 train 토큰 수 가정에 따라 대략 2 epoch 내외)이므로, 방향성 지표로 해석하는 것이 적절합니다.

---

## NGT는 무엇이 다른가? (메커니즘 요약)

일반 Transformer는 Q/K 내적(dot-product)으로 어텐션 점수를 계산합니다.

NGT는 기하(geometric) 스트림을 추가합니다:

- 각 토큰은 hidden state `h`(semantic)와 좌표 `z`(geometric)를 가집니다.
- 각 토큰은 학습되는 질량 `m`을 가지며 `Softplus`로 양수를 보장합니다.
- 어텐션 점수는 `z` 공간의 거리(및 질량 상호작용)에 의해 결정됩니다.
- radius cutoff(하드/소프트)로 거리 기반 sparsity를 학습할 수 있습니다.
- mass 기반 repulsion regularizer로 좌표 collapse를 억제합니다.

---

## 설치, 빠른 시작, 체크포인트

설치:

```bash
pip install -r requirements.txt
```

빠른 시작 (WikiText-103, 15k 스크리닝):

```bash
# WikiText-103 다운로드/캐시(HuggingFace datasets)
python prepare_data.py --dataset wikitext103

# 15k에서 vanilla + NGT(mass-in-value) 실행
bash run_wikitext103_25m.sh

# 채팅(체크포인트 config로 NGT/Vanilla 자동 감지)
python chat.py --checkpoint-path checkpoints/w3_25m/ngt_mass_in_value.pt_best.pt
python chat.py --checkpoint-path checkpoints/w3_25m/vanilla_25m.pt_best.pt
```

체크포인트 정책:

- `--checkpoint-path checkpoints/foo.pt`로 실행하면 다음 파일들이 저장됩니다.
- best 모델: `checkpoints/foo.pt_best.pt`
- last 모델: `checkpoints/foo.pt_last.pt`
- `--resume` 로드 순서: `*_last.pt` -> `*_best.pt` -> base 경로

Python 3.11+ 권장, 학습은 CUDA GPU를 권장합니다.

---

## 학습 (NGT)

전체 옵션은 `python train_shakespeare.py --help`를 참고하세요.

자주 쓰는 옵션:

- 데이터셋: `--dataset {shakespeare,wikitext103}`, `--data-path ...`
- 토크나이저: `--tokenizer {char,bpe,tiktoken}`
- BPE 옵션: `--bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json`
- 정규화: `--lambda-repulsion`, `--repulsion-interval`, `--no-repulsion`
- sparsity: `--no-radius-cutoff` 또는 `--use-soft-cutoff`
- 성능: `--use-rsqrt`, `--use-amp`, `--gradient-accumulation-steps`
- 스케줄: `--use-cosine-schedule --warmup-steps N`

예시:

```bash
python train_shakespeare.py --dataset wikitext103 --data-path data \
  --tokenizer bpe --bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json \
  --hidden-dim 512 --coord-dim 64 --num-layers 8 --num-heads 8 --mlp-dim 2048 \
  --block-size 512 --batch-size 16 --gradient-accumulation-steps 2 \
  --use-amp --use-cosine-schedule --warmup-steps 2000 \
  --checkpoint-path checkpoints/w3_ngt.pt
```

---

## 아티팩트 및 시각화 링크

요약/리포트:

- [최소 요약 (`reports/w3_25m_summary.md`)](reports/w3_25m_summary.md)
- [전체 요약 (`w3_25m_results/results/w3_25m/Summary.md`)](w3_25m_results/results/w3_25m/Summary.md)
- [Ablation 리포트 (`w3_25m_results/results/w3_25m/report.md`)](w3_25m_results/results/w3_25m/report.md)
- [결과 CSV (`w3_25m_results/results/w3_25m/results.csv`)](w3_25m_results/results/w3_25m/results.csv)

인터랙티브 HTML 시각화(Plotly 3D PCA):

- [coords_ngt_default.html](w3_25m_results_latest/results/w3_25m/coords_ngt_default.html)
- [coords_ngt_mass_in_value.html](w3_25m_results_latest/results/w3_25m/coords_ngt_mass_in_value.html)
- [coords_ngt_no_radius.html](w3_25m_results_latest/results/w3_25m/coords_ngt_no_radius.html)
- [coords_ngt_no_repulsion.html](w3_25m_results_latest/results/w3_25m/coords_ngt_no_repulsion.html)
- [coords_ngt_repulsion_interval_8.html](w3_25m_results_latest/results/w3_25m/coords_ngt_repulsion_interval_8.html)

TensorBoard:

```bash
tensorboard --logdir runs
```

---

## 보안 주의

체크포인트는 `torch.load(..., weights_only=False)`로 로드하며, Python pickle을 사용합니다. 신뢰할 수 없는 `.pt` 파일은 로드하지 마세요.

---

## 소개

안녕하세요. 저는 AI에 관심이 많은 한국의 학부생 **이태규(Taegyu Lee)**입니다.

대학원 진학을 목표로 개인 프로젝트 경험을 쌓기 위해 이 프로젝트를 시작했습니다. 아직 학부생 단계라 부족한 점이 많을 수 있으니, 언제든 PR이나 이슈를 주시면 감사히 반영하겠습니다.

연락처: `mjrror@korea.ac.kr`

---

## 라이선스

MIT (`LICENSE` 참고).

---

<div align="center">

**[맨 위로](#top)**

</div>
