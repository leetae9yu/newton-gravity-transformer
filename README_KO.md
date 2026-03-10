# 뉴턴 중력 트랜스포머 (NGT)

<a id="top"></a>

**[English](README.md)** | **[한국어](README_KO.md)**

### *"단어는 입자고, 어텐션은 중력이다"*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

NGT(Newton Gravity Transformer)는 토큰을 입자처럼 취급하는 실험적 Transformer 변형입니다. 각 토큰은 학습되는 **질량(mass)**과 **좌표(coordinates)**를 가지며, 어텐션은 잠재 공간에서의 거리 기반 **중력 커널(gravity kernel)**로 계산됩니다.

이 레포는 현재 학습, TensorBoard 로깅, `*_best.pt`/`*_last.pt` 체크포인트, 그리고 과거 실험에서 생성한 좌표 시각화 아티팩트를 중심으로 관리합니다.

---

## 프로젝트 포커스: 현재 ~6M WikiText-2 경로

현재 활성 코드 경로는 고정 BPE 토크나이저를 쓰는 WikiText-2 단일 경로이며, 대략 `~6M` 규모의 소형 모델을 기준으로 빠른 반복 실험과 안정성 분석을 진행하는 데 초점을 맞춥니다.

중요: 아래 벤치마크 표와 링크된 아티팩트는 과거 코드 버전에서 얻은 기록입니다. 참고 자료로는 유효하지만, 현재 브랜치의 대표 성능으로 간주하면 안 됩니다.

이 레포에 남아 있는 대규모 스크리닝 기록은 WikiText-103 + BPE-8192 + 약 25M 파라미터 스케일 기준이며, 현재의 주 개발 타깃은 아닙니다.

- 최소 요약: `reports/w3_25m_summary.md`
- 전체 스크리닝 아티팩트: `w3_25m_results/results/w3_25m/Summary.md`
- 사전학습 체크포인트(w3_25m): `https://huggingface.co/leetae9yu/newton-gravity-transformer/tree/main/checkpoints/w3_25m`

셰익스피어 데이터셋/체크포인트는 레거시 경로이며, 현재 프로젝트에서는 더 이상 사용하지 않습니다.

### 프로젝트 진행 흐름 (TinyShakespeare -> WikiText-103 -> 현재 WikiText-2)

- 초기 단계는 TinyShakespeare를 빠른 프로토타이핑용으로 사용했습니다.
- 이후 더 큰 규모 검증을 위해 WikiText-103 (~25M 파라미터 스케일)로 전환했습니다.
- 현재 브랜치는 더 빠른 디버깅과 촘촘한 ablation을 위해 `~6M` 규모의 WikiText-2 경로로 다시 초점을 옮겼습니다.
- 현재의 핵심 질문은 이 작은 스케일에서도 NGT가 vanilla transformer baseline과 경쟁할 수 있는지 확인하는 것입니다.

### 과거 25M 스크리닝 스냅샷 (w3_25m, seed=42, max_steps=15000)

val loss는 cross-entropy이며, perplexity는 `exp(loss)`입니다.

| run | 설정 | val loss @15000 | ppl @15000 | best val loss (step) |
|---|---|---:|---:|---:|
| vanilla | baseline | 4.5554 | 95.14 | 4.5524 (13500) |
| ngt_mass_in_value | `--mass-in-value` | 4.6635 | 106.01 | 4.6451 (13000) |
| ngt_no_repulsion | repulsion 비활성화(레거시 실행) | 4.7214 | 112.33 | 4.7214 (15000) |
| ngt_repulsion_interval_8 | `--repulsion-interval 8` | 4.7889 | 120.17 | 4.7748 (13000) |
| ngt_default | 기본값 | 4.7915 | 120.48 | 4.7762 (13000) |

같은 설정(`batch=16`, `accum=2`, `block=512`)에서의 처리량:

- vanilla: ~4.964 steps/s
- ngt_mass_in_value: ~0.852 steps/s
- ngt_no_radius: ~0.855 steps/s
- ngt_default / ngt_no_repulsion(레거시) / ngt_repulsion_interval_8: ~0.829-0.830 steps/s

본 결과는 과거 예산 제약 기반 15k 스크리닝(토크나이즈된 train 토큰 수 가정에 따라 대략 2 epoch 내외)이므로, 현재 코드나 현재 `~6M` 개발 브랜치의 대표 성능이라기보다 방향성 지표로 해석하는 것이 적절합니다.

---

## NGT는 무엇이 다른가? (메커니즘 요약)

일반 Transformer는 Q/K 내적(dot-product)으로 어텐션 점수를 계산합니다.

NGT는 기하(geometric) 스트림을 추가합니다:

- 각 토큰은 hidden state `h`(semantic)와 좌표 `z`(geometric)를 가집니다.
- 각 토큰은 학습되는 질량 `m`을 가지며 `Softplus`로 양수를 보장합니다.
- 어텐션 점수는 `z` 공간의 거리(및 질량 상호작용)에 의해 결정됩니다.
- radius cutoff로 거리 기반 sparsity를 학습할 수 있습니다.
- mass 기반 repulsion regularizer로 좌표 collapse를 억제합니다.

---

## 설치, 빠른 시작, 체크포인트

설치:

```bash
pip install -r requirements.txt
```

빠른 시작 (WikiText-2, 현재 ~6M 벤치마크 경로):

```bash
# WikiText-2 다운로드/캐시(HuggingFace datasets)
python prepare_data.py

# NGT 학습 실행 (기본값: WikiText-2 + BPE-8192, ~6M 경로)
python train.py --data-path data \
  --checkpoint-path checkpoints/ngt_wikitext2_bpe_8192.pt

```

현재 학습 경로는 WikiText-2로 고정되어 있으며, 대략 `~6M` 모델 스케일에 맞춰 실험이 진행됩니다.

체크포인트 정책:

- `--checkpoint-path checkpoints/foo.pt`로 실행하면 다음 파일들이 저장됩니다.
- best 모델: `checkpoints/foo.pt_best.pt`
- last 모델: `checkpoints/foo.pt_last.pt`

Python 3.11+ 권장, 학습은 CUDA GPU를 권장합니다.

---

## 학습 (현재 브랜치)

전체 옵션은 `python train.py --help`를 참고하고, 동일 경로의 vanilla baseline은 `python train_vanilla.py --help`를 참고하세요.

자주 쓰는 옵션:

- 데이터셋: `--dataset wikitext2`, `--data-path ...`
- 토크나이저: 고정 BPE 경로 (`--bpe-vocab-size`, `--tokenizer-path`)
- 정규화: `--repulsion`, `--lambda-repulsion`, `--repulsion-interval` (`--repulsion` 사용 시 기본값 `4`)
- 성능: gravity score는 rsqrt 기반 경로를 사용하며, 추가로 `--use-amp`, `--gradient-accumulation-steps`를 사용할 수 있습니다
- 스케줄: `--use-cosine-schedule --warmup-steps N`

현재 baseline 비교 실험은 다음 두 경로를 기준으로 합니다:

- NGT: `python train.py ...`
- Vanilla baseline: `python train_vanilla.py ...`

예시:

```bash
python train.py --data-path data \
  --checkpoint-path checkpoints/ngt_wikitext2_bpe_8192.pt
```

---

## 과거 아티팩트 및 시각화 링크

과거 실험 요약/리포트:

- [최소 요약 (`reports/w3_25m_summary.md`)](reports/w3_25m_summary.md)
- [전체 요약 (`w3_25m_results/results/w3_25m/Summary.md`)](w3_25m_results/results/w3_25m/Summary.md)
- [Ablation 리포트 (`w3_25m_results/results/w3_25m/report.md`)](w3_25m_results/results/w3_25m/report.md)
- [결과 CSV (`w3_25m_results/results/w3_25m/results.csv`)](w3_25m_results/results/w3_25m/results.csv)

인터랙티브 HTML 시각화(Plotly 3D PCA, 과거 결과):

- [coords_ngt_default.html](w3_25m_results_latest/results/w3_25m/coords_ngt_default.html)
- [coords_ngt_mass_in_value.html](w3_25m_results_latest/results/w3_25m/coords_ngt_mass_in_value.html)
- [coords_ngt_no_radius.html](w3_25m_results_latest/results/w3_25m/coords_ngt_no_radius.html)
- [coords_ngt_no_repulsion.html](w3_25m_results_latest/results/w3_25m/coords_ngt_no_repulsion.html) (레거시 파일명)
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
