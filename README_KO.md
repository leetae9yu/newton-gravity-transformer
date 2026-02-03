# 뉴턴 중력 트랜스포머 (NGT)

<a id="top"></a>

**[English](README.md)** | **[한국어](README_KO.md)**

### *"단어는 입자이고, 어텐션은 중력이다"*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

NGT는 dot-product attention 대신, 잠재 좌표 공간에서의 **거리**와 토큰의 **질량(mass)** 을 이용해 “중력 커널”로 어텐션을 구성하는 실험적 트랜스포머 구현입니다.

## 핵심 기능

- 중력 기반 어텐션(gamma: 세기, beta: bias, 헤드별 학습)
- 토큰별 질량 임베딩(`Softplus`로 양수 보장) + 좌표 임베딩(`z`)
- 레이어를 따라 좌표가 진화(coordinate evolution)
- 학습 가능한 반경 컷오프(하드/소프트)로 희소 어텐션
- 반발(Repulsion) 정규화: 질량 기반 + 거리 클램프(min_dist)로 NaN/Inf 방지
- 토크나이저: `char`, `bpe`(HF `tokenizers`), `tiktoken`
- TensorBoard 스칼라 + Projector 임베딩 로깅
- 체크포인트 안정화: `*_best.pt` + `*_last.pt`, `--resume` 로드 우선순위 지원
- `chat.py`에서 레거시 체크포인트 호환(누락된 `mass_emb`, 구버전 `coord_proj_next`)

---

## 설치

```bash
pip install -r requirements.txt
```

Python 3.11+ 권장입니다. CUDA GPU에서 학습이 훨씬 빠릅니다(CPU 학습도 가능하지만 느립니다).

---

## 빠른 시작 (TinyShakespeare)

```bash
# TinyShakespeare 다운로드
python prepare_data.py

# NGT 학습(기본 설정)
python train_shakespeare.py --max-steps 5000 --checkpoint-path checkpoints/shakespeare.pt

# Vanilla baseline 학습
python train_shakespeare_vanilla.py --max-steps 5000 --checkpoint-path checkpoints/vanilla_shakespeare.pt

# 채팅(체크포인트 config로 NGT/Vanilla 자동 판별)
python chat.py --checkpoint-path checkpoints/shakespeare.pt_best.pt
python chat.py --checkpoint-path checkpoints/vanilla_shakespeare.pt_best.pt
```

---

## 체크포인트 & 재시작(--resume)

`--checkpoint-path checkpoints/foo.pt` 를 주면 학습 중 다음 파일을 저장합니다:

- 베스트(검증 손실 기준): `checkpoints/foo.pt_best.pt`
- 마지막(최종): `checkpoints/foo.pt_last.pt`

`--resume` 는 `*_last.pt` → `*_best.pt` → 기본 경로 순서로 로드합니다.

---

## 학습 (NGT)

전체 옵션은 `python train_shakespeare.py --help` 를 참고하세요. 자주 쓰는 옵션:

- 데이터셋: `--dataset {shakespeare,wikitext103}`, `--data-path ...`
- 토크나이저: `--tokenizer {char,bpe,tiktoken}`
  - BPE 예: `--bpe-vocab-size 8192 --tokenizer-path data/tokenizer_bpe_8192.json`
- 정규화: `--lambda-repulsion`, `--repulsion-interval`, `--no-repulsion`
- 희소화: `--no-radius-cutoff` 또는 `--use-soft-cutoff`
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

## RunPod 워크플로우 (WikiText-103 ~25M)

- 실행 가이드: `GUIDE.md`
- 인수인계/세션 노트: `FUTURE.md`
- 러너 스크립트: `run_wikitext103_25m.sh` (`budget10` 모드로 스크리닝 + 리포트 생성)

`scp`/추가 포트 노출이 막혀있다면, 결과 다운로드는 `runpodctl send/receive`를 권장합니다:

```bash
# RunPod pod에서
tar -cJf /tmp/w3_25m_results.tar.xz results/w3_25m
runpodctl send /tmp/w3_25m_results.tar.xz
```

```powershell
# 로컬 Windows PowerShell에서
.\runpodctl.exe receive <CODE>
```

---

## TensorBoard (스칼라 + Projector)

기본적으로 `runs/...`에 로그가 생성됩니다.

```bash
tensorboard --logdir runs
```

Projector 임베딩을 남기려면 `--vis-interval` 을 설정하세요(미지정 시 `--eval-interval` 값으로 동작).

---

## 좌표 시각화 (3D PCA)

```bash
python visualize_coords.py --checkpoint-path checkpoints/shakespeare.pt_best.pt --output coords.html
```

Plotly 기반 HTML을 생성하며, 점 크기/색으로 질량을 표현합니다.

---

## 테스트

```bash
pytest -q
```

알려진 이슈: 기본값인 “하드” 반경 컷오프는 bool 마스킹을 사용하므로 `radius_param` 에 gradient가 흐르지 않습니다. 이 때문에 현재 `test_ngt.py`의 `test_gradient_flow` 가 실패합니다.

---

## 보안 주의

체크포인트는 `torch.load(..., weights_only=False)` 로 로드하며 내부적으로 pickle을 사용합니다. 신뢰할 수 없는 `.pt` 파일은 로드하지 마세요.

---

## 라이선스

MIT (`LICENSE` 참고).

---

<div align="center">

**[맨 위로](#top)**

</div>
