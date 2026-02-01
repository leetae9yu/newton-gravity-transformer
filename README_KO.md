# Newton Gravity Transformer (NGT)

**[English](README.md)** | **[한국어](README_KO.md)**

### *"단어는 입자이고, 어텐션은 중력이다"*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**NGT**는 표준 트랜스포머의 Dot-product Attention을 물리학 기반의 **중력 커널(Gravity Kernel)**로 대체하는 실험적 프로젝트입니다.

[아키텍처](#아키텍처) • [수학적 개념](#수학적-개념) • [설치 및 사용법](#설치-및-사용법) • [기여](#기여-및-피드백)

---

## 프로젝트 배경

안녕하세요! AI에 관심이 많은 학부생 **이태규**입니다.

이 프로젝트는 단순한 호기심에서 시작됐습니다: *"만약 의미적 관계가 운동과 중력의 법칙을 따른다면?"* 개발 과정 전체에서 **Vibe Coding** 방법론을 적용했습니다 — 추상적인 물리적 비유를 실제 작동하는 신경망 구조로 빠르게 전환하는 데 집중했습니다.

NGT는 단순한 모델이 아니라, **기하학적 해석 가능성(Geometric Interpretability)**에 대한 탐구입니다. 토큰에 질량과 좌표 같은 물리적 속성을 부여함으로써, 기존 어텐션 히트맵으로는 불가능한 방식으로 언어의 "중력"을 시각화할 수 있습니다.

---

## 핵심 개념

표준 트랜스포머가 벡터의 크기와 각도(내적)로 유사도를 계산하는 반면, NGT는 **거리($r$)**와 **질량($m$)**에 기반한 **메트릭 구조**를 도입합니다.

### 1. 중력 어텐션 (Gravity Attention)
통계적 패턴 매칭 대신, NGT는 토큰 간의 "인력"을 모델링합니다. 질량이 큰 토큰(예: 주어, 핵심 키워드)은 주변 토큰에 더 강한 중력을 행사하여 문맥 인식 궤도를 자연스럽게 형성합니다.

### 2. 학습 가능한 희소 어텐션 (Radius Cutoff)
NGT는 적응적 희소성 메커니즘을 구현합니다. 각 레이어는 자체적인 **상호작용 반경**을 학습합니다. 두 토큰 간의 잠재 거리가 이 학습 가능한 임계값을 초과하면 마스킹됩니다. 이를 통해 모델이 로컬 vs. 글로벌 어텐션의 범위를 동적으로 결정할 수 있습니다.

### 3. 좌표 진화 (Neural ODE 유사 흐름)
NGT에서 네트워크의 깊이는 시간의 흐름을 나타냅니다. 토큰은 의미 벡터만 업데이트하는 것이 아니라, 상호작용에 기반하여 잠재 매니폴드에서의 위치가 레이어를 거치며 진화합니다. 중력장 속을 이동하는 입자와 유사합니다.

---

## 아키텍처

### 이중 스트림 공진화 (Dual-Stream Co-Evolution)

NGT는 두 개의 동기화된 데이터 스트림을 유지합니다:
- **Hidden States ($h$):** 토큰의 의미적 "화물" (표준 트랜스포머 흐름)
- **Latent Coordinates ($z$):** 토큰의 기하학적 "위치" (기하학적 흐름)

```python
# "물리적" 어텐션 점수 공식
Score(i, j) = -gamma * (mass_i * mass_j) / (distance(i, j)**2 + epsilon)
```

토큰의 의미($h$)가 업데이트되면, 좌표($z$)의 이동을 유도하고, 이것이 다음 레이어의 어텐션 패턴을 변경합니다.

---

## 수학적 개념

### 1. 중력 어텐션 커널

$$\text{Score}_{ij} = -\gamma \cdot \frac{m_i \cdot m_j}{\|z_i - z_j\|^2 + \epsilon} + \beta$$

- **$\gamma$ (중력 상수)**: 전체 상호작용 강도를 제어하는 학습 가능한 파라미터
- **$\beta$ (중력 바이어스)**: 헤드별 학습 가능한 바이어스로, softmax의 dynamic range를 확장하여 점수가 양수/음수 모두 가능하게 한다
- **$m$ (질량)**: 토큰의 "중요도" 또는 "인력"을 나타내는 학습된 스칼라
- **$z$ (좌표)**: 저차원 잠재 매니폴드에서의 토큰 위치 $(d=16, 32)$

### 2. 적응적 희소성

$$\text{Mask}_{ij} = \mathbb{1}\left[\|z_i - z_j\|^2 > r^2\right]$$

모델은 학습을 통해 반경 $r$을 최적화하여, "중력적으로 중요한" 클러스터에 집중하면서 먼 노이즈를 무시할 수 있습니다.

---

## 설치 및 사용법

### 1. 시작하기
```bash
git clone https://github.com/leetae9yu/newton-gravity-transformer.git
cd newton-gravity-transformer
pip install -r requirements.txt
```

### 2. 빠른 실행 (TinyShakespeare)
```bash
# 데이터셋 다운로드
python prepare_data.py

# NGT 학습 (기본 설정)
python train_shakespeare.py --max-steps 5000

# 바닐라 트랜스포머 학습 (동일 하이퍼파라미터, 비교용)
python train_shakespeare_vanilla.py --max-steps 5000

# 학습된 모델과 대화 (체크포인트에서 NGT/Vanilla 자동 판별)
python chat.py --checkpoint-path checkpoints/shakespeare.pt_best.pt
python chat.py --checkpoint-path checkpoints/vanilla_shakespeare.pt_best.pt
```

### 3. 체크포인트 (5k 스텝)
`5k-v0.1` 디렉토리에서 체크포인트를 확인할 수 있습니다.
T4 무료 GPU로 5k 스텝을 학습했습니다 :)

### 4. 모델 설정 (기본값)

| 하이퍼파라미터 | 값 | 설명 |
|----------------|-------|-------------|
| `Total Parameters` | **4M** | 총 학습 가능 파라미터 |
| `vocab_size` | 65 | 문자 단위 어휘 (TinyShakespeare) |
| `hidden_dim` | 256 | 토큰 임베딩 및 히든 상태 차원 |
| `coord_dim` | 32 | 잠재 좌표 공간 차원 |
| `num_layers` | 6 | NGT 블록 수 (깊이) |
| `num_heads` | 8 | 중력 어텐션 헤드 수 |
| `mlp_dim` | 1024 | Feed-Forward Network 내부 레이어 차원 |
| `block_size` | 256 | 최대 문맥 길이 (시퀀스 길이) |
| `batch_size` | 64 | 학습 배치 크기 |

### 5. Ablation 실험 플래그

다음 플래그들을 통해 각 구성 요소의 영향을 개별적으로 측정하는 **ablation 실험**을 수행할 수 있습니다. 모든 플래그는 `train_shakespeare.py`에서 사용 가능하며, 일부는 `train_shakespeare_vanilla.py`에서도 사용 가능합니다.

#### 아키텍처 Ablation (NGT 전용)

| 플래그 | 기본값 | 설명 |
|------|---------|-------------|
| `--no-radius-cutoff` | Off (radius cutoff **활성화**) | 학습 가능한 반경 희소 마스킹을 비활성화. 거리에 관계없이 모든 토큰 쌍을 본다. |
| `--no-repulsion` | Off (repulsion **활성화**) | 반발 손실(repulsion loss)을 비활성화. 좌표는 여전히 진화하지만 서로 밀어내지 않는다. |

#### 연산 최적화 (NGT 전용)

| 플래그 | 기본값 | 설명 |
|------|---------|-------------|
| `--use-rsqrt` | Off | 나눗셈을 `rsqrt`(GPU 친화적 곱셈)로 대체. 중력 점수 계산의 핵심 병목을 제거한다. |
| `--mass-in-value` | Off | 질량을 어텐션 점수가 아닌 value 가중치에 적용. L×L 질량 외적을 제거. 질량이 "상호 인력" 대신 "전파력"이 된다. |
| `--use-soft-cutoff` | Off | 하드 반경 마스킹(bool + masked_fill)을 부드러운 ReLU 스타일 감쇠로 대체. 분기를 제거하여 GPU 처리량을 개선한다. |

#### 학습 최적화 (NGT & Vanilla 공통)

| 플래그 | 기본값 | 설명 |
|------|---------|-------------|
| `--use-amp` | Off | Automatic Mixed Precision(FP16/FP32)을 활성화. CUDA GPU에서 학습 속도 ~1.5-2배 향상. CPU에서는 자동 무시. |
| `--use-cosine-schedule` | Off | Cosine annealing LR 스케줄을 활성화. 학습률이 초기값에서 0 근처까지 부드럽게 감소. |
| `--warmup-steps` | 0 | Cosine decay 전 선형 워밍업 스텝 수. 초기 gravity 파라미터 안정화에 유용. |
| `--lambda-repulsion` | 0.05 | Repulsion loss 가중치. 좌표 공간에서 토큰이 서로 밀어내는 강도를 조절. |
| `--repulsion-interval` | 1 | N스텝마다 repulsion loss 계산. 좌표가 분산된 후 O(L²) 오버헤드를 줄일 수 있음. |
| `--data-path` | `data/input.txt` | 학습 텍스트 파일 경로. |

#### 예시: Ablation 실험 실행

```bash
# 기본 NGT
python train_shakespeare.py --checkpoint-path checkpoints/ngt_base.pt

# rsqrt만 적용
python train_shakespeare.py --use-rsqrt --checkpoint-path checkpoints/ngt_rsqrt.pt

# mass-in-value만 적용
python train_shakespeare.py --mass-in-value --checkpoint-path checkpoints/ngt_miv.pt

# soft cutoff만 적용
python train_shakespeare.py --use-soft-cutoff --checkpoint-path checkpoints/ngt_soft.pt

# 모든 최적화 + AMP + cosine 스케줄 + 워밍업
python train_shakespeare.py --use-rsqrt --mass-in-value --use-soft-cutoff --use-amp \
    --use-cosine-schedule --warmup-steps 200 \
    --checkpoint-path checkpoints/ngt_all_opt.pt

# 바닐라 베이스라인 + AMP
python train_shakespeare_vanilla.py --use-amp --checkpoint-path checkpoints/vanilla_amp.pt
```

> **참고:** 실험마다 다른 `--checkpoint-path`를 사용해야 결과가 덮어씌워지지 않습니다.

---

## 바닐라 트랜스포머 베이스라인

공정한 비교를 위해 동일한 하이퍼파라미터를 사용하는 표준 트랜스포머(`vanilla_model.py`)가 포함되어 있습니다. 유일한 차이점은 어텐션 메커니즘입니다:

| | NGT | Vanilla |
|---|---|---|
| 어텐션 | 중력 커널 (거리 + 질량) | Scaled Dot-Product (Q·K) |
| 위치 인코딩 | 좌표 임베딩 (레이어마다 진화) | 학습된 위치 임베딩 (히든에 더함) |
| 추가 구성 요소 | 질량 임베딩, 좌표 진화, 반경 컷오프 | 없음 |

두 모델 모두 동일한 FFN, LayerNorm 배치(pre-norm), 잔차 연결, 학습 설정을 공유합니다. TensorBoard 로그는 별도 디렉토리(`runs/ngt_experiment`와 `runs/vanilla_experiment`)에 기록되어 나란히 비교할 수 있습니다:

```bash
tensorboard --logdir runs
```

---

## 결과

### 생성 샘플 (v0.1, 5k 스텝)

> **ROMEO:**
> Will see rey did never the id their be very sound.
>
> **GREMIO:**
> Ay, do fairst, leady the dead his of Paul prison tame
> More execute mariage uper; where so do queen;
> All pergeant you wented, all in persaks and din man;
> The droubland ho great that soppsure
> The the ayou commmonand lord make your 's perjustce;
> On and them thou tirdd ust childred-shap him with estren,
> Or our where not war, we have sweet fr

> **ANGELO:**
> I say, to not I will not to for anin that say befrormanter?
>
> **QUEEN:**
> Has here; then imperated here to be no friends to steate
> To from beatie in a lleasure blow,
> And bad the pothrod my country; I gracious and of play
> This struing your means some was fith him.

모델은 희곡 대본 형식의 **구문 구조**(화자명, 줄바꿈)와 **셰익스피어 어휘**(thou, lord, queen, gracious)를 성공적으로 포착합니다. 문자 단위 토크나이저로 5,000 스텝만 학습했음에도, 중력 어텐션을 통한 구조적 이해의 발현을 보여줍니다.

<div align="center">
  <img src="assets/5k-step-loss.png" alt="학습 손실 곡선" width="600"/>
  <p><em>Figure 1: 학습 손실 곡선 (5k 스텝)</em></p>
</div>

---

## 기여 및 피드백

이 프로젝트는 학부생 개인 프로젝트로, 개선의 여지가 많습니다! 다음과 관련된 **PR**, **Issue**, 또는 자유로운 토론을 항상 환영합니다:

- **수학적 개선**: 중력 포텐셜 및 반발 파라미터 최적화
- **새로운 물리적 비유**: 전자기학, 유체역학, 열역학적 엔트로피 통합 아이디어
- **하드웨어 최적화**: 벡터화된 거리 계산의 성능 개선
- **해석 가능성 도구**: 잠재 매니폴드 진화를 시각화하는 새로운 방법

언제든 이슈를 열거나 PR을 제출해주세요!

---

## 변경 이력

### v0.2 — 아키텍처 및 안정성 전면 개선

**버그 수정**
- last 체크포인트에 ablation config 누락(`use_rsqrt`, `mass_in_value`, `use_soft_cutoff`) 수정 — `--resume` 시 실험 플래그가 리셋되는 문제 해결
- `chat.py`의 `generate()`에서 모델이 tuple을 반환할 때 처리하지 않는 버그 수정
- 좌표 진화가 attention 이전 hidden states를 사용하던 문제 수정 — 이제 상호작용 결과를 기반으로 좌표가 업데이트됨

**수치 안정성**
- softmax 후 불필요한 재정규화 제거 (gradient 왜곡 방지)
- 거리 계산 공식을 `||a||² + ||b||² - 2a·b`에서 직접 차이 `(z_i - z_j)²`로 변경 (catastrophic cancellation 방지)
- 모든 마스킹 값을 `-1e9`에서 `torch.finfo(dtype).min`으로 변경 (softmax에서 정확한 0 보장, AMP 안전)

**새 기능**
- **Gravity Bias**: 헤드별 학습 가능한 바이어스($\beta$)로 softmax dynamic range 확장
- **Weight Tying**: 입력 임베딩과 출력 projection 가중치 공유 (GPT-2/BERT 표준 기법), 파라미터 수 감소
- **Cosine LR Schedule**: `--use-cosine-schedule` + `--warmup-steps`로 부드러운 수렴
- **Repulsion 제어**: `--lambda-repulsion` (가중치) 및 `--repulsion-interval` (주기적 계산) 플래그 추가

**코드 품질**
- 공유 `FeedForward`와 `build_causal_mask`를 `common.py`로 추출, 4개 파일에서 중복 제거
- `VanillaTransformer`에 `max_seq_len` 초과 입력 truncation 추가 (NGT와 동일 동작)
- `chat.py` import 순서 PEP 8 준수로 수정
- 모든 `torch.load()`에 `weights_only=False` 명시 (PyTorch 2.6+ 호환)

**테스트**
- 16개 새 테스트 추가: VanillaTransformer, 토크나이저 왕복, ablation 조합, edge case(seq_len=1, max_seq_len 초과), soft cutoff, weight tying, gravity bias

---

## 로드맵

1. **고급 토크나이제이션**: 현재 NGT는 환경 제약(예: Google Colab T4 GPU)에서 메모리 오버헤드를 최소화하기 위해 **문자 단위 토크나이저**를 사용합니다. 하지만 *"단어는 입자"* 철학을 진정으로 실현하기 위해, **서브워드 토크나이저(BPE / Tiktoken)**로 전환할 계획입니다. 이를 통해 모델이 더 높은 수준의 의미 단위를 개별 물리적 입자로 다룰 수 있게 됩니다.

2. **Stable Diffusion과의 통합**: NGT의 가장 흥미로운 미래 단계 중 하나는 **Stable Diffusion과의 통합**입니다. NGT 좌표 공간을 Diffusion 모델의 잠재 공간과 정렬함으로써, "의미적 중력"을 이미지 생성에 활용하는 것을 목표로 합니다.

3. **시각화**: 현재 문자 단위 토크나이저를 사용하고 있어 시각화가 아직 불필요하다고 판단했습니다. 서브워드 토크나이저로 전환한 후, 잠재 매니폴드 진화를 이해할 수 있는 시각화 도구를 구현할 계획입니다.

~~4. **네이밍**: 코드베이스가 초기 이름 **HGT (Hierarchical Gravity Transformer)**에서 **NGT (Newton Gravity Transformer)**로 완전히 마이그레이션되었습니다. 잘 알려진 **Heterogeneous Graph Transformer**와의 혼동을 방지하기 위함입니다.~~

---

## 라이선스

이 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.

---

<div align="center">

**[맨 위로](#newton-gravity-transformer-ngt)**

</div>
