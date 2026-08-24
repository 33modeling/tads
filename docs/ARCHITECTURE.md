# TADS 설계 구조 (ARCHITECTURE)

이 문서는 저장소의 코드를 전부 읽고 확인한 사실만 기록한다. 추정이나 계획은 포함하지
않으며, 코드와 설정에 실제로 존재하는 동작만 서술한다.

- 대상 커밋 기준 트리: `tads/`, `baselines/`, `configs/`, `scripts/`, `tests/`
- 문서 3종: `docs/ARCHITECTURE.md`(설계), `docs/CODE.md`(코드 구조), `docs/USAGE.md`(사용법)

## 0. 문서화 배경 — README 상태

루트 `README.md`는 1줄 46바이트, 단어 6개(`# TADS — Trajectory-Anchored Data Selection`)
가 전부다. 설치, 실행, 알고리즘, 설정, 산출물 어느 것도 적혀 있지 않다. 반면 코드는
Python 63개 파일 14,168줄, YAML 74개 1,306줄, 셸 17개 2,636줄 규모다. 즉 저장소 이름만
보고는 무엇을 하는 코드인지 알 수 없는 상태이므로, 본 문서 묶음이 사실상 유일한 설명서
역할을 한다.

루트에는 `AUTO_EVAL_AGENT.md`(3,084줄, 약 203KB)가 있으나 이는 사람용 설명서가 아니라
"체크포인트가 생기면 자동으로 평가를 돌리는 원격 에이전트"를 위한 운영 지시서다. 문서
스스로 "사람용 설명서가 아니라 LLM이 그대로 실행할 수 있도록 경로/명령을 명시적으로
적어둔다"고 밝히고 있다. 게다가 일부 항목이 현재 코드와 어긋나 있다(§9 참조).

## 1. 저장소 성격 판정

코드를 근거로 판정하면 이 저장소는 다음 네 가지를 한 묶음으로 제공하는
**LLM instruction tuning용 데이터 선택(data selection) 연구 실험 프레임워크**다.

1. **TADS 알고리즘 구현** — Trajectory-Anchored Data Selection. 학습 중 매 epoch마다
   후보 데이터 풀을 점수화해 상위 B개만 골라 SFT(supervised fine-tuning)하는 결정적
   (deterministic) 선택 알고리즘. 구현체는 `tads/core/{trajectory_anchor,scorer,reward,selector}.py`.
2. **비교 baseline 6종** — NAIT, Data Agent(PPO), SelectIT, Q2Q/Cherry-LLM(IFD),
   LIMA, AlpaGasus. `baselines/<method>/` 아래에 각각 독립 entrypoint로 구현.
3. **벤치마크 평가 하니스** — MMLU, MMLU-Pro, GSM8K, SVAMP, HumanEval, MBPP, TyDiQA,
   XQuAD, BBH 9종 자체 구현 + lm-evaluation-harness 래퍼 1종, 총 10개 evaluator.
4. **실험 매트릭스 운영 도구** — 계층형 YAML 설정 74개, 데이터 다운로드/실행/평가/표
   생성 셸 스크립트 17개.

판정 근거가 되는 정량 지표:

| 영역 | 파일 수 | 줄 수 |
|---|---:|---:|
| `tads/core/` (알고리즘 코어) | 11 | 2,476 |
| `tads/evals/` (평가기) | 12 | 4,420 |
| `tads/data/` (데이터·프롬프트) | 3 | 770 |
| `tads/pipelines/` (SFT·선택 파이프라인) | 3 | 656 |
| `tads/modeling/` (모델 로딩·LoRA) | 3 | 399 |
| `tads/` 최상위 (`train.py`, `eval.py`, `__init__.py`) | 3 | 1,522 |
| `baselines/` (baseline 6종 + 패키지 초기화) | 19 | 3,246 |
| `tests/` | 7 | 440 |
| `scripts/*.py` | 2 | 239 |
| Python 합계 | 63 | 14,168 |
| `configs/**/*.yaml` | 74 | 1,306 |
| `scripts/*.sh` | 17 | 2,636 |

Git 이력은 main 기준 228 커밋, 최초 커밋 메시지는 "Initial commit: unify data-agent-dev
and tads-main into single TADS package"로, 두 개의 선행 저장소를 하나의 패키지로 합친
것이 출발점이다. 추적 파일 수는 160개다.

## 2. 목적과 범위

### 2.1 무엇을 푸는가

instruction tuning 데이터셋(기본값 Alpaca-GPT4 계열, Table 5 실험은 Evol-Instruct) 전체를
학습하는 대신, **선택 비율(selection_ratio)만큼만 골라 학습해도 전체 학습에 준하거나
그 이상의 벤치마크 성능을 얻는가**를 검증한다. 실험 설정의 표준 비율은 10%(`*_10`),
보조 실험으로 50%(`*_50`), 대조군으로 100%(`full_100`)와 무작위 선택(`random_10`)이
준비돼 있다.

### 2.2 범위 안

- 선택 알고리즘 학습 루프(선택 → SFT → 체크포인트)와 재개(resume)
- 다중 모델 패밀리 지원: Llama-2-7B, Qwen2.5-7B, Qwen2.5-0.5B, Mistral-7B, DeepSeek-7B
  (설정 파일 기준. Qwen2.5-14B 설정도 존재하나 어떤 실험에서도 참조하지 않음)
- full fine-tuning과 LoRA 두 가지 학습 모드
- 단일 GPU 및 DDP(torchrun) 실행
- 9개 벤치마크 자체 평가 + 결과 JSON → 논문형 표 생성

### 2.3 범위 밖(코드에 없음)

- 사전학습(pretraining), RLHF/DPO 등 선호도 학습
- 서빙/추론 API, 웹 UI
- 데이터 수집·정제 파이프라인(다운로드 스크립트는 공개 미러에서 파일을 받아오는 수준)
- 하이퍼파라미터 자동 탐색(스윕은 `--run_suffix`로 사람이 돌리는 구조)

## 3. 전체 아키텍처

계층은 네 겹이며, 위 계층만 아래 계층을 import한다. 역방향 import는 없다.

```
[ 진입점 계층 ]
  python -m tads.train            (method = random | full | tads)
  python -m tads.eval             (벤치마크 평가)
  python -m baselines.<m>.train   (m = nait | data_agent | selectit | q2q | lima | alpagasus)
        │
        ▼
[ 파이프라인 계층 ]  tads/pipelines/
  selection.py : epoch별 선택 dispatch + rank 간 선택 공유 + 선택 캐시
  sft.py       : 1 epoch SFT 루프, DataLoader 구성, DDP 동기화 정책
        │
        ▼
[ 코어 알고리즘 계층 ]  tads/core/
  trajectory_anchor.py : 층별 방향 v_l 추출(PCA + 부호 보정)
  reward.py            : 샘플별 L_i(CE loss), H_i(예측 엔트로피)
  scorer.py            : 풀 단위 R_i, 정규화, 최종 점수 s_i, top-B
  selector.py          : 후보 풀 1회 forward → 위 요소 조립 → 선택 인덱스
        │
        ▼
[ 인프라 계층 ]
  core/utils.py       : 설정 로더(defaults 체인·환경변수 보간), 로깅, 시드, DDP 헬퍼,
                        런타임 캐시 정리, 코어덤프 차단
  core/run_layout.py  : runs/<tag>/ 레이아웃, _latest 포인터, _complete sentinel
  core/timing.py      : 단계별(phase) 소요시간 계측·리포트
  core/schedulers.py  : cosine + warmup 스케줄러(transformers에서 vendoring)
  core/data_io.py     : JSON/JSONL/Parquet 원시 레코드 로더
  modeling/           : 토크나이저·모델 로딩, LoRA 설정, 평가용 로딩
  data/               : Alpaca 계열 토크나이즈, 모델 패밀리별 프롬프트 템플릿
  evals/              : 벤치마크 evaluator 레지스트리와 구현 10종
```

`baselines/*`는 코어·인프라 계층을 그대로 재사용한다(모델 로딩, 데이터셋 빌드, SFT 루프,
스케줄러, 타이머, 설정 로더). 자체적으로 갖는 것은 "무엇을 기준으로 고를지"에 해당하는
선택 로직뿐이다. 이 재사용 구조 덕분에 방법 간 비교에서 SFT 쪽 변수는 통제된다.

## 4. 컴포넌트와 책임

### 4.1 코어 알고리즘

| 모듈 | 책임 | 핵심 출력 |
|---|---|---|
| `core/trajectory_anchor.py` | probe 부분집합에 대해 층별 Δh를 모아 top-1 PCA로 방향 v_l 추출, 부호 보정, 히스토리 관리, 직렬화 | `v_by_layer`, λ1/λ2/gap, stability |
| `core/reward.py` | 배치 logits/labels에서 샘플별 응답 토큰 평균 CE loss(L_i)와 평균 예측 엔트로피(H_i) 계산 | `(r_loss, r_entropy, r_weight)` |
| `core/scorer.py` | 풀 단위 분산비 가중치 w, 합성 보상 R_i, alignment 최소-최대 정규화, 최종 점수 s_i, top-B 인덱스 | `R`, `w`, `s`, `topk indices` |
| `core/selector.py` | 후보 풀 전체를 1회 forward하며 위 값들을 누적하고 조립, 진단 로그와 통계 반환 | `selected_indices` + 진단 dict |

### 4.2 파이프라인

| 모듈 | 책임 |
|---|---|
| `pipelines/selection.py` | `method` 값에 따라 full/random/tads 분기. baseline 방법명이 들어오면 전용 entrypoint로 유도하는 오류를 던진다. rank 0만 무거운 선택을 수행하고 나머지 rank는 파일 sentinel을 폴링해 결과를 읽는다. 이전 실행이 남긴 선택 결과가 있으면 재사용한다. |
| `pipelines/sft.py` | 결정적 DataLoader 생성(단일 GPU에서도 epoch별로 셔플 순서가 달라지도록 시드에 epoch를 섞음), 1 epoch 학습, gradient accumulation 경계에서 clip/step/scheduler, 초기 몇 스텝의 진단 로그와 배리어, 마지막에 rank 평균 loss 집계 |

### 4.3 진입점

| 진입점 | 책임 |
|---|---|
| `tads/train.py` (921줄) | 설정 로드·오버라이드, DDP 초기화, run 디렉터리 결정, 재개 지점 탐색, 모델/토크나이저/데이터셋 준비, optimizer/scheduler 구성(8-bit AdamW 사전 프로브 포함), epoch 루프(선택 → SFT → 체크포인트 → `_latest` 갱신), 타이밍 리포트 |
| `tads/eval.py` (567줄) | 체크포인트 해석(명시 경로/run_tag/`_latest`), 평가 결과 run 레이아웃 생성, 벤치마크별 evaluator 호출(벤치 하나가 실패해도 나머지는 계속), 요약 JSON 원자적 기록, `_complete` sentinel과 `_latest` 갱신 |
| `baselines/<m>/train.py` | 방법별 선택 → 공유 SFT 루프 → `epoch_last/` 저장. 선택 결과는 파일로 캐시해 재실행 시 재사용 |

### 4.4 평가 계층

`tads/evals/base.py`가 추상 클래스 `BenchmarkEvaluator`와 데코레이터 기반 레지스트리를
제공하고, `tads/evals/__init__.py`가 각 구현 모듈을 eager import해 레지스트리를 채운다.
`tads.eval`은 이름 문자열로만 evaluator를 찾는다. 새 벤치마크 추가는 (1) 파일 생성,
(2) `@register("name")`, (3) `__init__.py`의 import 목록에 추가 세 단계로 끝난다.

모든 evaluator는 요약 dict에 `accuracy` 키를 채운다. 벤치마크마다 본래의 대표 지표
이름이 다르지만(정확도/EM/F1/pass@k), 표 생성기와 집계 도구가 벤치별 분기 없이 한 키만
읽으면 되도록 만든 교차 벤치 alias다.

## 5. 데이터·처리 흐름

### 5.1 학습 1 epoch (method=tads, DDP 기준)

```
epoch t 시작
  │
  ├─(rank 0) TrajectoryAnchor.update
  │     · probe 부분집합 무작위 추출 (seed + epoch*100 + 1)
  │     · 각 배치 forward(output_hidden_states=True)
  │     · 층 l마다 Δh_l = h_l[마지막 유효 토큰] - h_l[첫 토큰]  (Eq.1)
  │     · 32층 분을 GPU에서 stack 후 1회만 CPU 전송
  │     · 층별로 cat → top-1 PCA → v_l, λ1, λ2  (Eq.7~9)
  │     · 부호 보정: t=1이면 ⟨v_l, Δh̄_l⟩>0(공간적), t>1이면 ⟨v_l^t, v_l^{t-1}⟩>0(시간적)
  │     · 층별 원시 리스트를 처리 직후 폐기해 CPU 사용량이 단조 감소하도록 함
  │
  ├─(rank 0) collect_episode  — 후보 풀 전체 1회 forward
  │     · labels 포함 forward → logits, hidden_states
  │     · alignment: h̄_l = 패딩 제외 시퀀스 평균, align_i = (1/L) Σ_l ⟨h̄_l, v_l⟩
  │     · reward: 샘플 단위 L_i, H_i (배치 안에서 샘플별 루프, 메모리 상한 억제)
  │     · 루프 종료 후 풀 전체에 대해
  │           w   = Var(L) / (Var(L) + Var(H) + ε)          (Eq.4)
  │           R_i = w·L_i + (1-w)·H_i                        (Eq.3)
  │           ã_i = min-max(align) ∈ [0,1]
  │           s_i = R_i · (1 + λ·ã_i)                        (Eq.10)
  │     · k = max(1, ⌊N·selection_ratio⌋) 개를 topk로 선택
  │
  ├─ 선택 공유: rank 0이 인덱스 JSON을 원자적으로 쓰고 `.ready` sentinel 생성
  │             나머지 rank는 2초 간격 폴링(최대 6시간)으로 읽음. 이 구간에 NCCL
  │             collective 호출이 전혀 없다.
  │
  ├─ save_selection: `selected_indices_epoch{t}.json` 원자적 기록
  │
  ├─ SFT 1 epoch: Subset(dataset, selected) → DataLoader → forward/backward,
  │               grad_accum 경계에서 clip → step → scheduler → zero_grad(set_to_none)
  │
  └─(rank 0) 체크포인트: `epoch_last/`에 가중치·토크나이저·optimizer·scheduler·
              anchor 상태·env 메타·metrics를 저장 단계별 try/except로 기록하고,
              가중치와 optimizer가 모두 성공했을 때만 `_complete` sentinel을 원자적
              rename으로 남긴 뒤 `_latest` 포인터 갱신
```

`method=random`은 위 흐름에서 anchor/collect_episode 대신 시드 기반 순열의 앞 k개를
쓰고, `method=full`은 전체 인덱스를 그대로 쓴다. 즉 세 방법이 SFT·체크포인트 경로를
100% 공유하므로 선택 로직만 변수로 남는다.

### 5.2 데이터 준비 흐름

```
원시 파일(JSON/JSONL/Parquet/CSV) 또는 HF hub 이름
  → build_alpaca_dataset (교차 프로세스 파일 락으로 빌드 구간 직렬화)
  → tokenize_alpaca(prompt_style별 템플릿)
       · prompt와 response를 각각 따로 인코딩한 뒤 이어붙임
       · labels는 prompt 구간 전부 -100, response 구간만 학습 대상
       · prompt가 max_seq_len 이상이면 response 토큰을 최소 1개 남기도록 prompt를 자름
  → 고정 길이 패딩(attention_mask 0, labels -100)
  → HF Dataset(torch 포맷)
```

토크나이즈 캐시는 `data_cache/<model_key>/<prompt_style>/` 아래로 분리되어, 여러 실험을
동시에 돌려도 fingerprint가 섞이지 않는다.

### 5.3 평가 흐름

```
tads.eval
  → 체크포인트 결정 (--ckpt > --run_tag > _latest)
  → load_for_eval (LoRA면 adapter 병합 시도, 실패 시 래핑된 상태로 진행)
  → generation_config의 max_length/temperature/top_p/top_k를 None으로 정리
  → 벤치마크 순회: evaluator.evaluate(...) → 개별 결과 JSON 기록
       실패한 벤치는 failures 목록에 남기고 다음 벤치 계속
       벤치 사이에 gc + empty_cache
  → <experiment_label>-eval_summary.json 원자적 기록
  → _complete sentinel → _latest 갱신
```

결과 파일 이름은 `<experiment_label>-<bench>.json` 형식이고, `experiment_label`은
`cfg.experiment_name` → `<설정 상위 폴더명>_<설정 파일명>` → `<설정 파일명>` 순으로
결정된다. 여러 실험 결과를 한 폴더에 모아도 어느 (모델, 방법)의 수치인지 파일명만으로
구분할 수 있게 한 장치다.

## 6. 알고리즘 사양(코드 기준)

### 6.1 TADS

- 문맥화 벡터: `Δh_l(x) = h_l^{(K_x)}(x) − h_l^{(1)}(x)` (마지막 유효 토큰 − 첫 토큰)
- probe 집합 `D̃_t`에서 층별 공분산의 최상위 고유벡터가 `ṽ_l^{(t)}`
- 부호 보정: 첫 refresh는 평균 Δh̄_l과 내적이 양이 되도록, 이후는 직전 epoch의
  `v_l^{(t-1)}`과 내적이 양이 되도록 뒤집는다. 시간축 부호가 뒤집히면 alignment의
  의미가 epoch마다 반전되므로 필수 단계다.
- 후보 점수: `align_i = (1/L) Σ_l ⟨h̄_l(x_i), v_l⟩` → 최소-최대 정규화 → `s_i = R_i (1 + λ ã_i)`
- λ=0이면 `s_i = R_i`로 정확히 환원된다(합성 보상 단독 ranking).
- z-score, sigmoid, 학습되는 변환은 ranking 규칙에 들어가지 않는다. 코드 주석과
  `scorer.py`의 함수 배치가 이 원칙을 명시하고, z-score 변환(`calibrated_utility`)은
  ablation 전용으로만 남겨 두었다.

### 6.2 TADS와 NAIT의 설계상 차이

두 방법은 **v_l 추출을 동일한 Δh PCA로 공유**하지만 **후보 점수 계산이 다르다**.

| 구분 | TADS | NAIT (baseline) |
|---|---|---|
| v_l 추출 대상 | 학습 중인 모델 θ_{t-1}, epoch마다 갱신 | 고정 seed 집합, 학습 시작 전 1회 |
| 후보 투영 대상 | 시퀀스 평균 h̄_l | Δh_l (seed와 동일 정의) |
| 점수 결합 | R_i와 곱셈 결합 | 층별 내적 합만 사용 |
| 선택 시점 | epoch마다 다시 선택(dynamic) | 1회 선택 후 고정(static) |

`baselines/nait/direction.py`의 모듈 docstring이 이 차이를 명시적으로 기록해 두었다.

### 6.3 baseline 6종 요약

| baseline | 선택 기준 | 특징 |
|---|---|---|
| NAIT | seed 집합 Δh PCA 방향에 대한 Δh 투영 합 상위 K | 정적 선택, seed 파일 자동 생성 지원 |
| Data Agent | PPO actor가 뽑은 Beta 분포 표본 a_i 상위 K | 유일하게 학습되는 선택기. R은 PPO 학습 신호로만 사용 |
| SelectIT | rating 프롬프트에 대한 5-way 숫자 토큰 확률로 만든 불확실성 점수 | token 수준/sentence 수준 두 모드 |
| Q2Q(Cherry-LLM) | IFD = PPL(y\|x)/PPL(y) 상위 K, 범위 필터 적용 | 샘플당 조건부/무조건부 두 번 forward |
| LIMA | 선택 없음 — 소규모 수작업 큐레이션 데이터셋으로 교체 | 데이터 교체형 대조군 |
| AlpaGasus | 공개된 사전 필터 목록의 instruction 문자열 매칭 | 모델 forward 불필요 |

## 7. 분산 실행 설계

### 7.1 선택 구간에서 NCCL collective를 쓰지 않는 이유

`pipelines/selection.py`의 docstring에 실패 이력이 기록돼 있다. rank 0이 후보 풀 전체를
forward하는 동안(수십 분 단위) 다른 rank가 `dist.barrier()`에서 대기하면 NCCL watchdog
타임아웃에 걸려 communicator가 무너지고, 이후 모든 rank의 forward가 실패하며 rank 0은
체크포인트를 남기지 못한 채 종료된다. 그래서 이 구간은 다음과 같이 바뀌었다.

1. rank 0: 선택 인덱스를 `tmp → fsync → rename`으로 원자적 기록
2. rank 0: 별도의 `.ready` sentinel을 같은 방식으로 기록
3. 나머지 rank: `.ready`가 나타날 때까지 2초 간격 폴링(상한 6시간), 60초마다 진행 로그
4. 읽기 완료 후에도 배리어를 두지 않고 곧바로 SFT로 진입 — 첫 backward의 all_reduce가
   자연스러운 정렬 지점 역할을 한다.

같은 이유로 선택 파일 정리도 즉시 하지 않는다. rank 0이 읽기 직후 삭제하면 sentinel
확인과 실제 읽기 사이의 짧은 창에서 경쟁이 생기므로, **다음 epoch 진입 시점에 직전
4개 epoch 분을 청소**하는 지연 삭제 방식을 쓴다.

### 7.2 학습 루프의 동기화 정책

- `no_sync()` 기반 gradient accumulation 최적화는 기본 비활성. 첫 accumulation 경계에서
  hang이 관측된 이력이 있어 환경변수로만 켤 수 있다.
- 초반 3회의 optimizer step 경계에서는 명시적 배리어를 걸어 desync를 즉시 드러낸다.
  이후에는 배리어를 생략한다.
- 초반 10 스텝은 로그 레벨과 무관하게 항상 기록하고 `torch.cuda.synchronize()`를 호출해,
  숨은 CUDA 오류가 나중의 NCCL hang이 아니라 그 자리에서 예외로 드러나게 한다.
- NCCL 타임아웃은 기본 10분에서 120분으로 늘려 초기화한다.

### 7.3 두 개의 분산 구현이 공존한다

`tads/core/dist_utils.py`는 collective 기반의 "올바른" 선택 브로드캐스트 구현
(`broadcast_selection`, `all_gather_concat`, 전역 rank 판정 헬퍼)을 담고 있으나,
**프로덕션 경로에서는 호출되지 않는다.** 실제 학습은 위의 파일 폴링 방식을 쓴다.
이 모듈을 참조하는 것은 `tests/test_broadcast_selection.py` 하나뿐이다(§9 참조).

## 8. 설계 결정과 이유

코드 주석·구조에서 근거를 확인할 수 있는 결정만 정리한다.

| 결정 | 내용 | 이유 |
|---|---|---|
| 오프라인 기본값 | 진입점에서 `HF_*_OFFLINE=1`을 `setdefault` | 외부 HTTPS가 없는 실행 환경에서 HF 라이브러리가 메타데이터 갱신을 시도하다 캐시 락이 깨지는 실패를 차단 |
| 원자적 쓰기 + sentinel | 선택 파일, 설정 스냅샷, metrics, 요약, `_complete` 모두 tmp→fsync→rename | 중간에 죽어도 반쯤 쓰인 JSON이 남지 않게 하고, "완료 여부"를 별도 파일로 분리 |
| 저장 단계별 예외 격리 | 체크포인트 저장의 각 단계를 개별 try/except로 감싸고 실패 목록을 sidecar로 기록 | 한 단계 실패로 전체 epoch 산출물을 잃지 않기 위함. 단, 가중치·optimizer가 모두 성공해야만 sentinel을 쓴다 |
| `epoch_last/` 단일 디렉터리 | `tads.train`은 epoch별 디렉터리 대신 하나를 덮어씀. epoch 번호는 sentinel 내용에 기록 | 7B full-FT 체크포인트의 디스크 사용량 억제. baseline들은 여전히 `epoch_last/`만 남기지만 legacy `epoch_N/` 해석 경로도 유지 |
| `runs/<tag>/` 히스토리 | 실행마다 타임스탬프 태그 디렉터리, `_latest`가 최신을 가리킴 | 하이퍼파라미터를 바꿔 재실행해도 이전 결과를 덮어쓰지 않음. 재개는 같은 run_tag 안에서만 허용해 서로 다른 설정이 섞이지 않게 함 |
| `_latest` 이중 구현 | symlink 우선, 실패 시 `_latest.txt` 텍스트 파일 | symlink를 허용하지 않는 파일시스템 대비 |
| 8-bit optimizer 2단계 프로브 | 실제 1원소 텐서로 `AdamW8bit.step()`을 미리 실행해보고, 실패하면 fp32 AdamW로 자동 강등 | bitsandbytes의 CUDA 라이브러리 불일치는 첫 optimizer step에서야 드러나는데, TADS/Data Agent는 그 시점이 선택 단계 30분 뒤라 실행 전체를 날림 |
| BLAS/OMP 스레드 캡 | `import torch` **이전에** 16으로 고정 | 32개 층에 대해 `linalg.eigh`를 연속 호출할 때 스레드 생성이 폭주해 실패한 이력 |
| 코어덤프 차단 | 셸 `ulimit`이 아니라 Python에서 `RLIMIT_CORE=(0,0)` | 셸 스크립트를 거치지 않는 실행 경로가 흔하고, 7B DDP 코어덤프는 rank당 수백 GB 규모 |
| 데이터셋 빌드 락 | `cache_dir` 안의 파일 락으로 빌드 구간만 직렬화 | 동시 실행 잡들이 같은 캐시에 첫 토크나이즈를 쓰다가 Arrow 파일이 깨지는 문제 방지. 캐시가 준비된 뒤에는 비용이 사실상 0 |
| 모델 패밀리별 프롬프트 | `alpaca_default`, `qwen_chatml`, `mistral_instruct`, `llama_user_assistant`, `deepseek_user_assistant` | 학습과 평가의 템플릿을 일치시키지 않으면 SFT된 모델이 분포 밖 입력을 받는다. 평가 프롬프트 함수들이 같은 분기를 공유한다 |
| 평가 좌측 절단 | 모든 few-shot evaluator에서 `truncation_side="left"` | 기본값(오른쪽 절단)은 프롬프트 끝의 **테스트 문항**을 잘라내 예측 자체를 무의미하게 만든다. 왼쪽 절단은 demo를 앞에서 지운다 |
| 생성 결과 토큰 단위 슬라이싱 | `out[0, prompt_tok_len:]`로 자른 뒤 디코딩 | 디코딩 문자열을 프롬프트 길이로 자르면 BOS 자동 추가/특수토큰 제거 왕복 때문에 답 앞부분이 잘린다 |
| MBPP 서브프로세스 샌드박스 | 모델 코드를 별도 프로세스에서 실행하고 하드 타임아웃 | 인프로세스 `signal` 타임아웃은 C 확장 안에서 발동하지 않고, 모델 코드가 builtins를 오염시키면 이후 문항까지 오염됨 |
| `accuracy` alias | 모든 evaluator가 대표 지표를 `accuracy` 키에도 복사 | 집계기가 벤치마크별 분기 없이 한 키만 읽게 함 |
| 벤치 단위 실패 격리 | evaluator 하나가 실패해도 나머지 계속, 실패 내역을 요약 JSON에 기록 | 5~9개 벤치를 순차 실행하는데 중간 실패로 이미 끝난 결과까지 잃지 않기 위함 |
| baseline을 `tads.train`에서 분리 | `tads.train`은 random/full/tads만 처리하고, baseline 이름이 오면 전용 명령을 안내하는 오류를 던짐 | baseline마다 필요한 인자(seed 파일, 태그, 필터 파일)와 루프 구조가 달라 하나의 dispatch로 묶으면 조건 분기가 폭증 |
| 계층형 설정 | `defaults:` 리스트 → 깊은 병합 → 로컬 키가 최종 우선 | (base, method, model, mode) 축을 독립적으로 조합해 55개 실험 설정을 중복 없이 생성 |
| 스케줄러 vendoring | cosine+warmup 스케줄러를 transformers에서 복사 | transformers 5.0에서 optimization 모듈이 재편되며 공개 import 경로가 깨짐. LambdaLR 래퍼일 뿐이라 의존을 끊는 편이 안전 |

## 9. 코드에서 확인된 문제·불일치(설계 관점)

세부 근거와 수치는 `docs/CODE.md`에 정리했다. 여기서는 구조에 영향을 주는 항목만 든다.

1. **`scripts/run_main_7b.sh`의 기본 METHODS에 `data_agent_10`이 있는데 실행기는 항상
   `tads.train`이다.** `tads.train`은 `method: data_agent`를 거부하고 전용 entrypoint를
   안내하는 예외를 던지므로, 기본값 그대로 돌리면 4개 셀 중 1개가 반드시 실패한다.
   같은 문제를 `scripts/run_evol_7b.sh`는 `entry_for()` 분기로 해결해 두었다.
2. **DDP 지원 상태에 대한 서술이 스크립트끼리 어긋난다.** `run_main_7b.sh`는 torchrun
   DDP를 기본 모드로 쓰고, `run_evol_7b.sh` 주석은 "DDP 경로에 미해결 버그가 있으므로
   단일 프로세스가 지원되는 실행기"라고 적고 있다.
3. **분산 유틸이 이중 구현이다.** `core/dist_utils.py`의 collective 버전은 테스트에서만
   쓰이고, 실제 경로는 `pipelines/selection.py`의 파일 폴링 버전이다. 테스트가 통과해도
   운영 경로를 검증하지 못한다.
4. **`run_layout.resolve_eval_ckpt`가 있는데 `tads/eval.py`는 같은 로직을 인라인으로
   다시 구현했다.** 체크포인트 해석 규칙이 두 곳에 존재한다.
5. **baseline들은 `_complete` sentinel을 쓰지 않는다.** `epoch_last/`만 만들기 때문에
   `_latest` 기반 자동 해석(`find_latest_complete_epoch`)으로는 찾히지 않고, 평가 시
   `--ckpt`로 경로를 직접 줘야 한다. 실행 스크립트의 셸 헬퍼는 sentinel이 없는 legacy
   경우까지 처리하는 3단 폴백을 갖고 있다.
6. **문서와 코드의 지표 정의가 어긋난다.** `AUTO_EVAL_AGENT.md`의 지표 표는 TyDiQA의
   대표 키를 `accuracy_em`, XQuAD의 `accuracy`를 `macro_em`으로 적고 있으나, 현재 코드는
   두 벤치 모두 `accuracy`에 **F1**을 넣는다.
7. **`configs/methods/tads.yaml`과 일부 실험 YAML의 주석이 구버전 수식을 설명한다.**
   주석은 "Eq.8, R̃ = 풀 z-score"라고 쓰여 있지만 코드는 Eq.10과 원시 R을 쓰며 z-score를
   ranking에서 명시적으로 배제한다.
8. **`anchor` 설정의 깊은 병합 때문에 `layer_idx` 단일 층 모드가 사실상 도달 불가다.**
   `configs/base.yaml`이 `anchor.layer_indices: all`을 정의하므로, 실험 YAML이 `anchor:`
   블록에 `layer_idx`만 적어도 `layer_indices: all`이 병합되어 남는다. 그 결과 단일 층
   경로는 실행되지 않는다.

## 10. 확장 지점

| 확장 대상 | 방법 |
|---|---|
| 새 벤치마크 | `tads/evals/<name>.py`에 `BenchmarkEvaluator` 상속 클래스 작성 → `@register("<name>")` → `tads/evals/__init__.py` import 목록 추가. `evaluate()`는 `accuracy` 키를 포함한 dict 반환 |
| 새 선택 방법(baseline) | `baselines/<method>/`에 `train.py` 작성. 모델·데이터·SFT는 기존 모듈 재사용, 선택 로직만 신규. `configs/methods/<method>.yaml`에 `method:` 키와 하이퍼파라미터 정의 |
| 새 모델 패밀리 | `configs/models/<model>.yaml` 추가. 템플릿이 기존 5종과 다르면 `tads/data/sft_prompts.py`의 `tokenize_alpaca`와 각 `*_generation_prefix` 함수에 분기 추가. LoRA를 쓴다면 `lora.target_modules`를 반드시 명시(기본값은 Llama 계열 이름) |
| anchor 층 선택 변경 | `anchor.layer_indices`에 `"all"`, `"middle_to_last"`, 명시적 인덱스 리스트 지정 |
| 점수 규칙 ablation | `scorer.tads_score(calibrated_utility(R), ã, λ)`처럼 조합해 z-score 변형 사용 가능. 기본 경로는 원시 R |

## 11. 산출물 레이아웃

```
<output_root>/<output_subdir>/            # 실험 단위
  ├── _latest -> runs/<run_tag>           # symlink 또는 _latest.txt
  └── runs/<run_tag>/                     # 실행 단위
        ├── cfg.yaml, cfg.json            # 해석 완료된 설정 스냅샷
        ├── metrics.json                  # epoch별 loss + 선택 진단
        ├── selected_indices_epoch{N}.json
        ├── timing_breakdown.json         # 단계별 소요시간
        ├── logs/train_<method>_<ts>_r<rank>.log
        └── epoch_last/
              ├── (모델 가중치 + 토크나이저)
              ├── optimizer.pt, scheduler.pt
              ├── trajectory_anchor.pt, anchor_history.json   # tads 전용
              ├── env_meta.json
              ├── _complete                 # 내용 = epoch 번호
              └── _save_errors.json         # 저장 실패가 있었을 때만
```

평가 결과는 같은 규칙(`runs/<eval_tag>/` + `_latest` + `_complete`)을 그대로 따르며,
기본 위치는 `--out_dir` 미지정 시 `<ckpt>/eval/`이다.
