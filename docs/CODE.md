# TADS 코드 구조 (CODE)

파일별·함수별 역할과 핵심 로직을 실제 코드에서 확인한 수치와 함께 정리한다. 추정값은
쓰지 않았고, 계산으로 검증한 항목은 계산식을 함께 적었다.

## 1. 파일 트리와 규모

Python 63개 파일 14,168줄(`__pycache__` 제외), YAML 74개 1,306줄, 셸 17개 2,636줄.

```
.
├── README.md                 1줄 / 46바이트 / 6단어  ← 사실상 제목만 있는 껍데기
├── AUTO_EVAL_AGENT.md        3,084줄 / 약 203KB — 자동 평가 에이전트용 운영 지시서
├── pyproject.toml            패키징(setuptools), extras: eval / test
├── requirements.txt          런타임 + 평가 + 테스트 의존성(주석 포함)
├── .gitignore                체크포인트·로그·결과·`configs/env.yaml` 등 제외
├── tads/                     본체 패키지
├── baselines/                비교 baseline 6종
├── configs/                  계층형 YAML 74개
├── scripts/                  셸 17 + Python 2
└── tests/                    pytest 6개 파일 (수집 노드 25개)
```

파일별 줄 수(주요 파일만, 오름차순 아님):

| 파일 | 줄 |
|---|---:|
| `tads/train.py` | 921 |
| `tads/evals/tydiqa.py` | 821 |
| `tads/evals/mbpp.py` | 692 |
| `tads/evals/humaneval.py` | 607 |
| `tads/eval.py` | 567 |
| `tads/evals/mmlu_pro.py` | 540 |
| `tads/core/trajectory_anchor.py` | 525 |
| `tads/evals/xquad.py` | 453 |
| `tads/core/utils.py` | 448 |
| `tads/data/sft_prompts.py` | 434 |
| `tads/evals/bbh.py` | 409 |
| `tads/pipelines/selection.py` | 369 |
| `tads/core/selector.py` | 358 |
| `baselines/nait/train.py` | 339 |
| `tads/core/run_layout.py` | 338 |
| `tads/data/alpaca.py` | 332 |
| `baselines/selectit/train.py` | 323 |
| `tads/modeling/loader.py` | 307 |
| `baselines/alpagasus/train.py` | 303 |
| `tads/evals/mmlu.py` | 292 |
| `baselines/q2q/train.py` | 284 |
| `tads/pipelines/sft.py` | 282 |
| `baselines/data_agent/train.py` | 271 |
| `baselines/data_agent/select.py` | 268 |
| `baselines/selectit/score.py` | 259 |
| `baselines/nait/direction.py` | 256 |
| `baselines/q2q/score.py` | 251 |
| `baselines/data_agent/agent.py` | 237 |
| `tads/evals/gsm8k.py` | 227 |
| `tads/core/scorer.py` | 208 |
| `baselines/lima/train.py` | 197 |
| `tads/evals/svamp.py` | 178 |
| `baselines/lima/data.py` | 180 |
| `scripts/inspect_eval_data.py` | 180 |
| `tads/core/dist_utils.py` | 155 |
| `tads/core/timing.py` | 151 |
| `tads/evals/lm_harness.py` | 115 |
| `tads/core/data_io.py` | 107 |
| `tads/core/reward.py` | 106 |
| `tads/modeling/lora.py` | 65 |
| `tads/evals/base.py` | 58 |
| `tads/core/schedulers.py` | 51 |
| `scripts/make_table.sh` | 639 |
| `scripts/setup_env.sh` | 305 |

## 2. 의존 관계

import 방향은 단방향이다.

```
tads.train ──┬─→ tads.pipelines.{selection, sft}
             ├─→ tads.core.{run_layout, trajectory_anchor, utils, timing, schedulers}
             ├─→ tads.data.alpaca
             └─→ tads.modeling.loader

tads.pipelines.selection ─→ tads.core.{selector, trajectory_anchor, utils}
tads.core.selector       ─→ tads.core.{reward, scorer, trajectory_anchor, utils}
tads.core.scorer         ─→ (torch만)
tads.core.reward         ─→ (torch만)

tads.eval ──┬─→ tads.evals(registry) ─→ tads.evals.<bench> ─→ tads.data.sft_prompts
            ├─→ tads.modeling.loader
            └─→ tads.core.{utils, run_layout}

baselines.<m>.train ──┬─→ tads.core.{utils, schedulers, timing, data_io}
                      ├─→ tads.data.alpaca (lima만 baselines.lima.data)
                      ├─→ tads.modeling.loader
                      ├─→ tads.pipelines.sft
                      └─→ baselines.<m>.{direction|score|select|agent}
```

`tads.data.sft_prompts`는 학습 토크나이즈와 평가 프롬프트 생성 양쪽에서 쓰이는 유일한
공유 지점이다. `tads.core.dist_utils`는 어떤 프로덕션 모듈도 import하지 않는다(§7).

## 3. `tads/core` — 알고리즘 코어

### 3.1 `trajectory_anchor.py` (525줄)

`TrajectoryAnchor` 클래스 하나와 모듈 함수 1개.

| 요소 | 역할 |
|---|---|
| `_MAX_HISTORY = 50` | v/λ/stability 히스토리 보존 개수 상한 |
| `_resolve_layer_indices(spec, L)` | `"all"` → `range(L)`, `"middle_to_last"` → `range(L//2, L)`, list → 검증 후 그대로. `None`이면 예외를 던져 호출자가 legacy 경로로 폴백하게 함 |
| `__init__` | 기본값 `layer_idx=-1`, `layer_indices=None`, `max_samples_for_pca=1024`, `pca_batch_size=4`, `device="cuda"` |
| `is_fitted` / `is_multi_layer` | `v_by_layer` 비어있지 않은지 / `layer_indices_spec is not None` |
| `_pca_top1(delta)` | (N,H) 행렬 중심화 후 top-1 고유벡터. `N < H`면 (N,N) Gram 행렬로 `eigh`를 돌리고 `v = centredᵀ·u`를 정규화, 아니면 (H,H) 공분산에서 직접. λ1, λ2, 평균 μ 반환 |
| `update(model, dataset, seed, epoch)` | probe 추출 → forward 루프 → 층별 PCA → 부호 보정 → 상태 커밋 → 통계 반환 |
| `compute_alignment(states)` | 2-D/3-D 입력을 받아 alignment 계산 후 최소-최대 정규화. **어디서도 호출되지 않는다**(§7) |
| `get_history_summary` / `state_dict` / `load_state_dict` | 진단용 요약, 직렬화. `load_state_dict`는 multi-layer 이전 형식(`"v"` 단일 키) 체크포인트도 변환해 읽는다 |

`update()`의 세부 동작과 근거 수치:

- **probe 시드 오프셋 +1**: `g.manual_seed(seed + epoch*100 + 1)`. `pipelines/selection.py`의
  `_random_indices`가 `seed + epoch*100`을 쓰기 때문에, 오프셋이 없으면 두 난수열이
  동일해져 `random` 방법이 고르는 샘플과 probe 집합이 완전히 겹친다. 그 경우 anchor가
  "곧 학습될 바로 그 샘플들" 쪽으로 편향된다.
- **Δh 계산**: `hidden_states[li + 1]`(인덱스 0은 임베딩)에서 `first_h = h[:, 0, :]`,
  `last_h = h[bidx, lengths]`, `lengths = attention_mask.sum(1).clamp_min(1) - 1`.
- **전송 최적화**: 층별로 `.cpu()`를 호출하면 배치마다 32번 동기화된다. 현재 코드는
  32개 (B,H) 텐서를 GPU에서 `torch.stack` 후 한 번에 CPU로 옮긴다(배치당 동기화 1회).
- **부호 보정**: `prev_v_l = self.v_by_layer.get(li)`가 있으면 시간축
  (`dot(v_l, prev_v_l) < 0`이면 반전), 없으면 공간축(`dot(v_l, mu) < 0`이면 반전).
- **메모리 회수**: 층 처리 직후 `per_layer_deltas[li] = []`와 `del delta_l`로 원시 리스트와
  cat된 행렬을 즉시 놓아준다. 그 결과 CPU 사용량이 루프 진행에 따라 감소한다.
- **stability**: 이전과 현재 v_l의 L2 거리 평균. 층 집합이 바뀌었거나 첫 호출이면 NaN.
- **진행 로그**: forward는 1번째/20배수/마지막 배치에, PCA는 1번째/8배수/마지막 층에
  경과시간과 ETA를 남긴다.

**주석의 메모리 수치 오류(산술로 검증됨)**

- "각 층 리스트가 N_probe×H ≈ 540 MB (N_probe=1024, H=4096)": 실제는
  `1024 × 4096 × 4바이트 = 16,777,216 B ≈ 16.8 MB`다. 540 MB는 **32개 층 전체 합**
  (`16.78 MB × 32 ≈ 537 MB`)에 해당한다.
- "32개 층이면 dict만 ~17 GB": 위 계산대로면 약 0.54 GB다. 약 32배 과대 표기.
- `_MAX_HISTORY` 주석 "L=32, H=4096, fp32에서 항목당 ~50MB": 실제는
  `32 × 4096 × 4 = 524,288 B ≈ 0.52 MB`다. 약 100배 과대 표기.

세 수치 모두 코드 동작에는 영향이 없고 주석만 틀렸다.

### 3.2 `reward.py` (106줄)

- `compute_rewards(logits, labels, eps=1e-8)` → `(r_loss, r_entropy, r_weight)`
  - `shift_logits = logits[:, :-1, :]`, `shift_labels = labels[:, 1:]`
  - `resp_mask = (shift_labels != -100)`, `n_resp = resp_mask.sum(-1).clamp(min=1)`
  - **배치 안에서 샘플별 루프**를 돈다. 한 번에 처리하면 (B,T,V) fp32 텐서가 세 개
    필요하지만, 샘플 단위면 (T,V) 두 개면 된다. 주석은 Qwen2.5(V=151k),
    episode_batch_size=16 기준 엔트로피 피크가 약 15 GB → 약 1 GB로, Llama2(V=32k)에서는
    약 3 GB → 약 256 MB로 줄었다고 기록한다.
  - 엔트로피는 `log_softmax` 후 `-(exp(lp) * lp).sum(-1)`로 계산(수치 안정).
  - 반환되는 `r_weight`는 **배치 내부** 분산비다. batch_size=1이면 0이 된다.
    `selector.collect_episode`는 이 값을 버리고 풀 전체로 다시 계산한다.
- `composite_reward(r_loss, r_entropy, r_weight)` — 단순 가중합. 테스트에서만 사용된다.

### 3.3 `scorer.py` (208줄)

순수 함수 5개. GPU 불필요, 모두 CPU 텐서 연산.

| 함수 | 수식 / 규칙 | 상수 |
|---|---|---|
| `pool_reward(L, H, eps)` | `w = Var(L)/(Var(L)+Var(H)+ε)`, `R = w·L + (1−w)·H` | `eps=1e-8`, 표본 2개 미만이면 `w=0.5` |
| `calibrated_utility(R, eps)` | `(R − mean)/(std + ε)`. **ablation 전용** | `eps=1e-6`, `_STD_FLOOR=1e-4`. std가 floor 미만이면 스케일링을 건너뛰고 `R − mean`을 반환하며 경고 |
| `normalize_alignment(a, collapse_eps)` | 최소-최대 정규화. 폭이 임계 미만이면 전부 0.5로 채우고 `collapsed=True` | `collapse_eps=1e-8` |
| `tads_score(R, ã, λ)` | `s = R · (1 + λ·ã)`. 모양 불일치·음수 λ는 예외 | 부스트 범위 `[1, 1+λ]` |
| `select_top_b(scores, b)` | `scores.topk(b).indices` | `b<=0`, `b>N`, NaN/Inf는 모두 예외로 거부(조용한 축소 금지) |

`_STD_FLOOR` 가드의 근거: `eps=1e-6`으로 나누면 거의 균일한 풀에서 z-score가 1e6 규모로
폭발해 top-B가 부동소수점 반올림 잡음으로 결정된다. NaN/Inf가 아니라서 `select_top_b`의
검사에도 걸리지 않는다.

### 3.4 `selector.py` (358줄)

`collect_episode(...)` 하나가 본체이고, 보조로 `_flatten_cpu_float`, `_unwrap`이 있다.

기본 인자: `lam=0.0`, `use_anchor=False`, `batch_size=1`, `seed=42`, `epoch=0`,
`progress_interval=50`, `empty_cache_interval=10`.

처리 순서:

1. `model.eval()`, `use_cache=False` 강제(`_unwrap`으로 DDP/PEFT 래퍼 제거 후 config 수정)
2. `apply_anchor = use_anchor and anchor is not None and anchor.is_fitted`
3. anchor를 쓸 경우 층별 v_l을 fp32로 GPU에 미리 올려 캐시(`v_cache_gpu`)
4. 배치 루프
   - alignment: `mask_f`를 **fp32로 먼저 캐스팅**한 뒤 hidden state와 곱한다. fp16에서
     T=512 위치를 더하면 65,504 상한을 넘길 수 있어서다.
   - `valid_counts`가 0인 행이 있으면 경고를 남기고 `clamp_min(1)`로 NaN을 막는다.
   - 층별 `mean_h_l @ v_l`을 더한 뒤 마지막에 `len(layer_indices)`로 한 번 나눈다.
   - `compute_rewards`로 (L_i, H_i)를 얻어 CPU에 누적
   - `empty_cache_interval=10`마다 `torch.cuda.empty_cache()`
5. 루프 후: `pool_reward` → `normalize_alignment` → `tads_score` → `select_top_b`
6. `k = max(1, int(total_samples * selection_ratio))`
7. alignment가 collapse되면 ERROR 로그를 남긴다. 이 경우 `1 + λ·0.5`가 상수가 되어
   TADS가 조용히 합성 보상 단독 ranking으로 퇴화하기 때문이다.

반환 dict에는 `selected_indices`, `rewards`, `alignment`, `r_loss_mean`, `r_entropy_mean`,
`r_weight`, `var_loss`, `var_entropy`, `lam`, `use_anchor`, `align_mean`, `align_std`,
`alignment_collapsed`가 들어가고, 구버전 분석 스크립트 호환용 alias로
`rdiff_mean`(=r_loss_mean), `rconf_mean`(=r_entropy_mean), `r`(=r_weight)가 추가된다.

모듈 docstring은 구현이 논문 서술과 다른 지점을 명시한다. 논문은 probe와 후보 활성화를
**한 번의 forward**로 얻는다고 하지만 구현은 두 번 돈다(probe 전용 + 풀 전체). 수학적
결과는 동일하고 probe(약 1k)가 풀(약 50k) 대비 작아 벽시계 시간 5~10% 추가라고 적혀 있다.

### 3.5 `utils.py` (448줄)

| 함수 | 역할 / 세부 |
|---|---|
| `_DedupLogFilter` / `quiet_repeated_warnings()` | `(logger, level, message)` 조합당 1회만 통과. ERROR 이상은 항상 통과. `transformers`, `datasets`, `peft`, `accelerate`, `py.warnings` 5개 로거에 부착하고 `logging.captureWarnings(True)`로 stdlib 경고까지 흡수 |
| `_cuda_smoke_test()` | cuda에 1원소 텐서를 만들고 `add_` 후 `synchronize`. `torch.cuda.is_available()`가 True여도 커널 실행이 실패하는 환경을 진입 시점에 드러낸다 |
| `clear_runtime_caches()` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` setdefault → `gc.collect()` → `empty_cache()` → `ipc_collect()` → smoke test → 한 줄 요약 로그. `TADS_FRESH_DATA_CACHE` 값도 함께 로깅 |
| `disable_coredumps()` | `resource.setrlimit(RLIMIT_CORE, (0,0))`. `TADS_ENABLE_COREDUMPS=1`이면 건너뜀. ImportError/ValueError/OSError를 삼킨다 |
| `set_seed(seed)` | random / numpy / torch / cuda 전부 |
| `cuda_mem_str()` | `mem=<allocated>/<reserved>GB` 문자열 |
| `is_main_process` / `local_rank` / `world_size` / `rank` | DDP 헬퍼. `local_rank`는 환경변수 `LOCAL_RANK` |
| `setup_logger(log_dir, name, level)` | 기존 root 핸들러를 **close 후 제거**한 뒤 재구성. `force=True`만 쓰면 FileHandler fd가 새기 때문 |
| `backup_latest_if_exists(output_dir)` | `latest.pt`를 백업. **호출처 없음**(§7) |
| `_resolve_env` / `_ENV_PAT` | `${oc.env:VAR,default}` 패턴 치환. dict/list 재귀 |
| `_deep_merge(base, override)` | 중첩 dict는 재귀 병합, 그 외는 override 우선 |
| `_collect_path_candidates(ref, anchor)` | 절대경로 → cwd → 패키지 루트 → anchor 파일 기준 최대 10단계 상위 탐색 |
| `load_config(path)` | `defaults:`가 문자열이면 리스트로 변환, 리스트면 순서대로 재귀 로드 후 누적 병합, 마지막에 로컬 키 병합, 최종적으로 환경변수 보간 |

설정 우선순위(코드 기준, 뒤가 이김): `defaults` 리스트 앞 → 뒤 → 해당 파일의 로컬 키.
실험 YAML의 관례적 순서는 `base → methods/<m> → models/<model> → modes/<mode>`이므로,
겹치는 키는 **mode가 model을 이긴다**. `modes/full_ft.yaml`이 `learning_rate`를 일부러
설정하지 않는 이유가 여기에 있다(모델별 recipe를 살리기 위함).

### 3.6 `run_layout.py` (338줄)

`_RUN_TAG_PAT = ^[A-Za-z0-9._-]+$`로 태그를 제한한다(디렉터리명 겸 symlink 대상).

| 함수 | 역할 |
|---|---|
| `make_run_tag(suffix)` | `YYYYMMDD_HHMMSS` (+ `_<suffix>`) |
| `run_dir_for(output_dir, tag)` | `<output_dir>/runs/<tag>` |
| `list_runs(output_dir)` | `(tag, path)` 오름차순 |
| `resolve_latest(output_dir)` | symlink → 실제 디렉터리 → `_latest.txt` 순으로 해석 |
| `update_latest(output_dir, tag)` | 임시 symlink 생성 후 `os.replace`로 원자 교체. 실패하면 `_latest.txt`에 기록하고 `"textfile"` 반환 |
| `save_cfg_snapshot(run_dir, cfg)` | `cfg.yaml`과 `cfg.json` 둘 다 원자적으로 기록 |
| `load_cfg_snapshot(run_dir)` | 스냅샷 읽기. **호출처 없음**(§7) |
| `_is_sealed_ckpt_dir(p)` | `_complete` 존재 **그리고** `config.json` 또는 `adapter_config.json` 존재 |
| `_read_sentinel_epoch(p)` | `_complete` 내용을 int로. 실패 시 0 |
| `find_latest_complete_epoch(run_dir)` | `epoch_last/`가 sealed면 그것을, 아니면 legacy `epoch_N/` 중 sealed 최대 N |
| `resolve_eval_ckpt(...)` | 평가 대상 체크포인트 해석. **호출처 없음**(§7) |

### 3.7 나머지 코어 모듈

- **`schedulers.py`** — `get_cosine_schedule_with_warmup(optimizer, num_warmup_steps,
  num_training_steps, num_cycles=0.5, last_epoch=-1)`. transformers 구현을 vendoring한
  LambdaLR 래퍼. warmup 구간은 선형, 이후 `0.5·(1+cos(π·2·num_cycles·progress))`.
- **`timing.py`** — `PhaseTimer`. 카테고리는 `setup / data / selection / sft / checkpoint /
  misc` 6종이며 목록에 없는 값은 `misc`로 강등된다. `report()`는 phase별 누적/평균/비율과
  카테고리별 합계, 그리고 "tracked vs untracked"(`with timer.phase` 밖의 시간)를 낸다.
  `save_report()`는 tmp 후 `replace`, `log_table()`은 88칸 폭의 표를 로그로 남긴다.
- **`data_io.py`** — `read_records(path)`는 확장자로 분기하고, 알 수 없는 확장자는 첫
  4바이트가 `PAR1`이면 parquet으로 본다. JSON/JSONL 구분은 **첫 비공백 문자가 `[`인지**로
  하고, 파일은 `utf-8-sig`로 열어 BOM을 제거한다(BOM은 `str.isspace()`가 아니라서
  판별자를 망가뜨렸던 이력이 주석에 있다). `read_records_glob(spec)`은 `~` 확장 후
  정렬된 매치들을 순서대로 이어붙인다.
- **`dist_utils.py`** — `is_dist_initialized`, `get_global_rank`, `get_world_size`,
  `is_global_main`, `log_rank_predicates`, `all_gather_concat`, `broadcast_selection`.
  전역 rank만 사용하고 local rank 계열 판정을 금지한다는 규약이 docstring에 명시돼 있다.
  **테스트 외 호출처 없음**(§7).

## 4. `tads/data` — 데이터와 프롬프트

### 4.1 `alpaca.py` (332줄)

- `_LOCK_TIMEOUT_SEC = 4 * 60 * 60`(4시간). 첫 토크나이즈가 52K 샘플 기준
  `num_proc=4`에서 약 2분이라 상한은 "락이 걸려 죽은 상태"를 드러내기 위한 값이다.
- `_dataset_build_lock(cache_dir)` — `filelock.FileLock`을 `.tads_dataset_build.lock`에
  건다. filelock import 실패 시 경고 후 `nullcontext`로 폴백.
- `_LoggingFileLock` — 획득/해제 시각과 대기 시간을 로그로 남긴다(대기 중 무음 방지).
- `_resolve_data_files(spec)` — glob 메타문자가 있으면 확장, 없으면 존재 확인,
  없으면 같은 디렉터리에서 **basename의 첫 하이픈 앞 토큰**으로 시작하는
  `*.json/*.jsonl/*.parquet`를 찾는다. "디렉터리의 모든 json"이 아니라 접두사 기준인
  이유는 `valid.json`/`test.json`이 학습셋에 섞이는 것을 막기 위함이다.
- `verify_response_marker(tokenizer)` — `### Response:\n`을 인코딩해 더 긴 문자열
  안에서 재현되는지 확인하고 로그만 남기는 진단 함수. 토크나이즈 로직 자체는 마커
  탐색에 의존하지 않는다.
- `build_alpaca_dataset(...)` — 락 안에서 `_build_alpaca_dataset_locked` 호출.
  로컬 파일이 있으면 확장자로 loader를 고르고(`json`/`csv`/`text`/`parquet`), 없고
  오프라인 플래그가 켜져 있으면 **hub 다운로드를 거부하고** 조치 방법 3가지를 담은
  `FileNotFoundError`를 던진다. 마지막에 `Dataset.map(..., num_proc=4)`로 토크나이즈하고
  `set_format("torch")`. `TADS_FRESH_DATA_CACHE=1`이면 `load_from_cache_file=False`.

### 4.2 `sft_prompts.py` (434줄)

- 상수: `IM_START`/`IM_END`(문자열을 쪼개서 정의 — 파일 자체가 특수 토큰을 그대로 담지
  않게 함), `ASSISTANT_TAG`, `PROMPT_STYLE_BY_MODEL_ID`(6개 매핑),
  `ALPACA_PROMPT_PREFIX`, `_TEMPLATE_HAS_BOS = {"mistral_instruct"}`.
- `tokenize_alpaca(example, tokenizer, max_seq_len, prompt_style)` — 5개 스타일 분기:
  `alpaca_default`, `qwen_chatml`, `mistral_instruct`,
  `llama_user_assistant`/`deepseek_user_assistant`(동일 포맷).
  - pad id 결정: `pad_token_id` → 없으면 `eos_token_id` → 그것도 없으면 0. `num_proc>1`로
    토크나이저가 피클될 때 `pad_token`이 None으로 되돌아가는 HF 버전이 있어, 0으로
    바로 떨어지면 패딩 자리가 `<unk>`나 `!` 같은 토큰으로 채워진다.
  - prompt/response 별도 인코딩 후 연결. Llama-2 SentencePiece는 문맥 의존이라 전체
    문자열에서 마커를 찾는 방식이 성립하지 않는다.
  - `len(prompt_ids) >= max_seq_len`이면 경고 후 `prompt_ids[:max_seq_len-1]`로 잘라
    **response 토큰을 최소 1개 보장**한다. 그러지 않으면 labels 전체가 -100이 되어
    해당 샘플의 학습 신호가 0이 된다.
  - `labels = [-100]*len(prompt_ids) + response_ids`, 뒤에 패딩분 -100.
- `GSM8K_COT_8SHOT` — 8개 (질문, 풀이) 쌍. `build_cot_prompt_prefix(q)`가
  `Q: ...\n\n A: ...` 형태로 이어 붙이고 마지막에 테스트 문항을 `A: `로 끝낸다.
- 평가용 prefix 함수: `gsm8k_generation_prefix`, `tydiqa_user_block`,
  `tydiqa_generation_prefix`, `humaneval_generation_prefix`, `mbpp_generation_prefix`.
  모두 동일한 5개 스타일 분기를 갖는다.
  - `mistral_instruct`에서는 **앞의 `<s>`를 일부러 넣지 않는다**. 평가 쪽
    `tokenizer(prefix)`가 기본 `add_special_tokens=True`라 BOS가 자동으로 붙는데, 템플릿에도
    넣으면 학습 때 본 적 없는 이중 BOS가 된다.
  - TyDiQA의 `alpaca_default` 분기는 Alpaca 템플릿으로 감싸지 **않고** 원문 few-shot을
    그대로 반환한다. 감싸면 SFT 모델이 완결형 문장으로 답해 짧은 정답 구간 EM이 무너진다는
    설명이 주석에 있다.
  - HumanEval 주석은 감싸기 유무로 pass@10이 약 0.27과 0.08로 갈렸다고 기록한다.
    MBPP 주석은 감싸지 않은 경우 pass@1이 참조값 51.58 대비 약 10pt 낮은 41 근처였다고
    기록한다.

## 5. `tads/modeling`

- **`loader.py`**
  - `_resolve_local_path(path)` — 경로가 없으면 부모 디렉터리에서 대소문자 무시 동명
    후보를 찾아 하나면 그것을 쓰고 경고. 리눅스 대소문자 구분과 모델 디렉터리 명명
    관습 차이를 흡수한다.
  - `load_tokenizer(path)` — `trust_remote_code=True`, `local_files_only=True`,
    `pad_token`이 없으면 `eos_token`으로 채움.
  - `load_model(...)` — 기본 dtype `bfloat16`. DDP가 아니면 `device_map="auto"`,
    DDP면 `cuda:local_rank`로 옮긴 뒤 `DistributedDataParallel`로 감싼다.
    `find_unused_parameters`는 LoRA일 때만 True.
    `attn_implementation="flash_attention_2"`가 실패하면 `sdpa`로 재시도한다(이 폴백은
    flash를 명시적으로 요청했을 때만 동작하고, 그 외 예외는 그대로 올린다).
    gradient checkpointing은 `use_reentrant=False`로 켜고(구 경로는 PEFT와 충돌),
    `enable_input_require_grads()` 호출 후 `config.use_cache=False`로 고정한다.
  - `load_for_eval(...)` — adapter 유무로 모드 자동 판정. LoRA는 `merge_and_unload()`를
    시도하고 실패하면 래핑 상태로 진행. 마지막에 `generation_config`의
    `max_length/temperature/top_p/top_k`를 None으로 지운다. 평가기들이 매 호출
    `max_new_tokens`를 명시하는데 baked-in 기본값이 남아 있으면 생성 호출마다 경고가
    나와 로그가 묻힌다(벤치당 1.3K~6.5K회).
  - `get_hidden_size(model)` — DDP/PEFT 래퍼를 벗기고 `config.hidden_size` 반환.
- **`lora.py`** — `build_lora_config`. 기본값 `r=16`, `alpha=32`(`lora_alpha`도 허용),
  `dropout=0.05`(`lora_dropout`도 허용), `bias="none"`. `target_modules` 미지정 시
  Llama 계열 7개(`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`)로 폴백하며
  경고를 남긴다. `alpha/r > 64`면 오타 가능성 경고.
- **`__init__.py`** — `build_lora_config`만 모듈 수준 `__getattr__`로 지연 노출한다.
  full-FT 실행이 peft 버전 불일치로 죽지 않게 하려는 조치.

## 6. `tads/pipelines`

### 6.1 `selection.py` (369줄)

상수: `_POLL_INTERVAL_SEC = 2.0`, `_POLL_TIMEOUT_SEC = 6*60*60`(6시간).

- `_random_indices(n, ratio, seed, epoch)` — `seed + epoch*100` 시드로 순열 후 앞
  `max(1, int(n*ratio))`개.
- `_broadcast_selection(selected, epoch, output_dir)`
  - 분산이 아니면 리스트로 정규화해 그대로 반환.
  - rank 0: 직전 4개 epoch의 `_selection_epoch{N}.json` / `.ready`와 현재 epoch의 잔여
    sentinel을 먼저 지운 뒤, 선택 리스트를 tmp→fsync→rename, 이어서 `.ready`도 같은 방식.
  - 그 외 rank: `.ready`를 2초 간격 폴링, 60초마다 진행 로그, 상한 초과 시 예외.
    sentinel은 있는데 본체가 없으면 즉시 예외.
  - **끝에 배리어를 두지 않는다.** 첫 backward의 all_reduce가 정렬 지점이다.
- `select_indices(method, ...)`
  - `full` → 전체 인덱스, `random` → `_random_indices`.
  - `_BASELINE_METHODS = {data_agent, lima, nait, selectit, alpagasus, q2q}`에 속하면
    `python -m baselines.<m>.train --config ... --tag ...` 명령을 담은 `ValueError`.
  - 그 외 `tads`가 아니면 알 수 없는 method 오류.
  - **선택 캐시**: `<output_dir>/selected_indices_epoch{N}.json`이 있으면
    `collect_episode`를 건너뛰고 그 인덱스를 그대로 공유한다(`selection_cache_reused`
    플래그를 extras에 남김). 이 경로에서는 `anchor.update()`도 실행되지 않는다.
  - rank 0에서만 `anchor.update()` → `collect_episode` 실행. 예외는 traceback을 찍고
    다시 올린다. 진행 상황을 `print(..., flush=True)`로도 남기는데, 로거 버퍼링과 무관하게
    hang 위치를 특정하기 위한 장치다.
  - `output_dir`이 없으면 공유 파일을 `/tmp/tads_selection_share`에 쓴다.
- `save_selection(output_dir, epoch, selected)` — rank 0만, 원자적 기록.

### 6.2 `sft.py` (282줄)

- `_ALWAYS_LOG_FIRST = 10`, `_ALWAYS_BARRIER_FIRST_N_BOUNDARIES = 3`.
- `_collate(batch)` — `input_ids/attention_mask/labels`를 `torch.stack`.
- `make_dataloader(...)` — 생성기 시드 `seed + epoch*100`. 분산이면
  `DistributedSampler(shuffle=..., seed=seed)`를 만들고 `shuffle=False`로 전환.
  `worker_init_fn`은 `seed + epoch*100 + worker_id`. `TADS_DL_NUM_WORKERS`로 워커 수를
  덮어쓸 수 있다. **epoch를 시드에 섞는 이유**: 단일 GPU에서는 `set_epoch` 경로가 없어
  매 epoch 동일한 셔플 순서가 나온다.
- `sft_one_epoch(...)`
  - `DistributedSampler`면 `set_epoch(epoch)`.
  - 루프 진입 전에 `gc.collect()` + `empty_cache()` + `synchronize()`로 선택 단계의 잔여
    버퍼를 정리한다.
  - `no_sync()`는 `TADS_ENABLE_NO_SYNC`가 참일 때만 사용. 기본 비활성.
  - 경계 스텝: `clip_grad_norm_` → `optimizer.step()` → `scheduler.step()` →
    `zero_grad(set_to_none=True)`. 앞 3회 경계에서는 배리어와 대기시간 로그.
  - 초반 10 스텝은 항상 로그 + `cuda.synchronize()`.
  - 반환 직전 분산이면 평균 loss를 `all_reduce(SUM)` 후 world_size로 나눠 전역 평균으로
    만든다.

## 7. 미사용 코드·미참조 자산(실측)

전체 저장소에 대해 심볼 단위 grep과 AST 스캔으로 확인한 결과다.

### 7.1 정의만 있고 호출되지 않는 함수

| 심볼 | 위치 | 비고 |
|---|---|---|
| `TrajectoryAnchor.compute_alignment` | `core/trajectory_anchor.py:423` | 클래스 밖 어디에서도 호출되지 않는다. `selector.collect_episode`가 alignment를 직접 계산하기 때문. 게다가 3-D 경로는 `Σ_l ⟨states_l, v_l⟩`을 **L로 나누지 않는데**, selector는 나눈다. 두 경로의 스케일이 다르므로 되살릴 때 주의가 필요하다(최소-최대 정규화 뒤에는 순위가 같지만 값 자체는 다르다) |
| `core/utils.backup_latest_if_exists` | `core/utils.py:299` | 저장소 어디에서도 호출되지 않는다. 참조하는 `latest.pt` 파일명은 현재 체크포인트 레이아웃에 존재하지 않는다 |
| `core/run_layout.resolve_eval_ckpt` | `core/run_layout.py:263` | 76줄짜리 해석 로직 전체가 미사용. `tads/eval.py`가 동등한 규칙을 인라인으로 재구현했다 |
| `core/run_layout.load_cfg_snapshot` | `core/run_layout.py:179` | 저장 함수만 쓰이고 로드는 미사용 |
| `data/sft_prompts.alpaca_prompt_text` | `sft_prompts.py:68` | 미사용 |
| `data/sft_prompts.prompt_style_for_model_id` / `PROMPT_STYLE_BY_MODEL_ID` | `sft_prompts.py:26,36` | 미사용. 실제 `prompt_style`은 모델 YAML에서 온다 |
| `core/reward.composite_reward` | `reward.py:100` | 테스트에서만 호출. 실제 경로는 `scorer.pool_reward`가 담당 |
| `core/scorer.calibrated_utility` | `scorer.py:73` | 테스트에서만 호출. 코드 주석이 "ablation 전용, 기본 경로 아님"이라고 명시 |

### 7.2 테스트에서만 쓰이는 모듈 전체

`tads/core/dist_utils.py`(155줄) — `broadcast_selection`, `all_gather_concat`,
`is_global_main`, `log_rank_predicates`, `get_global_rank`, `get_world_size`가 모두
`tests/test_broadcast_selection.py`에서만 import된다. 프로덕션 선택 경로는
`pipelines/selection.py`의 파일 폴링 구현이다. 즉 이 테스트가 통과해도 실제 사용되는
선택 공유 경로는 검증되지 않는다.

### 7.3 미참조 설정 자산

- `configs/models/qwen2.5-14b.yaml` — 55개 실험 YAML 중 어느 것도 참조하지 않는다.
  (`sft_prompts.PROMPT_STYLE_BY_MODEL_ID`에 항목이 있고 `setup_env.sh`가 환경변수를
  내보내지만, 그 매핑 함수 자체가 미사용이다.)
- `configs/methods/*.yaml` 참조 횟수(정확 문자열 기준): `tads` 14, `data_agent` 13,
  `random` 13, `full` 9, `nait` 2, `alpagasus`/`lima`/`q2q`/`selectit` 각 1.
- `configs/models/*.yaml` 참조 횟수: `llama2-7b` 20, `qwen2.5-7b` 12,
  `qwen2.5-0.5b` 9, `deepseek-7b` 7, `mistral-7b` 7, `qwen2.5-14b` 0.
- `configs/modes/*.yaml`: `full_ft` 50, `lora` 5.

### 7.4 사용되지 않는 import (AST 스캔, `from __future__ import annotations` 제외)

| 파일 | 미사용 import |
|---|---|
| `tads/train.py` | `logging`, `sys` |
| `tads/eval.py` | `logging` |
| `tads/pipelines/selection.py` | `TrajectoryAnchor`, `local_rank`, `rank`, `world_size`, `Any`, `Dict`, `List`, `Optional`, `Tuple` |
| `tads/evals/bbh.py` | `os` |
| `tads/evals/mbpp.py` | `tempfile` |
| `tads/evals/xquad.py` | `string` |
| `baselines/nait/train.py` | `logging`, `numpy as np`, `time` |
| `baselines/selectit/train.py` | `glob`, `time` |
| `baselines/q2q/train.py` | `glob` |
| `baselines/alpagasus/train.py` | `glob` |
| `baselines/lima/data.py` | `glob`, `json`(함수 안에서 `import json as _json`로 다시 들여옴) |
| `tests/test_utils.py` | `os` |

`tads/evals/__init__.py`의 10개 모듈 import는 레지스트리를 채우기 위한 의도된
side-effect import다(미사용이 아님).

## 8. 실제 버그·불일치(근거 포함)

### 8.1 `scripts/run_main_7b.sh` 기본 실행이 반드시 1/4 실패한다

- `scripts/run_main_7b.sh:35` — `METHODS=${METHODS:-"full_100 random_10 data_agent_10 tads_10"}`
- 같은 파일의 `run_ddp()`와 `run_parallel_single_gpu()`는 항상 `-m tads.train`을 실행한다.
- `configs/experiments/main_7b/<model>/data_agent_10.yaml`은
  `configs/methods/data_agent.yaml`을 상속하므로 `method: data_agent`다.
- `tads/pipelines/selection.py:219-232` — `data_agent`는 `_BASELINE_METHODS`에 있어
  `ValueError`로 거부되고 `python -m baselines.data_agent.train`을 쓰라고 안내한다.

결과적으로 옵션 없이 `bash scripts/run_main_7b.sh`를 돌리면 4개 방법 중 `data_agent_10`
셀이 즉시 예외로 끝난다. 대비되는 사례로 `scripts/run_evol_7b.sh`는 `entry_for()` /
`extra_args_for()` 함수로 `data_agent*`를 `baselines.data_agent.train --tag DataAgent-PPO`로
분기시켜 같은 문제를 피한다.

### 8.2 `anchor.layer_idx` 단일 층 모드가 도달 불가

- `configs/base.yaml`의 `anchor` 블록은 `layer_indices: all`을 포함한다.
- `core/utils._deep_merge`는 중첩 dict를 재귀 병합한다.
- `configs/experiments/light_tads_05b.yaml`, `sanity_tads_7b.yaml`,
  `7b_fullft_tads_50.yaml`은 `anchor:` 블록에 `layer_idx: -1`만 적고 `layer_indices`를
  지우지 않는다.

따라서 병합 결과에는 `layer_indices: all`이 남고, `TrajectoryAnchor.is_multi_layer`가
True가 되어 `layer_idx`는 무시된다. 단일 층 경로(`selector.py`의 legacy 분기,
`trajectory_anchor.update`의 legacy 분기)는 이 저장소의 설정만으로는 실행되지 않는다.
단일 층을 실제로 쓰려면 실험 YAML에서 `anchor.layer_indices: null`을 명시해야 한다.

### 8.3 Q2Q의 IFD 프롬프트가 학습 프롬프트와 다르다

- `baselines/q2q/train.py`의 `_make_alpaca_prefix(ins)`는
  `alpaca_input_part("")`를 호출하므로 `input` 블록을 **항상 비운다**.
- 같은 파일 `_extract_ins_res(rec)`는 `input`이 있으면 `f"{ins}\n{inp}"`로 instruction에
  이어 붙인다.
- 반면 학습 경로(`sft_prompts.tokenize_alpaca`의 `alpaca_default`)는 input이 있을 때
  `", using the input below as context\n\n### Input:\n{inp}"` 블록을 넣는다.

즉 input 필드가 있는 샘플에서 IFD 점수를 매길 때 쓰는 프롬프트와 그 샘플이 실제로
학습될 때의 프롬프트가 다르다. IFD의 조건부 항 `PPL(y|x)`가 학습 조건과 어긋난다.

### 8.4 `make_table.sh`의 행 레이블과 데이터 출처가 어긋난다

`scripts/make_table.sh`의 `METHODS` 정의에서

```
("08",  "Composite-reward only (λ=0)",  "data_agent_10"),
```

라벨은 "합성 보상 단독(λ=0)" 즉 TADS의 ablation을 가리키지만, 실제로 읽는 디렉터리는
`data_agent_10`으로 PPO 기반 Data Agent baseline의 결과다. λ=0 ablation을 위한 별도
설정 디렉터리는 저장소에 없다(모든 `tads` 실험은 `configs/methods/tads.yaml`의
`lam: 1.0`을 쓴다).

### 8.5 문서와 코드의 대표 지표가 어긋난다

`AUTO_EVAL_AGENT.md`의 지표 표:

| bench | 문서가 명시한 키 |
|---|---|
| `tydiqa` | `accuracy_em` |
| `xquad` | `accuracy` (= `macro_em`) |

코드:

- `tads/evals/tydiqa.py` — `accuracy = f1_score`, EM은 `accuracy_em`에 별도 보관
- `tads/evals/xquad.py` — `accuracy = macro_f1`, EM은 `accuracy_em`/`macro_em`

`make_table.sh`는 `accuracy` 키만 읽으므로 표에는 F1이 들어간다(헤더도 "F1"로 적혀
있어 표 자체는 일관적). 어긋난 것은 `AUTO_EVAL_AGENT.md` 쪽이다.

### 8.6 설정 주석이 구버전 수식을 설명한다

`configs/methods/tads.yaml` 상단 주석과 `configs/experiments/main_7b/llama2/tads_10.yaml`,
`configs/experiments/evol_7b/*/tads_10.yaml` 주석은 아래처럼 적혀 있다.

```
s_i = R̃_i · (1 + λ · ã_i)     (Eq.8)
R̃_i = (R_i - R̄) / (σ_R + 1e-6)   calibrated utility (pool z-score)
```

현재 코드는 `scorer.tads_score(R, ã, λ)`에 **원시 R**을 넘기며(`selector.py:262-275`),
`scorer.py`의 docstring은 z-score를 ranking 규칙에서 배제한다고 명시한다.
`configs/experiments/main_05b/qwen25/tads_10.yaml` 주석만 현재 코드(Eq.10, 원시 R)와
일치한다.

### 8.7 baseline 체크포인트는 `_complete` sentinel을 쓰지 않는다

`baselines/{nait,data_agent,selectit,q2q,lima,alpagasus}/train.py`는 모두
`output_dir/epoch_last`에 `save_pretrained`만 하고 sentinel을 남기지 않는다. 따라서
`run_layout._is_sealed_ckpt_dir`가 False를 돌려주고 `find_latest_complete_epoch`로는
찾히지 않는다. 평가 시 `--ckpt <...>/epoch_last`를 직접 지정해야 하며, 각 baseline
설정 YAML 주석도 그 형태의 명령을 안내한다.

### 8.8 `--limit`의 의미가 벤치마크마다 다르다

- 과목/태스크/언어 **단위로** 적용: `mmlu`(57과목), `bbh`(태스크별), `xquad`(언어별)
- 전체 샘플 수에 적용: `gsm8k`, `svamp`, `humaneval`, `mbpp`, `tydiqa`, `mmlu_pro`

`--limit 10`을 주면 MMLU는 10개가 아니라 최대 570개를 평가한다. 코드에 이 차이를 알리는
경고나 문서화는 없다.

### 8.9 선택 캐시 재사용 시 anchor가 갱신되지 않는다

`pipelines/selection.py:247-277`의 캐시 재사용 경로는 `anchor.update()`를 건너뛴다.
같은 `--run_tag`로 재개할 때 이미 저장된 epoch의 선택은 재사용되므로 의도된 동작이지만,
그 epoch의 anchor refresh가 생략되어 다음 epoch의 **시간축 부호 보정**이 참조하는
`v_l^{(t-1)}`은 체크포인트에서 복원된 값이 된다. 부호 보정 체인이 한 칸 건너뛴다.

### 8.10 주석의 메모리 수치 오류 3건

§3.1에 계산과 함께 정리했다(각각 약 32배, 약 32배, 약 100배 과대 표기). 동작 영향은 없다.

### 8.11 죽은 분기 1건

`tads/evals/humaneval.py`의 `_strip_prose_preamble` 첫 번째 루프에는 `first_word`를
`_PROSE_OPENERS`와 비교하는 블록이 있으나 본문이 `pass`이고 곧바로 `break`한다. 실질적
동작은 "첫 비공백 줄이 코드처럼 보이면 그대로 반환, 아니면 두 번째 루프로"이며,
`_PROSE_OPENERS` 상수는 이 파일에서 실효적으로 쓰이지 않는다(같은 이름의 상수를 쓰는
`tads/evals/mbpp.py` 쪽 구현은 실제로 사용한다).

## 9. `tads/train.py` 상세 (921줄)

실행 순서와 각 단계의 근거 수치.

1. **import 이전 처리** — `OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_MAXIMUM` 5개 스레드 변수를
   `"16"`으로 `setdefault`. libgomp/MKL이 `import torch` 시점에 값을 읽으므로 이후에
   설정하면 효과가 없다. 이어서 `torchvision.io.VideoReader`가 없으면 빈 클래스를
   꽂아 둔다(transformers 5.0의 비디오 모델 레지스트리가 import 시점에 이를 참조).
2. `disable_coredumps()` → `clear_runtime_caches()` → 오프라인 환경변수 5종 setdefault
   → `quiet_repeated_warnings()`.
3. `parse_args()` — `--config`(필수), `--override`(0개 이상, 점 표기 중첩 지원),
   `--run_tag`, `--run_suffix`, `--list_runs`.
4. `_apply_overrides` — 값은 bool → int → float → str 순으로 시도. 중첩 키는 dict를
   만들며 내려가되, 중간에 dict가 아닌 값이 있으면 예외.
5. `_setup_ddp()` — `RANK`가 환경에 있으면 nccl 백엔드로 초기화, **timeout 120분**.
6. run 디렉터리 결정 → `cfg["output_dir"]`, `cfg["experiment_dir"]`, `cfg["run_tag"]`를
   설정 dict에 되돌려 넣는다. 병렬 실행 시 선택 공유 파일이 서로 덮어쓰는 것을 막기
   위한 조치다.
7. `save_cfg_snapshot(run_dir, cfg)`.
8. `find_latest_complete_epoch(run_dir)`로 재개 지점 탐색. 재개 시 체크포인트 내용으로
   `training_mode`를 재판정한다(`adapter_config.json` → lora, `config.json` → full).
   설정과 다르면 경고 후 체크포인트 쪽으로 맞춘다. LoRA면 base는 `model_path`에서,
   adapter는 체크포인트에서 읽도록 경로를 나눈다.
9. 데이터셋 캐시 경로는 `<data_cache>/<model_key>/<prompt_style>`.
   `dataset_subset_size`가 전체보다 작으면 시드 고정 순열로 부분집합을 만든다.
10. anchor는 `method == "tads" and is_main_process()`일 때만 생성. 이때
    `max_samples_for_pca` 기본값은 **2000**이다(설정에 `anchor` 키가 없을 때만 적용되며,
    `configs/base.yaml`은 1024를 준다).
11. optimizer:
    `approx_steps_per_epoch = max(1, int(n_total*ratio / batch_size / grad_accum / world_size))`,
    `total_steps = approx_steps_per_epoch * train_epochs`.
    `use_8bit_optimizer`가 참이면 **프로브 단계**로 1원소 파라미터에 대해
    `AdamW8bit.step()`을 실제 실행하고 `synchronize()`까지 한다. `(ImportError, OSError,
    RuntimeError, AttributeError)` 중 하나라도 나면 이 실행에 한해 8-bit를 끄고
    fp32 AdamW로 간다. 프로브를 통과한 뒤 실제 생성에서 실패하는 경우도 같은 예외
    집합으로 잡는다.
    `weight_decay` 기본 0.1, warmup step은 `max(1, int(total_steps*warmup_ratio))`.
12. 재개 시 복원: `env_meta.json`에서 bitsandbytes 버전을 비교해 다르면 경고,
    `optimizer.pt`/`scheduler.pt`/`trajectory_anchor.pt`는 `map_location="cpu"` +
    `weights_only=False`로 읽는다. CPU 경유 이유는 7B full-FT optimizer 상태가 약 14 GB라
    GPU 직접 로드 시 순간 2배가 되기 때문이다.
13. epoch 루프: `select_indices` → `save_selection` → 빈 선택이면 예외(빈 DataLoader에서
    DDP all_reduce가 hang하는 알려진 경로) → `Subset` → `make_dataloader` → `sft_one_epoch`.
14. 체크포인트(rank 0): `_safe(step_name, fn)` 헬퍼로 각 저장 단계를 감싼다. 저장 항목은
    모델, 토크나이저, `optimizer.pt`, `scheduler.pt`, `env_meta.json`,
    (tads면) `trajectory_anchor.pt`와 `anchor_history.json`, `metrics.json`.
    **모델과 optimizer가 모두 성공했을 때만** `_complete`에 epoch 번호를 쓰고 `_latest`를
    갱신한다. 실패가 있으면 `_save_errors.json`을 남긴다.
    `keep_last_n_checkpoints`는 `epoch_last` 레이아웃에서 무의미해 값이 있으면 로그만
    남기고 무시한다.
15. 루프 종료 후 `timing_breakdown.json` 저장 + 표 로그 + `destroy_process_group()`.

## 10. `tads/eval.py` 상세 (567줄)

- torchrun 하위 프로세스 판정: `LOCAL_RANK`가 있고 `WORLD_SIZE > 1`이며 `RANK != 0`일 때만
  조용히 종료한다. 예전에는 `RANK != 0`만 봤는데, 셸에 남은 `RANK`나 SLURM 환경 때문에
  단독 실행이 아무 로그 없이 끝나는 일이 있었다. 지금은 진입 즉시 pid와 세 변수를 출력한다.
- 체크포인트 우선순위: `--ckpt` > `--run_tag` > `_latest`. `--ckpt`와 `--run_tag`는 동시
  사용 금지. `--epoch N`은 `epoch_last/_complete` 내용이 N일 때만 매칭되고, 아니면
  legacy `epoch_N/`을 찾으며, 둘 다 없으면 어떤 값이 유효한지 알려주며 종료한다.
- 결과 레이아웃: 기본 `<ckpt>/eval/runs/<eval_tag>/`, `--flat`이면 히스토리 없이 평탄하게.
  `--eval_tag=latest`로 기존 run에 결과를 추가할 수 있다.
- 실험 라벨: `cfg.experiment_name` → `<설정 상위 폴더>_<파일명>` → `<파일명>`.
- 벤치마크마다 `try/except`로 감싸 실패를 `failures` 목록에 모으고, `finally`에서
  `gc.collect()`와 `empty_cache()`를 호출한다(9개 벤치 순차 실행 시 high-water mark 누적
  방지).
- 요약 JSON 원자적 기록 후 `_complete` sentinel → `_latest` 갱신. sentinel 실패 시
  `_latest`를 갱신하지 않는다(완료 오탐 방지).
- `warnings.filterwarnings`로 `max_new_tokens`/`do_sample`/`temperature` 관련 메시지
  3종을 정규식으로 억제한다.

## 11. `tads/evals` — 평가기 10종

`base.py`: `_REGISTRY` dict, `register(name)` 데코레이터(클래스에 `name` 속성도 주입),
`get_evaluator(name)`(미등록이면 등록 목록을 담은 KeyError), `list_evaluators()`,
추상 클래스 `BenchmarkEvaluator.evaluate(model, tokenizer, device, *, output_file,
limit, prompt_style, data_dir, **kwargs)`.

공통 규칙: 모든 evaluator가 `tokenizer.truncation_side = "left"`를 설정하고, 생성 결과는
`out[0, prompt_tok_len:]` 토큰 슬라이스 후 디코딩하며, 요약 dict에 `accuracy`와
`benchmark` 키를 채우고 `output_file`에 JSON을 쓴다.

| evaluator | few-shot | 생성 설정 | 채점 | 대표 지표(`accuracy`) |
|---|---|---|---|---|
| `mmlu` | 5-shot(dev 분할) | 생성 없음. 마지막 위치 logits에서 A/B/C/D 토큰 argmax. 입력 상한 2048 토큰(초과 시 좌측 절단 + 카운트) | 정답 인덱스 비교 | 전체 정확도(`overall_accuracy`와 동일) |
| `mmlu_pro` | 5-shot CoT(validation의 `cot_content`, 카테고리 매칭) | `max_new_tokens=512`, 입력 3072, greedy, stop `\nQuestion:` | A~J 문자 추출: "answer is X" 계열 3패턴(마지막 매치) → 마지막 괄호 문자 → 실패 시 None(오답 처리) | 카테고리 macro 평균 |
| `gsm8k` | 8-shot CoT | `max_new_tokens=256`, 입력 2048, greedy, stop `\nQ:` 등 4종 | "The answer is X" 마지막 매치 → `####` → 마지막 숫자. 정규화 후 문자열 비교, 실패 시 `float` 비교(오차 1e-6) | 정확도 |
| `svamp` | 8-shot CoT(gsm8k prefix 재사용) | 동일 | `gsm8k._grade` 재사용. 질문은 `Body + " " + Question` | 정확도 |
| `humaneval` | 0-shot(SFT 템플릿으로 감쌈) | `n_samples=20`을 `n_samples_per_batch=4`씩 5회 호출, `T=0.8`, `top_p=0.95`, `max_new_tokens=512`, 시드 42 | 후처리 6단계(코드펜스 → 프롬프트 에코 → 산문 서두 → 시그니처 재출력 → stop 시퀀스 → rstrip) 후 `human_eval` 패키지의 `evaluate_functional_correctness(k=[1,10], timeout=30, n_workers=4)` | `pass@10` |
| `mbpp` | 3-shot(`prompt` 분할 앞 3개) | greedy(`n_samples=1`), `max_new_tokens=512`, 입력 2048, stop 6종 | 완성 코드를 서브프로세스에서 `setup + code + tests` 한 번에 exec, 타임아웃 30초 | `pass@1`(codex 무편향 추정식) |
| `tydiqa` | 5-shot(같은 언어 train 데모) | `max_new_tokens=100`, 입력 2048, greedy | SQuAD식 정규화(소문자 → ASCII 구두점 제거 → 영어 관사 제거 → 공백 정리 → NFC) 후 EM과 토큰 F1 | F1 (EM은 `accuracy_em`) |
| `xquad` | 5-shot(영어 데모를 전 언어 공용) | `max_new_tokens=80`, 입력 2048, stop 6종 | 유니코드 카테고리 기반 구두점 처리 + 산문 서두 제거 후 EM/F1 | 언어 macro F1 |
| `bbh` | 공식 `cot-prompts/` 3-shot CoT, 없으면 자체 예시 `n_fewshot=5` 직답 | `max_new_tokens=256`, 입력 3072, stop `\nQ:` 등 | "the answer is X" → 마지막 줄의 (A)/True/False/Yes/No → 마지막 줄의 마지막 숫자 | 태스크 macro 평균 |
| `lm_harness` | 외부 | `python -m lm_eval` 자식 프로세스 실행 | 외부 harness 결과 JSON을 읽어 `results` 필드 첨부 | 없음(상태와 결과만 전달) |

각 evaluator의 방어 로직에서 눈여겨볼 부분:

- **`mmlu`**: `test-*.parquet`만 읽는다. `all` 설정의 디렉터리에는 dev/validation/
  auxiliary_train까지 있어서 전부 읽으면 테스트 풀이 오염된다. 스키마 검증(필수 4개 컬럼,
  `choices` 길이 4, `answer` 0~3)을 모델 로드 후 첫 문항 전에 수행한다.
- **`mbpp`**: `sanitized`와 `full` 설정의 컬럼명이 달라 별칭 표를 두고 내부적으로 `full`
  이름으로 정규화한다. `test_imports`만 있으면 `test_setup_code`를 합성한다.
  `_safe_list`는 numpy 배열에 `or`를 쓰면 truth value 예외가 나는 문제를 우회한다.
- **`tydiqa`**: 파일 해석이 parquet → legacy JSON → 키워드 glob 순서. 데이터 감사에서
  질문 보유율 50% 미만 또는 정답 보유율 10% 미만이면 실행을 거부하고, 95%/50% 미만이면
  경고만 낸다. train 분할이 없으면 0-shot으로 내려가며 `fewshot_fallback` 사유와
  `paper_faithful` 플래그를 결과 JSON에 기록한다.
- **`xquad`**: 언어 목록은 `NAIT_LANGUAGES` 11개 + 파일이 있을 때만 `ro`.
  영어 데모를 다른 언어에도 공용으로 쓰며(교차 언어 관례), 영어 자신만 자기 앞
  `n_fewshot`개를 데모로 쓰고 그만큼을 평가에서 뺀다. `empty_cache_interval=50`.
- **`bbh`**: 태스크 파일 전체를 먼저 열어 `{"examples": [{"input","target"}...]}` 구조를
  검증하고, 하나라도 어긋나면 모델 forward 전에 중단한다. `cot-prompts`가 하나도 없으면
  ERROR 로그로 "논문 재현 불가"를 알린다.
- **`mmlu_pro`**: 옵션이 `option_a..option_j` 컬럼으로 분리된 미러, JSON 문자열로
  저장된 미러, dict 형태 미러를 모두 흡수한다. `answer_index`만 있으면 문자를 유도한다.

## 12. `baselines/` 상세

### 12.1 `nait/`

- `direction.py`
  - `ALPACA_PROMPT_FULL` — instruction/input/output을 모두 포함한 전체 텍스트.
  - `extract_delta_from_seed(...)` — 우측 패딩으로 배치 forward, 층별
    `Δh = h[마지막 유효] − h[첫]`. 유효 길이 2 미만 행은 제외. 토크나이저의
    `padding_side`/`pad_token`을 바꿨다가 `finally`에서 원복한다.
    함수 시그니처 기본값은 `batch_size=8`이지만 `train.py`는 `nait.batch_size`
    (코드 폴백 2, 배포된 YAML 8)를 넘긴다. 100개 단위로 진행 로그.
  - `fit_directions(delta_per_layer)` — 층별 top-1 PCA + 공간 부호 보정. `N<H`면 Gram
    경로. `anchor`의 `_pca_top1`과 같은 수식이지만 별도 구현이다.
  - `score_candidates(...)` — `s_y = Σ_l ⟨Δh_l(y), v_l⟩`. docstring에 4가지 최적화가
    기록돼 있다: (1) 슬라이스 후 fp32 캐스팅, (2) 방향 벡터 GPU 캐시를 루프 밖으로,
    (3) 층마다 CPU 동기화하던 것을 배치당 1회로, (4) 음수 층 인덱스 사전 해석.
- `train.py` — seed JSON이 없으면 원본 데이터에서 무작위 표본을 뽑아 `seeds/mix_auto.json`에
  캐시한다(파일 락으로 동시 실행 보호). 코드 기본 `n_seeds=1500`, 배포된
  `configs/methods/nait.yaml`은 `n_seeds: 100000`으로 사실상 상한 해제.
  `layers` 기본값은 코드에서 `[-1]`, YAML에서 0~31 전체. 선택 결과와 점수를 파일로
  저장하고 재실행 시 재사용한다.

### 12.2 `data_agent/`

- `agent.py` — `ActorCritic`(공유 trunk 2층 + alpha/beta/value 헤드, `softplus + 1.0`),
  `PPOAgent`. 기본값: `hidden_dim=128`, `lr=3e-4`, `clip_eps=0.2`, `gamma=0.99`,
  `gae_lam=0.95`, `ppo_epochs=4`, `entropy_coef=0.0`, `value_coef=0.5`, `mb_size=1024`,
  `advantage_mode="group_relative"`, `value_clip=False`, grad clip 1.0.
  Beta 표본은 `clamp(1e-6, 1-1e-6)` 후 log_prob을 구한다(정확히 0이나 1이면 -inf가 되어
  다음 업데이트에서 ratio가 발산). `N < 2`면 예외.
  `load()`는 optimizer 상태 텐서를 수동으로 GPU로 옮긴다.
- `select.py` — 상태는 마지막 디코더 층의 시퀀스 평균. 보상은 논문식 정규화:
  `R_diff = min-max(L)`, `R_conf = H / max(H)`, `r = Var(R_diff)/(Var(R_diff)+Var(R_conf)+ε)`,
  `R = r·R_diff + (1−r)·R_conf`. **선택 점수는 `a_i`(Beta 표본) 단독**이며 R이나 R·a가
  아니다. docstring이 이 점을 굵게 명시한다.
- `train.py` — epoch마다 optimizer와 scheduler를 새로 만든다(선택 부분집합 크기가 매
  epoch 달라 총 step 수를 미리 확정할 수 없기 때문). `epoch_last/`에 `ppo_agent.pt`도 함께 저장.

### 12.3 `selectit/`

- `score.py`
  - `resolve_rating_token_ids(tokenizer)` — `convert_tokens_to_ids("1")` 우선, 실패 시
    `encode` 후 마지막 토큰. 5개 id가 서로 달라야 하고, 각각을 디코딩했을 때 원래
    숫자와 일치하는지까지 검증한다(다중 문자 토큰이 섞이면 점수가 무의미해짐).
  - `_double_softmax(p5)` — `exp(p/sum(p))` 후 재정규화. 공식 구현의 특이 동작을 그대로
    유지하되, 합이 0이거나 비정상이면 균등 분포를 반환해 NaN 전파를 막는다.
  - `_token_score_from_probs(p5, k=5)` — `(argmax+1) × (Σ_j (p[argmax] − p[j]) / (k−1))`.
  - `selectit_scores(...)` — `level="token"`이면 샘플당 템플릿 1개(순환 사용),
    `"sentence"`면 앞 5개 템플릿을 모두 쓰고 `avg / (1 + alpha·std)`로 합친다.
    기본 `alpha=0.2`, `batch_size=8`. 토크나이저 상태는 `finally`에서 원복.
  - `select_top_proportion(scores, proportion)` — 내림차순 상위 `max(1, int(n·p))`.
- `rating_prompts.txt` — 평가용 rating 프롬프트 9줄. sentence 수준은 최소 5개를 요구한다.
- `train.py` — 토크나이즈된 Dataset과 원시 레코드의 **길이가 다르면 즉시 예외**를 낸다
  (인덱스 정렬이 깨지면 점수와 샘플이 어긋나기 때문). 점수·선택 결과는 원자적으로 캐시.

### 12.4 `q2q/`

- `score.py` — `_batched_mean_nll_of_response(...)`가 우측 패딩으로 응답 구간 평균 NLL을
  계산한다. 타깃 마스크는 `start = max(0, n_prefix − 1)`부터 `L − 1`까지.
  `compute_ifd_scores(...)`는 조건부/무조건부 두 번 호출하고
  `IFD = exp(clamp(nll_cond − nll_uncond, −15, 15))`. 클램프는 이상치 억제용.
  `select_top_proportion_by_ifd(scores, p, ifd_low=0.5, ifd_high=1.0)`는 범위 필터 후
  상위 K, 남은 후보가 K보다 적으면 전체 유효값 상위 K로 폴백하며 경고한다.
- `train.py` — precursor 모델 단계를 생략하고 base 모델로 직접 IFD를 계산한다는 사실이
  파일 docstring과 `configs/methods/q2q.yaml` 주석에 모두 적혀 있다(`use_precursor: false`).

### 12.5 `lima/`

- `data.py` — 5가지 입력 스키마(A: 문자열 2원소 `conversations`, B: ShareGPT dict,
  C: ChatML/OpenAI `messages`, D: 평면 Alpaca, E: prompt/completion 쌍)를 흡수해
  `{instruction, input, output}`으로 정규화한다. 어느 것도 아니면 키 목록과 300자
  미리보기를 담은 `KeyError`를 던진다. 오프라인이면 hub 접근을 거부하고 조치 3가지를
  안내한다. 토크나이즈는 `tokenize_alpaca`를 그대로 쓴다(`num_proc=2`).
- `train.py` — 선택 로직 없음. `data_files` 해석 순서는 CLI → 환경변수 → cfg → hub.

### 12.6 `alpagasus/`

- `train.py` — 사전 필터 JSON을 읽어 instruction 문자열로 원본 인덱스를 역추적한다.
  `_extract_instruction`은 `input`이 있으면 `f"{ins}\n{inp}"`로 합친다.
  중복 instruction은 첫 인덱스만 쓰고(`setdefault`), 매칭 실패 수를 경고로 남기며,
  0개가 매칭되면 예외를 던진다. 결과 인덱스는 순서를 보존하며 중복 제거한다.
  필터 파일 경로는 CLI → 환경변수 → cfg 순으로 해석하고, 실패 시 세 출처의 값을 모두
  보여주는 오류 메시지를 낸다.

## 13. `configs/` 구조

```
configs/
├── base.yaml                 공통 기본값(경로·데이터·학습·선택·anchor·lora)
├── env.example.yaml          환경변수 예시(주석만)
├── methods/  9개             method별 키와 하이퍼파라미터
├── models/   6개             모델 경로·prompt_style·max_seq_len·learning_rate
├── modes/    2개             full_ft / lora
└── experiments/ 55개
      ├── (루트 9개)          7b_fullft_* 4, light_* 4, sanity_tads_7b
      ├── main_7b/  33개      llama2 12, qwen25 7, mistral 7, deepseek 7
      ├── main_05b/ 5개       qwen25
      └── evol_7b/  8개       llama2 4, qwen25 4
```

`base.yaml`의 주요 기본값: `prompt_style: alpaca_default`, `max_seq_len: 1024`,
`seed: 42`, `train_epochs: 3`, `batch_size: 2`, `grad_accum: 4`,
`learning_rate: 2.0e-5`, `warmup_ratio: 0.03`, `weight_decay: 0.1`,
`gradient_clip: 1.0`, `gradient_checkpointing: true`, `training_mode: full`,
`use_8bit_optimizer: false`, `attn_implementation: null`, `selection_ratio: 0.5`,
`episode_batch_size: 1`, `anchor: {layer_indices: all, layer_idx: -1,
max_samples_for_pca: 1024, pca_batch_size: 4}`, `lora: {r: 16, alpha: 32, dropout: 0.05,
target_modules: 7개}`.

`modes/full_ft.yaml`은 `batch_size: 8`, `grad_accum: 4`, `use_8bit_optimizer: true`를
주고 `learning_rate`/`warmup_ratio`는 **의도적으로 비운다**(모델 YAML의 recipe 우선).
`modes/lora.yaml`은 `batch_size: 4`, `grad_accum: 4`, `learning_rate: 2.0e-4`,
`warmup_ratio: 0.1`.

모델별 값: llama2-7b `max_seq_len 512 / lr 2e-5 / alpaca_default`,
qwen2.5-7b `512 / 1e-5 / qwen_chatml`, mistral-7b `512 / 1e-5 / warmup 0.10 /
mistral_instruct`, deepseek-7b `512 / 2e-5 / deepseek_user_assistant`,
qwen2.5-0.5b `1024 / qwen_chatml`(lr 미지정 → base의 2e-5),
qwen2.5-14b `2048 / qwen_chatml`(미참조).

`main_7b/llama2/tads_10.yaml`과 `evol_7b/*/tads_10.yaml`은 TADS 셀에만
`use_8bit_optimizer: false`, `warmup_ratio: 0.06`, `gradient_clip: 0.5`를 덧씌운다.
주석은 이 때문에 같은 매트릭스의 다른 방법과 SFT optimizer 조건이 달라진다는 점을
명시하고 있다.

## 14. `scripts/`

| 스크립트 | 줄 | 역할 |
|---|---:|---|
| `setup_env.sh` | 305 | 모든 경로 환경변수 export, 출력 디렉터리 생성, 존재하지 않는 경로를 **경고만** 하고 셸을 중단하지 않음. HF 캐시를 지정 위치로 리다이렉트, 오프라인 플래그 3종, 스레드 캡 5종, 코어덤프 정책, 토큰은 파일에 넣지 말라는 규약(`~/.tads_secrets.sh` 자동 source) |
| `run_main_7b.sh` | 126 | main 매트릭스 실행. 기본은 torchrun DDP 순차, `--parallel`은 GPU 1개당 1잡. `MASTER_PORT`는 셸 PID로 유도해 동시 실행 충돌을 줄임 |
| `run_evol_7b.sh` | 116 | Evol-Instruct 매트릭스. 단일 프로세스 실행, method별 entrypoint 분기 존재 |
| `run_eval_main_7b.sh` | 156 | 체크포인트 자동 탐색(`_latest` → `epoch_last/_complete` → sealed `epoch_N` → 아무 `epoch_N`) 후 `tads.eval` 실행 |
| `run_eval_evol_7b.sh` | 146 | 위와 동일 구조, 기본 벤치 9종 |
| `auto_eval_7b_fullft.sh` | 88 | legacy `7b_fullft` 트리 감시 루프(60초 간격). 평가 성공 시에만 완료 마커를 남기고, 초기 epoch 삭제는 `CLEANUP_EARLY_EPOCHS=1`일 때만 |
| `make_table.sh` | 639 | 결과 JSON 트리를 훑어 (set, model, method, bench) 버킷으로 모으고 markdown/csv/tsv 표를 출력. 셀에는 중복 제거한 모든 측정값을 내림차순으로 나열, W-AVG는 벤치별 최댓값의 단순 평균, Δ는 같은 (set, model)의 Full FT 대비 상대차 |
| `download_*.sh` | 58~155 | 10개 취득 스크립트(데이터셋 9 + 모델 1). MMLU와 GSM8K 전용 스크립트는 없다. 대부분 `curl` 재시도 + 부분파일(`.part`) 후 rename, 일부는 `datasets`/`huggingface_hub` 사용. 실패 시 수동 명령을 출력 |
| `merge_lora.py` | 59 | LoRA adapter를 base에 병합해 저장 |
| `inspect_eval_data.py` | 180 | mmlu_pro / svamp / mbpp / xquad 데이터의 실제 스키마와 첫 행을 덤프. 스키마 불일치 진단용 |

`make_table.sh`의 식별 로직이 특히 방어적이다. 결과 JSON에 기록된 `base_model` 문자열로
0.5B와 7B를 먼저 구분하고(경로만 보면 두 실험이 같은 디렉터리명을 쓰는 경우가 있음),
set(main_7b/main_05b/evol_7b/light)은 경로 세그먼트 → 실패 시 요약 JSON의 `ckpt` 경로
역방향 탐색으로 판정한다.

## 15. `tests/`

pytest 수집 노드 25개, 파일 6개. 마지막 실행 기록(`.pytest_cache`)에는 실패 항목이 없다.

| 파일 | 노드 | 내용 |
|---|---:|---|
| `test_scorer.py` | 8 | 분산비 w의 극단/균등 케이스, z-score 성질, 최소-최대 정규화와 collapse 플래그, λ=0 환원, λ=1에서 2배 부스트, top-B 순서, NaN 거부 |
| `test_data.py` | 6 | 5개 prompt_style 각각에 대해 길이·마스킹 불변식 검증(문자 단위 스텁 토크나이저 사용), prompt/response 분할 |
| `test_reward.py` | 3 | 반환 shape과 유한성, batch=1에서 `r_weight == 0`, 합성 보상 수식 |
| `test_utils.py` | 3 | `defaults` 체인, 리스트형 `defaults` 우선순위, 환경변수 보간과 기본값 |
| `test_evals_registry.py` | 3 | 레지스트리 등록 목록, 클래스 반환, 미등록 KeyError |
| `test_broadcast_selection.py` | 2 | torchrun 환경이 없으면 skip. rank 0의 top-B가 모든 rank에 동일하게 전파되는지, all_gather 결과가 동일한지 |

테스트가 다루지 않는 영역: 실제 프로덕션 선택 공유 경로(파일 폴링), `TrajectoryAnchor`의
PCA/부호 보정, `collect_episode` 조립, 모든 evaluator, 모든 baseline, run 레이아웃과
sentinel 로직.
