# TADS 사용 방법 (USAGE)

설치부터 학습·평가·표 생성·문제 해결까지, 코드에 실제로 존재하는 인자와 동작만 정리한다.
경로는 모두 환경변수 이름으로 표기한다(각 환경에 맞는 값을 직접 넣으면 된다).

> 루트 `README.md`는 제목 한 줄(6단어)이 전부이므로, 실행 방법에 관한 유일한 근거는
> 이 문서와 각 파일의 docstring이다.

## 1. 요구 사항

- Python 3.10 이상 (`pyproject.toml`의 `requires-python`)
- CUDA GPU. 7B full fine-tuning은 8-bit optimizer 사용을 전제로 설정돼 있다
  (`configs/modes/full_ft.yaml`의 `use_8bit_optimizer: true`)
- 필수 패키지: `torch>=2.0`, `torchvision`, `transformers>=4.40`, `datasets>=2.14`,
  `peft>=0.10`, `accelerate>=0.25`, `pyyaml>=6.0`, `numpy>=1.24,<2.0`, `tqdm>=4.66`,
  `sentencepiece`, `bitsandbytes>=0.43`, `filelock>=3.12`
- 평가 추가 패키지: `lm-eval>=0.4`, `pandas`, `pyarrow>=14.0`, `human-eval>=1.0`
- 테스트: `pytest>=7`

`torchvision`은 LLM 학습에 직접 쓰이지 않지만 **반드시 설치해야 한다.**
transformers 5.x의 비디오 모델 레지스트리가 import 시점에 torchvision C++ op을
조회하는데, 없으면 `LlamaForCausalLM`을 포함한 모든 `*ForCausalLM`의 지연 로딩이
연쇄 실패한다. torch와 같은 CUDA 인덱스에서 함께 설치해야 한다.

## 2. 설치

```bash
# 저장소 루트에서
python -m venv .venv && source .venv/bin/activate

# (a) requirements 기반 — 학습 + 평가 + 테스트 의존성이 모두 들어 있다
pip install -r requirements.txt

# (b) 패키지로 설치 — extras 선택 가능
pip install -e .              # 학습만
pip install -e ".[eval]"      # + 평가(lm-eval, pandas, pyarrow, human-eval)
pip install -e ".[test]"      # + pytest
```

`pyproject.toml`의 패키지 탐색 범위는 `tads*`뿐이다. `baselines`, `scripts`, `tests`는
설치 대상이 아니므로 **저장소 루트에서 `python -m baselines.<method>.train` 형태로
실행**해야 한다.

## 3. 환경 변수 설정

```bash
source scripts/setup_env.sh
```

이 스크립트는 값을 export하고, 존재하지 않는 경로에 대해 **경고만** 출력한다(셸을
중단하지 않는다). 기본값을 바꾸려면 source 하기 **전에** 해당 변수를 export하면 된다.

### 3.1 경로 변수

| 변수 | 용도 |
|---|---|
| `MODEL_PATH_LLAMA2_7B`, `MODEL_PATH_QWEN25_7B`, `MODEL_PATH_QWEN25_05B`, `MODEL_PATH_QWEN25_14B`, `MODEL_PATH_MISTRAL_7B`, `MODEL_PATH_DEEPSEEK_7B` | 각 base 모델 디렉터리 |
| `ALPACA_DATA_FILES` | 학습 데이터 파일 또는 glob (`.json`/`.jsonl`/`.parquet`/`.csv`) |
| `ALPACA_DATASET_NAME` | 위 파일이 없을 때 사용할 HF hub 데이터셋 이름 |
| `EVOL_INSTRUCT_DATA_FILES` | Evol-Instruct 실험용 데이터 |
| `LIMA_DATA_FILES` | LIMA 로컬 미러 |
| `ALPAGASUS_FILTERED_FILE` | AlpaGasus 사전 필터 JSON |
| `OUTPUT_ROOT` | 체크포인트 루트 |
| `DATA_CACHE` | 토크나이즈 캐시 및 HF 캐시 루트 |
| `EVAL_RESULTS_ROOT` | TADS 계열(random/full/tads) 평가 결과 루트 |
| `NAIT_EVAL_RESULTS_ROOT`, `SELECTIT_EVAL_RESULTS_ROOT`, `LIMA_EVAL_RESULTS_ROOT`, `ALPAGASUS_EVAL_RESULTS_ROOT`, `Q2Q_EVAL_RESULTS_ROOT` | baseline별 평가 결과 루트(방법별 이력을 분리하기 위함) |
| `MMLU_DATA_DIR`, `MMLU_PRO_DATA_DIR`, `GSM8K_DATA_DIR`, `SVAMP_DATA_DIR`, `HUMANEVAL_DATA_DIR`, `MBPP_DATA_DIR`, `TYDIQA_DATA_DIR`, `XQUAD_DATA_DIR`, `BBH_DATA_DIR` | 벤치마크 데이터 디렉터리 |

`setup_env.sh`는 HF 캐시(`HF_HOME`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE`,
`TRANSFORMERS_CACHE`)를 `DATA_CACHE` 하위로 돌린다. 홈 디렉터리 용량이 작거나 여러 잡이
동시에 도는 환경에서 캐시 경합을 피하기 위한 조치다.

### 3.2 토큰

gated 데이터셋(LIMA 등)이 필요하면 셋 중 하나:

1. `huggingface-cli login` (권장, 파일에 토큰이 남지 않음)
2. 셸 rc에 `export HF_TOKEN=...`
3. `~/.tads_secrets.sh`에 `export HF_TOKEN=...` 후 `chmod 600`
   — `setup_env.sh`가 자동으로 source한다

`scripts/setup_env.sh`는 git으로 추적되므로 실제 토큰을 이 파일에 직접 적어서는 안 된다.

### 3.3 동작 토글 (실제로 코드가 읽는 것)

| 변수 | 기본 | 효과 | 읽는 위치 |
|---|---|---|---|
| `TADS_ENABLE_COREDUMPS` | 0 | 1이면 `RLIMIT_CORE` 차단을 건너뛴다 | `core/utils.disable_coredumps` |
| `TADS_FRESH_DATA_CACHE` | 0 | 1이면 토크나이즈 캐시를 무시하고 재생성 | `data/alpaca.build_alpaca_dataset` |
| `TADS_DL_NUM_WORKERS` | (미설정) | DataLoader 워커 수를 강제 | `pipelines/sft.make_dataloader` |
| `TADS_ENABLE_NO_SYNC` | 0 | 1이면 DDP `no_sync()` 기반 grad accumulation 최적화 활성 | `pipelines/sft.sft_one_epoch` |
| `TADS_GSM8K_USE_SFT_WRAP` | 0 | 1이면 GSM8K 8-shot 프롬프트를 SFT 템플릿으로 감쌈 | `evals/gsm8k.py` |
| `TADS_MBPP_USE_SFT_WRAP` | 0 | 1이면 MBPP 3-shot 프롬프트를 SFT 템플릿으로 감쌈 | `evals/mbpp.py` |

**주의**: `scripts/setup_env.sh` 주석은 `TADS_DDP_BACKEND`, `TADS_DDP_FIND_UNUSED`,
`TADS_DDP_STATIC_GRAPH`, `TADS_DDP_BROADCAST_BUFFERS`, `TADS_NCCL_REINIT` 5개도 설명하지만
**Python 코드 어디에서도 읽지 않는다**(전체 `.py` grep으로 확인). 설정해도 아무 효과가
없다. DDP 백엔드는 `tads/train.py`에 nccl로 고정돼 있고, `find_unused_parameters`는
`tads/modeling/loader.py`에서 학습 모드(LoRA 여부)로만 결정된다.

오프라인 관련 변수 3종(`HF_DATASETS_OFFLINE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`)은
`setup_env.sh`와 Python 진입점 양쪽에서 1로 설정된다. 일회성으로 온라인이 필요하면
명령 앞에 붙여 하위 프로세스에만 적용한다.

```bash
HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -m tads.eval ...
```

## 4. 데이터 준비

```bash
bash scripts/download_qwen25_05b.sh   [target_dir]   # 소형 모델(스모크 테스트용)
bash scripts/download_humaneval.sh    [target_dir]   # 164문항 검증까지 수행
bash scripts/download_mbpp.sh         [target_dir]   # sanitized/{test,prompt}-*.parquet
bash scripts/download_mmlu_pro.sh     [target_dir]   # test-*.parquet + validation-*.parquet
bash scripts/download_svamp.sh        [target_dir]
bash scripts/download_tydiqa.sh       [target_dir]   # HF parquet(구 GCS URL은 403)
bash scripts/download_xquad.sh        [target_dir]   # 12개 언어 JSON
bash scripts/download_alpagasus.sh    [target_dir]   # 사전 필터 JSON
bash scripts/download_evol_instruct.sh [target_dir]
bash scripts/download_lima.sh         [target_dir]   # gated: 로그인 + 약관 동의 필요
```

인자를 생략하면 대응하는 `*_DATA_DIR` 환경변수를 쓴다. 대부분 `.part` 임시 파일로 받은
뒤 rename하며, 실패하면 수동으로 실행할 명령을 출력한다. MMLU와 GSM8K 전용 다운로드
스크립트는 없다(기존에 준비된 디렉터리를 가리키게 하는 방식).

데이터 스키마 문제로 평가가 거부될 때는 실제 파일 구조를 먼저 확인한다.

```bash
python scripts/inspect_eval_data.py               # mmlu_pro svamp mbpp xquad 전부
python scripts/inspect_eval_data.py mbpp          # 하나만
```

컬럼 목록, dtype, 첫 행 샘플을 출력하므로 "미러가 다른 스키마인지"를 바로 판별할 수 있다.

## 5. 학습 실행

### 5.1 TADS / random / full — `tads.train`

```bash
# 단일 GPU
CUDA_VISIBLE_DEVICES=0 python -m tads.train \
    --config configs/experiments/main_05b/qwen25/tads_10.yaml

# DDP 4장
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
    -m tads.train --config configs/experiments/main_7b/llama2/tads_10.yaml
```

인자:

| 인자 | 설명 |
|---|---|
| `--config <yaml>` | 필수 |
| `--override k=v [k=v ...]` | 최상위 또는 점 표기 중첩 키를 덮어쓴다. 값은 bool → int → float → str 순으로 해석 |
| `--run_tag <tag>` | 결과를 `<output_dir>/runs/<tag>/`에 쓴다. 기존 태그를 주면 **그 run을 재개**한다. `latest`를 주면 `_latest`가 가리키는 run을 재개 |
| `--run_suffix <s>` | 자동 타임스탬프 태그 뒤에 `_<s>`를 붙인다(스윕용). `--run_tag`가 있으면 무시 |
| `--list_runs` | `<output_dir>/runs/` 목록과 `_latest`를 출력하고 종료 |

override 예:

```bash
python -m tads.train --config configs/experiments/main_7b/llama2/tads_10.yaml \
    --override selection_ratio=0.3 tads.lam=0.5 anchor.max_samples_for_pca=2048 \
               episode_batch_size=8 train_epochs=1
```

하이퍼파라미터 스윕은 run_suffix로 구분한다.

```bash
python -m tads.train --config <cfg> --run_suffix=lr2e5
python -m tads.train --config <cfg> --run_suffix=lr5e5 --override learning_rate=5e-5
python -m tads.train --config <cfg> --list_runs
```

`tads.train`이 처리하는 `method` 값은 `random`, `full`, `tads` 세 가지뿐이다. 그 외
값(`data_agent`, `nait`, `selectit`, `lima`, `alpagasus`, `q2q`)이 들어오면 전용 실행
명령을 안내하는 오류로 종료한다.

### 5.2 baseline 6종

```bash
# NAIT — seed 파일이 없으면 학습 데이터에서 자동 생성해 캐시한다
python -m baselines.nait.train \
    --config configs/experiments/main_7b/llama2/nait_10.yaml \
    --tag NAIT-Mix [--seed_path seeds/mix.json] [--n_seeds 1500]

# Data Agent (PPO)
python -m baselines.data_agent.train \
    --config configs/experiments/main_7b/llama2/data_agent_10.yaml \
    --tag DataAgent-PPO

# SelectIT
python -m baselines.selectit.train \
    --config configs/experiments/main_7b/llama2/selectit_10.yaml \
    --tag SelectIT-Token [--rating-prompts-file baselines/selectit/rating_prompts.txt]

# Q2Q / Cherry-LLM (IFD)
python -m baselines.q2q.train \
    --config configs/experiments/main_7b/llama2/q2q_10.yaml --tag Q2Q-Top10

# LIMA (데이터 교체형, 선택 없음)
python -m baselines.lima.train \
    --config configs/experiments/main_7b/llama2/lima.yaml --tag LIMA \
    [--data_files /path/to/lima.jsonl]

# AlpaGasus (사전 필터 목록 매칭)
python -m baselines.alpagasus.train \
    --config configs/experiments/main_7b/llama2/alpagasus.yaml \
    --tag AlpaGasus-ChatGPT-9k [--filtered_file /path/to/filtered.json]
```

`--tag`는 산출 디렉터리 이름의 일부가 된다: `<output_root>/<output_subdir>/<method>_<tag_slug>/`
(`tag_slug`는 소문자 + 하이픈을 밑줄로 치환). 예를 들어 `--tag DataAgent-PPO`는
`data_agent_dataagent_ppo/`가 된다.

baseline은 모두 **선택 결과를 파일로 캐시**한다(`selected_indices.json`,
`scores.json`/`ifd_scores.json`). 같은 출력 디렉터리로 재실행하면 점수 계산 단계를
건너뛰고 SFT부터 시작한다. 점수를 다시 계산하려면 이 파일들을 지운다.

baseline은 `epoch_last/`만 만들고 `_complete` sentinel을 쓰지 않는다. 따라서 평가할 때는
`--ckpt <...>/epoch_last`를 **직접 지정**해야 한다.

### 5.3 매트릭스 일괄 실행

```bash
# main 매트릭스 (기본: torchrun DDP 순차)
bash scripts/run_main_7b.sh --gpus 0,1,2,3
MODELS="llama2 qwen25" METHODS="tads_10 random_10" bash scripts/run_main_7b.sh --gpus 0,1

# 1 GPU당 1잡 병렬
bash scripts/run_main_7b.sh --gpus 0,1,2,3 --parallel

# Evol-Instruct 매트릭스 (단일 프로세스, method별 entrypoint 자동 분기)
bash scripts/run_evol_7b.sh --gpus 0
```

**`run_main_7b.sh`의 기본 `METHODS`에는 `data_agent_10`이 들어 있는데, 이 스크립트는 항상
`tads.train`을 호출한다.** `tads.train`은 `data_agent`를 거부하므로 그 셀은 예외로 끝난다.
회피 방법은 둘 중 하나다.

```bash
# (1) data_agent를 빼고 돌린 뒤
METHODS="full_100 random_10 tads_10" bash scripts/run_main_7b.sh --gpus 0,1,2,3
# (2) Data Agent는 전용 entrypoint로 따로 실행
CUDA_VISIBLE_DEVICES=0 python -m baselines.data_agent.train \
    --config configs/experiments/main_7b/llama2/data_agent_10.yaml --tag DataAgent-PPO
```

`run_evol_7b.sh`는 같은 문제를 `entry_for()` 분기로 이미 처리해 두었으므로 기본값 그대로
사용할 수 있다.

두 스크립트의 실행 방식이 다르다는 점도 알아 둘 필요가 있다. `run_main_7b.sh`는 torchrun
DDP가 기본이고, `run_evol_7b.sh` 주석은 "DDP 경로에 미해결 버그가 있어 단일 프로세스가
지원되는 실행기"라고 적고 있다. 두 서술이 저장소 안에서 서로 어긋난다.

## 6. 산출물 확인

```
<OUTPUT_ROOT>/<output_subdir>/
  ├── _latest -> runs/<run_tag>          # symlink 또는 _latest.txt
  └── runs/<run_tag>/
        ├── cfg.yaml, cfg.json           # 해석 완료된 설정
        ├── metrics.json                 # epoch별 loss + 선택 진단
        ├── selected_indices_epoch{N}.json
        ├── timing_breakdown.json
        ├── logs/
        └── epoch_last/                  # 가중치·optimizer·scheduler·anchor·_complete
```

- 학습 완료 여부: `epoch_last/_complete`의 내용(정수)이 `cfg.json`의 `train_epochs`와
  같은지 확인한다.
- `metrics.json`의 각 행에는 `epoch`, `method`, `selected_n`, `n_total`, `train_loss`,
  `elapsed_sec`가 들어가고, TADS면 `r_loss_mean`, `r_entropy_mean`, `r_weight`,
  `lam`, `use_anchor`, `align_mean`, `align_std`, `anchor_stats`가 추가된다.
- `timing_breakdown.json`은 phase별/카테고리별(`setup`, `data`, `selection`, `sft`,
  `checkpoint`, `misc`) 소요 시간과 미계측 시간을 담는다. 방법 간 "선택 오버헤드 대
  순수 학습 시간" 비교에 쓰인다.

## 7. 평가 실행

```bash
python -m tads.eval \
    --config configs/experiments/main_7b/llama2/tads_10.yaml \
    --ckpt <run_dir>/epoch_last \
    --benchmarks mmlu,gsm8k,humaneval,tydiqa,bbh \
    --out_dir <EVAL_RESULTS_ROOT>/main_7b/llama2/tads_10/
```

인자 전체:

| 인자 | 기본 | 설명 |
|---|---|---|
| `--config` | (필수) | `model_path`, `prompt_style`, 벤치 경로를 여기서 읽는다 |
| `--ckpt` | 없음 | 체크포인트 디렉터리. 생략 시 `_latest` 해석 |
| `--run_tag` | 없음 | `<output_dir>/runs/<tag>` 선택. `--ckpt`와 동시 사용 불가 |
| `--epoch N` | 없음 | `epoch_last/_complete`가 N일 때만 매칭. 아니면 legacy `epoch_N/` 탐색 |
| `--list_runs` | - | 학습 run 목록 출력 후 종료 |
| `--benchmarks` | `mmlu` | 쉼표 구분. 등록된 이름: `bbh, gsm8k, humaneval, lm_harness, mbpp, mmlu, mmlu_pro, svamp, tydiqa, xquad` |
| `--out_dir` | `<ckpt>/eval/` | 결과 루트 |
| `--eval_tag` | 타임스탬프 | `latest`를 주면 기존 평가 run에 결과를 추가 |
| `--eval_suffix` | `""` | 자동 태그에 접미사 |
| `--list_eval_runs` | - | 평가 run 목록 출력 후 종료 |
| `--flat` | off | 히스토리 레이아웃 없이 `out_dir`에 직접 쓴다(일회성 전용) |
| `--limit N` | 없음 | 샘플 수 제한. **벤치마크마다 의미가 다르다**(아래 주의) |
| `--training_mode` | 자동 | `full`/`lora` 강제 |
| `--cuda_device` | 0 | GPU 인덱스 |
| `--<bench>_data_dir` | cfg 값 | 벤치별 데이터 경로 개별 지정 |
| `--harness_task` | `mmlu` | `lm_harness` 전용 |
| `--lm_eval_path` | 없음 | lm-eval 포크를 `PYTHONPATH`에 추가 |

**`--limit`의 적용 단위 차이**

- 과목/태스크/언어 단위로 적용: `mmlu`(57과목), `bbh`(태스크별), `xquad`(언어별)
- 전체 샘플 수로 적용: `gsm8k`, `svamp`, `humaneval`, `mbpp`, `tydiqa`, `mmlu_pro`

`--limit 10`을 주면 MMLU는 10문항이 아니라 최대 570문항을 평가한다. 스모크 테스트를 할
때는 이 차이를 감안해야 한다.

결과 파일:

```
<out_dir>/runs/<eval_tag>/
  ├── <label>-<bench>.json               # 벤치별 상세(문항 단위 결과 포함)
  ├── <label>-humaneval_completions.jsonl# HumanEval 생성 결과(오프라인 재채점용)
  ├── <label>-eval_summary.json          # 전체 요약 + failures
  ├── logs/
  └── _complete
```

`<label>`은 `cfg.experiment_name` → `<설정 상위 폴더>_<파일명>` → `<파일명>` 순으로
결정된다. 벤치 하나가 실패해도 나머지는 계속 실행되며, 실패 내역은 요약 JSON의
`failures` 배열에 `{benchmark, error}` 형태로 남는다.

각 벤치의 대표 지표는 요약 dict의 `accuracy` 키에 통일해 들어간다.

| bench | `accuracy`가 담는 값 |
|---|---|
| `mmlu` | 전체 정확도 |
| `mmlu_pro` | 카테고리 macro 평균 |
| `gsm8k`, `svamp` | 정확도 |
| `humaneval` | pass@10 (pass@1은 진단용으로 별도 보관) |
| `mbpp` | pass@1 |
| `tydiqa` | F1 (EM은 `accuracy_em`) |
| `xquad` | 언어 macro F1 (EM은 `accuracy_em`/`macro_em`) |
| `bbh` | 태스크 macro 평균 |

### 7.1 매트릭스 일괄 평가

```bash
bash scripts/run_eval_main_7b.sh --gpus 0
bash scripts/run_eval_main_7b.sh --gpus 0,1,2,3 --parallel
MODELS="llama2" METHODS="tads_10" BENCHMARKS="mmlu,gsm8k" \
    bash scripts/run_eval_main_7b.sh --gpus 0
bash scripts/run_eval_evol_7b.sh --gpus 0        # 기본 9벤치
```

두 스크립트는 체크포인트를 `_latest` → `epoch_last/_complete` → sealed `epoch_N` →
아무 `epoch_N` 순으로 찾는다. baseline 산출물처럼 sentinel이 없는 경우에도 마지막 폴백이
동작한다.

### 7.2 결과 표 생성

```bash
bash scripts/make_table.sh <EVAL_RESULTS_ROOT>            # markdown (기본)
bash scripts/make_table.sh <EVAL_RESULTS_ROOT> csv
bash scripts/make_table.sh <EVAL_RESULTS_ROOT> tsv
```

- (table-set, model) 쌍마다 표를 하나씩 세로로 쌓아 출력한다.
  인식하는 set은 `main_7b`, `main_05b`, `evol_7b`, `light`이고, 어디에도 속하지 않는
  파일은 마지막에 `(no-set)`으로 묶인다.
- 각 셀은 같은 (방법, 벤치)에 대한 **모든 측정값**을 소수 2자리 중복 제거 후 내림차순
  나열한다.
- `W-AVG`는 벤치별 최댓값의 단순 평균, `Δ`는 같은 (set, model)의 `full_100` 대비 상대차다.
- 표준출력에는 표만, 표준에러에는 셀별 원본 파일 경로와 파싱 실패 목록이 나온다.
  집계가 이상할 때는 stderr를 먼저 본다.

```bash
bash scripts/make_table.sh <root> markdown > table.md 2> table.debug.txt
```

**주의**: 표의 `08 Composite-reward only (λ=0)` 행은 실제로는 `data_agent_10` 디렉터리를
읽는다. λ=0 ablation 전용 실험 설정은 저장소에 없으므로 이 행의 수치는 Data Agent(PPO)
baseline 결과다. λ=0 실험을 직접 돌리려면 `--override tads.lam=0`으로 별도 run을 만들고
표 정의를 수정해야 한다.

## 8. 설정 변경 포인트

### 8.1 계층 구조와 우선순위

실험 YAML은 `defaults:` 리스트로 부모를 상속한다. 관례적 순서는 다음과 같다.

```yaml
defaults:
  - configs/base.yaml
  - configs/methods/<method>.yaml
  - configs/models/<model>.yaml
  - configs/modes/<full_ft|lora>.yaml
```

병합 규칙(`tads/core/utils.py`):

1. `defaults` 리스트를 순서대로 로드해 누적 병합한다. **뒤에 오는 파일이 이긴다.**
2. 중첩 dict(`anchor`, `lora`, `tads`, `data_agent` 등)는 **깊은 병합**이다.
3. 마지막으로 그 파일 자신의 최상위 키가 부모를 덮는다.
4. 최종 결과에서 `${oc.env:VAR,default}` 패턴을 환경변수로 치환한다.

깊은 병합 때문에 생기는 함정 하나: `configs/base.yaml`이 `anchor.layer_indices: all`을
정의하므로, 실험 YAML이 `anchor:` 블록에 `layer_idx`만 적어도 `layer_indices: all`이
남아 다층 모드로 동작한다. 단일 층 모드를 쓰려면 `anchor.layer_indices: null`을 명시해야
한다.

### 8.2 자주 바꾸는 키

| 키 | 위치 | 의미 |
|---|---|---|
| `selection_ratio` | 실험 YAML | 선택 비율. 실제 개수는 `max(1, int(N × ratio))` |
| `train_epochs` | 실험 YAML | epoch 수. 배포된 실험은 모두 3 |
| `tads.lam` | `methods/tads.yaml` | anchor 가중치 λ. 부스트 범위는 `[1, 1+λ]`. 0이면 합성 보상 단독 ranking |
| `tads.use_anchor` | `methods/tads.yaml` | false면 λ와 무관하게 anchor를 끈다 |
| `anchor.layer_indices` | `base.yaml` / 실험 | `"all"`, `"middle_to_last"`, 명시적 인덱스 리스트, `null`(단일 층) |
| `anchor.max_samples_for_pca` | `base.yaml` / 실험 | probe 크기. 기본 1024 |
| `anchor.pca_batch_size` | `base.yaml` / 실험 | probe forward 배치 |
| `episode_batch_size` | method/실험 | 후보 풀 forward 배치. 클수록 빠르고 메모리를 많이 쓴다 |
| `batch_size`, `grad_accum` | mode/실험 | SFT 배치. 유효 배치 = `batch_size × grad_accum × world_size` |
| `learning_rate`, `warmup_ratio`, `weight_decay`, `gradient_clip` | model/mode/실험 | optimizer 관련 |
| `use_8bit_optimizer` | mode/실험 | true면 bitsandbytes AdamW8bit(사전 프로브 후 실패 시 자동 강등) |
| `gradient_checkpointing` | `base.yaml` | 기본 true. 끄면 속도가 오르고 메모리를 더 쓴다 |
| `attn_implementation` | `base.yaml`/model | `null`, `"sdpa"`, `"flash_attention_2"`, `"eager"` |
| `max_seq_len` | model/실험 | 학습 시퀀스 길이. 7B 모델 설정은 512, 0.5B는 1024 |
| `prompt_style` | model | 5종 중 하나. 학습과 평가가 같은 값을 쓴다 |
| `dataset_subset_size` | 실험 | 후보 풀 자체를 줄인다(스모크 테스트용) |
| `training_mode` | mode/실험 | `full` 또는 `lora` |
| `lora.r`, `lora.alpha`, `lora.dropout`, `lora.target_modules` | mode/실험 | LoRA 설정. Llama 계열이 아니면 `target_modules`를 반드시 지정 |
| `output_subdir` | 실험 | `OUTPUT_ROOT` 아래 실험 디렉터리 이름 |
| `experiment_name` | 실험(선택) | 평가 결과 파일 접두사 |

### 8.3 새 실험 설정 추가

```yaml
# configs/experiments/my_exp/llama2/tads_30.yaml
defaults:
  - configs/base.yaml
  - configs/methods/tads.yaml
  - configs/models/llama2-7b.yaml
  - configs/modes/full_ft.yaml

output_subdir: my_exp/llama2/tads_30
selection_ratio: 0.3
train_epochs: 3
tads:
  lam: 0.5
```

`make_table.sh`가 표에 포함하려면 set 이름이 `KNOWN_SETS`에, 방법 디렉터리 이름이
`METHODS`에 등록돼 있어야 한다. 새 축을 추가하려면 해당 스크립트의 상단 목록을 고친다.

## 9. 테스트

```bash
pytest                                    # 수집 노드 25개
pytest tests/test_scorer.py -v            # 점수 규칙만
pytest -k "alpaca_default"                # prompt_style 하나만

# 분산 테스트는 torchrun이 필요하다(그 외 환경에서는 자동 skip)
torchrun --standalone --nproc-per-node=2 tests/test_broadcast_selection.py
```

테스트는 `tads.core.scorer`, `tads.core.reward`, `tads.core.utils`의 설정 로더,
`tads.data.sft_prompts`의 토크나이즈, evaluator 레지스트리, 그리고
`tads.core.dist_utils`의 collective 유틸을 다룬다. **실제 학습에서 쓰이는 파일 폴링 기반
선택 공유 경로, `TrajectoryAnchor`, `collect_episode`, 모든 evaluator 본체, 모든 baseline은
테스트 범위 밖이다.**

## 10. 문제 해결

### 10.1 설치·import

| 증상 | 원인과 조치 |
|---|---|
| `LlamaForCausalLM` 등 모든 모델 로딩이 연쇄 실패 | torchvision 미설치 또는 torch와 버전 불일치. torch와 같은 CUDA 인덱스에서 재설치. 진입점에 `torchvision.io.VideoReader` 스텁이 있지만 근본 해결은 정상 설치 |
| `from transformers import get_cosine_schedule_with_warmup` 계열 오류 | 해당 없음. 이 저장소는 `tads/core/schedulers.py`에 자체 구현을 갖고 있어 transformers 버전 변화에 영향받지 않는다 |
| `pandas.read_parquet` 실패 | `pyarrow>=14.0`가 없다. `pip install pyarrow` (pandas가 자동으로 끌어오지 않는다) |
| HumanEval 채점 단계에서 `RuntimeError` | `pip install human-eval` (하이픈). 생성 결과는 이미 `*_completions.jsonl`에 저장돼 있어 나중에 따로 채점할 수 있다 |

### 10.2 학습 중

| 증상 | 원인과 조치 |
|---|---|
| `libgomp: Thread creation failed: Resource temporarily unavailable` | anchor PCA 구간에서 스레드가 폭주. 진입점이 `import torch` 전에 스레드 변수 5종을 16으로 고정하지만, 자체 실행기를 만들었다면 같은 처리를 앞에 넣어야 한다 |
| `bitsandbytes early probe FAILED` 로그 후 fp32로 진행 | bnb 휠이 노드의 CUDA 런타임과 맞지 않는다. 실행은 fp32 AdamW로 계속된다(느리고 메모리를 더 쓰지만 정확도는 동일). 맞는 휠을 설치하면 8-bit로 복귀 |
| resume 후 loss가 갑자기 평탄 | bnb 버전 불일치로 optimizer 상태 복원이 실패했을 수 있다. `epoch_last/env_meta.json`의 `bitsandbytes` 값과 현재 버전을 비교하는 경고 로그를 확인 |
| CUDA smoke test FAILED (ERROR 로그) | `torch.cuda.is_available()`는 True인데 커널 실행이 실패. libcudart 버전 불일치, CUDA 심볼릭 링크 손상, 드라이버 노후가 흔한 원인 |
| "OOM at step ~50" 형태의 후반부 OOM | 할당자 단편화. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`가 적용됐는지 확인(진입점이 setdefault로 넣지만, CUDA 컨텍스트가 이미 만들어진 뒤라면 무효) |
| 선택 단계에서 몇 십 분간 로그 없음 | 정상일 수 있다. `TrajectoryAnchor.update`는 20배치마다, `collect_episode`는 50배치마다 진행률과 ETA를 남긴다. 그 로그조차 없으면 hang |
| `TADS alignment COLLAPSED` ERROR | alignment 최대-최소 폭이 1e-8 미만. anchor PCA의 고윳값 간격이 사라졌거나 층 간 내적이 상쇄됐다. 그 epoch의 TADS는 합성 보상 단독 ranking과 같아진다. `anchor.layer_indices`를 좁히거나 probe 크기를 키워 확인 |
| `selected indices is empty` 예외 | `selection_ratio × N`이 0으로 내려갔다. 비율이나 `dataset_subset_size`를 확인 |
| DDP에서 SFT 첫 스텝 후 멈춤 | `TADS_ENABLE_NO_SYNC=1`을 켰다면 끈다(기본은 꺼짐). 초반 3개 경계의 배리어 로그로 어느 rank가 늦는지 확인 |
| worker rank가 `timed out ... waiting for ...ready` | rank 0이 선택 중 죽었다. rank 0 로그를 먼저 본다. 대기 상한은 6시간 |
| `ready sentinel present but selection file missing` | 이전 버전의 즉시 삭제 경합 증상. 현재 코드는 다음 epoch 진입 시점에 지연 삭제하므로, 이 오류가 보이면 공유 디렉터리에 옛 파일이 섞였는지 확인 |
| 같은 epoch 선택이 재계산되지 않음 | 의도된 동작이다. `<run_dir>/selected_indices_epoch{N}.json`이 있으면 재사용한다. 재계산하려면 그 파일을 지운다(단, 이 경로에서는 anchor 갱신도 함께 생략된다) |
| `method=... is a comparison baseline` 오류 | `tads.train`으로 baseline을 돌렸다. 오류 메시지에 적힌 전용 명령을 사용 |
| 여러 잡이 같은 캐시에서 Arrow 오류 | 데이터셋 빌드 락이 동작하지 않는 파일시스템(fcntl 잠금 불가)이거나 filelock 미설치. `filelock`을 설치하고, 잡마다 `DATA_CACHE`를 분리 |
| 프롬프트 스타일이나 `max_seq_len`을 바꿨는데 결과가 그대로 | 토크나이즈 캐시 재사용. `TADS_FRESH_DATA_CACHE=1`로 강제 재생성 |

### 10.3 평가 중

| 증상 | 원인과 조치 |
|---|---|
| 아무 로그 없이 즉시 종료 | 셸에 `RANK`/`WORLD_SIZE`가 남아 torchrun 하위 프로세스로 오인됐을 수 있다. 현재 코드는 진입 즉시 pid와 세 변수를 출력하므로 그 줄이 없다면 외부 요인(OOM killer 등)이다 |
| `No _latest pointer under ...` | 학습을 먼저 돌리거나 `--ckpt`로 경로를 직접 지정. baseline 산출물은 sentinel이 없어 항상 `--ckpt`가 필요하다 |
| `--epoch N`이 거부됨 | `tads.train`은 `epoch_last/`만 남긴다. 오류 메시지가 sentinel에 기록된 실제 epoch 번호를 알려준다 |
| MMLU 스키마 오류 | `cais/mmlu`의 `all` 설정 parquet(`test-*`, `dev-*`)이 필요하다. `option_a/option_b` 형태 미러는 거부된다 |
| MMLU 로그에 truncated 경고 다수 | 5-shot 프롬프트가 2048 토큰을 넘겨 좌측 절단됐다. 요약 JSON의 `truncated_prompts`로 개수 확인 |
| TyDiQA가 0-shot으로 내려감 | train 분할이 없어 같은 언어 데모를 만들지 못했다. 요약 JSON의 `fewshot_fallback`과 `paper_faithful` 필드로 확인 가능 |
| TyDiQA가 실행 자체를 거부 | 질문 보유율 50% 미만 또는 정답 보유율 10% 미만인 데이터. 다른 데이터셋을 가리키고 있을 가능성이 높다 |
| BBH가 "NOT paper-faithful" ERROR | `cot-prompts/*.txt`가 없어 직답 few-shot으로 대체됐다. 공식 배포본을 받아 `bbh/<task>.json`과 `cot-prompts/<task>.txt`가 함께 있게 한다 |
| HumanEval pass@10이 0에 가까움 | 요약 JSON의 `first_samples`(앞 3문항의 원본/후처리 결과)를 먼저 본다. 산문만 있으면 프롬프트 스타일 문제, 잘린 코드면 `max_new_tokens` 부족(`truncated_pct`가 5% 이상이면 경고가 뜬다) |
| MBPP pass@1이 낮음 | `exec_timeout`(기본 30초)과 `use_sft_wrap` 설정을 확인. per-problem 결과의 `error` 필드에 실패 원인이 남는다 |
| 여러 벤치를 돌리다 후반에 OOM | 벤치 사이에 gc와 empty_cache를 하지만 누적이 심하면 벤치를 나눠서 실행. XQuAD/BBH/HumanEval이 특히 무겁다 |
| 결과가 표에 안 잡힘 | `make_table.sh`가 model/method/set을 경로와 `base_model`로 판정한다. stderr의 `no_method` / `no_model` 카운트와 셀별 출처 목록을 확인 |

### 10.4 디스크·정리

- 코어덤프는 기본으로 차단된다(Python에서 `RLIMIT_CORE=(0,0)`). 디버깅 때문에
  `TADS_ENABLE_COREDUMPS=1`로 켠다면, 7B DDP 기준 rank당 수백 GB가 생길 수 있으므로
  반드시 여유 있는 볼륨에서 실행한다.
- `tads.train`은 run당 `epoch_last/` 하나만 유지한다. `keep_last_n_checkpoints`는
  이 레이아웃에서 무의미해 값이 있어도 로그만 남기고 무시된다.
- 과거 레이아웃의 `epoch_N/`이 남아 있어도 자동 삭제하지 않는다(마이그레이션을 명시적으로
  두기 위함). 수동으로 정리한다.
- `.gitignore`가 `checkpoints/`, `results/`, `logs/`, `cache/`, `*.pt`, `*.safetensors`,
  `metrics.json`, `selected_indices*.json`, `configs/env.yaml`을 제외한다. 산출물이
  저장소에 섞이지 않도록 출력 루트를 저장소 밖에 두는 편이 안전하다.

## 11. 빠른 스모크 테스트 절차

전체 매트릭스를 돌리기 전에 소형 모델로 경로를 점검하는 순서다.

```bash
source scripts/setup_env.sh
bash scripts/download_qwen25_05b.sh

# 1) 후보 풀을 줄이고 1 epoch만
CUDA_VISIBLE_DEVICES=0 python -m tads.train \
    --config configs/experiments/light_tads_05b.yaml \
    --run_suffix=smoke \
    --override train_epochs=1 dataset_subset_size=2000 \
               anchor.max_samples_for_pca=128 episode_batch_size=8

# 2) run 목록과 sealed epoch 확인
python -m tads.train --config configs/experiments/light_tads_05b.yaml --list_runs

# 3) 가벼운 벤치 하나로 평가 경로 확인
CUDA_VISIBLE_DEVICES=0 python -m tads.eval \
    --config configs/experiments/light_tads_05b.yaml \
    --benchmarks gsm8k --limit 20 --out_dir ./_smoke_eval

# 4) 표 생성 경로 확인
bash scripts/make_table.sh ./_smoke_eval markdown
```

`light_*` 설정은 LoRA + 0.5B 조합이라 단일 GPU에서도 빠르게 끝난다. 다만
`anchor.layer_idx`를 지정해도 `base.yaml`의 `layer_indices: all`이 살아남아 다층 모드로
동작한다는 점(§8.1)은 이 설정에도 그대로 적용된다.
