# GaugePack v1 NN-LUT training tools

이 디렉터리는 frozen GaugePack VFU의 비선형 함수 네 개를 생성·검증하는
개별 실행 파일과 통합 실행 파일을 모은다. 모든 PWL page는 15개 boundary와
16개 segment를 사용한다.

| 파일 | frozen 입출력 계약 | 생성 방식 |
|---|---|---|
| `gaugepack_gelu_nnlut_train.py` | FFN1 raw accumulator+bias → terminal `S8 [-127,127]` | calibration 학습형 continuous PWL; M18/C48 compile 전 reference |
| `gaugepack_qexp_nnlut_train.py` | raw `d_int=score_int-rowmax_int:S25` → `E7 [0,127]` | `s_Q×s_K/sqrt(head_dim)`을 QEXP 계수에 fold하고, 기존 U8 boundary/calibration trace로 E7 segment-local Direct32 refit |
| `gaugepack_softmax_recip_nnlut_train.py` | `L:U13 [127,8128]` → `R≈2^23/L`, `RNE>>8` | frozen geometric relative-minimax + M18/C48 integer search + 전 legal-domain certificate |
| `gaugepack_rsqrt_nnlut_train.py` | `D27=RNE((128Q-S²)>>4)` → `rho:U8` | calibration 학습형 continuous PWL; M18/C48 compile 전 reference |
| `gaugepack_all_nnlut_train.py` | 위 네 실행 파일의 공통 진입점 | 단일 op forwarding, 전체 self-test, ordered JSON plan |

## 데이터 분리 규칙

- calibration trace만 scale 선택과 fitting에 사용한다.
- disjoint train-search trace는 fitting 종료 후 metric 계산에만 사용한다.
- SST-2 validation 872개는 이 도구들에 넣지 않는다.
- GELU 입력 `x_int32`에는 FFN1 integer bias가 이미 포함되어 있어야 한다.
- RSQRT 입력은 full `D`가 아니라 `D27=RNE_even(D/16)`이다.
- QEXP는 현재 실제 E7 artifact flow라서 source U8 artifact 안의 calibration/search
  trace와 boundary를 사용한다.
- QEXP input은 scale하지 않은 raw QK difference code다. Artifact의
  `score_scale_folded = query_scale×key_scale/sqrt(head_dim)`이 scaled-attention을
  이미 포함하므로 runtime에서 `1/sqrt(head_dim)`을 다시 적용하지 않는다.
- reciprocal은 gradient training 대상이 아니라 frozen analytic/minimax compiler다.

## 빠른 검증

```bash
python gaugepack_all_nnlut_train.py --self-test
```

개별 검증:

```bash
python gaugepack_gelu_nnlut_train.py --self-test
python gaugepack_qexp_nnlut_train.py --self-test
python gaugepack_softmax_recip_nnlut_train.py --self-test
python gaugepack_rsqrt_nnlut_train.py --self-test
```

## 개별 학습/생성 예시

GELU:

```bash
python gaugepack_gelu_nnlut_train.py \
  --calibration-npz layer0_ffn1_acc_calibration.npz \
  --search-npz layer0_ffn1_acc_search.npz \
  --accumulator-scale 0.0003125 \
  --output-scale 0.03125 \
  --site-name layer0_ffn1_gelu \
  --out-dir artifacts/gelu_layer0
```

QEXP E7 artifact:

```bash
python gaugepack_qexp_nnlut_train.py \
  --base-artifact-dir nnlut_sst2_gaugepack_all_fixed \
  --out-dir nnlut_sst2_gaugepack_e7_qexp \
  --overwrite-output
```

Reciprocal:

```bash
python gaugepack_softmax_recip_nnlut_train.py \
  --out-dir artifacts/softmax_reciprocal
```

RSQRT:

```bash
python gaugepack_rsqrt_nnlut_train.py \
  --calibration-npz layer0_attention_D27_calibration.npz \
  --search-npz layer0_attention_D27_search.npz \
  --site-name layer0_attention_output_ln \
  --out-dir artifacts/rsqrt_layer0
```

## 통합 실행 plan

`gaugepack_all_nnlut_train.py`는 각 개별 파일을 canonical implementation으로
그대로 호출한다. 로직을 통합 파일에 복제하지 않아 개별 실행과 통합 실행이
서로 달라지는 문제를 막는다.

```json
{
  "jobs": [
    {
      "operator": "gelu",
      "args": [
        "--calibration-npz", "layer0_ffn1_acc_calibration.npz",
        "--search-npz", "layer0_ffn1_acc_search.npz",
        "--accumulator-scale", "0.0003125",
        "--output-scale", "0.03125",
        "--out-dir", "artifacts/gelu_layer0"
      ]
    },
    {
      "operator": "qexp",
      "args": [
        "--base-artifact-dir", "nnlut_sst2_gaugepack_all_fixed",
        "--out-dir", "nnlut_sst2_gaugepack_e7_qexp",
        "--overwrite-output"
      ]
    },
    {
      "operator": "reciprocal",
      "args": ["--out-dir", "artifacts/softmax_reciprocal"]
    },
    {
      "operator": "rsqrt",
      "args": [
        "--calibration-npz", "layer0_attention_D27_calibration.npz",
        "--search-npz", "layer0_attention_D27_search.npz",
        "--out-dir", "artifacts/rsqrt_layer0"
      ]
    }
  ]
}
```

```bash
python gaugepack_all_nnlut_train.py \
  --plan nnlut_jobs.json \
  --report nnlut_jobs_report.json
```

## 의존성

- Python 3.10+
- NumPy
- PyTorch 2.x: GELU와 RSQRT 학습에 필요

QEXP와 reciprocal flow는 NumPy만 사용한다.
