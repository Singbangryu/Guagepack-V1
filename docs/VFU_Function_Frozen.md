# GaugePack VFU 기능 및 산술 — Frozen

> **파일:** `VFU_Function_Frozen.md`  
> **상태:** **FROZEN — VFU 산술/계산 그래프 기준 문서**  
> **최종 갱신:** 2026-08-27  
> **범위:** GaugePack BERT-Tiny VFU — Requantization, GELU, Softmax, LayerNorm  
> **최상위 결정 원장:** `gaugepack-master-work.md` 최신 FROZEN  
> **상위 아키텍처 기준:** `GaugePack_VFU_Architecture_Frozen_v1.md`  
> **목적:** 이 파일 하나만 읽고 VFU의 계산 그래프, CORE16 4-stage 동작, operand mapping, bitwidth, scale, RNE/SAT 위치, compiler 책임을 파악할 수 있게 한다.

> **중요:** LayerNorm residual add는 DSP/PRE 연산이 아니다. `main` branch만 DSP를 통해 requant한 뒤, **S3 POST-ALU에서 `main8 + original_skip8 -> z:S9`**를 수행한다.

> **2026-08-24 승계 감사 주의:** 이 문서의 식/bitwidth/RNE/SAT 계약은 FROZEN이다.
> 하지만 production coefficient payload, current RTL build, scratch/controller,
> physical layout, per-command QK/PV granularity, PMPU no-touch/fifo contract와 최신
> padded E2E는 완료됐거나 동결됐다는 뜻이 아니다. `OPEN-SPAD-001`,
> `OPEN-SCHED-001`, `OPEN-PMPU-001`, `OPEN-PMPU-MR-001`, `OPEN-PMPU-SEL-001`을 포함한 구현/검증 상태는 Master §6/§7과
> `GaugePack_WorkMaster_Succession_2026-08-24.md`를 따른다.

> **2026-08-25 memory hierarchy — `SPAD-20260825-001`, `MEM-20260825-001`:** S-pad는 VFU-private
> intermediate-only scratch다. PMPU가 사용할 E7/context/PV용 corner-turned V/final activation은
> PMPU-visible Logical-DRAM에 먼저 commit한 뒤 PMPU가 reload한다. Logical-DRAM은
> on-chip operand-store abstraction이며 logical bank별 BRAM/URAM 선택은
> `OPEN-MEM-PHYS-001`이다. 아래 계산 그래프의
> `direct E7 PV`는 E7를 normalized `E/L`로 바꾸지 않는 산술 의미이며
> `S-pad→PMPU` 직접 물리 경로를 뜻하지 않는다.

> **2026-08-27 hybrid scratch 정합화 — `SPAD-20260825-004`,
> `SPAD-20260827-005`:** Dynamic V transpose는 Intermediate S-pad가 아니라 dedicated
> `vfu_vcorner_ff`가 담당한다. Intermediate S-pad의 resident는 raw Score와 LayerNorm
> `z/T`뿐이며, schedule-robust production memory leaf는 16 banks × 512 depth × 32-bit
> opaque container, synchronous 1R1W로 둔다. Exact Score/LN address overlay는 계속
> `OPEN-SPAD-001`이다.

> **VFU memory client — `VFU-MEM-20260825-001`:** PMPU는 operand-store read-only지만
> VFU는 read/write client다. Residual/skip tensor를 읽고 final tensor를 commit-write한다.
> RQ `M/C/F`, NN-LUT page, LN gamma/beta처럼 command 동안 고정되는 작은 parameter는
> tensor stream과 분리된 config/parameter bank에서 preload해 local register에 latch하며,
> 매 beat main URAM에서 다시 읽지 않는다.

---

## 0. 이 문서를 읽는 순서

VFU 연산은 항상 다음 순서로 본다.

```text
알고리즘 / 함수
      ↓
정수 계산 그래프
      ↓
값(Value) + 폭(Width) + Scale
      ↓
CORE16 S0/S1/S2/S3 mapping
      ↓
Compiler 계수
      ↓
Architectural output / scratch
```

각 화살표에는 가능하면 다음 네 가지를 함께 적는다.

```text
value     : 실제 정수식이 무엇인가?
width     : signed/unsigned 몇 bit인가?
scale     : real value와 어떤 관계인가?
precision : 어디서 RNE / shift / saturation이 일어나는가?
```

비트폭만 보면 함수의 의미를 잃기 쉽고, 식만 보면 RTL에서 실제로 들어가는지 확인하기 어렵다. 이 문서는 두 관점을 같이 본다.

---

# 1. VFU 전체 구조 한 장 요약

```text
                           GaugePack VFU

PMPU / Scratch source
        │
        ▼
  2-entry input skid
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              CORE16                                      │
│                         16 lanes, II = 1                                 │
│                                                                          │
│  S0                  S1                  S2                  S3           │
│  SOURCE / PRE   →   DSP INPUT REG   →  DSP PREG       →   POST/COMMIT   │
│                                           │                              │
│                                      P = A×B+C                           │
│                                                                          │
│                  A27 / B18 / C48  →  P48                                │
│                                                                          │
│  OP_RQ_RES only:                                                         │
│  original skip8 ───────── sideband delay/alignment ───────────► S3      │
│                                                               POST add   │
└──────────────────────────────────────────────────────────────────────────┘
        │
        ├── final S8/U8 code
        ├── Softmax E / L / R / context
        ├── LayerNorm S/Q metadata
        └── LayerNorm z / T scratch
```

공통 DSP spine은 다음과 같다.

```text
A : signed 27 bit
B : signed 18 bit
C : signed 48 bit
P : signed 48 bit

P = A × B + C
```

예외적으로 LayerNorm MomentPack ACC에서는 DSP P-feedback을 사용한다.

---

# 2. 공통 수치 규약

## 2.1 Signed activation 규약

```text
NARROW_S8 = [-127, +127]
8'h80 = -128 → illegal activation code
```

적용 대상:

- Q/K/V activation
- generic requant output
- GELU output
- residual branch `main8`, `skip8`
- Softmax context
- LayerNorm final output

Softmax QEXP는 8-bit byte로 운반하지만 의미는 signed activation이 아니라 다음과 같다.

```text
E7 = U7 [0,127]
physical byte bit7 = 0
```

## 2.2 공통 비트폭

| 항목 | 폭 | 의미 |
|---|---:|---|
| PMPU accumulator ingress | S32 | 외부 container |
| DSP A | S27 | active arithmetic operand |
| DSP B | S18 | multiplier / coefficient |
| DSP C | S48 | bias / PWL intercept |
| DSP P | S48 | `A×B+C` 결과 |
| Generic/PWL multiplier `M` | S18 | compiler generated |
| Generic/PWL intercept `C` | S48 | compiler generated |
| Shift amount `F/shamt` | U6 | shift sideband |
| Segment index | U4 | 16 segments |
| PWL range | 2 bit/lane | middle / low / high |
| low/high tail code | raw 8 bit | operator별 의미가 다름 |

---

# 3. CORE16 4-stage 기능

## 3.1 S0 — SOURCE / PRE

역할:

- source 선택
- 작은 stateless arithmetic
- DSP operand를 만들기 위한 전처리
- PWL segment boundary compare 입력 생성

대표 연산:

```text
pass x
score - rowmax
wire_encode(2^23 + z)
(z << 7) - S
Q << 7 wiring
```

`OP_RQ_RES`에서 `original_skip8:S8`은 PRE arithmetic 입력이 아니다. S0에서 함께 capture된 뒤 S3까지 **sideband로 latency 정렬만** 한다.

## 3.2 S1 — DSP INPUT / COEFFICIENT

역할:

- A27 / B18 / C48 register
- segment별 `M/C` 선택
- feature별 `M_gamma/C_beta` 선택
- op/page/shift sideband 정렬
- `OP_RQ_RES`의 skip8 sideband 정렬

`skip8`은 DSP A/B/C 어디에도 연결되지 않는다.

## 3.3 S2 — DSP PREG

기본:

```text
P = A × B + C
```

MomentPack ACC:

```text
P ← P + A × B
```

`OP_RQ_RES`에서도 S2까지의 DSP 연산은 generic RQ와 동일하다.

```text
P = main × M_RQ + C_RQ
```

이 시점까지 residual addition은 일어나지 않는다.

## 3.4 S3 — POST / COMMIT

역할:

- RNE right shift
- S8/E7/U8 clamp
- PWL low/high tail select
- `OP_RQ_RES` residual add
- Moment S/Q field decode
- raw S25 bypass
- scratch / architectural write

`OP_RQ_RES`의 정확한 S3 순서:

```text
P:S48
  ↓ RNE >> F_RQ
main_rounded
  ↓ NARROW_S8 saturation
main8:S8
  + original_skip8:S8
  ↓ exact signed add
z:S9 [-254,254]
```

`z`에는 post-add RNE/SAT을 적용하지 않는다.

---

# 4. 전체 OP 요약

| OP | S0~S2 연산 | S3 POST / architectural output |
|---|---|---|
| `OP_RQ` | `P=x×M+C` | `RNE>>F → NARROW_S8` |
| `OP_RQ_RES` | **main only:** `P=main×M_RQ+C_RQ`, skip은 sideband | `main8=RNE/SAT(P)`, 이후 **POST add** `main8+original_skip8 → z:S9`, post-add SAT 없음 |
| `OP_GELU` | PWL `P=x×M_g+C_g` | terminal GELU `S8` |
| `OP_QEXP` | `d=score-rowmax`, PWL | `E7 [0,127]` |
| `OP_SM_RECIP_RAW` | PWL `L×M_r+C_r` | `RNE>>8 → positive R:S18` |
| `OP_SM_CONTEXT` | `N×R` | fixed `RNE>>23 → S8` |
| `OP_LN_MOMENT_INIT` | `wire(2^23+z)×z` | Moment P 초기화 |
| `OP_LN_MOMENT_ACC` | `P←P+wire(2^23+z)×z` | 마지막 beat에서 `S:S16`, `Q:U23` decode |
| `OP_LN_D` | `128Q-S²` | `RNE>>4 → D27` |
| `OP_LN_RSQRT` | D27 PWL | `rho:U8` |
| `OP_LN_NORM` | `(128z-S)×rho` | raw `T:S25` |
| `OP_LN_AFFINE` | `T×M_gamma+C_beta` | final `RNE>>F → S8` |

---

# 5. Generic Requantization

## 5.1 함수

PMPU는 raw integer linear accumulator를 내보내고 VFU가 encoder 내부 requant를 담당한다.

```text
x : PMPU S32 container

P = x × M + C
q = NARROW_S8(RNE_even(P / 2^F))
```

Coefficient contract:

```text
M : S18
C : S48
F : U6
```

실제 DSP A에는 compiler/range certificate로 A27-safe value가 들어가야 한다.

## 5.2 4-stage

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `x:S32 container` → active `x:S27` |
| S1 | `A=S27(x)`, `B=S18(M)`, `C=S48(C)` |
| S2 | `P:S48 = A×B+C` |
| S3 | `RNE(P>>F)` → `NARROW_S8` |

---

# 6. GELU

## 6.1 계산 구조

GELU는 별도 polynomial unit을 두지 않는다. FFN1 raw accumulator를 16-segment Direct32/Q-Map PWL로 **terminal S8 code에 직접 mapping**한다.

```text
FFN1 PMPU accumulator:S32
        │
        ▼
   segment select
        │
        ▼
P = x × M_g[seg] + C_g[seg]
        │
        ▼
POST decode / tail / clamp
        │
        ▼
GELU output:S8 [-127,127]
```

중간 GELU real/fixed tensor를 따로 materialize하지 않는다.

## 6.2 4-stage

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `x:S32 container`, active PWL domain A27-safe, segment compare |
| S1 | `A:S27=x`, `B:S18=M_g[seg]`, `C:S48=C_g[seg]` |
| S2 | `P:S48=x×M_g+C_g` |
| S3 | middle decode + low/high tail select → final `S8 [-127,127]` |

## 6.3 PWL metadata

```text
boundary[15] : S27
seg index     : U4
M[16]         : S18
C[16]         : S48
shift         : U6
low_code      : raw8, GELU에서는 S8 의미
high_code     : raw8, GELU에서는 S8 의미
```

---

# 7. Softmax

## 7.1 수학적 변환

원래 scaled-attention context는:

```text
Softmax((QK^T) / sqrt(H)) × V
```

GaugePack v1의 `H=head_dim=64`이고, PMPU는 별도의 `1/sqrt(H)` 곱셈 없이
raw integer dot-product를 출력한다.

```text
score_int    = Σ(q_code × k_code)
s_score      = s_Q × s_K / sqrt(H)
scaled_score ≈ score_int × s_score
```

`1/sqrt(H)`는 runtime RSQRT 연산이 아니라 compile-time 상수다. 이 상수와
Q/K quantization scale을 합친 `s_score` 전체를 QEXP page의 계수에 fold한다.
`H=64`라서 `1/sqrt(H)=1/8`이지만, QK 뒤에 별도 `>>3` stage를 두지 않는다.
LayerNorm의 RSQRT page와도 무관하다.

GaugePack v1은 explicit probability `E/L`를 materialize하지 않는다.

```text
d_int = score_int - rowmax_int
E_j   ≈ exp(d_int × s_score) 를 E7로 양자화
L   = Σ E_j
N   = Σ E_j V_j
context ≈ N / L
```

항등식:

```text
Σ ((E/L) × V) = (Σ E×V) / L = N/L
```

따라서 normalization/division은 PV 이후 terminal context boundary로 미룬다.

## 7.2 전체 계산 그래프

```text
PMPU QK
   │
   ▼
score:S24 ──────────────┐
   │                    │
   └── rowmax:S24       │
            │           │
            └─────┬─────┘
                  ▼
         d=score-rowmax:S25
                  │
                  ▼
   QEXP 16-seg PWL (s_score fused)
                  │
                  ▼
             E7 [0,127]
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
  E7 L-DRAM commit      L=ΣE:U13
          │                │
          │                ▼
          │         RAW reciprocal PWL
          │         R≈2^23/L:S18+
          │                │
          ▼                │
  L-DRAM reload→PMPU PV    │
   N=Σ(E×V):S21            │
          │                │
          └────────┬───────┘
                   ▼
                N × R
                   │
                   ▼
                P:S48
                   │
             fixed RNE >>23
                   ▼
           context:S8 [-127,127]
```

## 7.3 QK / rowmax

```text
score  : S24 raw QK integer accumulator
rowmax : S24
d      : score-rowmax = S25
```

Invalid key는 rowmax 계산에서 제외한다.

PMPU의 DOT1 score drain은 tile 안에서 key column을 `15→0` 순서로 내보낸다.
따라서 `seq_len<16`이면 첫 physical score beat가 invalid일 수 있다. 각 16-query
tile 시작 시 rowmax를 `S24_MIN=24'sh800000`으로 초기화하고,
`key_valid=1`인 beat만 max-update한다. 첫 physical beat의 score로 초기화하는
방식은 금지한다. Legal command에는 valid key가 반드시 있으므로 별도
`rowmax_has_valid` state/output은 두지 않는다.

`s_score`는 positive constant이므로 raw integer score에서 rowmax를 구해도
scaled score에서 구한 rowmax와 같은 key를 선택한다. 따라서 rowmax 앞에
별도 scale 회로가 필요 없다.

### Softmax mask / scheduling 계약

GaugePack v1은 한 번에 16개 query row와 현재 key 하나의 score를 처리한다.

```text
command : seq_len[6:0]  (BERT input의 실제 token 수, legal 2...64)
schedule: seq_len_padded = 16 × ceil(seq_len/16), one of 16/32/48/64
beat    : key_valid = (key_index < seq_len)  (1 bit)
lanes   : 같은 key_valid를 16개 query lane에 broadcast
```

QK ingest에서는 `key_index=o_feat[8:0]`이고, score replay에서는 현재 replay address가
`key_index`다. QK는 full-width `o_feat<N_sched` 및 `o_feat<64`를 검사한 뒤에만
`o_feat[5:0]`를 score address로 사용한다. 6-bit truncate 후 검사하거나 beat counter로
reverse-drain 주소를 재구성하지 않는다. 두 phase가 같은 `seq_len` 비교 계약을 사용한다.
PMPU의 QK data는 S32 container이므로 각 lane에서
`acc[31:24]=={8{acc[23]}}`를 assert한 뒤 `acc[23:0]`를 score S24로 pack한다.
상위비트 확인 없는 truncate/saturation은 금지한다.

Host는 token tensor의 valid prefix `seq_len`을 보존하고 suffix를 `seq_len_padded`까지
padding한다. `M/N/K`는 PMPU가 실제로 처리하는 scheduled matrix dimension이며:

```text
Q/K/V, output projection, FFN : M = seq_len_padded
QK per head                   : M=N=seq_len_padded, K=head_dim
PV per head                   : M=seq_len_padded, N=head_dim, K=seq_len_padded
```

QK의 `N_sched=seq_len_padded`는 padded key dimension이다. `seq_len`은 그중 유효한 token
prefix 길이이며 scheduled `M/N/K`에서 역추론하지 않는다. `seq_len`은 첫 QK score beat가
들어오기 전에 VFU Softmax controller가 latch하고 score/rowmax, score replay,
QEXP/rowsum이 끝날 때까지 유지한다. Generic RQ, GELU, residual, LayerNorm은
token 간 reduction을 하지 않으므로 `seq_len`을 사용하지 않는다.

위 `M/N/K`는 padded tensor/job 축의 의미다. Current PMPU에서 한 QK command의
`M_cmd=seq_len_padded`를 주면 여러 `o_tr` query tile이 연속 drain된다. 현재 Softmax
row state와 raw-score S-pad live set은 한 16-query tile 기준으로 작성된 부분이 있으므로, 실제 attention command를
`M_cmd=16`으로 tile-loop할지 multi-tile buffering을 지원할지는 Master
`OPEN-SCHED-001`이며 이 문서의 산술식으로 임의 결정하지 않는다. `SPAD-20260825-001`에
따라 PMPU-visible Logical-DRAM은 feature-major `addr=feature*MT+tr`를 사용하고,
S-pad는 private layout이다. Commit path가 두 layout 사이 pack/transpose를 소유한다.
또한 co-worker가 수정하기로 한 external PMPU row-dimension register `M_r`의 새
commit/diff는 아직 확인되지 않았다. 이 `M_r`는 §7.6 reciprocal slope와 다른 신호다.
`OPEN-PMPU-MR-001`을 감사하기 전에는 current `M_r→MT` 동작을 최종 schedule의
근거로 확정하지 않는다.

16개 physical query lane은 모두 실행한다. 마지막 tile의
`query_index>=seq_len` row는 padded/don't-care output이지만 다른 query row와
reduction state를 공유하지 않으므로 valid row를 오염시키지 않는다. 따라서
Softmax datapath에는 `query_valid`와 lane별 `pair_valid[15:0]`가 없다.

Legal Softmax command는 `[CLS]`와 `[SEP]`를 포함한다.

```text
2 <= seq_len <= 64
seq_len_padded = N_sched = 16 × ceil(seq_len/16)
seq_len <= N_sched <= 64
```

값 64를 그대로 표현해야 하므로 `seq_len`은 U7이다. U6로 줄이면 64가 0으로
alias되므로 금지한다. `seq_len_padded/N_sched`도 64를 표현해야 하므로 U7이다.

GaugePack v1 BERT encoder의 valid token은 index 0부터 연속된 prefix이고 padding은
suffix다. 따라서 일반 `key_mask[63:0]`, causal/lane별 attention mask는 지원하지
않는다. Masking은 `seq_len`으로 판정하는 key 축에만 적용한다.
Padded input code를 0으로 채우더라도 projection bias 이후 key/value가 0이라고
보장되지 않으므로 이 key-axis mask는 생략할 수 없다.

Acceptance criteria:

```text
1. seq_len_padded == 16/32/48/64 and seq_len <= seq_len_padded < seq_len+16
2. QK는 query tile마다 정확히 seq_len_padded개의 scheduled key beat를 처리
3. key_index>=seq_len인 beat는 rowmax/rowsum을 갱신하지 않고 QEXP E=0
4. valid query prefix [0,seq_len)의 context는 unpadded variable-S golden과 일치
5. padded query suffix [seq_len,seq_len_padded)의 output은 correctness 비교 대상에서 제외
```

## 7.4 QEXP

### 함수

```text
if !key_valid  : E = 0
else if d == 0 : E = 127
else           : E = QEXP_PWL(d; s_score fused)

E ∈ [0,127]
```

QEXP page의 analytic teacher/compile 목적함수는:

```text
E_ref(d) = clamp(RNE_even(127 × exp(d × s_score)), 0, 127)
s_score  = s_Q × s_K / sqrt(head_dim)
```

이때 page metadata의 `score_scale_folded`는 반드시 위 `s_score` 전체를
의미한다. PMPU/VFU에서 `d`를 미리 `1/sqrt(head_dim)`로 scale한 뒤
이 page에 넣으면 attention scale이 두 번 적용되므로 금지한다.

### 4-stage

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `score:S24 - rowmax:S24 → d:S25`, segment select |
| S1 | `A:S27=signext(d)`, `B:S18=M_e[seg]`, `C:S48=C_e[seg]` |
| S2 | `P:S48=d×M_e+C_e` |
| S3 | PWL decode → E7 clamp, invalid key → 0, endpoint `d=0` → 127 |

출력:

```text
E semantic : U7 [0,127]
physical   : 8-bit byte, bit7=0
```

## 7.5 Rowsum

`2<=seq_len<=64`:

```text
input contract:
  valid_i      = E beat transaction-valid
  key_valid_i  = controller가 replay key index와 seq_len으로 생성한 scalar

update:
  fire && clear_i              -> L = key_valid_i ? E : 0
  fire && !clear_i && key_valid_i -> L = L + E
  fire && !key_valid_i         -> 누적값 유지

주의:
  key_valid_i는 산술 update만 gate한다.
  clear_i/last_i/result_valid_o는 scheduled beat 기준으로 진행한다.

L = Σ E_j, 0 <= j < seq_len
invalid key의 E_j = 0
L_max(seq_len) = seq_len × 127 <= 64 × 127 = 8128
```

Legal command에는 valid key가 하나 이상 있다. 각 query row의 rowmax 위치는
`d=0`이고 QEXP endpoint가 정확히 `E=127`이므로 모든 query lane에서:

```text
L_min = 127
L_max = 8128
```

따라서:

```text
L : U13
legal domain : 127 ... 8128
L=0          : illegal command/state
```

## 7.6 RAW reciprocal

목표:

```text
R ≈ 2^23 / L
```

현재 v1:

```text
16-segment raw geometric minimax PWL
NO CLZ
NO L normalization
NO row-dependent exponent
NO variable final context shift
```

연산:

```text
P_r = L × M_r[seg] + C_r[seg]
R   = RNE_even(P_r >> 8)
```

비트폭:

```text
L   : U13
M_r : S18
C_r : S48
P_r : S48
R   : positive S18
```

### 4-stage

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `L:U13`, segment select |
| S1 | `A:S27=L`, `B:S18=M_r`, `C:S48=C_r` |
| S2 | `P_r:S48` |
| S3 | `RNE(P_r>>8)` → positive `R:S18` |

## 7.7 PV numerator

```text
N = Σ(E × V)
E ∈ [0,127]
V ∈ [-127,127]
```

범위 증명:

```text
|N| <= 127L <= 1,032,256
```

따라서:

```text
N semantic            : S21
PMPU external container: S32
N FIFO                 : semantic S21
```

PV output에는 intermediate requant를 넣지 않는다.

## 7.8 Context apply

```text
P = N × R
context = NARROW_S8(RNE_even(P >> 23))
```

### 4-stage

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `N:S21`, `R:S18` |
| S1 | `A:S27=signext(N)`, `B:S18=R`, `C=0` |
| S2 | `P:S48=N×R` |
| S3 | fixed `RNE>>23` → `NARROW_S8` context |

최종 context scale:

```text
s_context = s_V
```

---

# 8. LayerNorm

## 8.1 원래 식

Hidden size `H=128`.

```text
μ  = (1/H) Σ z_real
σ² = (1/H) Σ(z_real-μ)²
y  = γ (z_real-μ) / sqrt(σ²+ε) + β
```

Residual 입력 정수 코드는:

```text
z_real = s_z × z
```

라고 하자.

다음을 정의한다.

```text
S = Σz
Q = Σz²
D = 128Q - S²
n = 128z - S
```

그러면 input scale이 소거되어:

```text
normalized = n / sqrt(D)
```

이 된다. 이 항등식이 현재 LayerNorm 정수 datapath의 핵심이다.

---

## 8.2 전체 계산 그래프

```text
PMPU main:S32
    │
    │ main-only RQ → native skip scale
    ▼
 main8:S8
    │
    │
    │                         original skip8:S8
    │                               │
    │                               │ sideband only through S0~S2
    │                               │
    └───────────────────────┐       │
                            │       │
                            ▼       ▼
                      S3 POST-ALU
                  main8 + original skip8
                            │
                            ▼
                          z:S9
                            │
                            ▼
                    G23 MomentPack
                    │             │
                    ▼             ▼
                  S:S16         Q:U23
                    └──────┬──────┘
                           ▼
                    D=128Q-S²:U30
                           │
                        RNE >>4
                           ▼
                         D27
                           │
                    RSQRT 1/sqrt(16x)
                           │
                           ▼
                       rho:U8
                           │
 z:S9 ──→ n=128z-S:S17 ────┘
                 │
                 ▼
            T=n×rho:S25
         RAW / NO SHIFT / NO RNE
                 │
                 ▼
       P=T×M_gamma+C_beta:S48
                 │
          final RNE >> F_final
                 ▼
        LayerNorm output:S8
```

---

## 8.3 LayerNorm 6 phases

```text
L0 MAIN RQ + POST-ALU RESIDUAL
L1 MOMENTPACK STATS
L2 D
L3 RSQRT
L4 NORMALIZE
L5 AFFINE
```

`SKIP_ALIGN` pass는 현재 v1에 없다. Skip은 producer가 이미 만든 original narrow-S8 tensor 그대로 사용한다.

---

## 8.4 L0 — MAIN RQ + POST-ALU RESIDUAL

이 phase는 **두 부분**으로 나눠서 이해해야 한다.

### A. S0~S2: Main branch RQ

Main만 native skip scale로 requant한다.

```text
P = main_raw × M_RQ + C_RQ
main8 = NARROW_S8(RNE(P >> F_RQ))
```

비트폭:

```text
main raw : S32 container / A27-certified active range
M_RQ     : S18
C_RQ     : S48, 현재 일반적으로 0
F_RQ     : U6
main8    : S8 [-127,127]
```

### B. S3: POST-ALU residual add

```text
z = main8 + original_skip8
```

비트폭:

```text
main8         : S8 [-127,127]
original skip : S8 [-127,127]
z             : S9 [-254,254]
```

Residual add 이후에는:

```text
NO RNE
NO SAT
NO requant
```

### 4-stage

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `main:S32` source, `original_skip8:S8`은 sideband capture만 |
| S1 | `A:S27=main`, `B:S18=M_RQ`, `C:S48=C_RQ`; skip은 delay만 |
| S2 | `P:S48=main×M_RQ+C_RQ`; residual add 아직 없음 |
| S3 | `P → RNE>>F → NARROW_S8 = main8`, 이후 **POST-ALU**에서 `main8+original_skip8 → z:S9`, post-add SAT 없음 |

정확한 mental model:

```text
                DSP spine
main S32 ─→ A27×M18+C48 ─→ P48
                              │
                              ▼
                         S3 POST-ALU
                              │
                         RNE + S8 SAT
                              │
                              ▼
                           main8:S8
                              │
original skip8:S8 ────────────+
                              │ exact S9 add
                              ▼
                            z:S9
```

---

## 8.5 L1 — G23 MomentPack

필요한 통계:

```text
S = Σz
Q = Σz²
```

비트폭:

```text
z ∈ [-254,254]
S_max = 128×254 = 32512      → S16
Q_max = 128×254² = 8,258,048 < 2^23 → U23
```

Pack identity:

```text
Pmom = Σ z(2^23 + z)
     = (S << 23) + Q
```

DSP mapping:

```text
A27 = wire_encode(2^23 + z)
B18 = sign_extend(z)
C48 = 0
```

INIT:

```text
P = A×B
```

ACC:

```text
P ← P + A×B
```

Final decode:

```text
Q = Pmom[22:0]   → U23
S = Pmom >>> 23  → S16
```

Moment product는 semantic S32 정도이고 packed total은 S39 bound 안에 있어 P48에 들어간다.

### 4-stage — INIT

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `z:S9` → wire `2^23+z` |
| S1 | `A:S27`, `B:S18`, `C=0` |
| S2 | `P=A×B`, PREG에 저장 |
| S3 | ordinary data write 없음, PREG가 Moment state가 됨 |

### 4-stage — ACC

| Stage | 데이터 / 연산 |
|---|---|
| S0 | 다음 `z:S9` |
| S1 | 동일한 `A:S27`, `B:S18`, P-feedback enable |
| S2 | `P←P+A×B` |
| S3 | 마지막 beat에서 `S:S16`, `Q:U23` decode/commit |

---

## 8.6 L2 — D

```text
D = 128Q - S²
```

현재 BERT-Tiny에서는 integerized epsilon:

```text
epsilon_D = 0
```

비트폭:

```text
Q      : U23
Q << 7 : U30
S      : S16
D      : U30 semantic
```

D를 signed positive container로 보면 S31이 필요하므로 signed A27에 그대로 넣을 수 없다.

따라서:

```text
D27 = RNE_even(D >> 4)
```

최악값에서도 D27은 positive S27 범위에 들어간다.

DSP mapping:

```text
A = sign_extend(S)
B = -sign_extend(S)
C = zero_extend(Q << 7)
P = A×B + C
  = 128Q - S²
```

`C`의 `Q<<7`은 unsigned positive value이므로 sign-extension하지 않고 zero-extension한다.

### 4-stage

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `S:S16`, `Q:U23`, `Q<<7:U30` wiring |
| S1 | `A:S27=S`, `B:S18=-S`, `C:S48=zeroext(Q<<7)` |
| S2 | `P:S48`, semantic `D:U30` |
| S3 | `RNE_even(D>>4)` → `D27:U27 / positive A27-safe` |

---

## 8.7 L3 — RSQRT

D를 `>>4`했기 때문에 RSQRT page가 받는 입력은:

```text
x = D27 ≈ D/16
```

원래 필요한 값은 `1/sqrt(D)`이므로 page teacher는:

```text
rho_real ≈ 1 / sqrt(16x)
```

를 학습한다.

출력:

```text
rho_code : U8 [0,255]
rho_real ≈ rho_code × s_rho
```

`D=0`을 위한 별도 detector나 zero override는 두지 않는다. Constant row는
`n=128z-S=0`이므로 page가 내는 finite clamped `rho` 값과 무관하게 다음
NORMALIZE의 `T=n×rho`가 정확히 0이 된다.

### 4-stage

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `D27:U27`, segment select |
| S1 | `A:S27=D27`, `B:S18=M_rsqrt[seg]`, `C:S48=C_rsqrt[seg]` |
| S2 | `P:S48=D27×M+C` |
| S3 | PWL decode / RNE / U8 clamp → `rho:U8` |

---

## 8.8 L4 — NORMALIZE

먼저:

```text
n = 128z - S
  = (z<<7) - S
```

범위 증명:

```text
|n| <= 65024
n : S17
```

그 다음:

```text
T = n × rho
```

비트폭:

```text
n   : S17
rho : U8
T   : S25
```

최대 bound:

```text
65024 × 255 = 16,581,120 < 2^24
```

따라서 S25면 충분하다.

**중요:** T는 raw S25 그대로 다음 phase로 넘긴다.

```text
NO >>8
NO intermediate RNE
NO intermediate SAT
NO intermediate requant
```

Scale 의미:

```text
rho_real ≈ rho_code × s_rho
normalized ≈ T × s_rho
```

즉 `rho`의 scale은 T 이후에도 살아 있고 마지막 affine coefficient에 fold된다.

### 4-stage

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `z:S9`, `S:S16` → `n=(z<<7)-S:S17` |
| S1 | `A:S27=signext(n)`, `B:S18=zero/sign-safe ext(rho:U8)`, `C=0` |
| S2 | `P:S48`, semantic product `T:S25` |
| S3 | `p_i[24:0]`를 raw `T:S25`로 그대로 commit, shift/RNE/SAT 없음 |

---

## 8.9 L5 — AFFINE + final requant

원래:

```text
y_real = gamma × normalized + beta
```

그리고:

```text
normalized ≈ T × s_rho
```

따라서 compiler가 다음을 fold한다.

```text
M_gamma ≈ 2^F × gamma × s_rho / s_y
C_beta  ≈ 2^F × beta / s_y
```

실제 DSP 연산:

```text
P = T × M_gamma + C_beta
```

마지막:

```text
y8 = NARROW_S8(RNE_even(P >> F_final))
```

비트폭:

```text
T         : S25
M_gamma   : S18
C_beta    : S48
F_final   : U6
P         : S48
y8        : S8 [-127,127]
```

### 4-stage

| Stage | 데이터 / 연산 |
|---|---|
| S0 | `T:S25` replay, feature index |
| S1 | `A:S27=signext(T)`, `B:S18=M_gamma[h]`, `C:S48=C_beta[h]` |
| S2 | `P:S48=T×M_gamma+C_beta` |
| S3 | `RNE>>F_final` → `NARROW_S8` final output |

---

## 8.10 LayerNorm stage matrix — row=path, column=stage

| Path | S0 — SOURCE/PRE | S1 — DSP operand | S2 — DSP PREG | S3 — POST/COMMIT |
|---|---|---|---|---|
| `L0 RQ_RES` | `main:S32`, skip:S8 sideband capture | `A27=main`, `B18=M_RQ`, `C48=C_RQ`; skip delay only | `P48=main×M+C`, residual add 없음 | `P→RNE/S8=main8`, 이후 **POST add** `main8+original_skip8→z:S9` |
| `L1 MOMENT_INIT` | `z:S9 → wire(2^23+z)` | `A27`, `B18=z`, `C=0` | `P=A×B` | PREG state 유지 |
| `L1 MOMENT_ACC` | next `z:S9` | 동일 operand + P-feedback | `P←P+A×B` | final beat `S:S16`, `Q:U23` |
| `L2 D` | `S:S16`, `Q:U23`, `Q<<7:U30` | `A27=S`, `B18=-S`, `C48=zeroext(Q<<7)` | `P48`, semantic `D:U30` | `RNE>>4 → D27` |
| `L3 RSQRT` | `D27:U27`, segment | `A27=D27`, `B18=M`, `C48=C` | `P48` | PWL decode → `rho:U8` |
| `L4 NORM` | `z:S9`, `S:S16 → n:S17` | `A27=n`, `B18=rho`, `C=0` | `P48`, semantic `T:S25` | raw `T:S25`, no shift |
| `L5 AFFINE` | `T:S25` | `A27=T`, `B18=M_gamma`, `C48=C_beta` | `P48` | final `RNE>>F → S8` |

---

# 9. 함수별 I/O 빠른 참조표

| Function | 입력 | 핵심 연산 | 출력 | 출력 의미 |
|---|---|---|---|---|
| Generic RQ | S32 container, active A27-safe | `x×M18+C48` | S8 | quantized activation |
| RQ_RES DSP | main S32 | `main×M_RQ+C_RQ` | P48 → main8:S8 | main branch requant |
| RQ_RES POST | main8:S8 + original skip:S8 | exact add | z:S9 | residual/LN input |
| GELU | S32/A27-safe | 16-seg PWL | S8 | terminal GELU code |
| QEXP | raw d:S25 | 16-seg PWL, `s_Q×s_K/sqrt(H)` fused | E7 | unnormalized scaled-exp code |
| SM reciprocal | L:U13 | 16-seg raw PWL | positive S18 | `≈2^23/L` |
| SM context | N:S21 + R:S18 | multiply | S8 | `N/L`, scale=s_V |
| LN Moment | z:S9 | G23 P-feedback | S16 + U23 | S/Q |
| LN D | S16 + U23 | `128Q-S²` | D27 | D/16 rounded |
| LN RSQRT | D27 | 16-seg PWL | U8 | `1/sqrt(D)` code |
| LN NORM | z:S9, S:S16, rho:U8 | `(128z-S)×rho` | S25 | raw normalized code |
| LN AFFINE | T:S25 | `T×M_gamma+C_beta` | S8 | final LayerNorm activation |

---

# 10. Scratch / lifetime

S-pad는 architectural tensor store가 아니라 VFU phase-local intermediate store다.
PMPU operand와 다음 layer가 소비할 final code는 Logical-DRAM에 commit한다.

## 10.1 Softmax

```text
Score    : 16 × S24 = 384 meaningful bits
E7       : 16 × 8-bit physical commit beat; Logical-DRAM tensor로 기록
Context  : 16 × S8 final commit beat; Logical-DRAM tensor로 기록
```

Softmax에서 S-pad의 필수 resident 값은 replay가 필요한 raw Score다. E7는 QEXP/rowsum과
동시에 Logical-DRAM commit stream으로 내보낼 수 있으며 Context는 final S8 생성 후
Logical-DRAM에 commit한다. 구현상 skid/pack staging을 둘 수는 있지만 PMPU-visible
resident tensor로 취급하지 않는다.

## 10.2 LayerNorm overlay

```text
original skip S8 : producer-owned residual buffer
z S9             : 16 × 9  = 144 meaningful bits
T S25            : 16 × 25 = 400 meaningful bits
```

LayerNorm은 동일 scratch address 영역을 phase별로 재사용한다.

```text
L0 write z
  ↓
L1 Moment reads z
  ↓
L2/L3 scalar metadata
  ↓
L4 replay z, overwrite with T
  ↓
L5 read T, write final S8
```

## 10.3 Dynamic V corner-turn

Runtime physical transpose는 PMPU `P·V`를 위한 dynamic V 16×16 corner-turn 하나뿐이다.
V projection output tile은 dedicated `2×16×16×S8` ping-pong FF leaf
`vfu_vcorner_ff`에 feature-column 방향으로 기록하고 token-row 방향으로 drain한 뒤
PMPU-native `PV_WGT16` layout으로 Logical-DRAM에 commit한다. V-corner는 Intermediate
S-pad의 address/data/port를 사용하지 않는다. Static INT4 W는 Host/DMA가 이미 `W^T`
layout으로 초기 적재하므로 VFU runtime transpose 대상이 아니다.

## 10.4 Intermediate S-pad container

Intermediate S-pad는 raw Score와 LayerNorm `z/T`를 위한 16-bank memory다. 한 accepted
beat의 16 lane은 동일 logical key/feature address를 사용하므로 bank별 독립 address나
lane별 write-enable은 필요하지 않다. Source adapter가 다음처럼 32-bit container를 만든다.

```text
Score S24 -> S32 sign extension
z S9      -> S32 sign extension
T S25     -> S32 sign extension
```

Production memory leaf의 기본 geometry는 `16 banks × 512 depth × 32-bit`, synchronous
`1R1W`, compile-time `SPAD_RD_LAT={1,2}` 지원과 default `2`다. Depth 512는 full padded-M
phase-serial schedule과 repeated-M16 schedule을 모두 수용하며, `512×36` BRAM 후보에서
depth 128로 줄여도 bank당 primitive 수가 줄지 않는다. Memory leaf는 signedness나 mode를
해석하지 않고 32-bit pattern만 보존한다. Data array는 reset하지 않으며 same-bank
same-address read/write 결과에 의존하지 않는다.

Score/LN mode, exact address overlay, rowmax/S/Q/L/R side state, controller sideband pipeline,
Logical-DRAM commit은 memory leaf 밖에 남고 `OPEN-SPAD-001` 및 관련 controller 결정에서
동결한다. BRAM/URAM primitive 자체도 합성 결과 전에는 기능 계약이 아니다.

---

# 11. Precision이 실제로 줄어드는 지점

VFU에서 shift가 존재한다고 해서 모두 같은 의미가 아니다. **precision-reducing boundary는 계산 그래프에 반드시 명시한다.**

| 위치 | 동작 | 의미 |
|---|---|---|
| Generic RQ | `RNE >> F` | accumulator → activation quantization |
| RQ_RES main | `RNE >> F_RQ` | main을 skip native scale로 맞춤 |
| QEXP | PWL output quantization | E7 code 생성 |
| SM reciprocal | `RNE >> 8` | reciprocal code 생성 |
| SM context | `RNE >> 23` | `N×R` → context S8 |
| LN D | `RNE >> 4` | U30 D를 A27-safe D27로 축소 |
| LN RSQRT | PWL output quantization | rho U8 생성 |
| LN NORM | **없음** | raw T:S25 유지 |
| LN AFFINE | final `RNE >> F_final` | 최종 S8 output |

특히:

```text
T = (128z-S) × rho
```

뒤에는 `>>8` 같은 중간 shift를 넣지 않는다.

---

# 12. Architecture가 고정하는 것 / Compiler가 생성하는 것

## 12.1 Architecture 고정 항목

```text
CORE16 = 16 lanes
DSP contract = A27 / B18 / C48 / P48
NARROW_S8 = [-127,127]

Softmax:
score S24 raw QK accumulator
rowmax S24
d S25
seq_len U7, legal 2...64
key_valid = (key_index < seq_len), scalar broadcast
QEXP score scale = s_Q × s_K / sqrt(head_dim), page-folded
E7 [0,127]
L U13
N S21
R S18
context shift = 23

LayerNorm:
main-only RQ to native skip scale
residual add owner = S3 POST-ALU
skip is original producer-owned S8
z S9, no post-add SAT
MomentPack G23
S S16
Q U23
D U30
D shift = 4
D27
RSQRT U8
n S17
T raw S25
NO T intermediate shift
M_gamma S18
C_beta S48
F_final U6
final NARROW_S8
```

Compiler는 위 representation 자체를 임의로 바꾸면 안 된다.

예를 들어 T에서 width concern이 생기더라도:

```text
S17 × U8 → S25 → A27
```

이라는 architecture proof를 먼저 확인해야 한다. 임의로 `T>>8`을 넣고 `M_gamma<<8`로 보상하는 것은 architecture 변경이다.

## 12.2 Compiler 생성 항목

```text
Generic/Main RQ:
  M_RQ
  C_RQ
  F_RQ

GELU/QEXP/RSQRT page:
  boundary[15]
  M[16]
  C[16]
  shift
  low_code/high_code

QEXP site:
  head_dim
  s_score = s_Q × s_K / sqrt(head_dim)
  metadata.score_scale_folded = s_score
  input domain = raw d_int (runtime rescale 없음)

Softmax reciprocal:
  fixed raw-v1 boundaries/M/C

LayerNorm site:
  rsqrt page
  s_rho
  M_gamma[128]
  C_beta[128]
  F_final

그리고 각 경로의 P48 range certificate
```

---

# 13. 폐기된 경로 / 다시 섞지 말 것

## 13.1 LayerNorm 폐기 항목

```text
7-pass SKIP_ALIGN
separate skip requant pass
aligned_skip scratch tensor
external S_acc / Q_acc fabric accumulator
intermediate T requant / >>8
explicit normalized tensor materialization
residual add in PRE/DSP
```

현재는:

```text
6 phases
main-only RQ
original skip is sideband
S3 POST-ALU residual add
G23 MomentPack P-feedback
D>>4 → D27
RSQRT 1/sqrt(16x)
raw T:S25
folded final affine
```

## 13.2 Softmax 비-baseline 항목

```text
exact DIV4
CLZ + denominator normalization
row-dependent final shift
U8 XOR/recenter
Vsum correction
explicit probability E/L tensor
PV intermediate requant
raw-v2 calibrated reciprocal as baseline
separate runtime score × 1/sqrt(head_dim) stage
QEXP에 fold된 attention scale의 중복 적용
```

---

# 14. Block diagram용 canonical 계산 그래프

## 14.1 Pointwise / Q-Map

```text
x
│
├─ segment compare
│
▼
A27 × M18 + C48
│
▼
P48
│
▼
operator-specific POST
│
├─ S8      : RQ / GELU / context / final LN
├─ E7      : QEXP
├─ U8      : RSQRT
├─ S18     : Softmax reciprocal
└─ raw S25 : LN normalize
```

## 14.2 Softmax

```text
seq_len U7 → key_valid=(key_index<seq_len), key 축만 gate
QK raw S24  (s_Q×s_K/sqrt(H)는 QEXP page에 fold)
  ↓ rowmax S24
score-rowmax S25
  ↓
QEXP → E7
  ├──────────────→ L U13 → recip → R S18 ──────┐
  │                                             │
  └→ PMPU PV → N S21 ──────────────────────────×
                                                ↓
                                              P48
                                                ↓ RNE>>23
                                           context S8
```

## 14.3 LayerNorm

```text
main S32 → DSP RQ → P48
                    │
                    ▼
               S3 POST-ALU
                    │
                 main8 S8
                    │
original skip S8 ───+
                    │ exact add
                    ▼
                   z S9
                    │
                    ▼
              MomentPack G23
                │        │
                ▼        ▼
              S16       Q23
                └───┬────┘
                    ▼
                  D U30
                    ↓ RNE>>4
                  D27
                    ↓ RSQRT 1/sqrt(16x)
                 rho U8
                    │
z S9 → n=128z-S:S17 ─┘
          │
          ▼
      n×rho → T:S25 raw
          │
          ▼
T×M_gamma+C_beta → P48
          │
          ↓ final RNE/SAT
     LN output:S8
```

---

# 15. RTL / Compiler 변경 전 검토 체크리스트

산술 변경을 제안하기 전에 아래를 먼저 답한다.

1. semantic equation이 무엇인가?
2. 해당 edge에 실제 어떤 integer value가 있는가?
3. signed인가 unsigned인가?
4. 최소 안전 bitwidth는 몇 bit인가?
5. code의 scale은 무엇인가?
6. 다음 DSP A27/B18/C48에 실제로 들어가는가?
7. shift는 수학적 scale 변환인가, 실제 precision loss인가?
8. shift를 넣으면 그 scale은 어디에서 보상되는가?
9. scale compensation이 버린 low bits까지 복구하는가? 보통 아니다.
10. 이 위치에 RNE가 필요한가?
11. saturation이 합법적인가?
12. compiler가 이 representation을 바꿀 권한이 있는가?
13. frozen Python model과 bit-true인가?
14. residual이라면 add가 정확히 어느 stage 소유인가? 현재는 S3 POST-ALU다.

---

# 16. Source-of-truth 우선순위

충돌 시 다음 순서로 본다.

1. Project owner의 현재 대화 내 명시적 지시
2. Library `gaugepack-master-work.md`의 최신 `FROZEN` 결정
3. `GaugePack_VFU_Architecture_Frozen_v1.md` — 명시적 `OPEN`을 제외한 frozen architecture/dataflow
4. `VFU_Function_Frozen.md` — VFU arithmetic / calculation graph 기준 문서
5. current compiler artifacts / manifests — 실제 production coefficient 값
6. current PRE/POST/CORE RTL/TB — 구현 및 closure 증거
7. older RevA/RevB / 7-pass spreadsheets / Git의 stale Frozen mirror — history only

Master, `VFU_Function_Frozen.md`, frozen architecture가 충돌하면 조용히 한쪽을
선택하지 말고 **contract mismatch**로 취급한다. Master에 `OPEN`으로 기록하고
판정 후 두 Frozen 문서를 같은 revision decision으로 정렬한다.

---

# 17. 초압축 Cheat Sheet

```text
CORE16
S0 PRE → S1 A27/B18/C48 → S2 P48 → S3 POST

RQ
S32 → A27×M18+C48 → P48 → RNE>>F → S8

RQ_RES
main S32 → DSP RQ → P48
                    ↓ S3 RNE/SAT
                  main8 S8
original skip S8 ───+
                    ↓ POST exact add
                   z S9

GELU
S32/A27 → PWL M18/C48 → P48 → terminal S8

SOFTMAX
seq_len U7 → scalar key_valid=(key_index<seq_len)
score raw S24  [QEXP에 s_Q×s_K/sqrt(H) fold]
rowmax S24
  ↓
d S25 → QEXP → E7
              ↓
           L U13 → R S18 (~2^23/L)
E7×V → N S21
N×R → P48 → RNE>>23 → context S8

LAYERNORM
main RQ → S3 POST add with original skip → z S9
z → MomentPack → S16/Q23
D=128Q-S² → U30 → RNE>>4 → D27
D27 → RSQRT[1/sqrt(16x)] → rho U8
n=128z-S → S17
n×rho → raw T S25   <<< NO SHIFT
T×M_gamma+C_beta → P48 → final RNE → S8
```

---

## 최종 mental model

VFU는 GELU/Softmax/LayerNorm 전용 회로를 따로 모아놓은 것이 아니라,

```text
하나의 16-lane integer transform spine
+
operator별 PRE 의미
+
operator별 POST 의미
+
reduction / metadata state
```

로 이해한다.

특히 가장 중요한 설계 규칙은 다음이다.

> **산술 의미를 바꾸는 shift, RNE, SAT, residual add를 구현 세부사항 속에 숨기지 않는다. 계산 그래프에 반드시 stage와 함께 명시한다.**
