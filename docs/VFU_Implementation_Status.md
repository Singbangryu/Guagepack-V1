# GaugePack VFU Current Implementation Status

> **문서 성격:** Git source/TB/model의 구현·검증 현황 snapshot
> **감사일:** 2026-08-27
> **감사 기준 source HEAD:** `d41d7da926aab16c7efe619fbdc730ef2da33ace`
> **정본 baseline:** Master v22 / Architecture v20 / Function v13
> **대상:** `rtl/vfu`, `tb/vfu`, `model`

이 문서는 현재 repository에 실제로 존재하는 구현과 확인된 증거를 정리합니다.
기능 또는 구조 계약을 새로 정의하지 않으며, 계약은 최신 Master와 Frozen
Function/Architecture가 우선합니다. Source가 존재한다는 사실과 검증 완료를 구분하고,
unit test PASS를 전체 VFU·합성·hardware 검증으로 확대하지 않습니다.

## 1. 상태 표기

| 표기 | 의미 |
|---|---|
| `IMPLEMENTED` | 구현 source가 존재함. 그 자체로 correctness를 뜻하지 않음 |
| `RTL_SIM_PASS` | 명시한 standalone RTL simulation 범위가 PASS |
| `MASTER_ACCEPTED` | Work-master가 task 계약과 독립 증거를 대조해 수락 |
| `BUILD_BLOCKED` | 현재 clean source에서 compile/elaboration을 재현할 수 없음 |
| `DRAFT` | 참고용 초안. production build source가 아님 |
| `NOT_STARTED` | 현재 HEAD에 수락 가능한 구현 source가 없음 |
| `NOT_RUN` | 해당 검증을 실행하지 않음 |

## 2. 현재 구현 요약

| 영역 | 파일 | 구현 상태 | 현재 증거와 한계 |
|---|---|---|---|
| 외부 VFU op | `rtl/vfu/vfu_external_ops.vh` | `IMPLEMENTED` | 3-bit `VFU_RQ/GELU/SM/SM_NL/LN` spelling과 내부 sequence 주석이 존재. `OPEN-OP-001`이 남아 있어 최종 controller ABI/lifecycle로 간주하지 않음 |
| 내부 micro-op | `rtl/vfu/vfu_internal_op_defs.vh` | `IMPLEMENTED` | 4-bit `VFU_OP_*`와 clamp mode 정의. 여러 leaf가 아직 삭제된 옛 이름 `vfu_defs.vh`를 include함 |
| 16-segment selector | `rtl/vfu/gaugepack_vfu_segment_gen.v` | `IMPLEMENTED` | standalone compile PASS. 조합형 range/segment selector source 존재; 독립 TB는 없음 |
| PRE S0 | `rtl/vfu/vfu_pre_alu16_s0.v` | `IMPLEMENTED`, `BUILD_BLOCKED` | `vfu_pre_alu_lane`과 registered `vfu_pre_alu16` 존재. missing include 때문에 현재 TB compile 실패 |
| DSP S1/S2 lane | `rtl/vfu/vfu_dsp_lane.v` | `IMPLEMENTED`, `BUILD_BLOCKED` | `DSP48E2` lane source와 TB 존재. missing include에서 먼저 실패하며 Vivado primitive simulation/synthesis는 이번 감사에서 실행하지 않음 |
| POST helpers | `rtl/vfu/post_alu/vfu_rne_shift48.v`, `vfu_clamp_wrapper.v`, `vfu_residual_add.v` | `IMPLEMENTED`, `BUILD_BLOCKED` | RNE/residual standalone compile PASS. Clamp는 old header include 때문에 blocked. 통합 POST TB도 실행 불가 |
| POST lane | `rtl/vfu/post_alu/vfu_post_alu_lane.v` | `IMPLEMENTED`, `BUILD_BLOCKED`, `SPEC_MISMATCH` | 단일 lane source 존재. `force_zero_i` 기반 LN `D=0→rho=0` override가 Function v13과 충돌 |
| Softmax rowmax | `rtl/vfu/vfu_rowmax16.v` | `IMPLEMENTED`, `RTL_SIM_PASS` | 16×S24 rowmax와 scalar `key_valid_i`. 현재 directed row-unit TB PASS; 최종 독립 acceptance는 아직 OPEN |
| QEXP commit + rowsum | `rtl/vfu/vfu_exp_commit16.v` | `IMPLEMENTED`, `RTL_SIM_PASS` | invalid key의 E=0과 16×U13 rowsum. 현재 directed row-unit TB PASS; `e_o`의 “E scratch” 주석은 stale |
| V corner-turn | `rtl/vfu/vfu_vcorner_ff.sv` | `IMPLEMENTED`, `RTL_SIM_PASS`, `MASTER_ACCEPTED` | 2×16×16×S8 FF ping-pong, local column→row transpose, opaque 5-bit tag. Standalone simulation과 independent 128-seed stress 수락 |
| Intermediate S-pad leaf | 예정 `rtl/vfu/vfu_spad_mem16.v` | `NOT_STARTED` | `16×512×32`, common-address 1R1W, `RD_LAT=1/2` 계약과 coder task만 READY. 현재 HEAD에는 source/TB 없음 |
| CORE16/16-lane active wrapper | 예정 | `NOT_STARTED` | `vfu_core16/vfu_core_lane`, `vfu_coeff`, `vfu_input_unit`, active 16-lane POST wrapper와 coefficient/sideband alignment가 없음 |
| 함수별 controller | 예정 | `NOT_STARTED` | RQ/GELU/Softmax/LayerNorm FSM과 얇은 dispatcher가 없음 |
| Scratch typed integration | 예정 `vfu_scratch` | `NOT_STARTED` | Score/LN typed adapter, address overlay, side state, z→T lifecycle가 아직 OPEN |
| Shared commit | 예정 `vfu_commit_unit` | `NOT_STARTED`, `DEFERRED` | ACT16/PV_WGT16 address, acceptance/visibility, completion realization이 아직 동결되지 않음 |
| VFU top/PMPU integration | 예정 | `NOT_STARTED` | command latch, Logical-DRAM client, PMPU selector/latency/fifo schedule 연결이 없음 |

## 3. Active source 구조

```text
TOP-visible op header
  vfu_external_ops.vh

private CORE leaf sources
  vfu_internal_op_defs.vh
  gaugepack_vfu_segment_gen.v
  vfu_pre_alu16_s0.v
  vfu_dsp_lane.v
  post_alu/
    vfu_rne_shift48.v
    vfu_clamp_wrapper.v
    vfu_residual_add.v
    vfu_post_alu_lane.v

Softmax side units
  vfu_rowmax16.v
  vfu_exp_commit16.v

V transpose leaf
  vfu_vcorner_ff.sv
```

이 구조는 아직 하나의 active `vfu_core16` 또는 `vfu_top`으로 연결되어 있지 않습니다.
특히 coefficient select, PRE/DSP/POST sideband alignment, memory read latency와 function FSM은
별도 구현 대상입니다.

## 4. Draft 격리

다음 파일은 production 구현 증거로 사용하지 않습니다.

| 파일 | 상태/이유 |
|---|---|
| `rtl/vfu/vfu_draft/vfu_post_alu16_draft.v` | `DRAFT`; active POST lane과 중복 module 이름을 포함하며 old `force_zero` 계약 사용 |
| `rtl/vfu/vfu_draft/vfu_softmax_draft.v` | Draft TB는 PASS하지만 `DRAFT/SUPERSEDED`; current stored-score/commit/controller의 production 증거가 아님 |
| `tb/vfu/vfu_draft/tb_vfu_softmax_draft.sv` | `DRAFT`; production acceptance evidence가 아님 |

현재 repository에는 active source만 고르는 manifest/build script, Makefile, synthesis Tcl,
CI workflow가 없습니다. Wildcard로 `rtl/vfu/**/*.v`를 compile하면 draft의 중복 module 또는
stale interface가 섞일 수 있으므로, 통합 전 explicit source manifest가 필요합니다.

## 5. 2026-08-27 재현 결과

### 5.1 RTL

| 대상 | 결과 | 범위 |
|---|---|---|
| `tb_vfu_row_units` | `PASS` | reverse key order 15→0, `seq_len=5`, invalid-key E mask, rowmax/rowsum directed check |
| `tb_vfu_vcorner_ff` | `PASS` | 기본 seed 10 groups/160 beats, permutation, tag, CE/output stall, simultaneous fill/drain |
| V-corner independent stress | `PASS` | 앞선 Master acceptance의 추가 128-seed standalone simulation |
| `tb_vfu_pre_alu16` | `BUILD_BLOCKED` | `vfu_defs.vh` not found |
| `tb_vfu_post_alu_lane` | `BUILD_BLOCKED` | `vfu_defs.vh` not found |
| `tb_vfu_dsp_lane` | `BUILD_BLOCKED` | `vfu_defs.vh` not found; 이후 primitive elaboration은 평가하지 않음 |

Row-unit TB의 현재 PASS는 기본 동작 증거입니다. 다음 항목은 아직 충분히 검사하지 않습니다.

- `clear_i` 재시작과 CE stall의 조합
- padded final beat와 `last_i/result_valid_o` 정렬 변형
- U13 최대 `64×127=8128` 경계
- back-to-back row와 randomized scoreboard
- 최신 Function/Architecture 기준의 독립 acceptance

또한 이 TB의 failure branch는 `$fatal` 대신 `FAIL`을 출력하고 `$finish`하므로, 자동화할
때는 출력 문자열을 검사하거나 TB를 exit-failing 형태로 강화해야 합니다.

### 5.2 Model / NN-LUT tools

| 파일/flow | 현재 결과 | 한계 |
|---|---|---|
| `gaugepack_qexp_nnlut_train.py --self-test` | `PASS` | E7 reference artifact flow이며 production M18/C48 bundle 자체는 아님 |
| `gaugepack_softmax_recip_nnlut_train.py --self-test` | `PASS` | legal `L=127..8128` exhaustive certificate와 frozen snapshot match |
| GELU trainer self-test | `NOT_RUN` | 현재 실행 환경에 PyTorch가 없어 import에서 중단 |
| RSQRT trainer self-test | `NOT_RUN`, `SPEC_MISMATCH` | PyTorch가 없고 exported `D0_policy`가 old zero-bypass 계약 |
| unified NN-LUT self-test | `NOT_RUN` | GELU import 단계에서 PyTorch 부재로 중단 |
| `gaugepack_frozen_e2e_inference.py --self-test` | `NOT_RUN` | PyTorch 부재. Source는 key-only mask를 사용하지만 explicit `seq_len_padded` scheduled E2E는 없음 |

GELU/QEXP/RSQRT의 최종 production coefficient bundle/manifest와 RTL coefficient bank는
현재 repository에 없습니다. Reciprocal은 frozen M/C snapshot과 software certificate를
가지지만 전체 VFU integration 증거는 아닙니다.

## 6. 현재 알려진 code/spec mismatch

1. **삭제된 header 이름**
   `vfu_internal_op_defs.vh`로 rename했지만 PRE/DSP/POST RTL과 세 TB가 여전히
   `vfu_defs.vh`를 include합니다. 현재 clean leaf build를 막습니다.

2. **LayerNorm `D=0` override**
   Function v13은 별도 `D==0` detector/zero override를 금지합니다. 그러나
   `vfu_post_alu_lane.v`와 그 TB에는 `force_zero_i`가 남아 있습니다.

3. **RSQRT compiler metadata**
   `gaugepack_rsqrt_nnlut_train.py`는 `D0_policy: bypass page and emit rho=0`을 export해
   Function v13의 constant-row 자연 경로와 충돌합니다.

4. **E destination 주석**
   `vfu_exp_commit16.v`의 `e_o`는 아직 “E scratch write data”로 설명됩니다. Current
   memory hierarchy에서 E7는 PMPU-visible Logical-DRAM commit stream입니다. Datapath의
   key mask 동작은 유지할 수 있지만 destination ownership 주석/연결은 갱신해야 합니다.

5. **`lane_valid[15:0]` 의미**
   PRE의 lane-valid pass-through가 generic occupancy인지 stale query/pair mask인지 아직
   동결되지 않았습니다. Softmax query masking 용도로 사용하는 것은 금지됩니다.

6. **Build/source manifest 부재**
   active source와 `vfu_draft`를 분리하는 재현 가능한 compile manifest가 없습니다.

## 7. 합성·통합·hardware 상태

| 항목 | 상태 |
|---|---|
| V-corner Vivado OOC resource/timing | `NOT_RUN` |
| PRE/DSP/POST active chain synthesis | `NOT_RUN` |
| S-pad inference/resource/timing | `NOT_STARTED` |
| Full VFU synthesis / 200 MHz closure | `NOT_STARTED` |
| PMPU + Logical-DRAM integration simulation | `NOT_STARTED` |
| padded BERT-Tiny E2E RTL/model equivalence | `NOT_STARTED` |
| KV260 hardware execution | `NOT_STARTED` |

이번 환경의 PATH에는 Vivado/XSim, Verilator, Yosys가 없었고 repository에도 synthesis
script/report가 없었습니다. 따라서 source attribute나 `DSP48E2` instantiation만 보고
resource mapping 또는 timing을 추정하지 않습니다.

따라서 현재 “VFU가 구현 완료됐다”라고 표현하면 안 됩니다. 정확한 표현은
**일부 arithmetic/row-state/V-corner leaf가 구현되었고, V-corner standalone RTL만
Master acceptance를 받았으며 controller/memory/commit/top integration은 미구현**입니다.

## 8. 다음 구현 경계

현재 발행된 다음 task는 schedule-independent S-pad memory leaf입니다.

```text
rtl/vfu/vfu_spad_mem16.v
tb/vfu/tb_vfu_spad_mem16.sv

16 banks × 512 depth × 32-bit
independent rd_addr / wr_addr
each address broadcast to all 16 banks
full 512-bit vector
synchronous 1R1W
RD_LAT = 1 or 2, production default 2
```

이 leaf에는 Score/LN address generator, side state, V-corner, PMPU, ready/error protocol,
Logical-DRAM commit을 넣지 않습니다. Typed `vfu_scratch`와 shared commit은 각각의 OPEN
결정을 닫은 뒤 별도 task로 구현합니다.

## 9. 재현 명령

```bash
# Row units
iverilog -g2012 -Wall -s tb_vfu_row_units \
  -o /tmp/tb_vfu_row_units.vvp \
  rtl/vfu/vfu_rowmax16.v \
  rtl/vfu/vfu_exp_commit16.v \
  tb/vfu/tb_vfu_row_units.sv
vvp /tmp/tb_vfu_row_units.vvp

# V-corner
iverilog -g2012 -Wall -s tb_vfu_vcorner_ff \
  -o /tmp/tb_vfu_vcorner_ff.vvp \
  rtl/vfu/vfu_vcorner_ff.sv \
  tb/vfu/tb_vfu_vcorner_ff.sv
vvp /tmp/tb_vfu_vcorner_ff.vvp

# NumPy-only NN-LUT checks
python3 model/nnlut_training/gaugepack_qexp_nnlut_train.py --self-test
python3 model/nnlut_training/gaugepack_softmax_recip_nnlut_train.py --self-test
```

## 10. 갱신 규칙

- Source 추가만으로 `VERIFIED`를 쓰지 않습니다.
- TB PASS에는 simulator, source list, test 범위와 제외 범위를 같이 기록합니다.
- Vivado synthesis를 실행하지 않았으면 resource/timing을 `NOT_RUN`으로 둡니다.
- Draft source는 active manifest와 구현 표에 넣지 않습니다.
- Frozen Function/Architecture가 바뀌면 이 문서는 구현 영향만 갱신하고 계약을 재정의하지
  않습니다.
- Git HEAD가 바뀔 때 감사 기준 commit과 상태 표를 함께 갱신합니다.
