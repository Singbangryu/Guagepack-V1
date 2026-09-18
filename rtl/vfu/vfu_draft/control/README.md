# VFU 공통 / Softmax 컨트롤 초안

공통 명령 제어와 함수별 실행 제어를 연결해 보는 **draft RTL**이다. 실제 `VFU_TOP`, CORE 연산,
메모리 또는 commit 경로를 완성한 코드는 아니다. 기존 production manifest에는 넣지 않는다.
`tb_vfu_control_hierarchy_draft.sv`에 common + 기존 RQ/GELU + Softmax의 named-port 연결 예시가 있다.
LN은 완료 입력만 연결할 자리를 두었으며, LN 컨트롤러 구현은 포함하지 않는다.

## 읽는 순서와 역할

| 파일 | 역할 |
|---|---|
| `vfu_common_control_draft.v` | 명령 설정 저장, `COMMON_IDLE → SETUP → CORE_RUN → COMMON_IDLE`, 하위 선택 |
| 기존 `rtl/vfu/vfu_rq_gelu_control.v` | Owner의 기존 `STREAM → DRAIN`; 수정 없이 사용 |
| `vfu_softmax_control_draft.v` | SM의 score/MAX → QEXP/E/L → reciprocal/R, 별도 SM_NL의 context 처리 순서 |
| `tb/vfu/vfu_draft/control/tb_vfu_control_hierarchy_draft.sv` | 세 컨트롤러 연결, 선택된 opcode/완료 전달, R 수명 확인 |

공통 제어는 하위 phase를 다시 추적하지 않는다. RQ/GELU는 저장된 외부 op를 내부 RQ/GELU op로
정적으로 바꾸고, Softmax는 자신의 phase에 따라 QEXP/RECIP/CONTEXT op를 낸다. 공유 wrapper가
선택된 하위의 launch/op/제어를 CORE와 저장 경로로 전달한다. coefficient/page 선택도 wrapper의
남은 배선이다. 저장된 layer/site(`num_ln`)와 선택된 phase의 CORE op를 사용하며, 이 초안에서
새 page encoding을 정의하지 않는다.

## 공통 제어

| 상태 | 동작 | 다음 경계 |
|---|---|---|
| COMMON_IDLE | 합법적인 `start_i`에서 설정 저장, `cmd_accept_o` 발생 | SETUP |
| SETUP | 선택된 하위 `init=1`, 모든 `active=0` | 초안에서는 정확히 1 cycle 후 CORE_RUN |
| CORE_RUN | 선택된 하위 `active=1`; 실행·대기·drain 모두 포함 | 선택된 `done` 사건에서 직접 IDLE |

`busy_o`는 SETUP/CORE_RUN에서 높다. `done_o`는 CORE_RUN에서 선택된 하위의 완료를 조합으로
전달한다. 완료가 높은 cycle에도 busy, active와 descriptor는 유지되고, 다음 edge에 IDLE로 간다.
추가 DONE 상태/레지스터는 없다. 새 명령은 그 뒤 idle cycle에 받는다. reset은 `rst_ni` 동기 active-low이며
idle reset을 전제로 한다. 기존 RQ/GELU의 `rstn_i`에는 이름을 맞춰 연결한다.

| Port/group | I/O | Width | Description |
|---|---|---:|---|
| `clk_i`, `rst_ni`, `start_i` | I | 1 each | Clock, synchronous active-low reset, one-cycle idle command request. |
| `op_i` / `op_o` | I / O | 3 | External command; held output uses RQ=0, GELU=1, SM=2, SM_NL=3, LN=4. |
| `m_i`, `n_i` / corresponding `_o` | I / O | 10 each | Padded tokens M=16/32/48/64 and command feature count N=1..512. |
| `seq_len_i` / `seq_len_o` | I / O | 7 | Original sequence length 2..64. |
| `layer_i`, `num_ln_i` / corresponding `_o` | I / O | 1 each | Held layer and LN-site configuration. |
| `req_mult_i` / `req_mult_o` | I / O | signed 18 | Held RQ multiplier. |
| `req_shamt_i` / `req_shamt_o` | I / O | 6 | Held RQ shift. |
| `transpose_i`, `write_sel_i` / corresponding `_o` | I / O | 1, 3 | Held route flag and opaque memory destination. |
| `write_base_i`, `res_base_i`, `ln_base_i` / corresponding `_o` | I / O | 16 each | Held logical-memory bases in existing word units. |
| `rq_gelu_done_i`, `softmax_done_i`, `ln_done_i` | I | 1 each | Child completion events; only selected child can finish the command. |
| `cmd_accept_o`, `setup_o`, `busy_o`, `done_o` | O | 1 each | Acceptance, setup window, command lifetime and selected completion. |
| `rq_gelu_init_o`, `softmax_init_o`, `ln_init_o` | O | 1 each | Selected child's initialization, even while inactive. |
| `rq_gelu_active_o`, `softmax_active_o`, `ln_active_o` | O | 1 each | Selected child's execution permission through drain. |
| `mt_o`, `total_beats_o` | O | 3, 12 | M/16 and (M/16)*N, including count 2048. |
| `rq_gelu_core_op_o` | O | 4 | Static internal RQ/GELU mapping, meaningful for that selected group only. |

외부 설정은 수락 edge에 저장하고 완료까지 유지한다. IDLE 복귀로도 저장값이나 Softmax R을 clear하지 않는다.
reserved op와 busy 중 start는 simulation에서 오류로 검사하며 런타임 복구 프로토콜은 제공하지 않는다.
1-cycle SETUP은 아직 준비 완료 handshake가 없는 초안 선택이다. 입력은 실제 active까지 FIFO/adapter가
보존해야 한다. 이 상태를 추가했다고 PMPU 첫 결과 또는 final outflow 안전성이 해결되는 것은 아니다.

## Softmax 순서

SM은 head 전체의 score를 먼저 저장한 뒤 query tile별 QEXP와 reciprocal을 진행한다.
진행 중 PMPU를 멈추는 스케줄 대신 full-head score 보관을 전제로 한 초안이다. Score 저장은 실제 accepted
write edge와 rowmax 입력이 같은 `score_fire`를 사용한다. 지연된 write queue는 이 인터페이스에 없다.

| Phase | 하는 일 | 나가는 조건 |
|---|---|---|
| SM_SCORE | MT×S score vector 저장과 MAX 누산 | 마지막 실제 score acceptance |
| SM_MAX_DRAIN | 앞에서 도착한 이벤트도 포함해 필요한 MAX write를 모음 | 모든 tile MAX 저장 완료 |
| SM_EXP | tile replay 시작 pulse, 반환된 score를 QEXP에 투입 | 해당 tile의 S번째 launch |
| SM_EXP_DRAIN | QEXP op 유지, 최종 L과 해당 tile 마지막 E write를 기다림 | 두 사건 모두 관측 |
| SM_RECIP | 현재 L의 reciprocal 요청 | `recip_valid && recip_ready` |
| SM_RECIP_DRAIN | reciprocal op 유지, MAX slot을 R로 바꾼 완료를 기다림 | 다음 tile로 이동 또는 마지막 tile이면 SM done |
| NL_STREAM | 각 PV numerator의 실제 tile에 대응하는 R로 SM_CONTEXT 실행 | MT×64번째 실제 launch |
| NL_DRAIN | 새 입력 차단, context 최종 write 대기 | `commit_done_i`에서 SM_NL done |

`S=16*MT=ceil16(seq_len)`이고 MT=1..4다. SM에는 N=S, 별도 SM_NL에는 N=64를 사용한다.
공통 제어의 M을 `keys_i`에 연결한다. `keys_i`는 64도 표현해야 하므로 7-bit이고, key index는 6-bit다.

| Port/group | I/O | Width | Description |
|---|---|---:|---|
| `clk_i`, `rst_ni`, `init_i`, `active_i` | I | 1 each | Clock, reset, setup initialization and execution permission. |
| `op_i`, `mt_i`, `seq_len_i`, `keys_i` | I | 3, 3, 7, 7 | Held SM/SM_NL mode and full-head geometry. |
| `score_valid_i`, `score_tile_i`, `score_key_i` | I | 1, 2, 6 | Available tagged score; global query tile and actual key index. |
| `max_store_done_i`, `max_store_tile_i` | I | 1, 2 | Unique post-write MAX event and its delayed tile tag. |
| `replay_valid_i` | I | 1 | Current tile score return, accepted when replay is enabled. |
| `rowsum_valid_i`, `e_tile_done_i` | I | 1 each | Current tile final L availability and last E write completion. |
| `recip_ready_i`, `recip_store_done_i` | I | 1 each | Reciprocal launch acceptance and later R tile write completion. |
| `nl_valid_i`, `nl_tile_i`, `nl_r_valid_i` | I | 1, 2, 1 | Numerator availability, actual global tile, and matching valid typed R. |
| `commit_done_i` | I | 1 | Registered final context write completion, for SM_NL only. |
| `score_en_o`, `score_fire_o` | O | 1 each | Score permission and shared FIFO/store/rowmax acceptance event. |
| `rowmax_clear_o`, `rowmax_last_o`, `score_key_valid_o` | O | 1 each | First/final accepted score in tile and actual-key padding predicate. |
| `replay_start_o`, `replay_en_o`, `replay_tile_o` | O | 1, 1, 2 | Accepted whole-tile replay request, return permission and tile. |
| `replay_key_o`, `replay_key_valid_o` | O | 6, 1 | Ascending replay key and padding predicate. |
| `replay_first_o`, `replay_tile_last_o`, `replay_command_last_o` | O | 1 each | Qualified QEXP metadata for L and E-write boundaries. |
| `recip_valid_o` | O | 1 | Reciprocal launch request held until accepted. |
| `nl_en_o`, `nl_fire_o`, `nl_last_o` | O | 1 each | Numerator permission, accepted launch and final launch. |
| `core_op_o`, `state_clear_o`, `done_o` | O | 4, 1, 1 | Phase-selected CORE opcode, new-SM state invalidation and completion. |

실제 연결 시 지켜야 할 경계는 다음과 같다.

- score는 query tile 0,1,… 순서로 연속해서 온다고 가정한다. tile 내 key 순서는 바뀔 수 있고,
  16개 묶음의 15→0 순서도 허용한다. 각 tile은 S개 key를 정확히 한 번씩 가져야 한다.
  rowmax clear/last는 key 숫자가 아니라 실제 accepted count로 결정한다.
- `max_store_tile_i`는 rowmax 결과에 정렬된 지연 tag다. 새 tile이 들어오는 순간의 입력 tag를
  이전 MAX write에 쓰면 안 된다. 넓은 upstream 좌표는 범위를 확인한 뒤 2/6-bit로 줄인다.
- `replay_start_o`는 항상 수락되는 whole-tile 요청이고 별도 request-ready가 없다. 요청과 같은
  cycle의 반환도 가능하지만 요청 전 반환은 금지한다. adapter는 현재 tile의 score를 key 오름차순으로
  S개 반환하고, `replay_en && replay_valid`에서 data/metadata가 함께 CORE에 들어간다.
- first/tile-last/key-valid를 CORE 결과까지 정렬해 exp helper에 전달한다. 예를 들어 MT=2이면
  tile0 마지막 QEXP는 `tile_last=1, command_last=0`, tile1 마지막은 둘 다 1이다.
  tile-last는 rowsum 완료에, command-last는 head 전체 E ACT16 마지막 write에 사용한다.
  CORE의 한 last bit를 두 의미에 무조건 공용으로 쓰지 말고 별도 sideband를 정렬한다.
- `e_tile_done_i`는 adapter가 만드는 tile 마지막 write의 등록된 사건이다. ACT16의 command
  `commit_done`가 매 tile마다 나온다고 가정하지 않는다. L/E 사건은 서로 다른 cycle이어도 된다.
  L은 reciprocal이 소비할 때까지 보존한다. padding key도 QEXP/commit 횟수에는 포함하며 E는 zero-mask한다.
- `recip_ready_i`는 wrapper가 준비된 L을 CORE에 투입할 수 있다는 의미다. CORE 자체에 ready/CE를
  추가하지 않는다. R write 완료는 reciprocal launch보다 최소 1 cycle 뒤여야 하고, SM_NL context
  완료도 마지막 numerator launch 이후의 등록된 사건이어야 한다. zero-latency 완료 모델은 지원하지 않는다.
- `state_clear_o`는 새 SM init에서만 old MAX/R metadata를 무효화한다. 각 tile의 마지막 QEXP와 E/L
  drain 이후 MAX를 R로 교체한다. SM done 후에도 R은 IDLE, 다른 명령 및 SM_NL init을 통과해 보존된다.
  caller는 동일 head/geometry의 SM_NL까지 R 소유권을 유지해야 한다. 이 초안에는 head ID/tag 검증이 없다.
- SM_NL의 `nl_r_valid_i`는 **현재 numerator tile에 대해 valid이고 R 타입**이라는 뜻이다.
  state FF의 조합 read를 가정하며, 실제 read latency가 다르면 adapter가 operand를 정렬해야 한다.
  feature 좌표/중복 여부, lane mask, memory address와 최종 tensor coverage는 wrapper/TB 책임이다.

## 실행과 검증 범위

Icarus Verilog 12의 RTL simulation으로 실행한다. repository root에서:

```sh
bash tb/vfu/vfu_draft/control/run_control_draft.sh
```

별도 설치 경로는 환경변수로 지정할 수 있다.

```sh
IVERILOG=/path/to/iverilog VVP=/path/to/vvp IVL_DIR=/path/to/ivl \
  bash tb/vfu/vfu_draft/control/run_control_draft.sh
```

runner는 common, Softmax, hierarchy TB를 실행하고 common의 reserved-op/busy-start 오류 검사도 확인한다.
기존 RQ/GELU 파일에 timescale이 없어서 Icarus가 inherited-timescale warning을 낼 수 있으며 파일은 그대로 둔다.
common TB는 9개 명령과 실제 RQ/GELU 4546-beat 입력, 2048-beat count, 설정 유지와 지연 commit을 검사한다.
Softmax TB는 S=16/32/48/64와 padding 경계, 입력 bubble, MAX/E/L/R 완료 지연, 별도 SM_NL과 R 수명을 검사한다.
hierarchy TB는 `SM → RQ → SM_NL → LN(stub) → GELU`의 실제 제어 연결을 검사한다.

테스트의 scratch write/replay, MAX/R slot, L/E/commit 완료는 **boundary model**이다. MAX/R marker는
산술 계산값이 아니다. 아직 실제 CORE/NN-LUT 수치 결과, coefficient page, S-pad/state64/ACT16 통합,
주소 배치, Value transpose commit, PMPU/FIFO 무손실, LN, 합성/200 MHz/보드 동작을 검증하지 않았다.
