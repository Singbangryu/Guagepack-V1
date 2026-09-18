# VFU 공통 / Softmax / LayerNorm 컨트롤 초안

공통 명령 제어와 함수별 실행 제어를 연결해 보는 **draft RTL**이다. 실제 `VFU_TOP`, CORE 연산,
메모리 또는 commit 경로를 완성한 코드는 아니다. 기존 production manifest에는 넣지 않는다.
`tb_vfu_control_hierarchy_draft.sv`에 common + 기존 RQ/GELU + Softmax의 named-port 연결 예시가 있다.
`tb_vfu_ln_hierarchy_draft.sv`는 여기에 실제 LN 제어와 rho 저장 helper를 연결한다.
두 예제 모두 데이터 연산과 메모리 완료는 명시적인 boundary model이다.

## 읽는 순서와 역할

| 파일 | 역할 |
|---|---|
| `vfu_common_control_draft.v` | 명령 설정 저장, `COMMON_IDLE → SETUP → CORE_RUN → COMMON_IDLE`, 하위 선택 |
| 기존 `rtl/vfu/vfu_rq_gelu_control.v` | Owner의 기존 `STREAM → DRAIN`; 수정 없이 사용 |
| `vfu_softmax_control_draft.v` | SM의 score/MAX → QEXP/E/L → reciprocal/R, 별도 SM_NL의 context 처리 순서 |
| `tb/vfu/vfu_draft/control/tb_vfu_control_hierarchy_draft.sv` | 세 컨트롤러 연결, 선택된 opcode/완료 전달, R 수명 확인 |
| `vfu_ln_control_draft.v` | LN 입력 저장 후 tile별 Moment → D → RSQRT → NORM → AFFINE 제어 |
| `vfu_ln_rho16_draft.v` | 현재 LN tile의 16×U8 rho, tile tag, valid 저장 |
| `tb/vfu/vfu_draft/control/tb_vfu_ln_hierarchy_draft.sv` | common/RQ/SM/LN/rho 연결, `SM → LN → SM_NL`, 별도 R과 rho 수명 확인 |

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

## LayerNorm 순서

LN은 N=128, MT=1..4를 전제로 한다. 먼저 명령 전체의 `MT*128` RQ_RES 결과 z를
scratch에 저장한 뒤, tile 0부터 차례로 아래 연산을 한다. 최대 512 vector word를 사용하고,
NORM에서 다 읽은 z 위치를 T로 덮어쓰는 **초안 schedule**이다. 별도 full-command T bank를
추가하지 않는다. 실제 read/write adapter 및 배치는 아직 구현하지 않았다.

| 제어 구간 | CORE op | launch 수 | 다음 구간으로 가는 사건 |
|---|---|---:|---|
| RQ_STREAM → RQ_DRAIN | RQ_RES | 명령 전체 MT×128 | 마지막 z write와 이전 write가 모두 완료 |
| MOMENT_INIT → INIT_DRAIN | LN_MOMENT_INIT | tile당 1 | INIT 결과가 실제 S3에서 retire |
| MOMENT_ACC → MOMENT_DRAIN | LN_MOMENT_ACC | tile당 127 | 마지막 Moment의 S/Q capture |
| D_REQUEST → D_DRAIN | LN_D | tile당 1 | 해당 tile의 D 결과 도착 |
| RSQRT_REQUEST → RSQRT_DRAIN | LN_RSQRT | tile당 1 | rho helper가 해당 tile 결과를 저장 |
| NORM_STREAM → NORM_DRAIN | LN_NORM | tile당 128 | 해당 tile의 마지막 T write 완료 |
| AFFINE_STREAM → AFFINE_DRAIN | LN_AFFINE | tile당 128 | tile 최종 write 완료; 마지막 tile은 command commit_done도 필요 |

INIT는 feature 0 한 개, ACC는 feature 1..127을 처리한다. INIT와 ACC 사이에 실제 drain을
두어 op/page를 유지한다. INIT의 last는 0이고 S/Q capture를 발생시키지 않는다. ACC의 마지막
feature에서만 Moment last를 보낸다. 마지막 tile AFFINE의 tile-done과 command commit-done은
서로 다른 cycle에 올 수 있으며 둘 다 확인해야 LN done이다. 공통 controller는 그동안 LN active,
설정과 busy를 유지하고 하위 phase를 직접 세지 않는다.

Replay는 tile당 네 burst로 분리한다. `replay_start`는 항상 수락되는 요청이며 별도 ready가 없다.
반환은 요청 뒤 정확히 아래 수만큼 오름차순으로 온다. 중간 bubble은 가능하지만 INIT drain을
넘어 미리 읽어 온 ACC 데이터를 보관하는 숨은 buffer를 전제하지 않는다.

| Replay pass | start feature | 요청 beats |
|---|---:|---:|
| Moment INIT | 0 | 1 |
| Moment ACC | 1 | 127 |
| NORM | 0 | 128 |
| AFFINE | 0 | 128 |

### LN control과 rho helper 포트

아래 포트는 외부 system TOP 핀이 아니라 VFU 내부 draft 연결이다.

| LN control port/group | I/O | Width | Description |
|---|---|---:|---|
| `clk_i`, `rst_ni`, `init_i`, `active_i` | I | 1 each | Clock, synchronous active-low reset, inactive setup initialization, selected execution. |
| `mt_i` | I | 3 | Held tile count 1..4; features are fixed to 128. |
| `rq_valid_i`, `z_store_done_i` | I | 1 each | Paired main/skip is ready; all command z writes have completed. |
| `replay_valid_i` | I | 1 | Ordered current-burst vector and its required operands are ready. |
| `init_retire_i`, `moment_capture_i` | I | 1 each | Tagged INIT S3 retirement; final ACC captured persistent S/Q. |
| `scalar_ready_i`, `d_result_valid_i` | I | 1 each | Singleton operand-adapter readiness; current-tile D result is held. |
| `rho_store_done_i`, `rho_match_valid_i` | I | 1 each | Registered rho capture event; stored rho matches current tile. |
| `t_store_done_i`, `affine_tile_done_i`, `commit_done_i` | I | 1 each | Final T store, tile final architectural write, registered command final write. |
| `rq_en_o`, `rq_fire_o`, `rq_last_o` | O | 1 each | Ingress permission, actual launch, command-final ingress marker. |
| `replay_start_o`, `replay_en_o`, `replay_fire_o` | O | 1 each | Always-accepted burst request, return permission, actual replay launch. |
| `replay_tile_o` | O | 2 | Current global token-tile index. |
| `replay_start_feature_o`, `replay_feature_o`, `replay_beats_o` | O | 7, 7, 8 | Requested first feature, current return feature, requested count. |
| `replay_first_o`, `replay_tile_last_o` | O | 1 each | Qualified first burst return and feature127 boundary. |
| `scalar_valid_o`, `scalar_fire_o` | O | 1 each | D/RSQRT request and accepted singleton launch. |
| `core_op_o`, `core_last_o`, `done_o` | O | 4, 1, 1 | Phase-selected op, qualified operation-specific last, final LN completion. |

| Rho helper port/group | I/O | Width | Description |
|---|---|---:|---|
| `clk_i`, `rst_ni`, `clear_i` | I | 1 each | Clock, reset and LN-only initialization. |
| `capture_i`, `capture_tile_i`, `capture_data_i` | I | 1, 2, 512 | Qualified LN_RSQRT retirement with aligned tile and sixteen zero-extended U8 containers. |
| `rd_tile_i` | I | 2 | Requested current tile. |
| `rho_o`, `match_valid_o`, `stored_o` | O | 128, 1, 1 | Packed rho; matching validity; registered one-cycle capture receipt. |

LN replay는 request cycle 다음 STREAM부터 반환을 받는다. 완료 입력은 실제 positive-latency
result/write 뒤 해당 drain에 전달한다. AFFINE의 tile/command 완료만 마지막 launch와 같은 cycle에
관측된 경우도 모아 두지만, 일반 CORE/memory 연결에서 latency가 0이라는 뜻은 아니다.
helper는 invalid/tag mismatch일 때 rho=0을 보이고, clear와 capture가 동시에 오면 simulation 오류다.

### LN 데이터와 저장 수명

| 연산 | CORE operand 연결 | 결과/수명 |
|---|---|---|
| RQ_RES | main S32, 별도로 정렬한 native-scale S8 skip, 공통의 held M/F와 C=0 | z:S9를 S32 container로 확장해 tagged scratch write |
| Moment INIT/ACC | z; 기존 G23 MomentPack datapath | 마지막 ACC에서 기존 CORE의 S:S16/Q:U23 출력 capture |
| D | lane별 S를 sign-extend해 src0, Q를 zero-extend해 src1 | CORE 결과 D27을 다음 RSQRT가 사용할 때까지 보존 |
| RSQRT | D27과 해당 layer/site의 coefficient page | 실제 RSQRT 결과의 lane별 low byte를 rho helper에 저장 |
| NORM | z→src0, 보존된 S→src1, rho→rsqrted | raw T:S25를 이미 읽은 z 위치에 저장 |
| AFFINE | T와 feature별 M_gamma/C_beta/F | NARROW_S8 결과만 architectural commit 경로로 전달 |

S/Q는 기존 CORE가 마지막 Moment 이후 유지하므로 이 초안은 별도 S/Q bank를 만들지 않는다.
D도 CORE output이 다음 valid 결과까지 유지하는 것을 이용한다. 그러나 rho는 NORM 결과가
CORE output을 바꾸어도 필요하므로 별도 128-bit register helper가 필요하다. rho는 한 tile만
저장한다. 다음 tile의 rho로 교체하기 전 이전 tile의 NORM 사용이 모두 끝난다.

`rho_o[lane*8 +: 8] = capture_data_i[lane*32 +: 8]`이다. 512-bit 결과의 low 128-bit를
그대로 연결하면 네 lane의 container만 선택하므로 잘못된 연결이다. helper에는 실제 LN RSQRT
S3 결과일 때만 capture를 넣고, 결과와 함께 지연된 tile tag를 넣는다. LN init/reset은 rho valid만
무효화한다. **Softmax R은 다른 저장소**이며 LN init, LN 결과, rho capture가 이를 지우지 않는다.

### LN adapter가 연결해야 하는 경계

- 입력 `rq_valid`는 main과 skip이 실제 소비 경계에 정렬되어 준비됐다는 뜻이다. FIFO의
  nonempty만 연결해서는 residual read latency가 해결되지 않는다. 실제 accepted ingress의
  tile/feature tag를 z write까지 전달한다. 입력 feature 순서는 16개 단위 15→0이어도 된다.
- replay 반환의 valid는 data, operand와 필요한 coefficient가 함께 준비됐다는 뜻이다.
  scalar ready는 D/RSQRT singleton을 넣을 operand adapter의 준비를 뜻하며 CORE에 새 ready/CE를
  추가하지 않는다. 모든 요청과 결과의 op/tile tag는 실제 파이프라인을 따라 정렬해야 한다.
- INIT retire, final Moment capture, D valid, rho stored, T stored는 서로 다른 완료 사건이다.
  NORM은 저장된 rho의 valid와 현재 tile 일치를 요구한다. unrelated CORE 결과로 rho를 덮어쓰면 안 된다.
- `core_last`의 중간 phase 의미를 ACT16 command-last와 구분한다. AFFINE feature127은 모든
  tile에서 tile-last지만, 마지막 tile에서만 command-last다. 중간 z/D/rho/T 결과는 final commit 대상이 아니다.
- NORM read 후 T overwrite를 하며 같은 주소의 read/write가 같은 cycle에 발생하면 안 된다.
  반환 데이터에 원래 주소를 붙여 유지하고, overwrite한 z를 다시 읽지 않는다. AFFINE은 모든 T write
  완료 뒤 시작한다. TB의 `tile*128+feature`는 설명용 주소이며 production scratch 배치를 확정하지 않는다.
- common은 N=128을 LN 전용으로 검사하지 않으므로 실제 wrapper가 LN geometry를 확인해야 한다.
  tile index 2-bit와 MT count 3-bit, feature index 7-bit와 burst count 8-bit를 구별한다.
  공통 logical base는 계속 16-bit다. 물리 메모리 용량을 늘린다는 뜻은 아니다.

coefficient page/72-bit parameter ABI, residual read 배선, S3 sideband 정렬, scratch/ACT16 주소와
실제 numerical CORE 연결은 다음 integration 작업이다. 이 파일들은 production VFU_TOP이 아니다.

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

runner는 common, Softmax, 기존 hierarchy, LN unit 및 LN hierarchy TB를 실행하고
common의 reserved-op/busy-start 및 LN의 잘못된 MT, clear/capture 충돌, U8 container,
잘못된 완료/반환 사건 오류 검사도 확인한다.
기존 RQ/GELU 파일에 timescale이 없어서 Icarus가 inherited-timescale warning을 낼 수 있으며 파일은 그대로 둔다.
common TB는 9개 명령과 실제 RQ/GELU 4546-beat 입력, 2048-beat count, 설정 유지와 지연 commit을 검사한다.
Softmax TB는 S=16/32/48/64와 padding 경계, 입력 bubble, MAX/E/L/R 완료 지연, 별도 SM_NL과 R 수명을 검사한다.
기존 hierarchy TB는 `SM → RQ → SM_NL → LN(stub) → GELU`를 그대로 검사한다.
LN unit TB는 MT=1/4/2/3/4의 5개 명령과 bubble, 각 drain, 최종 완료 순서를 검사한다.
LN hierarchy TB는 MT=2의 `SM → LN → SM_NL → RQ → GELU`에서 실제 LN controller와 rho helper를
사용한다. scratch z/T marker, S/Q/D 생존 구간과 rho low-byte packing은 boundary model로 확인한다.

테스트의 scratch write/replay, MAX/R slot, L/E/commit 완료는 **boundary model**이다. MAX/R marker는
산술 계산값이 아니다. 아직 실제 CORE/NN-LUT 수치 결과, coefficient page, S-pad/state64/ACT16 통합,
주소 배치, Value transpose commit, PMPU/FIFO 무손실, LN 수치 정확도, 합성/200 MHz/보드 동작을 검증하지 않았다.
