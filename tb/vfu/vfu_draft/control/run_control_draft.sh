#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Override executable paths for a local tool install;
# IVL_DIR is optional and only needed by relocated Icarus distributions.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"
iverilog_bin="${IVERILOG:-iverilog}"
vvp_bin="${VVP:-vvp}"
ivl_args=()
if [[ -n "${IVL_DIR:-}" ]]; then
    ivl_args=(-B "$IVL_DIR")
fi
sim_dir="$(mktemp -d)"
trap 'rm -rf "$sim_dir"' EXIT

for tb_top in tb_vfu_common_control_draft tb_vfu_softmax_control_draft tb_vfu_control_hierarchy_draft tb_vfu_ln_control_draft tb_vfu_ln_hierarchy_draft; do
    "$iverilog_bin" "${ivl_args[@]}" -g2012 -Wall -I rtl/vfu \
        -s "$tb_top" -o "$sim_dir/$tb_top.vvp" \
        rtl/vfu/vfu_draft/control/vfu_common_control_draft.v \
        rtl/vfu/vfu_rq_gelu_control.v \
        rtl/vfu/vfu_draft/control/vfu_softmax_control_draft.v \
        rtl/vfu/vfu_draft/control/vfu_ln_control_draft.v \
        rtl/vfu/vfu_draft/control/vfu_ln_rho16_draft.v \
        "tb/vfu/vfu_draft/control/$tb_top.sv"
    "$vvp_bin" "$sim_dir/$tb_top.vvp"
done

# These runs must fail in the intended simulation-only contract check.
for negative_case in 1 2; do
    if "$vvp_bin" "$sim_dir/tb_vfu_common_control_draft.vvp" \
        "+NEGATIVE=$negative_case" > "$sim_dir/negative.log" 2>&1; then
        cat "$sim_dir/negative.log"
        printf '%s\n' 'ERROR: illegal command was not rejected.' >&2
        exit 1
    fi
    negative_log="$(cat "$sim_dir/negative.log")"
    if [[ "$negative_case" == 1 && "$negative_log" != *'common draft: reserved external opcode'* ]] ||
       [[ "$negative_case" == 2 && "$negative_log" != *'common draft: start while busy'* ]]; then
        cat "$sim_dir/negative.log"
        printf '%s\n' 'ERROR: negative test failed for an unexpected reason.' >&2
        exit 1
    fi
done
printf '%s\n' '[PASS] expected common-controller illegal-command checks'

# LN negatives check adapter contracts, not a runtime recovery mechanism.
ln_cases=(bad_mt bad_clear bad_u8 bad_event bad_replay)
ln_messages=(
    'LN MT must be 1..4'
    'LN rho clear and capture conflict'
    'LN rho capture requires zero-extended U8 containers'
    'LN final S/Q capture outside Moment drain'
    'LN replay returned outside its prepared active burst'
)
for case_index in "${!ln_cases[@]}"; do
    if "$vvp_bin" "$sim_dir/tb_vfu_ln_control_draft.vvp" \
        "+NEG=${ln_cases[$case_index]}" > "$sim_dir/ln_negative.log" 2>&1; then
        cat "$sim_dir/ln_negative.log"
        printf '%s\n' 'ERROR: illegal LN adapter event was not rejected.' >&2
        exit 1
    fi
    negative_log="$(cat "$sim_dir/ln_negative.log")"
    if [[ "$negative_log" != *"${ln_messages[$case_index]}"* ]]; then
        cat "$sim_dir/ln_negative.log"
        printf '%s\n' 'ERROR: LN negative test failed for an unexpected reason.' >&2
        exit 1
    fi
done
printf '%s\n' '[PASS] expected LN geometry/event/rho contract checks'
