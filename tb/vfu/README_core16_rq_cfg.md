# CORE16 RQ configuration test

`tb_vfu_core16_rq_cfg.sv` runs the real CORE, PRE, DSP16 and POST pipeline with
the original vendor DSP48E2 model. Its second module is a **test-only** synthetic
coefficient page; exclude the empty production `vfu_coeff_page16.v` when compiling
this test. No vendor sources or generated simulation files belong in the repository.

The caller supplies signed18 `req_mult_i` and unsigned6 `req_shamt_i` before the
first valid RQ/RQ_RES input and holds both through drain. The test changes them only
between drained commands. Input edge n produces S0 after n and S3 after n+3;
that input's `skip_s1_i` is supplied before edge n+2.

## Run

Requirements: Bash, Icarus Verilog 12.0, `timeout`, and original `DSP48E2.v` and
`glbl.v` from Xilinx/XilinxUnisimLibrary commit
`1c8e05fd1e9a79ceb8b996a0996674122eed086f`. Put those two files in `UNISIM_DIR`.
The TB waits for natural `glbl.GSR` release before completing local reset.

Run from the repository root. Normal installations need only `UNISIM_DIR`.
For a relocated Icarus installation, also set `IVERILOG`, `VVP`, and `IVL_DIR`
to its compiler, runtime, and directory containing the ivl/VPI modules.

```bash
set -euo pipefail
: "${UNISIM_DIR:?Set UNISIM_DIR to the directory containing DSP48E2.v and glbl.v}"
IVERILOG="${IVERILOG:-iverilog}"
VVP="${VVP:-vvp}"
compile_options=()
runtime_options=()
if [ -n "${IVL_DIR:-}" ]; then
    compile_options=(-B "$IVL_DIR")
    runtime_options=(-M "$IVL_DIR")
fi
build_dir="$(mktemp -d)"

"$IVERILOG" "${compile_options[@]}" -g2012 -Wall -I rtl/vfu \
    -s tb_vfu_core16_rq_cfg -s glbl -o "$build_dir/sim.vvp" \
    rtl/vfu/gaugepack_vfu_segment_gen.v \
    rtl/vfu/vfu_pre_alu16_s0.v \
    rtl/vfu/vfu_dsp16_s1s2.v \
    rtl/vfu/post_alu/vfu_rne_shift48.v \
    rtl/vfu/post_alu/vfu_clamp_wrapper.v \
    rtl/vfu/post_alu/vfu_residual_add.v \
    rtl/vfu/post_alu/vfu_post_alu16_s3.v \
    rtl/vfu/vfu_core16.v \
    tb/vfu/tb_vfu_core16_rq_cfg.sv \
    "$UNISIM_DIR/DSP48E2.v" "$UNISIM_DIR/glbl.v"
timeout 60s "$VVP" "${runtime_options[@]}" "$build_dir/sim.vvp"
```

Success prints `[PASS] CORE16 RQ config` with 21 commands, 84 beats and 1344
lane comparisons. A mismatch or timeout exits nonzero.

## Coverage and limits

- RQ/RQ_RES page M/C/F are all X, so known outputs require external M/F and C=0.
- Zero, positive and negative multipliers, signed18 endpoints, shifts 0/1/2/17/31/47/48/63,
  signed RNE ties, symmetric S8 clamps, and unsaturated S9 residuals.
- All 16 lanes, consecutive beats, bubbles, changing skips, complete drain, and
  S0/S3 valid, operation, feature/tile, lane/key validity and last alignment.
- LN_AFFINE with identical inputs and different or unknown RQ configuration,
  GELU with nonzero synthetic page C, and SM_CONTEXT's PRE multiplier/fixed shift.
- Reference rounding uses unsigned magnitude arithmetic; it does not instantiate
  production RNE/clamp helpers or read expected values from DUT internals.

This is bounded functional simulation with synthetic coefficients. It does not
validate real coefficient pages, full S32 accumulator input range (the unchanged
PRE operand is S27), all micro-ops, PMPU/Top/FSM/commit integration, XSim startup,
synthesis, timing, or hardware execution.
