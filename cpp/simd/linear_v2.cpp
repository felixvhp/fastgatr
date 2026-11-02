#include <torch/extension.h>
#include <map>
#include <tuple>
#include <torch/script.h>
#include <cstring>
#include <vector>
#include <arm_neon.h>
#include "_cycle_counter.hpp"

// ---------------------------------------------------------------------------
// equi_linear v2
//
// Why this file exists:
//   The v1 SIMD equi_linear (linear.cpp) vectorizes across the 16-blade
//   dimension. That axis does not map cleanly to NEON lanes: only three blade
//   groups (1-4, 5-8, 11-14) are contiguous, so the hot loop ends up doing
//   3 vector FMAs plus ~9 scalar FMAs, then 12 vgetq_lane_f32 extracts per
//   output channel. Roughly half the math stays scalar and the lane extracts
//   move data SIMD->GPR on the critical path. On top of that the weight walk
//   is strided: the inner loop runs over channels_in, which is the *outer*
//   stride of packed_weight[i][o][9], so consecutive iterations jump
//   channels_out*9 floats through memory and thrash the cache as channels_out
//   grows.
//
//   v2 changes the vectorized axis: lanes = 4 output channels. For a fixed
//   input channel, every output channel applies the same 9 packed weights to
//   the same 16 input blades. So we broadcast each input blade (a scalar) and
//   FMA it into 4 output channels at once. Every blade -- including the "off"
//   contributions that scatter in v1 -- becomes a clean vector FMA. No lane
//   extracts, no scalar tail.
//
//   To make the 4-output-channel loads contiguous we repack the weights to
//   [channels_in][9][channels_out] (pack_equi_linear_weight_v2). Then for a
//   fixed (i, weight_index) the channels_out values are adjacent in memory and
//   load with a single vld1q_f32.
// ---------------------------------------------------------------------------

// Input:
//   weight: [channels_out, channels_in, 9]
// Output:
//   packed: [channels_in, 9, channels_out]   (note the transposed last two axes)
// Why:
//   Lays the weights out so the v2 kernel can load 4 output channels with one
//   contiguous vld1q_f32. The fixed scale factors are folded in here once, so
//   the forward pass does no per-element scaling.
torch::Tensor pack_equi_linear_weight_v2(torch::Tensor weight) {
    TORCH_CHECK(weight.dim() == 3 && weight.size(2) == 9,
                "pack_equi_linear_weight_v2: weight must have shape [out, in, 9]");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32,
                "pack_equi_linear_weight_v2: float32 only");

    weight = weight.contiguous();
    const int channels_out = (int)weight.size(0);
    const int channels_in = (int)weight.size(1);

    // [channels_in, 9, channels_out]
    auto packed = torch::empty({channels_in, 9, channels_out}, weight.options());
    const float* __restrict weight_ptr = weight.data_ptr<float>();
    float* __restrict packed_ptr = packed.data_ptr<float>();

    const float scale[9] = {
        1.0f,                    // 0
        0.5f,                    // 1
        0.40824829046386301f,    // 2  (1/sqrt(6))
        0.5f,                    // 3
        1.0f,                    // 4
        1.0f,                    // 5
        0.57735026918962576f,    // 6  (1/sqrt(3))
        0.57735026918962576f,    // 7
        1.0f,                    // 8
    };

    for (int o = 0; o < channels_out; ++o) {
        for (int i = 0; i < channels_in; ++i) {
            const float* __restrict weight_oi = weight_ptr + (o * channels_in + i) * 9;
            for (int k = 0; k < 9; ++k) {
                // packed[i][k][o]
                packed_ptr[(i * 9 + k) * channels_out + o] = weight_oi[k] * scale[k];
            }
        }
    }
    return packed;
}

// Input:
//   x:             [B, T, channels_in, 16]
//   packed_weight: [channels_in, 9, channels_out]  (from pack_equi_linear_weight_v2)
//   bias:          [channels_out] or nullptr
// Output:
//   out:           [B, T, channels_out, 16]
// Why:
//   Vectorizes across 4 output channels. For each (b,t) and each group of 4
//   output channels we keep 16 vector accumulators (one per blade, lanes = the
//   4 output channels), stream over channels_in, and FMA the broadcast input
//   blades into them. The leftover output channels (channels_out % 4) are
//   handled by a scalar tail.
void equi_linear_v2_kernel(float* __restrict out, const float* __restrict x,
    const float* __restrict packed_weight, const float* __restrict bias,
    int B, int T, int channels_in, int channels_out) {

    // packed_weight stride helpers: packed[i][k][o] = packed_weight[(i*9 + k)*channels_out + o]
    const int CO = channels_out;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const int batch_token = b * T + t;
            const float* __restrict x_row = x + batch_token * channels_in * 16;
            float* __restrict out_row = out + batch_token * channels_out * 16;

            // --- Vectorized body: 4 output channels at a time. ---
            int o = 0;
            for (; o + 4 <= channels_out; o += 4) {
                // One accumulator per blade; each holds 4 output channels.
                float32x4_t acc0  = vdupq_n_f32(0.0f);
                float32x4_t acc1  = vdupq_n_f32(0.0f);
                float32x4_t acc2  = vdupq_n_f32(0.0f);
                float32x4_t acc3  = vdupq_n_f32(0.0f);
                float32x4_t acc4  = vdupq_n_f32(0.0f);
                float32x4_t acc5  = vdupq_n_f32(0.0f);
                float32x4_t acc6  = vdupq_n_f32(0.0f);
                float32x4_t acc7  = vdupq_n_f32(0.0f);
                float32x4_t acc8  = vdupq_n_f32(0.0f);
                float32x4_t acc9  = vdupq_n_f32(0.0f);
                float32x4_t acc10 = vdupq_n_f32(0.0f);
                float32x4_t acc11 = vdupq_n_f32(0.0f);
                float32x4_t acc12 = vdupq_n_f32(0.0f);
                float32x4_t acc13 = vdupq_n_f32(0.0f);
                float32x4_t acc14 = vdupq_n_f32(0.0f);
                float32x4_t acc15 = vdupq_n_f32(0.0f);

                for (int i = 0; i < channels_in; i++) {
                    const float* __restrict x_channel = x_row + i * 16;
                    // Base of this input channel's weights; the 9 weight slices
                    // for blade-group k are at offset k*CO, each 4 output
                    // channels loaded contiguously at +o.
                    const float* __restrict pbase = packed_weight + (i * 9) * CO + o;

                    const float32x4_t w0 = vld1q_f32(pbase + 0 * CO);
                    const float32x4_t w1 = vld1q_f32(pbase + 1 * CO);
                    const float32x4_t w2 = vld1q_f32(pbase + 2 * CO);
                    const float32x4_t w3 = vld1q_f32(pbase + 3 * CO);
                    const float32x4_t w4 = vld1q_f32(pbase + 4 * CO);
                    const float32x4_t w5 = vld1q_f32(pbase + 5 * CO);
                    const float32x4_t w6 = vld1q_f32(pbase + 6 * CO);
                    const float32x4_t w7 = vld1q_f32(pbase + 7 * CO);
                    const float32x4_t w8 = vld1q_f32(pbase + 8 * CO);

                    // Broadcast input blades and FMA into the 4 output channels.
                    // Mapping mirrors the v1 math exactly (see linear.cpp).
                    //   weight slice -> contributions
                    //   w0: blade 0
                    //   w1: blades 1,2,3,4   (diagonal)
                    //   w2: blades 5,6,7,8,9,10
                    //   w3: blades 11,12,13,14
                    //   w4: blade 15 (diagonal)
                    //   w5: blade 1   (off, reads x[0])
                    //   w6: blades 5,6,7   (off, reads x[2],x[3],x[4])
                    //   w7: blades 11,12,13 (off, reads x[8],x[9],x[10])
                    //   w8: blade 15  (off, reads x[14])
                    acc0  = vfmaq_n_f32(acc0,  w0, x_channel[0]);

                    acc1  = vfmaq_n_f32(acc1,  w1, x_channel[1]);
                    acc2  = vfmaq_n_f32(acc2,  w1, x_channel[2]);
                    acc3  = vfmaq_n_f32(acc3,  w1, x_channel[3]);
                    acc4  = vfmaq_n_f32(acc4,  w1, x_channel[4]);

                    acc5  = vfmaq_n_f32(acc5,  w2, x_channel[5]);
                    acc6  = vfmaq_n_f32(acc6,  w2, x_channel[6]);
                    acc7  = vfmaq_n_f32(acc7,  w2, x_channel[7]);
                    acc8  = vfmaq_n_f32(acc8,  w2, x_channel[8]);
                    acc9  = vfmaq_n_f32(acc9,  w2, x_channel[9]);
                    acc10 = vfmaq_n_f32(acc10, w2, x_channel[10]);

                    acc11 = vfmaq_n_f32(acc11, w3, x_channel[11]);
                    acc12 = vfmaq_n_f32(acc12, w3, x_channel[12]);
                    acc13 = vfmaq_n_f32(acc13, w3, x_channel[13]);
                    acc14 = vfmaq_n_f32(acc14, w3, x_channel[14]);

                    acc15 = vfmaq_n_f32(acc15, w4, x_channel[15]);

                    // Off-diagonal contributions.
                    acc1  = vfmaq_n_f32(acc1,  w5, x_channel[0]);
                    acc5  = vfmaq_n_f32(acc5,  w6, x_channel[2]);
                    acc6  = vfmaq_n_f32(acc6,  w6, x_channel[3]);
                    acc7  = vfmaq_n_f32(acc7,  w6, x_channel[4]);
                    acc11 = vfmaq_n_f32(acc11, w7, x_channel[8]);
                    acc12 = vfmaq_n_f32(acc12, w7, x_channel[9]);
                    acc13 = vfmaq_n_f32(acc13, w7, x_channel[10]);
                    acc15 = vfmaq_n_f32(acc15, w8, x_channel[14]);
                }

                if (bias != nullptr) {
                    acc0 = vaddq_f32(acc0, vld1q_f32(bias + o));
                }

                // Scatter: out[o+lane][blade]. The 4 lanes are 4 output
                // channels, each 16 floats apart in out_row.
                float blade[16][4];
                vst1q_f32(blade[0],  acc0);
                vst1q_f32(blade[1],  acc1);
                vst1q_f32(blade[2],  acc2);
                vst1q_f32(blade[3],  acc3);
                vst1q_f32(blade[4],  acc4);
                vst1q_f32(blade[5],  acc5);
                vst1q_f32(blade[6],  acc6);
                vst1q_f32(blade[7],  acc7);
                vst1q_f32(blade[8],  acc8);
                vst1q_f32(blade[9],  acc9);
                vst1q_f32(blade[10], acc10);
                vst1q_f32(blade[11], acc11);
                vst1q_f32(blade[12], acc12);
                vst1q_f32(blade[13], acc13);
                vst1q_f32(blade[14], acc14);
                vst1q_f32(blade[15], acc15);

                for (int lane = 0; lane < 4; ++lane) {
                    float* __restrict out_o = out_row + (o + lane) * 16;
                    for (int k = 0; k < 16; ++k) {
                        out_o[k] = blade[k][lane];
                    }
                }
            }

            // --- Scalar tail: remaining (channels_out % 4) output channels. ---
            for (; o < channels_out; ++o) {
                float acc[16];
                for (int k = 0; k < 16; ++k) acc[k] = 0.0f;

                for (int i = 0; i < channels_in; i++) {
                    const float* __restrict x_channel = x_row + i * 16;
                    const float* __restrict pw = packed_weight + (i * 9) * CO + o;
                    const float w0 = pw[0 * CO];
                    const float w1 = pw[1 * CO];
                    const float w2 = pw[2 * CO];
                    const float w3 = pw[3 * CO];
                    const float w4 = pw[4 * CO];
                    const float w5 = pw[5 * CO];
                    const float w6 = pw[6 * CO];
                    const float w7 = pw[7 * CO];
                    const float w8 = pw[8 * CO];

                    acc[0]  += w0 * x_channel[0];
                    acc[1]  += w1 * x_channel[1];
                    acc[2]  += w1 * x_channel[2];
                    acc[3]  += w1 * x_channel[3];
                    acc[4]  += w1 * x_channel[4];
                    acc[5]  += w2 * x_channel[5];
                    acc[6]  += w2 * x_channel[6];
                    acc[7]  += w2 * x_channel[7];
                    acc[8]  += w2 * x_channel[8];
                    acc[9]  += w2 * x_channel[9];
                    acc[10] += w2 * x_channel[10];
                    acc[11] += w3 * x_channel[11];
                    acc[12] += w3 * x_channel[12];
                    acc[13] += w3 * x_channel[13];
                    acc[14] += w3 * x_channel[14];
                    acc[15] += w4 * x_channel[15];

                    acc[1]  += w5 * x_channel[0];
                    acc[5]  += w6 * x_channel[2];
                    acc[6]  += w6 * x_channel[3];
                    acc[7]  += w6 * x_channel[4];
                    acc[11] += w7 * x_channel[8];
                    acc[12] += w7 * x_channel[9];
                    acc[13] += w7 * x_channel[10];
                    acc[15] += w8 * x_channel[14];
                }

                float* __restrict out_o = out_row + o * 16;
                if (bias != nullptr) acc[0] += bias[o];
                for (int k = 0; k < 16; ++k) out_o[k] = acc[k];
            }
        }
    }
}

// Input:
//   x:             [B, T, channels_in, 16]
//   packed_weight: [channels_in, 9, channels_out]  (from pack_equi_linear_weight_v2)
//   basis:         passed in to match the v1 API, not read here.
//   bias:          optional [channels_out]
// Output:
//   [B, T, channels_out, 16]
// Why:
//   Python-callable wrapper for the v2 packed path. Pack weights once with
//   pack_equi_linear_weight_v2, cache, then call this many times.
torch::Tensor asl_equi_linear_v2(torch::Tensor x, torch::Tensor packed_weight, torch::Tensor basis,
    std::optional<torch::Tensor> bias = std::nullopt) {
    CYCLE_SCOPE("equi_linear");
    x = x.contiguous();
    packed_weight = packed_weight.contiguous();
    basis = basis.contiguous();
    TORCH_CHECK(packed_weight.dim() == 3 && packed_weight.size(1) == 9,
                "equi_linear_v2: packed weight must have shape [in, 9, out]");

    int B = x.size(0);
    int T = x.size(1);
    int channels_in = x.size(2);
    int channels_out = (int)packed_weight.size(2);
    TORCH_CHECK(packed_weight.size(0) == channels_in,
                "equi_linear_v2: packed weight input channels must match x");
    auto out = torch::empty({B, T, channels_out, 16}, x.options());

    const float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
    if (bias.has_value()) {
        bias_contig = bias.value().contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    equi_linear_v2_kernel(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        packed_weight.data_ptr<float>(),
        bias_ptr,
        B, T, channels_in, channels_out
    );
    return out;
}

#include <pybind11/pybind11.h>

PYBIND11_MODULE(linear_v2, m) {
    m.def("pack_equi_linear_weight", &pack_equi_linear_weight_v2,
          "Pack and pre-scale equi_linear weights into [in, 9, out] layout (v2)");
    m.def("equi_linear_packed", &asl_equi_linear_v2,
          "pin-equivariant linear map, vectorized across output channels (v2)",
          pybind11::arg("x"), pybind11::arg("packed_weight"), pybind11::arg("basis"),
          pybind11::arg("bias") = std::nullopt);
    BIND_CYCLE_COUNTER(m);
}
