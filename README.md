# Custom GPU Compute Kernels for LLM Inference

Fused SiLU and LayerNorm kernels in Triton, benchmarked against eager PyTorch and `torch.compile` on an NVIDIA T4 GPU.

## 1. Key Results (LayerNorm)

| Hidden Dim (N) | Eager (GB/s) | torch.compile (GB/s) | Triton (GB/s) | Triton vs Eager | Triton vs compile |
|----------------|--------------|----------------------|---------------|-----------------|-------------------|
| 768            | 206.6        | 216.0                | 224.0         | 1.08×           | 1.04×             |
| 1024           | 220.9        | 220.0                | 224.0         | 1.01×           | 1.02×             |
| 2048           | 226.6        | 234.6                | 228.6         | 1.01×           | 0.97×             |
| 3072           | 194.3        | 240.1                | 229.5         | 1.18×           | 0.96×             |
| 4096           | 175.5        | 241.7                | 229.6         | 1.31×           | 0.95×             |
| 8192           | 163.1        | 239.1                | 231.8         | 1.42×           | 0.97×             |

![LayerNorm Benchmark](layernorm_benchmark.png)

**Conclusion:** For $N \le 2048$ the Triton kernel is on par with eager `F.layer_norm`. For $N \ge 3072$ eager PyTorch degrades (down to 163 GB/s at $N=8192$) while the Triton kernel stays flat at ~224-232 GB/s, i.e. 1.2-1.4× higher effective bandwidth. `torch.compile` (TorchInductor) is the strongest baseline: for $N \ge 2048$ it is 3-5% faster than the hand-written kernel (≈240 GB/s).

## 2. SiLU Performance
A Triton kernel for the SiLU (Swish) activation was also implemented to establish memory bandwidth limits. The custom kernel achieved strict parity with PyTorch's native ATen dispatch across the entire benchmark range ($2^{12} - 2^{23}$ elements). Both implementations flattened out at ~228 GB/s (representing ~71% of the nominal 320 GB/s NVIDIA T4 memory bandwidth limit), with smaller sizes being purely constrained by kernel launch overhead.

## 3. Method
* **Effective Bandwidth:** Calculated using the theoretical minimum data transfer required to solve the operation: `(2 * M * N + 2 * N [for weight/bias]) * bytes_per_element / execution_time`. This measures algorithmic effective bandwidth, not the actual hardware DRAM traffic.
* **Benchmarking:** Utilized `triton.testing.do_bench` for accurate timing (which handles warmup and `torch.cuda.synchronize()`). 
* **Parameters:** Benchmarks were run using FP16 precision with a fixed sequence/batch size of $M = 4096$. 
* `torch.compile` was initialized outside the benchmarking loop with `dynamic=False` to prevent recompilation overhead across different $N$ shapes.

## 4. Profiler Finding & Hypothesis Verification
The initial hypothesis assumed that eager PyTorch executes multiple kernels during the LayerNorm forward pass, which would explain the bandwidth drop. 

To verify this, the PyTorch CUDA profiler was used. **The profiler disproved this hypothesis**, showing that eager PyTorch actually dispatches a single fused kernel (`vectorized_layer_norm_kernel`) for the forward pass. However, execution times varied significantly: 377-379 µs for PyTorch versus 283 µs for Triton (at $N=M=4096$).

## 5. Correctness Testing
Before benchmarking, the kernels were tested using `torch.testing.assert_close`:
* **Shapes:** Tested on both power-of-two and irregular shapes (e.g., $N=1000, 3072$) to ensure the masking logic handles padding correctly without out-of-bounds reads or incorrect results.
* **Precision:** Tested in FP32 (`atol=1e-4`, `rtol=1e-4`) and FP16 (`atol=1e-2`, `rtol=1e-2`).
* **Inputs:** Weight ($\gamma$) and Bias ($\beta$) initialized as random tensors.

## 6. Limitations
* **Hardware:** Benchmarked on a single NVIDIA T4. Memory wall gaps may behave differently on Ampere/Hopper architectures.
* **Scope:** Only the forward pass (inference) was implemented.
* **Metrics:** Values reflect a single benchmark run (no variance analysis). "Effective bandwidth" does not reflect actual measured DRAM traffic.
* **Root Cause Unresolved:** While the profiler proved PyTorch uses a single kernel, the exact cause of the eager implementation's degradation at large $N$ remains unverified.

## 7. Next Steps
* Inspect `torch.compile` (TorchInductor) generated code to compare its memory access patterns against the custom Triton kernel and identify the source of the 3-5% performance advantage.
* Implement `@triton.autotune` to dynamically select optimal `num_warps` and `BLOCK_SIZE`.
* Extend the kernel to compute the backward pass.
* Implement **RMSNorm** (Root Mean Square Normalization), standard in LLaMA architecture.

## 8. Environment & Reproduction
* **PyTorch Version:** 2.11.0+cu130
* **Triton Version:** 3.6.0
* **Hardware:** NVIDIA T4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milana-sadokha/llm-triton-kernels/blob/main/triton_llm_kernels.ipynb)
