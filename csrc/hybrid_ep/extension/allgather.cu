// SPDX-License-Identifier: MIT 
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include "allgather.cuh"
#include <cuda/ptx>

#define MAX_BLOCKS 256
#define TIMEOUT 20000000000ull

// ============================================================================
// Original kernel: uint4 store-based allgather (kept as fallback)
// ============================================================================
template<int SHARED_SIZE = 1024>
__global__ void ag_nvl_kernel(
    void** dst_buffers_all_ranks, 
    void* src, 
    int bytes_per_rank, 
    int64_t *iter_id_ptr, // Normal GPU memory
    unsigned long long *flag_nvl_ptr, // Register memory on rank 0 
    unsigned long long *flag_sm_ptr, // Normal GPU memory
    int rank_idx, 
    int rank_num
) {
    int is_last_SM = 0;
    uint4** dst_list_ptr = reinterpret_cast<uint4**>(dst_buffers_all_ranks);
    uint4* src_ptr = reinterpret_cast<uint4*>(src);
    auto iter_id = *iter_id_ptr;
    iter_id ++ ; // increment iter_id

    __shared__ uint4 shared_data[SHARED_SIZE];
    // Compute the data size assigned to each SM
    int uint4_per_rank = bytes_per_rank / sizeof(uint4);  // uint4 = 16 bytes
    int chunk_size = (uint4_per_rank + gridDim.x - 1) / gridDim.x;
    int chunk_start = blockIdx.x * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, uint4_per_rank);

    int loop_time = (chunk_end - chunk_start + SHARED_SIZE - 1) / SHARED_SIZE;
    for(int i = 0; i < loop_time; i++) {
        int start_idx = chunk_start + i * SHARED_SIZE;
        int end_idx = min(start_idx + SHARED_SIZE, chunk_end);

        // Load the data from src to shared_data
        for(int j = threadIdx.x; j < end_idx - start_idx; j += blockDim.x) {
            shared_data[j] = src_ptr[start_idx + j];
        }
        __syncthreads();

        // Copy the data from src to dst
        for(int j = 0; j < rank_num; j++) {
            auto dst_rank = (rank_idx + j) % rank_num;
            auto dst_ptr = dst_list_ptr[dst_rank];
            for(int k = threadIdx.x; k < end_idx - start_idx; k += blockDim.x) {
                auto local_offset = start_idx + k;
                auto dst_offset = local_offset + rank_idx * uint4_per_rank;
                dst_ptr[dst_offset] = shared_data[k];
            }
        }
    }

    __syncthreads();
    __threadfence();

    if(threadIdx.x == 0) {
        unsigned long long value_to_add = blockIdx.x == 0 ? MAX_BLOCKS - gridDim.x + 1 : 1;
        auto old_val_sm_sync = atomicAdd(flag_sm_ptr, value_to_add);  
        is_last_SM = (gridDim.x == 1 || old_val_sm_sync + value_to_add == iter_id * MAX_BLOCKS);
    }

    __threadfence_system();
    if(is_last_SM) {
      // Update the flag_nvl_ptr
      asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;"
                    :
                    : "l"(__cvta_generic_to_global(flag_nvl_ptr)), "n"(1)
                    : "memory");
      *iter_id_ptr = iter_id;
      auto expected = iter_id * rank_num;
      clock_t s = clock64();
      unsigned long long flag_data = 0;

      // Wait for the flag_nvl_ptr to be updated from all ranks in nvl domain
      do{
        asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                      : "=l"(flag_data)
                      : "l"(__cvta_generic_to_global(flag_nvl_ptr))
                      : "memory");
        if (clock64() - s > 2ull * TIMEOUT) {
          printf("HYBRID-EP ALLGATHER TIMEOUT:SM %d [%d]:expecting %llu got %llu\n", blockIdx.x,
                  threadIdx.x, (unsigned long long)expected, flag_data);
          break;
        }
      }while(flag_data < expected);
    }
}

// ============================================================================
// TMA-based allgather kernel: all-to-all with cp.async.bulk
//
// Design:
//   - Each SM owns a contiguous byte-range of the local src data.
//   - The data is loaded from GMEM into SMEM in tiles, then each tile is
//     TMA-bulk-copied to all R destination buffers concurrently.
//   - TMA S2G (shared-to-global) writes go over NVLink to IPC-mapped remote
//     buffers. Blackwell TMA supports ~128 outstanding ops per SM, so all
//     R writes per tile are fully concurrent.
//   - Completion: all SMs converge via local atomicAdd, then the last SM
//     does fence.release.sys + remote atomic flag to signal all peers.
//
// SMEM layout (multi-warp version):
//   - NUM_PIPELINES independent pipelines, each with double-buffered tiles.
//   - Each pipeline is owned by one warp.
//   - pipeline[p].tile[0..1]: double-buffered tile data
//   - pipeline[p].mbar[0..1]: mbarriers for G2S completion tracking
//   - Total SMEM: NUM_PIPELINES * (2 * TILE_BYTES + 16) bytes
//
// Thread structure:
//   - NUM_PIPELINES warps (e.g., 4 warps = 128 threads). Each warp's elected
//     thread issues TMA commands for its pipeline. Warps work on independent
//     sub-chunks of the SM's assigned data range.
// ============================================================================

// TILE_BYTES per pipeline: 16 KB per tile, double-buffered = 32 KB per pipeline.
// With 4 pipelines: 128 KB SMEM total + mbarriers. Well within 228 KB.
static constexpr int AG_TMA_TILE_BYTES = 16384;
static constexpr int AG_TMA_NUM_PIPELINES = 4;

__global__ void __launch_bounds__(AG_TMA_NUM_PIPELINES * 32, 1)
ag_nvl_tma_kernel(
    void** dst_buffers_all_ranks,
    void* src,
    int bytes_per_rank,
    int64_t* iter_id_ptr,
    unsigned long long* flag_nvl_ptr,  // NVLink-accessible flag on rank 0
    unsigned long long* flag_sm_ptr,   // Local SM convergence flag
    int rank_idx,
    int rank_num
) {
    constexpr int TILE_BYTES = AG_TMA_TILE_BYTES;
    constexpr int NUM_PIPELINES = AG_TMA_NUM_PIPELINES;
    // Per-pipeline SMEM: 2 tiles + 2 mbarriers (8B each, aligned)
    constexpr int PIPELINE_SMEM = 2 * TILE_BYTES + 16;

    extern __shared__ char smem_raw[];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp owns a pipeline with its own tile buffers and mbarriers.
    char* my_smem = smem_raw + warp_id * PIPELINE_SMEM;
    char* tiles[2] = {my_smem, my_smem + TILE_BYTES};
    uint64_t* mbars[2] = {
        reinterpret_cast<uint64_t*>(my_smem + 2 * TILE_BYTES),
        reinterpret_cast<uint64_t*>(my_smem + 2 * TILE_BYTES + 8)
    };

    auto iter_id = *iter_id_ptr;
    iter_id++;

    // Each SM handles a contiguous byte-range of src, split among pipelines.
    int sm_chunk_bytes = (bytes_per_rank + gridDim.x - 1) / gridDim.x;
    sm_chunk_bytes = (sm_chunk_bytes + 15) & ~15;
    int sm_chunk_start = blockIdx.x * sm_chunk_bytes;
    int sm_chunk_end = min(sm_chunk_start + sm_chunk_bytes, bytes_per_rank);
    if (sm_chunk_start >= bytes_per_rank) {
        sm_chunk_start = sm_chunk_end = 0;
    }
    int sm_remaining = sm_chunk_end - sm_chunk_start;

    // Split SM's chunk among warps (pipelines).
    int warp_chunk_bytes = (sm_remaining + NUM_PIPELINES - 1) / NUM_PIPELINES;
    warp_chunk_bytes = (warp_chunk_bytes + 15) & ~15;
    int warp_chunk_start = sm_chunk_start + warp_id * warp_chunk_bytes;
    int warp_chunk_end = min(warp_chunk_start + warp_chunk_bytes, sm_chunk_end);
    if (warp_chunk_start >= sm_chunk_end) {
        warp_chunk_start = warp_chunk_end = 0;
    }
    int warp_remaining = warp_chunk_end - warp_chunk_start;
    int num_tiles = (warp_remaining + TILE_BYTES - 1) / TILE_BYTES;

    char* src_bytes = reinterpret_cast<char*>(src);

    bool is_leader = elect_sync(~0u);

    if (is_leader) {
        cuda::ptx::mbarrier_init(mbars[0], 1u);
        cuda::ptx::mbarrier_init(mbars[1], 1u);
    }
    __syncwarp();

    int parity[2] = {0, 0};

    for (int t = 0; t < num_tiles; t++) {
        int slot = t & 1;
        int tile_offset = warp_chunk_start + t * TILE_BYTES;
        int tile_size = min(TILE_BYTES, warp_chunk_end - tile_offset);
        int tma_size = (tile_size + 15) & ~15;

        // Issue TMA G2S
        if (is_leader) {
            cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release,
                cuda::ptx::scope_cta,
                cuda::ptx::space_shared,
                mbars[slot],
                static_cast<uint32_t>(tma_size));

            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_shared,
                cuda::ptx::space_global,
                reinterpret_cast<void*>(tiles[slot]),
                reinterpret_cast<const void*>(src_bytes + tile_offset),
                static_cast<uint32_t>(tma_size),
                mbars[slot]);
        }

        // Wait for previous S2G to finish reading SMEM
        if (t > 0 && is_leader) {
            cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
        }

        // Wait for G2S completion
        if (is_leader) {
            while (!cuda::ptx::mbarrier_try_wait_parity(mbars[slot], parity[slot])) {}
            parity[slot] ^= 1;
        }
        __syncwarp();

        // Issue S2G to all destination ranks
        if (is_leader) {
            int dst_offset_in_rank = rank_idx * bytes_per_rank + tile_offset;

            for (int r = 0; r < rank_num; r++) {
                char* dst_ptr = reinterpret_cast<char*>(
                    reinterpret_cast<void**>(dst_buffers_all_ranks)[r]);

                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    reinterpret_cast<void*>(dst_ptr + dst_offset_in_rank),
                    reinterpret_cast<const void*>(tiles[slot]),
                    static_cast<uint32_t>(tma_size));
            }
            cuda::ptx::cp_async_bulk_commit_group();
        }
    }

    // Drain all S2G writes for this warp.
    if (is_leader) {
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
    }
    __syncwarp();

    // ---- Cross-SM convergence + cross-rank signaling ----
    // All warps in the block must converge before signaling.
    __syncthreads();

    // System-scope fence to ensure all NVLink writes are globally visible.
    asm volatile("fence.release.sys;" ::: "memory");

    int is_last_SM = 0;
    if (threadIdx.x == 0) {
        unsigned long long value_to_add =
            blockIdx.x == 0 ? MAX_BLOCKS - gridDim.x + 1 : 1;
        auto old_val = atomicAdd(flag_sm_ptr, value_to_add);
        is_last_SM =
            (gridDim.x == 1 || old_val + value_to_add == iter_id * MAX_BLOCKS);
    }

    if (is_last_SM) {
        // Signal all peers via NVLink atomic on rank 0's flag
        asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;"
                     :
                     : "l"(__cvta_generic_to_global(flag_nvl_ptr)), "n"(1)
                     : "memory");
        *iter_id_ptr = iter_id;
        auto expected = iter_id * rank_num;
        clock_t s = clock64();
        unsigned long long flag_data = 0;

        // Spin-wait for all ranks to complete
        do {
            asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                         : "=l"(flag_data)
                         : "l"(__cvta_generic_to_global(flag_nvl_ptr))
                         : "memory");
            if (clock64() - s > 2ull * TIMEOUT) {
                printf("HYBRID-EP TMA ALLGATHER TIMEOUT: SM %d expecting %llu got %llu\n",
                       blockIdx.x, (unsigned long long)expected, flag_data);
                break;
            }
        } while (flag_data < expected);
    }
}

void CustomAllgather::launch(torch::Tensor src, int ag_sms, cudaStream_t stream) {
    auto bytes_per_rank = src.numel() * src.element_size();
    auto rank_num = num_of_ranks_per_node;
    assert(rank_idx >= 0 && rank_idx < rank_num);
    assert(rank_num <= MAX_NUM_OF_RANKS_PER_NODE);
    assert(bytes_per_rank % 16 == 0);  // TMA minimum alignment

    // Use TMA kernel by default for intra-node (NVLink) allgather.
    // The TMA kernel writes to IPC-mapped NVLink peer buffers via cp.async.bulk;
    // this only works within a single NVLink domain (multi-node uses NCCL instead,
    // gated by num_of_nodes > 1 in executor.cu, before this function is called).
    // Set HYBRID_EP_USE_AG_NVL_LEGACY=1 to fall back to the original uint4-store kernel.
    const char* legacy_env = getenv("HYBRID_EP_USE_AG_NVL_LEGACY");
    bool use_legacy = legacy_env && (legacy_env[0] == '1');

    if (use_legacy) {
        int block_size = 1024;
        ag_nvl_kernel<<<ag_sms, block_size, 0, stream>>>(
            dst_buffers_all_ranks_gpu,
            src.data_ptr(),
            bytes_per_rank,
            iter_id_ptr,
            flag_nvl_ptr,
            flag_sm_ptr,
            rank_idx,
            rank_num
        );
    } else {
        // TMA kernel: NUM_PIPELINES warps per block, each with double-buffered tiles
        constexpr int TILE_BYTES = AG_TMA_TILE_BYTES;
        constexpr int NUM_PIPELINES = AG_TMA_NUM_PIPELINES;
        constexpr int PIPELINE_SMEM = 2 * TILE_BYTES + 16;
        int smem_bytes = NUM_PIPELINES * PIPELINE_SMEM;
        int block_threads = NUM_PIPELINES * 32;

        // Set max dynamic SMEM for this kernel if needed
        static bool smem_configured = false;
        if (!smem_configured) {
            cudaFuncSetAttribute(
                ag_nvl_tma_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes);
            smem_configured = true;
        }

        ag_nvl_tma_kernel<<<ag_sms, block_threads, smem_bytes, stream>>>(
            dst_buffers_all_ranks_gpu,
            src.data_ptr(),
            bytes_per_rank,
            iter_id_ptr,
            flag_nvl_ptr,
            flag_sm_ptr,
            rank_idx,
            rank_num
        );
    }
}

void CustomAllgather::init(pybind11::object process_group, int rank_idx, BufferConfig buffer_config, ExtendedMemoryAllocator* allocator) {
    this->rank_idx = rank_idx;
    this->num_of_ranks_per_node = buffer_config.num_of_ranks_per_node;
    this->num_of_experts_per_rank = buffer_config.num_of_experts_per_rank;
    this->num_of_tokens_per_rank = buffer_config.max_num_of_tokens_per_rank;
    this->num_of_nodes = buffer_config.num_of_nodes;
    this->allocator = allocator;
    this->process_group = process_group;
}

bool CustomAllgather::grow_buffer_config(const HybridEpConfigInstance& config, BufferConfig& buf_config) {
    bool changed = false;
    changed |= grow_to(buf_config.num_of_ranks_per_node, config.num_of_ranks_per_node);
    changed |= grow_to(buf_config.num_of_experts_per_rank, config.num_of_experts_per_rank);
    changed |= grow_to(buf_config.max_num_of_tokens_per_rank, config.max_num_of_tokens_per_rank);
    changed |= grow_to(buf_config.num_of_nodes, config.num_of_nodes);
    return changed;
}

void CustomAllgather::update_config(BufferConfig config) {
    this->num_of_ranks_per_node = config.num_of_ranks_per_node;
    this->num_of_experts_per_rank = config.num_of_experts_per_rank;
    this->num_of_tokens_per_rank = config.max_num_of_tokens_per_rank;
    this->num_of_nodes = config.num_of_nodes;
}

void CustomAllgather::allocate_buffers() {
    allocate_ag_buffer();
}

void CustomAllgather::allocate_ag_buffer() {
    // Allocate the output buffer
    auto num_of_expert = num_of_experts_per_rank * num_of_ranks_per_node * num_of_nodes;
    auto gathered_elets = num_of_expert * num_of_tokens_per_rank * num_of_ranks_per_node * num_of_nodes;
    auto gathered_bytes = gathered_elets * sizeof(bool);
    allocator->allocate(&dst_buffer, gathered_bytes);

    if(num_of_nodes == 1) {
        // Allocate the nvl sync flag on the rank 0
        if(rank_idx == 0) {
            allocator->allocate((void**)&flag_nvl_ptr, sizeof(unsigned long long));
            CUDA_CHECK(cudaMemset(flag_nvl_ptr, 0, sizeof(unsigned long long)));
        }

        // Allocate the sm sync flag
        CUDA_CHECK(cudaMalloc((void**)&flag_sm_ptr, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(flag_sm_ptr, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc((void**)&iter_id_ptr, sizeof(int64_t)));
        CUDA_CHECK(cudaMemset(iter_id_ptr, 0, sizeof(int64_t)));

        // Allocate the dst_buffers_all_ranks
        dst_buffers_all_ranks = (void**)malloc(num_of_ranks_per_node * sizeof(void*));
        CUDA_CHECK(cudaMalloc((void**)&dst_buffers_all_ranks_gpu, num_of_ranks_per_node * sizeof(void*)));

        // Get handle of the nvl sync flag
        MemHandle handles[2];
        allocator->get_handle(&handles[0], dst_buffer);
        if (rank_idx == 0) {
            allocator->get_handle(&handles[1], flag_nvl_ptr);
        }
        // Pack handles into tensor
        ag_handles = torch::empty({static_cast<int64_t>(sizeof(handles))},
                                                torch::dtype(torch::kUInt8).device(torch::kCPU));
        memcpy(ag_handles.data_ptr<uint8_t>(), handles, sizeof(handles));

        open_ag_handles();
    }
}

void CustomAllgather::open_ag_handles() {
    if(num_of_nodes > 1 ) return;

    // Use Python's torch.distributed APIs through py::object
    auto torch_distributed = py::module_::import("torch.distributed");    
    // Move tensors to CUDA for communication
    auto ag_handles_cuda = ag_handles.cuda();  
    // Get world size from process group
    int world_size = process_group.attr("size")().cast<int>();
    // Create empty tensors for allgather output
    py::list ag_handles_output_list;
  
    for (int i = 0; i < world_size; i++) {
        ag_handles_output_list.append(torch::empty_like(ag_handles_cuda));
    }
    // Perform allgather using Python API
    torch_distributed.attr("all_gather")(ag_handles_output_list, ag_handles_cuda, process_group);
  
    // Convert back to C++ vectors and move to CPU
    std::vector<torch::Tensor> ag_handles_cpu_tensors;
    for (int i = 0; i < world_size; i++) {
        ag_handles_cpu_tensors.push_back(ag_handles_output_list[i].cast<torch::Tensor>().cpu());
    }
    
    // Open the flag_nvl_ptr handle
    if (rank_idx != 0) {
        MemHandle flag_nvl_ptr_handle;
        // Only rank 0 will allocate memory for this flag
        memcpy(&flag_nvl_ptr_handle, ag_handles_cpu_tensors[0].data_ptr<uint8_t>() + sizeof(MemHandle),
            sizeof(MemHandle));
        allocator->open_handle((void**)(&flag_nvl_ptr), &flag_nvl_ptr_handle);
    }

    // Open the dst_buffers_all_ranks handles
    for (int i = 0; i < num_of_ranks_per_node; i++) {
        MemHandle dst_buffer_handle;
        // Extract the handles from the tensor.
        memcpy(&dst_buffer_handle, ag_handles_cpu_tensors[i].data_ptr<uint8_t>(), sizeof(MemHandle));
        if(i != rank_idx) {
            allocator->open_handle((void**)(&dst_buffers_all_ranks[i]), &dst_buffer_handle);
        } else {
            // For local rank, use direct pointer assignment (more efficient, no IPC overhead)
            dst_buffers_all_ranks[i] = dst_buffer;
        }
    }

    CUDA_CHECK(cudaMemcpy(dst_buffers_all_ranks_gpu, dst_buffers_all_ranks, num_of_ranks_per_node * sizeof(void*), cudaMemcpyHostToDevice));
}

void CustomAllgather::destroy() {
    if(num_of_nodes == 1) {
        if (flag_nvl_ptr != nullptr) {
            if(rank_idx == 0) {
                allocator->free(flag_nvl_ptr);
            } else {
                allocator->close_handle(flag_nvl_ptr);
            }
            flag_nvl_ptr = nullptr;
        }
        
        if (flag_sm_ptr != nullptr) {
            CUDA_CHECK(cudaFree(flag_sm_ptr));
            flag_sm_ptr = nullptr;
        }
        
        if (iter_id_ptr != nullptr) {
            CUDA_CHECK(cudaFree(iter_id_ptr));
            iter_id_ptr = nullptr;
        }
    
        // Close remote memory handles (not locally allocated, just mapped)
        if (dst_buffers_all_ranks != nullptr) {
            for(int i = 0; i < num_of_ranks_per_node; i++) {
                if(i != rank_idx) {
                    allocator->close_handle(dst_buffers_all_ranks[i]);
                }
            }
            free(dst_buffers_all_ranks);
            dst_buffers_all_ranks = nullptr;
        }
        
        // Free the GPU buffer
        if (dst_buffers_all_ranks_gpu != nullptr) {
            CUDA_CHECK(cudaFree(dst_buffers_all_ranks_gpu));
            dst_buffers_all_ranks_gpu = nullptr;
        }
    }
    
    if (dst_buffer != nullptr) {
        allocator->free(dst_buffer);
        dst_buffer = nullptr;
    }
}

void * CustomAllgather::get_output_buffer() {
    return dst_buffer;
}

CustomAllgather::~CustomAllgather() {
    destroy();
}