#ifndef MUMAX3_METAL_RUNTIME_H
#define MUMAX3_METAL_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum mr_status {
    MR_SUCCESS = 0,
    MR_ERROR_UNAVAILABLE = 1,
    MR_ERROR_INVALID_ARGUMENT = 2,
    MR_ERROR_OUT_OF_MEMORY = 3,
    MR_ERROR_NOT_FOUND = 4,
    MR_ERROR_COMPILE = 5,
    MR_ERROR_PIPELINE = 6,
    MR_ERROR_COMMAND = 7,
    MR_ERROR_INTERNAL = 8
};

enum mr_arg_kind {
    MR_ARG_BUFFER = 1,
    MR_ARG_FLOAT32 = 2,
    MR_ARG_INT32 = 3,
    MR_ARG_UINT32 = 4,
    MR_ARG_UINT8 = 5,
    MR_ARG_FLOAT64 = 6,
    MR_ARG_INT64 = 7,
    MR_ARG_UINT64 = 8
};

/*
 * grid_* is the CUDA threadgroup count and block_* the threadgroup size, so
 * grid * block threads are dispatched and the kernel needs a bounds guard.
 * threads_* optionally overrides that with the exact thread count: Metal then
 * builds a partial trailing threadgroup and the kernel can index
 * thread_position_in_grid without a guard. Zero means "use grid * block".
 */
typedef struct mr_grid {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;
    uint32_t block_x;
    uint32_t block_y;
    uint32_t block_z;
    uint32_t threads_x;
    uint32_t threads_y;
    uint32_t threads_z;
} mr_grid;

/*
 * Scalar values are stored little-endian in bits. For MR_ARG_BUFFER, buffer
 * is a shared-allocation address (interior pointers are accepted) and size is
 * the optional minimum accessible byte span.
 */
typedef struct mr_arg {
    uint32_t kind;
    uint32_t size;
    const void *buffer;
    uint64_t bits;
} mr_arg;

typedef struct mr_device_info {
    char name[256];
    uint32_t unified_memory;
    uint32_t reserved;
    uint64_t max_threads_x;
    uint64_t max_threads_y;
    uint64_t max_threads_z;
    uint64_t recommended_max_working_set_size;
    uint64_t current_allocated_size;
    uint64_t tracked_allocation_size;
    uint64_t tracked_peak_allocation_size;
} mr_device_info;

/*
 * Opaque Metal objects borrowed from the runtime. They remain valid only
 * between mr_begin_external and mr_end_external. Objective-C++ adapters may
 * bridge-cast these values to id<MTLDevice>, id<MTLCommandQueue>, and
 * id<MTLCommandBuffer>. Adapters normally append work without committing the
 * command buffer. Framework wrappers such as MPSCommandBuffer may internally
 * call commitAndContinue; in that case the adapter must pass the wrapper's
 * final live root command buffer to mr_end_external so the runtime can adopt
 * it without recommitting the original buffer.
 */
typedef struct mr_external_context {
    void *device;
    void *queue;
    void *command_buffer;
    uint64_t token;
} mr_external_context;

typedef struct mr_buffer_view {
    void *buffer;
    size_t offset;
    size_t length;
} mr_buffer_view;

typedef void mr_completion;

typedef struct mr_runtime_stats {
    uint64_t command_buffer_submissions;
    uint64_t external_root_adoptions;
    uint64_t full_drains;
    uint64_t completion_records;
    uint64_t completion_queries;
    uint64_t completion_query_hits;
    uint64_t completion_waits;
    uint64_t completion_wait_submissions;
    /*
     * Arithmetic-only fillers submitted on a private queue right before the
     * host blocks, so the GPU does not fall out of its performance state across
     * the host's decision window. They touch no simulation buffer.
     */
    uint64_t keepalive_submissions;
} mr_runtime_stats;

int mr_initialize(char **error_message);
int mr_shutdown(char **error_message);
int mr_get_device_info(mr_device_info *info, char **error_message);

int mr_register_source(const void *source, size_t length, char **error_message);
int mr_register_library(const void *library, size_t length, char **error_message);

int mr_alloc(size_t bytes, void **pointer, char **error_message);
int mr_free(void *pointer, char **error_message);
int mr_get_address_range(const void *pointer,
                         void **base,
                         size_t *bytes,
                         char **error_message);
int mr_copy(void *dst, const void *src, size_t bytes, char **error_message);
int mr_copy_to_device(void *dst, const void *src, size_t bytes, char **error_message);
/*
 * Writes the destination without first draining the queue. Allocations are
 * shared memory, so the store is visible to the GPU immediately and therefore
 * also to work that is already encoded. Only safe when the caller guarantees
 * that no in-flight kernel reads the destination range, for example by
 * rotating through a ring of upload slots.
 */
int mr_copy_to_device_unordered(void *dst,
                                const void *src,
                                size_t bytes,
                                char **error_message);
int mr_copy_to_host(void *dst, const void *src, size_t bytes, char **error_message);
/*
 * Reads the source without first draining the queue. Allocations are shared
 * memory, so no blit is involved and the host sees whatever the GPU has already
 * written. Only safe when the caller has separately proven that every kernel
 * writing the source range has completed, for example by waiting a completion
 * token recorded immediately after those kernels were encoded.
 */
int mr_copy_to_host_unordered(void *dst,
                              const void *src,
                              size_t bytes,
                              char **error_message);
int mr_fill(void *dst, uint8_t value, size_t bytes, char **error_message);
int mr_fill_u32(void *dst, uint32_t value, size_t count, char **error_message);

int mr_launch(const char *name,
              mr_grid grid,
              const mr_arg *args,
              size_t arg_count,
              char **error_message);

/*
 * Resolve a kernel name to a dense integer handle once, so the dispatch path
 * does not build an NSString and hash a dictionary on every launch. Handles are
 * stable for the lifetime of the process and are invalidated by mr_shutdown.
 */
int mr_register_kernel(const char *name, uint32_t *handle, char **error_message);
int mr_launch_handle(uint32_t handle,
                     mr_grid grid,
                     const mr_arg *args,
                     size_t arg_count,
                     char **error_message);
int mr_flush(char **error_message);
int mr_synchronize(char **error_message);

/*
 * Capture the current ordered-queue tail without submitting it. The returned
 * token strongly retains the exact command buffer that contains all work
 * encoded before this call. Query is nonblocking. Wait commits the token only
 * when it is still the runtime's live current buffer, then waits just that
 * command buffer; a stale/adopted MPS root is never recommitted.
 */
int mr_record_completion(mr_completion **completion, char **error_message);
int mr_query_completion(mr_completion *completion,
                        int *complete,
                        char **error_message);
int mr_wait_completion(mr_completion *completion, char **error_message);
void mr_release_completion(mr_completion *completion);

int mr_get_runtime_stats(mr_runtime_stats *stats, char **error_message);
int mr_reset_runtime_stats(char **error_message);

int mr_begin_external(mr_external_context *context, char **error_message);
int mr_resolve_buffer_locked(const void *pointer,
                             size_t minimum_bytes,
                             mr_buffer_view *view,
                             char **error_message);
/*
 * final_command_buffer is null when the borrowed command buffer did not
 * change. If an external framework committed it and continued on another
 * command buffer, final_command_buffer must be that framework's current live
 * root buffer on context->queue. The runtime retains the replacement before
 * returning and records the already-committed original for later error
 * collection; it never commits the original a second time.
 */
int mr_end_external(mr_external_context *context,
                    void *final_command_buffer,
                    int encoder_failed,
                    char **error_message);

void mr_free_error(char *error_message);

#ifdef __cplusplus
}
#endif

#endif
