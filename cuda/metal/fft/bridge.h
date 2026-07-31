#ifndef MUMAX3_METAL_FFT_BRIDGE_H
#define MUMAX3_METAL_FFT_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
enum mf_status {
    MF_SUCCESS = 0,
    MF_ERROR_INVALID_ARGUMENT = 1,
    MF_ERROR_UNAVAILABLE = 2,
    MF_ERROR_RUNTIME = 3,
    MF_ERROR_ENCODING = 4
};

enum mf_transform {
    MF_C2C = 41,
    MF_R2C = 42,
    MF_C2R = 44
};

/*
 * active_outer is the number of entries along the second-to-last transform axis
 * that can hold non-zero data, or 0 when every entry can. MuMax3 zero-pads the
 * magnetisation, so the trailing rows of the padded array are always zero and
 * their Hermitian-axis transform is always zero. The Hermitian-axis stage can
 * therefore be restricted to the leading active_outer rows, which is exact
 * rather than an approximation. It requires those rows to be a contiguous prefix
 * of the buffer, so the caller must pass 0 unless that holds.
 */
void *mf_plan_create(const int64_t *dimensions,
                     size_t rank,
                     int64_t batch,
                     int32_t transform,
                     int64_t active_outer,
                     char **error_message);

int mf_plan_execute(void *plan,
                    const void *input,
                    void *output,
                    int32_t direction,
                    char **error_message);

int mf_plan_destroy(void *plan, char **error_message);

void mf_free_error(char *error_message);

#ifdef __cplusplus
}
#endif

#endif
