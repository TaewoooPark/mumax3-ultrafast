#ifndef MUMAX3_METAL_VKFFT_BRIDGE_H
#define MUMAX3_METAL_VKFFT_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void *mvk_plan_create(int64_t nx,
                      int64_t ny,
                      int64_t active_x,
                      int64_t active_y,
                      int inverse_only,
                      char **error_message);

int mvk_plan_append(void *plan,
                    const void *buffer,
                    int inverse,
                    char **error_message);

void mvk_plan_destroy(void *plan);
void mvk_free_error(char *error_message);

#ifdef __cplusplus
}
#endif

#endif
