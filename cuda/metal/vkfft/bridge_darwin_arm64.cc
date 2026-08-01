#include "bridge.h"
#include "../metal_runtime.h"

#include <cstdlib>
#include <cstring>

// VkFFT 1.3.4, pinned in third_party/vkfft/README.mumax3.md. This translation
// unit is intentionally compiled without Objective-C ARC: metal-cpp performs
// its own retain/release calls and cannot safely share ARC ownership inference.
#define VKFFT_BACKEND 5
#include "third_party/vkfft/vkFFT/vkFFT.h"

namespace {

struct MVKPlan {
    VkFFTApplication application;
    pfUINT bytes;
};

void setTextError(char **destination, const char *message) {
    if (destination != nullptr) {
        *destination = ::strdup(message);
    }
}

void setVkError(char **destination, const char *prefix, VkFFTResult result) {
    if (destination == nullptr) {
        return;
    }
    const char *detail = getVkFFTErrorString(result);
    const size_t prefixLength = std::strlen(prefix);
    const size_t detailLength = std::strlen(detail);
    char *message = static_cast<char *>(
        std::malloc(prefixLength + detailLength + 3));
    if (message == nullptr) {
        return;
    }
    std::memcpy(message, prefix, prefixLength);
    message[prefixLength] = ':';
    message[prefixLength + 1] = ' ';
    std::memcpy(message + prefixLength + 2, detail, detailLength + 1);
    *destination = message;
}

void copyRuntimeError(char **destination, char *runtimeError) {
    if (destination != nullptr) {
        *destination = runtimeError == nullptr
                           ? ::strdup("unknown Metal runtime failure")
                           : ::strdup(runtimeError);
    }
    if (runtimeError != nullptr) {
        mr_free_error(runtimeError);
    }
}

}  // namespace

extern "C" void *mvk_plan_create(int64_t nx,
                                  int64_t ny,
                                  int64_t active_x,
                                  int64_t active_y,
                                  int inverse_only,
                                  char **error_message) {
    if (error_message != nullptr) {
        *error_message = nullptr;
    }
    if (nx <= 1 || ny <= 1 || active_x <= 0 || active_x >= nx ||
        active_y <= 0 || active_y >= ny) {
        setTextError(error_message, "invalid VkFFT 2D plan arguments");
        return nullptr;
    }

    mr_external_context context = {};
    char *runtimeError = nullptr;
    int runtimeStatus = mr_begin_external(&context, &runtimeError);
    if (runtimeStatus != MR_SUCCESS) {
        copyRuntimeError(error_message, runtimeError);
        return nullptr;
    }

    MVKPlan *plan = static_cast<MVKPlan *>(std::calloc(1, sizeof(MVKPlan)));
    VkFFTResult result = VKFFT_SUCCESS;
    if (plan == nullptr) {
        result = VKFFT_ERROR_MALLOC_FAILED;
    } else {
        VkFFTConfiguration configuration = {};
        configuration.FFTdim = 2;
        configuration.size[0] = static_cast<pfUINT>(nx);
        configuration.size[1] = static_cast<pfUINT>(ny);
        plan->bytes = static_cast<pfUINT>(2 * (nx / 2 + 1) * ny) *
                      sizeof(float);
        // Metal permits the actual buffer to be supplied through launch
        // parameters, but VkFFT still uses bufferSize while deriving strides
        // and binding blocks. Keep this plan-owned value alive with the app.
        configuration.bufferSize = &plan->bytes;
        configuration.device = reinterpret_cast<MTL::Device *>(context.device);
        configuration.queue = reinterpret_cast<MTL::CommandQueue *>(context.queue);
        configuration.performR2C = 1;
        configuration.normalize = 0;
        configuration.makeForwardPlanOnly = inverse_only ? 0 : 1;
        configuration.makeInversePlanOnly = inverse_only ? 1 : 0;

        // The meaningful magnetization occupies the leading rectangle. Native
        // padding skips the stale tail left by earlier in-place inverse FFTs.
        // VkFFT 1.3.4's Metal backend miscompiles axis-1 native padding in a
        // single-direction R2C/C2R application. Axis 0 is correct; the caller
        // explicitly clears the trailing Y rows before each forward transform.
        configuration.performZeropadding[0] = 1;
        configuration.fft_zeropad_left[0] = static_cast<pfUINT>(active_x);
        configuration.fft_zeropad_right[0] = configuration.size[0];
        (void)active_y;

        // specifyOffsetsAtLaunch intentionally stays zero. VkFFT 1.3.4's
        // Metal generator emits an undeclared PushConsts reference otherwise.
        result = initializeVkFFT(&plan->application, configuration);
    }

    // initializeVkFFT cleans up its own partial state when it fails. Record a
    // fully initialized app separately so an mr_end_external failure can
    // release it without either leaking or double-deleting partial state.
    const bool initialized = result == VKFFT_SUCCESS;
    char *endError = nullptr;
    const int endStatus = mr_end_external(&context, nullptr, 0, &endError);
    if (endStatus != MR_SUCCESS && result == VKFFT_SUCCESS) {
        if (error_message != nullptr) {
            *error_message = endError;
            endError = nullptr;
        }
        result = VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
    }
    if (endError != nullptr) {
        mr_free_error(endError);
    }

    if (result != VKFFT_SUCCESS) {
        if (error_message != nullptr && *error_message == nullptr) {
            setVkError(error_message, "initializeVkFFT", result);
        }
        if (initialized) {
            deleteVkFFT(&plan->application);
        }
        std::free(plan);
        return nullptr;
    }
    return plan;
}

extern "C" int mvk_plan_append(void *opaque_plan,
                                const void *buffer_pointer,
                                int inverse,
                                char **error_message) {
    if (error_message != nullptr) {
        *error_message = nullptr;
    }
    if (opaque_plan == nullptr || buffer_pointer == nullptr) {
        setTextError(error_message, "invalid VkFFT launch arguments");
        return -1;
    }

    MVKPlan *plan = static_cast<MVKPlan *>(opaque_plan);
    mr_external_context context = {};
    char *runtimeError = nullptr;
    int runtimeStatus = mr_begin_external(&context, &runtimeError);
    if (runtimeStatus != MR_SUCCESS) {
        copyRuntimeError(error_message, runtimeError);
        return -1;
    }

    int status = 0;
    mr_buffer_view bufferView = {};
    runtimeStatus = mr_resolve_buffer_locked(buffer_pointer,
                                             static_cast<size_t>(plan->bytes),
                                             &bufferView,
                                             &runtimeError);
    if (runtimeStatus != MR_SUCCESS) {
        copyRuntimeError(error_message, runtimeError);
        status = -1;
    } else if (bufferView.offset != 0) {
        setTextError(error_message,
                     "VkFFT in-place buffer must start at allocation offset zero");
        status = -1;
    } else {
        // computeCommandEncoder follows Cocoa's non-create rule. Drain its
        // autorelease at the end of this append instead of calling release()
        // directly, which over-releases when an outer pool is present.
        NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
        MTL::CommandBuffer *commandBuffer =
            reinterpret_cast<MTL::CommandBuffer *>(context.command_buffer);
        MTL::ComputeCommandEncoder *encoder =
            commandBuffer->computeCommandEncoder();
        if (encoder == nullptr) {
            setTextError(error_message,
                         "VkFFT failed to create a Metal compute encoder");
            status = -1;
        } else {
            MTL::Buffer *buffer =
                reinterpret_cast<MTL::Buffer *>(bufferView.buffer);
            VkFFTLaunchParams launch = {};
            launch.commandBuffer = commandBuffer;
            launch.commandEncoder = encoder;
            launch.buffer = &buffer;
            const VkFFTResult result = VkFFTAppend(&plan->application,
                                                    inverse > 0 ? 1 : -1,
                                                    &launch);
            encoder->endEncoding();
            if (result != VKFFT_SUCCESS) {
                setVkError(error_message, "VkFFTAppend", result);
                status = static_cast<int>(result);
            }
        }
        pool->drain();
    }

    char *endError = nullptr;
    const int endStatus = mr_end_external(&context,
                                          nullptr,
                                          status != 0,
                                          &endError);
    if (endStatus != MR_SUCCESS) {
        if (status == 0) {
            copyRuntimeError(error_message, endError);
            endError = nullptr;
            status = -1;
        }
        if (endError != nullptr) {
            mr_free_error(endError);
        }
    }
    return status;
}

extern "C" void mvk_plan_destroy(void *opaque_plan) {
    if (opaque_plan == nullptr) {
        return;
    }
    MVKPlan *plan = static_cast<MVKPlan *>(opaque_plan);
    deleteVkFFT(&plan->application);
    std::free(plan);
}

extern "C" void mvk_free_error(char *error_message) {
    std::free(error_message);
}
