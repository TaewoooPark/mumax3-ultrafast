#include "bridge.h"
#include "../metal_runtime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <Availability.h>
#include <stdlib.h>
#include <string.h>

/*
 * One cached MPSGraphTensorData view of a caller buffer. The buffer is retained
 * so its address cannot be recycled underneath a cached view, which is what
 * makes comparing addresses a sound identity test. Shapes are plan-owned and
 * therefore compared by pointer as well.
 */
@interface MFTensorSlot : NSObject {
@public
    id<MTLBuffer> buffer;
    size_t offset;
    NSArray<NSNumber *> *shape;
    MPSDataType dataType;
    MPSGraphTensorData *data;
}
@end

@implementation MFTensorSlot
@end

@interface MFPlan : NSObject {
@public
    int32_t transform;
    BOOL oddLastDimension;
    size_t realBytes;
    size_t HermitianBytes;
    NSArray<NSNumber *> *realShape;
    NSArray<NSNumber *> *HermitianShape;
    NSArray<NSNumber *> *axes;
    // A separable transform split into the Hermitian (last, contiguous) axis
    // and the remaining axes. MPSGraph produces bit-identical results either
    // way, but is far slower when asked for all axes in one operation: for a
    // 1024x1024 padded R2C on an M4 the combined form costs 255 us against
    // 47 us + 41 us for the two stages.
    NSArray<NSNumber *> *primaryAxes;
    NSArray<NSNumber *> *leadingAxes;
    // Shapes for the Hermitian-axis stage when the zero-padded tail is skipped.
    // Equal to realShape/HermitianShape when no rows are skipped.
    NSArray<NSNumber *> *activeRealShape;
    NSArray<NSNumber *> *activeHermitianShape;
    BOOL skipsPaddedRows;
    MPSGraph *forwardGraph;
    MPSGraphTensor *forwardInput;
    MPSGraphTensor *forwardOutput;
    MPSGraph *inverseGraph;
    MPSGraphTensor *inverseInput;
    MPSGraphTensor *inverseOutput;
    MPSGraph *stageGraph;
    MPSGraphTensor *stageInput;
    MPSGraphTensor *stageOutput;
    id<MTLBuffer> scratch;
    /*
     * mumax3 hands a plan the same demag allocations for the life of a run, so
     * rebuilding the tensor views on every execute was pure allocation: an
     * RK45DP step issues 36 of them, and an offset view allocates an MPSNDArray
     * and its descriptor on top of the MPSGraphTensorData. Views are cached per
     * (buffer, offset, shape, type).
     */
    NSMutableArray<MFTensorSlot *> *tensorSlots;
}
@end

@implementation MFPlan
@end

static void mf_set_error(char **destination, NSString *message) {
    if (destination == nullptr) {
        return;
    }
    const char *text = message == nil ? "unknown Metal FFT failure" : message.UTF8String;
    *destination = strdup(text == nullptr ? "unknown Metal FFT failure" : text);
}

static void mf_copy_runtime_error(char **destination, char *runtimeError) {
    if (runtimeError == nullptr) {
        mf_set_error(destination, @"unknown Metal runtime failure");
        return;
    }
    if (destination != nullptr) {
        *destination = strdup(runtimeError);
    }
    mr_free_error(runtimeError);
}

static NSArray<NSNumber *> *mf_shape(const int64_t *dimensions,
                                     size_t rank,
                                     int64_t batch,
                                     BOOL Hermitian) {
    NSMutableArray<NSNumber *> *shape = [NSMutableArray arrayWithCapacity:rank + (batch > 1 ? 1 : 0)];
    if (batch > 1) {
        [shape addObject:@(batch)];
    }
    for (size_t axis = 0; axis < rank; ++axis) {
        int64_t value = dimensions[axis];
        if (Hermitian && axis + 1 == rank) {
            value = value / 2 + 1;
        }
        [shape addObject:@(value)];
    }
    return [shape copy];
}

static NSArray<NSNumber *> *mf_axes(size_t rank, int64_t batch) {
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:rank];
    NSInteger offset = batch > 1 ? 1 : 0;
    for (size_t axis = 0; axis < rank; ++axis) {
        [result addObject:@(offset + (NSInteger)axis)];
    }
    return [result copy];
}

/*
 * MPSGraph's combined multi-axis transform sustains full bandwidth while the
 * strided axis is at most 512 long and then collapses. Padded R2C on an M4,
 * marginal cost per transform and the implied single-pass bandwidth:
 *
 *   {1, 512,  512}    24.1 us   87 GB/s
 *   {1, 512, 1024}    36.6 us  115 GB/s   long axis contiguous, fine
 *   {1, 640,  640}    60.9 us   54 GB/s
 *   {1, 768,  768}   116.7 us   40 GB/s
 *   {1,1024, 1024}   260.4 us   32 GB/s   long axis strided, collapsed
 *
 * Running the Hermitian axis and the remaining axes as two separate transforms
 * avoids that path. It is bit-identical, verified against the combined form to
 * exactly 0 difference in both directions, so this only selects a schedule.
 *
 * The split is not free: it doubles the number of MPSGraph encodes, which costs
 * more than it saves while the combined form is still healthy. Whole-demag
 * timings on this M4 put the crossover between 512 and 1024, but run-to-run
 * spread on the same binary reached 35% at these sizes, so the threshold is set
 * from the bandwidth measurements above, which were reproducible, rather than
 * from the noisier end-to-end numbers. Splitting therefore starts above 512.
 */
static const int64_t mf_split_threshold = 512;

/*
 * Restrict the Hermitian-axis stage to the leading active rows. The rows the
 * caller declares inactive are zero, so their transform is zero; the stage that
 * follows still reads the full array, because those zeros are exactly the
 * zero-padding the convolution depends on.
 *
 * Only the Hermitian axis can be shortened this way. The zeros inside an active
 * row are the padding along that axis and must be transformed.
 */
static void mf_active_shapes(MFPlan *plan, int64_t activeOuter) {
    plan->activeRealShape = plan->realShape;
    plan->activeHermitianShape = plan->HermitianShape;
    plan->skipsPaddedRows = NO;
    if (activeOuter <= 0 || plan->leadingAxes.count == 0) {
        return;
    }
    NSUInteger count = plan->realShape.count;
    if (count < 2) {
        return;
    }
    NSUInteger outer = count - 2;
    if (activeOuter >= plan->realShape[outer].longLongValue) {
        return;  // nothing to skip
    }
    // The active rows must be a contiguous prefix, so every axis outside the
    // one being shortened has to be a single plane.
    for (NSUInteger axis = 0; axis < outer; ++axis) {
        if (plan->realShape[axis].longLongValue != 1) {
            return;
        }
    }
    NSMutableArray<NSNumber *> *real = [plan->realShape mutableCopy];
    NSMutableArray<NSNumber *> *herm = [plan->HermitianShape mutableCopy];
    real[outer] = @(activeOuter);
    herm[outer] = @(activeOuter);
    plan->activeRealShape = [real copy];
    plan->activeHermitianShape = [herm copy];
    plan->skipsPaddedRows = YES;
}

static void mf_split_axes(MFPlan *plan, const int64_t *dimensions, size_t rank) {
    NSUInteger count = plan->axes.count;
    // Without a split the single graph must still transform every axis.
    plan->primaryAxes = plan->axes;
    plan->leadingAxes = @[];
    if (count < 2) {
        return;
    }
    int64_t longestLeading = 0;
    for (size_t axis = 0; axis + 1 < rank; ++axis) {
        if (dimensions[axis] > longestLeading) {
            longestLeading = dimensions[axis];
        }
    }
    if (longestLeading >= mf_split_threshold) {
        plan->leadingAxes =
            [plan->axes subarrayWithRange:NSMakeRange(0, count - 1)];
        plan->primaryAxes = @[plan->axes[count - 1]];
    }
}

static size_t mf_element_count(NSArray<NSNumber *> *shape) {
    size_t result = 1;
    for (NSNumber *dimension in shape) {
        result *= dimension.unsignedLongLongValue;
    }
    return result;
}

static void mf_build_forward_graph(MFPlan *plan) {
    plan->forwardGraph = [MPSGraph new];
    if (plan->transform == MF_R2C) {
        plan->forwardInput = [plan->forwardGraph placeholderWithShape:plan->activeRealShape
                                                             dataType:MPSDataTypeFloat32
                                                                 name:@"mumax3_fft_r2c_input"];
        MPSGraphFFTDescriptor *descriptor = [MPSGraphFFTDescriptor descriptor];
        descriptor.inverse = NO;
        descriptor.scalingMode = MPSGraphFFTScalingModeNone;
        descriptor.roundToOddHermitean = plan->oddLastDimension;
        plan->forwardOutput = [plan->forwardGraph realToHermiteanFFTWithTensor:plan->forwardInput
                                                                          axes:plan->primaryAxes
                                                                    descriptor:descriptor
                                                                          name:@"mumax3_fft_r2c_output"];
        if (plan->leadingAxes.count != 0) {
            plan->stageGraph = [MPSGraph new];
            plan->stageInput = [plan->stageGraph placeholderWithShape:plan->HermitianShape
                                                            dataType:MPSDataTypeComplexFloat32
                                                                name:@"mumax3_fft_r2c_stage_input"];
            MPSGraphFFTDescriptor *stage = [MPSGraphFFTDescriptor descriptor];
            stage.inverse = NO;
            stage.scalingMode = MPSGraphFFTScalingModeNone;
            plan->stageOutput = [plan->stageGraph fastFourierTransformWithTensor:plan->stageInput
                                                                            axes:plan->leadingAxes
                                                                      descriptor:stage
                                                                            name:@"mumax3_fft_r2c_stage_output"];
        }
    } else {
        plan->forwardInput = [plan->forwardGraph placeholderWithShape:plan->realShape
                                                             dataType:MPSDataTypeComplexFloat32
                                                                 name:@"mumax3_fft_c2c_forward_input"];
        MPSGraphFFTDescriptor *descriptor = [MPSGraphFFTDescriptor descriptor];
        descriptor.inverse = NO;
        descriptor.scalingMode = MPSGraphFFTScalingModeNone;
        plan->forwardOutput = [plan->forwardGraph fastFourierTransformWithTensor:plan->forwardInput
                                                                            axes:plan->axes
                                                                      descriptor:descriptor
                                                                            name:@"mumax3_fft_c2c_forward_output"];
    }
}

static void mf_build_inverse_graph(MFPlan *plan) {
    plan->inverseGraph = [MPSGraph new];
    MPSGraphFFTDescriptor *descriptor = [MPSGraphFFTDescriptor descriptor];
    descriptor.inverse = YES;
    descriptor.scalingMode = MPSGraphFFTScalingModeNone;
    descriptor.roundToOddHermitean = plan->oddLastDimension;

    if (plan->transform == MF_C2R) {
        if (plan->leadingAxes.count != 0) {
            plan->stageGraph = [MPSGraph new];
            plan->stageInput = [plan->stageGraph placeholderWithShape:plan->HermitianShape
                                                            dataType:MPSDataTypeComplexFloat32
                                                                name:@"mumax3_fft_c2r_stage_input"];
            MPSGraphFFTDescriptor *stage = [MPSGraphFFTDescriptor descriptor];
            stage.inverse = YES;
            stage.scalingMode = MPSGraphFFTScalingModeNone;
            plan->stageOutput = [plan->stageGraph fastFourierTransformWithTensor:plan->stageInput
                                                                            axes:plan->leadingAxes
                                                                      descriptor:stage
                                                                            name:@"mumax3_fft_c2r_stage_output"];
        }
        plan->inverseInput = [plan->inverseGraph placeholderWithShape:plan->activeHermitianShape
                                                             dataType:MPSDataTypeComplexFloat32
                                                                 name:@"mumax3_fft_c2r_input"];
        plan->inverseOutput = [plan->inverseGraph HermiteanToRealFFTWithTensor:plan->inverseInput
                                                                          axes:plan->primaryAxes
                                                                    descriptor:descriptor
                                                                          name:@"mumax3_fft_c2r_output"];
    } else {
        plan->inverseInput = [plan->inverseGraph placeholderWithShape:plan->realShape
                                                             dataType:MPSDataTypeComplexFloat32
                                                                 name:@"mumax3_fft_c2c_inverse_input"];
        plan->inverseOutput = [plan->inverseGraph fastFourierTransformWithTensor:plan->inverseInput
                                                                            axes:plan->axes
                                                                      descriptor:descriptor
                                                                            name:@"mumax3_fft_c2c_inverse_output"];
    }
}

static MPSGraphTensorData *mf_tensor_data(id<MTLBuffer> buffer,
                                          size_t offset,
                                          NSArray<NSNumber *> *shape,
                                          MPSDataType dataType) {
    if (offset == 0) {
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                       shape:shape
                                                    dataType:dataType];
    }
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        MPSNDArrayDescriptor *descriptor = [MPSNDArrayDescriptor descriptorWithDataType:dataType
                                                                                  shape:shape];
        descriptor.preferPackedRows = YES;
        MPSNDArray *array = [[MPSNDArray alloc] initWithBuffer:buffer
                                                       offset:offset
                                                   descriptor:descriptor];
        return [[MPSGraphTensorData alloc] initWithMPSNDArray:array];
    }
#endif
    return nil;
}

/*
 * Bound on distinct views one plan keeps alive. A 2D demag plan needs at most
 * the full and active shapes of its input and output plus two scratch views; the
 * cap only guards against an unexpected caller pattern turning the cache into
 * unbounded retention, and dropping it costs one rebuild.
 */
static const NSUInteger mf_tensor_slot_limit = 8;

static MPSGraphTensorData *mf_tensor_data_cached(MFPlan *plan,
                                                 id<MTLBuffer> buffer,
                                                 size_t offset,
                                                 NSArray<NSNumber *> *shape,
                                                 MPSDataType dataType) {
    if (plan->tensorSlots == nil) {
        plan->tensorSlots = [NSMutableArray array];
    }
    for (MFTensorSlot *slot in plan->tensorSlots) {
        if (slot->buffer == buffer && slot->offset == offset &&
            slot->shape == shape && slot->dataType == dataType) {
            return slot->data;
        }
    }
    MPSGraphTensorData *data = mf_tensor_data(buffer, offset, shape, dataType);
    if (data == nil) {
        return nil;
    }
    if (plan->tensorSlots.count >= mf_tensor_slot_limit) {
        [plan->tensorSlots removeAllObjects];
    }
    MFTensorSlot *slot = [MFTensorSlot new];
    slot->buffer = buffer;
    slot->offset = offset;
    slot->shape = shape;
    slot->dataType = dataType;
    slot->data = data;
    [plan->tensorSlots addObject:slot];
    return data;
}

extern "C" void *mf_plan_create(const int64_t *dimensions,
                                  size_t rank,
                                  int64_t batch,
                                  int32_t transform,
                                  int64_t active_inner,
                                  int64_t active_outer,
                                  char **error_message) {
    @autoreleasepool {
        (void)active_inner;
        if (dimensions == nullptr || rank == 0 || rank > 3 || batch < 1) {
            mf_set_error(error_message, @"rank must be 1...3 and batch must be positive");
            return nullptr;
        }
        if (transform != MF_R2C && transform != MF_C2R && transform != MF_C2C) {
            mf_set_error(error_message, @"only single-precision R2C, C2R, and C2C transforms are supported");
            return nullptr;
        }
        for (size_t axis = 0; axis < rank; ++axis) {
            if (dimensions[axis] < 1) {
                mf_set_error(error_message, @"all FFT dimensions must be positive");
                return nullptr;
            }
        }
        if (@available(macOS 14.0, *)) {
            @try {
                MFPlan *plan = [MFPlan new];
                plan->transform = transform;
                plan->oddLastDimension = dimensions[rank - 1] % 2 != 0;
                plan->realShape = mf_shape(dimensions, rank, batch, NO);
                plan->HermitianShape = mf_shape(dimensions, rank, batch, YES);
                plan->axes = mf_axes(rank, batch);
                mf_split_axes(plan, dimensions, rank);
                // Skipping padded rows needs the two-stage form, so it only
                // applies where mf_split_axes already chose to split. Forcing a
                // split just to skip rows loses: measured on an M4, whole-demag
                // time went from 167 to 209 us at padded 128x128 and 219 to 278
                // us at 256x256, because the extra MPSGraph encode costs more
                // than halving an already cheap Hermitian pass saves.
                mf_active_shapes(plan, active_outer);
                plan->realBytes = mf_element_count(plan->realShape) * sizeof(float);
                plan->HermitianBytes = mf_element_count(plan->HermitianShape) * 2 * sizeof(float);

                if (transform == MF_R2C || transform == MF_C2C) {
                    mf_build_forward_graph(plan);
                }
                if (transform == MF_C2R || transform == MF_C2C) {
                    mf_build_inverse_graph(plan);
                }
                return (__bridge_retained void *)plan;
            } @catch (NSException *exception) {
                mf_set_error(error_message, exception.reason);
                return nullptr;
            }
        }
        mf_set_error(error_message, @"MPSGraph FFT requires macOS 14 or newer");
        return nullptr;
    }
}

extern "C" int mf_plan_execute(void *opaquePlan,
                                 const void *input,
                                 void *output,
                                 int32_t direction,
                                 char **error_message) {
    @autoreleasepool {
        if (opaquePlan == nullptr || input == nullptr || output == nullptr) {
            mf_set_error(error_message, @"plan and buffers must be non-null");
            return MF_ERROR_INVALID_ARGUMENT;
        }
        if (@available(macOS 14.0, *)) {
            // Continue below. Keeping the availability check here produces a
            // clear runtime error when an otherwise relocatable binary is
            // started on an older macOS release.
        } else {
            mf_set_error(error_message, @"MPSGraph FFT requires macOS 14 or newer");
            return MF_ERROR_UNAVAILABLE;
        }

        MFPlan *plan = (__bridge MFPlan *)opaquePlan;
        const BOOL inverse = plan->transform == MF_C2R ||
                             (plan->transform == MF_C2C && direction > 0);
        const size_t inputBytes = plan->transform == MF_R2C
                                      ? plan->realBytes
                                      : (plan->transform == MF_C2R ? plan->HermitianBytes : plan->realBytes * 2);
        const size_t outputBytes = plan->transform == MF_R2C
                                       ? plan->HermitianBytes
                                       : (plan->transform == MF_C2R ? plan->realBytes : plan->realBytes * 2);

        mr_external_context context = {};
        char *runtimeError = nullptr;
        int runtimeStatus = mr_begin_external(&context, &runtimeError);
        if (runtimeStatus != MR_SUCCESS) {
            mf_copy_runtime_error(error_message, runtimeError);
            return MF_ERROR_RUNTIME;
        }

        int result = MF_SUCCESS;
        MPSCommandBuffer *mpsCommandBuffer = nil;
        id<MTLCommandBuffer> finalCommandBuffer = nil;
        @try {
            mr_buffer_view inputView = {};
            runtimeStatus = mr_resolve_buffer_locked(input, inputBytes, &inputView, &runtimeError);
            if (runtimeStatus != MR_SUCCESS) {
                mf_copy_runtime_error(error_message, runtimeError);
                result = MF_ERROR_RUNTIME;
            }

            mr_buffer_view outputView = {};
            if (result == MF_SUCCESS) {
                runtimeStatus = mr_resolve_buffer_locked(output, outputBytes, &outputView, &runtimeError);
                if (runtimeStatus != MR_SUCCESS) {
                    mf_copy_runtime_error(error_message, runtimeError);
                    result = MF_ERROR_RUNTIME;
                }
            }

            if (result == MF_SUCCESS) {
                MPSGraph *graph = inverse ? plan->inverseGraph : plan->forwardGraph;
                MPSGraphTensor *inputTensor = inverse ? plan->inverseInput : plan->forwardInput;
                MPSGraphTensor *outputTensor = inverse ? plan->inverseOutput : plan->forwardOutput;
                NSArray<NSNumber *> *inputShape =
                    plan->transform == MF_R2C ? plan->realShape :
                    (plan->transform == MF_C2R ? plan->HermitianShape : plan->realShape);
                NSArray<NSNumber *> *outputShape =
                    plan->transform == MF_R2C ? plan->HermitianShape :
                    (plan->transform == MF_C2R ? plan->realShape : plan->realShape);
                MPSDataType dataType =
                    plan->transform == MF_R2C ? MPSDataTypeFloat32 :
                    (plan->transform == MF_C2R ? MPSDataTypeComplexFloat32 : MPSDataTypeComplexFloat32);
                MPSDataType outputDataType =
                    plan->transform == MF_C2R ? MPSDataTypeFloat32 : MPSDataTypeComplexFloat32;

                id<MTLBuffer> inputBuffer = (__bridge id<MTLBuffer>)inputView.buffer;
                id<MTLBuffer> outputBuffer = (__bridge id<MTLBuffer>)outputView.buffer;
                MPSGraphTensorData *inputData = mf_tensor_data_cached(plan, inputBuffer, inputView.offset, inputShape, dataType);
                MPSGraphTensorData *outputData = mf_tensor_data_cached(plan, outputBuffer, outputView.offset, outputShape, outputDataType);
                if (inputData == nil || outputData == nil) {
                    mf_set_error(error_message, @"non-zero FFT buffer offsets require an Xcode 16+/macOS 15 build");
                    result = MF_ERROR_UNAVAILABLE;
                } else {
                    id<MTLCommandBuffer> commandBuffer =
                        (__bridge id<MTLCommandBuffer>)context.command_buffer;
                    mpsCommandBuffer =
                        [MPSCommandBuffer commandBufferWithCommandBuffer:commandBuffer];

                    /*
                     * A separable R2C/C2R runs as two stages through a
                     * plan-owned scratch buffer: the Hermitian axis and then
                     * the remaining axes. The results are bit-identical to
                     * asking MPSGraph for every axis at once, but the combined
                     * form schedules far worse - 255 us against 47 + 41 us for
                     * a 1024x1024 padded R2C on an M4. The scratch adds no
                     * traffic: the second stage reads it instead of reading the
                     * destination it would otherwise update in place.
                     */
                    MPSGraph *stageGraph = plan->stageGraph;
                    if (stageGraph != nil && plan->scratch == nil) {
                        id<MTLDevice> metalDevice =
                            (__bridge id<MTLDevice>)context.device;
                        /*
                         * Shared rather than Private so the padded tail can be
                         * zeroed once here. On a forward transform that skips
                         * padded rows the Hermitian stage only ever writes the
                         * leading prefix, and the stage after it reads the whole
                         * buffer, so the tail has to start at zero and stays
                         * zero for the life of the plan. Private measured no
                         * faster than Shared on this unified memory anyway.
                         */
                        plan->scratch =
                            [metalDevice newBufferWithLength:plan->HermitianBytes
                                                     options:MTLResourceStorageModeShared];
                        if (plan->scratch == nil || plan->scratch.contents == nullptr) {
                            plan->scratch = nil;
                            mf_set_error(error_message,
                                         @"failed to allocate the Metal FFT stage buffer");
                            result = MF_ERROR_RUNTIME;
                        } else {
                            memset(plan->scratch.contents, 0, plan->HermitianBytes);
                        }
                    }

                    if (result != MF_SUCCESS) {
                        // fall through to mr_end_external below
                    } else if (stageGraph == nil) {
                        [graph encodeToCommandBuffer:mpsCommandBuffer
                                               feeds:@{inputTensor: inputData}
                                    targetOperations:nil
                                   resultsDictionary:@{outputTensor: outputData}
                                 executionDescriptor:nil];
                    } else {
                        /*
                         * The Hermitian stage works on the leading active rows,
                         * the other stage on the whole array. Both views start at
                         * the same address because the active rows are a
                         * contiguous prefix, which is what mf_active_shapes
                         * checks before enabling this.
                         */
                        MPSGraphTensorData *scratchFull =
                            mf_tensor_data_cached(plan,
                                                  plan->scratch,
                                                  0,
                                                  plan->HermitianShape,
                                                  MPSDataTypeComplexFloat32);
                        MPSGraphTensorData *scratchActive = scratchFull;
                        if (plan->skipsPaddedRows) {
                            scratchActive =
                                mf_tensor_data_cached(plan,
                                                      plan->scratch,
                                                      0,
                                                      plan->activeHermitianShape,
                                                      MPSDataTypeComplexFloat32);
                        }
                        if (inverse) {
                            // Inverse over the leading axes needs every row,
                            // including the padding. The Hermitian stage then
                            // only has to reconstruct the rows copyUnPad reads.
                            MPSGraphTensorData *realOut = outputData;
                            if (plan->skipsPaddedRows) {
                                realOut = mf_tensor_data_cached(plan,
                                                                outputBuffer,
                                                                outputView.offset,
                                                                plan->activeRealShape,
                                                                MPSDataTypeFloat32);
                            }
                            if (realOut == nil) {
                                mf_set_error(error_message,
                                             @"non-zero FFT buffer offsets require an Xcode 16+/macOS 15 build");
                                result = MF_ERROR_UNAVAILABLE;
                            } else {
                                [stageGraph encodeToCommandBuffer:mpsCommandBuffer
                                                           feeds:@{plan->stageInput: inputData}
                                                targetOperations:nil
                                               resultsDictionary:@{plan->stageOutput: scratchFull}
                                             executionDescriptor:nil];
                                [graph encodeToCommandBuffer:mpsCommandBuffer
                                                      feeds:@{inputTensor: scratchActive}
                                           targetOperations:nil
                                          resultsDictionary:@{outputTensor: realOut}
                                        executionDescriptor:nil];
                            }
                        } else {
                            // The padded rows are zero, so their Hermitian
                            // transform is zero and the scratch tail already
                            // holds it. Only the active rows are transformed.
                            MPSGraphTensorData *realIn = inputData;
                            if (plan->skipsPaddedRows) {
                                realIn = mf_tensor_data_cached(plan,
                                                               inputBuffer,
                                                               inputView.offset,
                                                               plan->activeRealShape,
                                                               MPSDataTypeFloat32);
                            }
                            if (realIn == nil) {
                                mf_set_error(error_message,
                                             @"non-zero FFT buffer offsets require an Xcode 16+/macOS 15 build");
                                result = MF_ERROR_UNAVAILABLE;
                            } else {
                                [graph encodeToCommandBuffer:mpsCommandBuffer
                                                      feeds:@{inputTensor: realIn}
                                           targetOperations:nil
                                          resultsDictionary:@{outputTensor: scratchActive}
                                        executionDescriptor:nil];
                                [stageGraph encodeToCommandBuffer:mpsCommandBuffer
                                                           feeds:@{plan->stageInput: scratchFull}
                                                targetOperations:nil
                                               resultsDictionary:@{plan->stageOutput: outputData}
                                             executionDescriptor:nil];
                            }
                        }
                    }
                }
            }
        } @catch (NSException *exception) {
            mf_set_error(error_message, exception.reason);
            result = MF_ERROR_ENCODING;
        } @finally {
            /*
             * Any method using an MPSCommandBuffer may internally call
             * commitAndContinue. Apple's contract requires asking the wrapper
             * for its current live root afterwards; returning the borrowed
             * object would make the runtime commit an already-committed buffer.
             * Keep this in @finally so a graph that commits before throwing is
             * handed off safely as well.
             */
            if (mpsCommandBuffer != nil) {
                finalCommandBuffer = mpsCommandBuffer.rootCommandBuffer;
            }
        }

        char *endError = nullptr;
        const int endStatus = mr_end_external(
            &context,
            (__bridge void *)finalCommandBuffer,
            result != MF_SUCCESS,
            &endError);
        if (endStatus != MR_SUCCESS) {
            if (result == MF_SUCCESS) {
                mf_copy_runtime_error(error_message, endError);
                result = MF_ERROR_RUNTIME;
            } else if (endError != nullptr) {
                mr_free_error(endError);
            }
        }
        return result;
    }
}

extern "C" int mf_plan_destroy(void *opaquePlan, char **error_message) {
    (void)error_message;
    if (opaquePlan == nullptr) {
        return MF_SUCCESS;
    }
    @autoreleasepool {
        MFPlan *plan = (__bridge_transfer MFPlan *)opaquePlan;
        (void)plan;
    }
    return MF_SUCCESS;
}

extern "C" int mf_test_commit_and_continue(char **error_message) {
    @autoreleasepool {
        mr_external_context context = {};
        char *runtimeError = nullptr;
        int runtimeStatus = mr_begin_external(&context, &runtimeError);
        if (runtimeStatus != MR_SUCCESS) {
            mf_copy_runtime_error(error_message, runtimeError);
            return MF_ERROR_RUNTIME;
        }

        int result = MF_SUCCESS;
        MPSCommandBuffer *mpsCommandBuffer = nil;
        id<MTLCommandBuffer> finalCommandBuffer = nil;
        @try {
            id<MTLCommandBuffer> commandBuffer =
                (__bridge id<MTLCommandBuffer>)context.command_buffer;
            mpsCommandBuffer =
                [MPSCommandBuffer commandBufferWithCommandBuffer:commandBuffer];
            [mpsCommandBuffer commitAndContinue];
        } @catch (NSException *exception) {
            mf_set_error(error_message, exception.reason);
            result = MF_ERROR_ENCODING;
        } @finally {
            if (mpsCommandBuffer != nil) {
                finalCommandBuffer = mpsCommandBuffer.rootCommandBuffer;
            }
        }

        char *endError = nullptr;
        const int endStatus = mr_end_external(
            &context,
            (__bridge void *)finalCommandBuffer,
            result != MF_SUCCESS,
            &endError);
        if (endStatus != MR_SUCCESS) {
            if (result == MF_SUCCESS) {
                mf_copy_runtime_error(error_message, endError);
                result = MF_ERROR_RUNTIME;
            } else if (endError != nullptr) {
                mr_free_error(endError);
            }
        }
        return result;
    }
}

extern "C" void mf_free_error(char *error_message) {
    free(error_message);
}
