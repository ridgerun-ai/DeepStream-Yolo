/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <string.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>

#include <gst/gst.h>
#include <glib.h>

#include "infer_custom_process.h"
#include "nvbufsurface.h"
#include "nvdsmeta.h"

#include "utils.h"

// enable debug log
#define ENABLE_DEBUG 1

namespace dsis = nvdsinferserver;

#if ENABLE_DEBUG
#define LOG_DEBUG(fmt, ...) fprintf(stdout, "%s:%d" fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...)
#endif

#define LOG_ERROR(fmt, ...) fprintf(stderr, "%s:%d" fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#ifndef INFER_ASSERT
#define INFER_ASSERT(expr)                                                     \
    do {                                                                       \
        if (!(expr)) {                                                         \
            fprintf(stderr, "%s:%d ASSERT(%s) \n", __FILE__, __LINE__, #expr); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)
#endif

// --- model parameters ---
#define NET_INPUT_WIDTH     640.0f
#define NET_INPUT_HEIGHT    640.0f
#define DETECTION_THRESH    0.25f
#define NMS_IOU_THRESH      0.45f

#define CSTR(str) (str).empty() ? "" : (str).c_str()

// constant values definition
static const std::vector<std::string> kClassLabels = {
    "person",        "bicycle",       "car",           "motorbike",
    "aeroplane",     "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "sofa",          "pottedplant",   "bed",
    "diningtable",   "toilet",        "tvmonitor",     "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush",
};

extern "C" dsis::IInferCustomProcessor* CreateInferServerCustomProcess(
    const char* config, uint32_t configLen);

namespace {
using namespace dsis;

std::string
memType2Str(InferMemType t)
{
    switch (t) {
    case InferMemType::kGpuCuda:
        return "kGpuCuda";
    case InferMemType::kCpu:
        return "kCpu";
    case InferMemType::kCpuCuda:
        return "kCpuPinned";
    default:
        return "Unknown";
    }
}

std::string
dataType2Str(dsis::InferDataType t)
{
    switch (t) {
    case InferDataType::kFp32:
        return "kFp32";
    case InferDataType::kFp16:
        return "kFp16";
    case InferDataType::kInt8:
        return "kInt8";
    case InferDataType::kInt32:
        return "kInt32";
    case InferDataType::kInt16:
        return "kInt16";
    case InferDataType::kUint8:
        return "kUint8";
    case InferDataType::kUint16:
        return "kUint16";
    case InferDataType::kUint32:
        return "kUint32";
    case InferDataType::kFp64:
        return "kFp64";
    case InferDataType::kInt64:
        return "kInt64";
    case InferDataType::kUint64:
        return "kUint64";
    case InferDataType::kString:
        return "kString";
    case InferDataType::kBool:
        return "kBool";
    default:
        return "Unknown";
    }
}

// return buffer description string
std::string
strOfBufDesc(const dsis::InferBufferDescription& desc)
{
    std::stringstream ss;
    ss << "*" << desc.name << "*, shape: ";
    for (uint32_t i = 0; i < desc.dims.numDims; ++i) {
        if (i != 0) {
            ss << "x";
        } else {
            ss << "[";
        }
        ss << desc.dims.d[i];
        if (i == desc.dims.numDims - 1) {
            ss << "]";
        }
    }
    ss << ", dataType:" << dataType2Str(desc.dataType);
    ss << ", memType:" << memType2Str(desc.memType);
    return ss.str();
}

}  // namespace

class YoloV8CustomPostprocessor : public dsis::IInferCustomProcessor {
private:
    std::map<uint64_t, std::vector<float>> _streamFeedback;
    std::mutex _streamMutex;
public:
    ~YoloV8CustomPostprocessor() override = default;
    /** override function
     * Specifies supported extraInputs memtype in extraInputProcess()
     */
    void supportInputMemType(dsis::InferMemType& type) override { type = dsis::InferMemType::kCpu; }

    /** override function
     * check whether custom loop process needed.
     * If return True, extraInputProcess() and inferenceDone() runs in order per stream_ids
     * This is usually for LSTM loop purpose. FasterRCNN does not need it.
     * The code for requireInferLoop() conditions just sample when user has
     * a LSTM-like Loop model and requires loop custom processing.
     * */
    bool requireInferLoop() const override { return false; }

    /**
     * override function
     * Do custom processing on extra inputs.
     * @primaryInput is already preprocessed. DO NOT update it again.
     * @extraInputs, do custom processing and fill all data according the tensor shape
     * @options, it has most of the common Deepstream metadata along with primary data.
     *           e.g. NvDsBatchMeta, NvDsObjectMeta, NvDsFrameMeta, stream ids...
     *           see infer_ioptions.h to see all the potential key name and structures
     *           in the key-value table.
     */
    NvDsInferStatus extraInputProcess(
        const std::vector<dsis::IBatchBuffer*>& primaryInputs,
        std::vector<dsis::IBatchBuffer*>& extraInputs,
        const dsis::IOptions* options) override
    {
        return NVDSINFER_SUCCESS;
    }

    /** override function
     * Custom processing for inferenced output tensors.
     * output memory types is controlled by gst-nvinferserver config file
     *     config_triton_inferserver_primary_fasterRCNN.txt:
     *       infer_config {
     *         backend {  output_mem_type: MEMORY_TYPE_CPU }
     *     }
     * User can even attach parsed metadata into GstBuffer from this function
     */
    NvDsInferStatus inferenceDone(
        const dsis::IBatchArray* outputs, const dsis::IOptions* inOptions) override 
    {
        std::vector<uint64_t> streamIds;
        INFER_ASSERT(
            inOptions->getValueArray(OPTION_NVDS_SREAM_IDS, streamIds) == NVDSINFER_SUCCESS);
        INFER_ASSERT(!streamIds.empty());
        uint32_t batchSize = streamIds.size();
        std::vector<std::vector<NvDsInferObjectDetectionInfo>> parsedBboxes(batchSize);

        // add 2 output tensors into map
        std::unordered_map<std::string, const dsis::IBatchBuffer*> tensors;
        for (uint32_t iTensor = 0; iTensor < outputs->getSize(); ++iTensor) {
            const dsis::IBatchBuffer* outTensor = outputs->getBuffer(iTensor);
            INFER_ASSERT(outTensor);
            auto desc = outTensor->getBufDesc();
            LOG_DEBUG("out tensor: %s, desc: %s \n", CSTR(desc.name), strOfBufDesc(desc).c_str());
            tensors.emplace(desc.name, outTensor);
        }

        // boxes dim format is [batch, n_candidates, 4]
        auto boxes = tensors["bboxes"];
        // scores dim format is [batch, n_candidates, n_classes]
        auto scores = tensors["scores"];
        INFER_ASSERT(boxes && scores);
        auto boxDesc = boxes->getBufDesc();
        auto scoreDesc = scores->getBufDesc();
        INFER_ASSERT(boxDesc.dims.numDims == 3 && boxDesc.dims.d[2] == 4);
        INFER_ASSERT(
            scoreDesc.dims.numDims == 3 && scoreDesc.dims.d[2] == (int)kClassLabels.size());
        float* boxesPtr = (float*)boxes->getBufPtr(0);
        float* scoresPtr = (float*)scores->getBufPtr(0);

        // Number of boxes and classes inferred from score tensor shape
        int numBoxes = 0;
        int numClasses = 0;

        if (scoreDesc.dims.numDims == 3) {
            numBoxes = scoreDesc.dims.d[1];
            numClasses = scoreDesc.dims.d[2];
        } else {
            LOG_ERROR("Unexpected tensor rank for scores");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }

        for (int i = 0; i < numBoxes; ++i) {
            // Get class scores for the i bbox
            const float* scorePtr = scoresPtr + i * numClasses;

            // Get bbox data in (rxc, ryc, rw, rh) format
            const float* bbox = boxesPtr + i * 4;

            // Find class with highest probability (argmax)
            int maxIndex = 0;
            float maxProb = scorePtr[0];
            for (int c = 1; c < numClasses; ++c) {
                if (scorePtr[c] > maxProb) {
                    maxProb = scorePtr[c];
                    maxIndex = c;
                }
            }

            // Apply detection threshold
            if (maxProb < DETECTION_THRESH) continue;

            // Extract and convert relative bbox to absolute corner format
            float rxc = bbox[0];
            float ryc = bbox[1];
            float rw  = bbox[2];
            float rh  = bbox[3];

            float netW = NET_INPUT_WIDTH;
            float netH = NET_INPUT_HEIGHT;

            float x1 = (rxc - rw / 2.f) * netW;
            float y1 = (ryc - rh / 2.f) * netH;
            float x2 = (rxc + rw / 2.f) * netW;
            float y2 = (ryc + rh / 2.f) * netH;

            x1 = clamp(x1, 0, netW);
            y1 = clamp(y1, 0, netH);
            x2 = clamp(x2, 0, netW);
            y2 = clamp(y2, 0, netH);

            NvDsInferObjectDetectionInfo obj;
            obj.left = x1;
            obj.width = clamp(x2 - x1, 0, netW);
            obj.top = y1;
            obj.height = clamp(y2 - y1, 0, netH);

            if (obj.width < 1 || obj.height < 1) {
                continue;
            }

            obj.classId = maxIndex;
            obj.detectionConfidence = maxProb;

            LOG_DEBUG(
                "cid: %u, obj [%.2f, %.2f, %.2f, %.2f], score:%.2f\n", obj.classId,
                obj.top, obj.left, rw, rh, maxProb);

            parsedBboxes[0].emplace_back(obj);
        }

        // Apply NMS filter
        auto nmsFiltered = applyNMS(parsedBboxes[0]);

        attachObjMeta(inOptions, nmsFiltered, 0);

        return NVDSINFER_SUCCESS;
    }

    /** override function
     * Receiving errors if anything wrong inside lowlevel lib
     */
    void notifyError(NvDsInferStatus s) override
    {
        std::unique_lock<std::mutex> locker(_streamMutex);
        _streamFeedback.clear();
    }

private:
    /**
     * attach bounding boxes into NvDsBatchMeta and NvDsFrameMeta
     */
    NvDsInferStatus attachObjMeta(
        const dsis::IOptions* inOptions, const std::vector<NvDsInferObjectDetectionInfo>& objs,
        uint32_t batchIdx);

    std::vector<NvDsInferObjectDetectionInfo> applyNMS(
        const std::vector<NvDsInferObjectDetectionInfo>& inputObjs);
};

/** Implementation to Create a custom processor for DeepStream Triton
 * plugin(nvinferserver) to do custom extra input preprocess and custom
 * postprocess on triton based models.
 */
extern "C" {
dsis::IInferCustomProcessor* 
CreateInferServerCustomProcess(const char* config, uint32_t configLen) 
{
    return new YoloV8CustomPostprocessor();
} 
}
/**
 * attach bounding boxes into NvDsBatchMeta and NvDsFrameMeta
 */
NvDsInferStatus
YoloV8CustomPostprocessor::attachObjMeta(
    const dsis::IOptions* inOptions, const std::vector<NvDsInferObjectDetectionInfo>& detectObjs,
    uint32_t batchIdx)
{
    INFER_ASSERT(inOptions);
    GstBuffer* gstBuf = nullptr;
    NvDsBatchMeta* batchMeta = nullptr;
    std::vector<NvDsFrameMeta*> frameMetaList;
    NvBufSurface* bufSurf = nullptr;
    std::vector<NvBufSurfaceParams*> surfParamsList;
    int64_t unique_id = 0;

    // get GstBuffer
    if (inOptions->hasValue(OPTION_NVDS_GST_BUFFER)) {
        INFER_ASSERT(inOptions->getObj(OPTION_NVDS_GST_BUFFER, gstBuf) == NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(gstBuf);

    // get NvBufSurface
    if (inOptions->hasValue(OPTION_NVDS_BUF_SURFACE)) {
        INFER_ASSERT(inOptions->getObj(OPTION_NVDS_BUF_SURFACE, bufSurf) == NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(bufSurf);

    // get NvDsBatchMeta
    if (inOptions->hasValue(OPTION_NVDS_BATCH_META)) {
        INFER_ASSERT(inOptions->getObj(OPTION_NVDS_BATCH_META, batchMeta) == NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(batchMeta);

    // get all frame meta list into vector<NvDsFrameMeta*>
    if (inOptions->hasValue(OPTION_NVDS_FRAME_META_LIST)) {
        INFER_ASSERT(
            inOptions->getValueArray(OPTION_NVDS_FRAME_META_LIST, frameMetaList) ==
            NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(batchIdx < frameMetaList.size());  // batchsize

    // get unique_id
    if (inOptions->hasValue(OPTION_NVDS_UNIQUE_ID)) {
        INFER_ASSERT(inOptions->getInt(OPTION_NVDS_UNIQUE_ID, unique_id) == NVDSINFER_SUCCESS);
    }

    // get all surface params list into vector<NvBufSurfaceParams*>
    if (inOptions->hasValue(OPTION_NVDS_BUF_SURFACE_PARAMS_LIST)) {
        INFER_ASSERT(
            inOptions->getValueArray(OPTION_NVDS_BUF_SURFACE_PARAMS_LIST, surfParamsList) ==
            NVDSINFER_SUCCESS);
    }
    INFER_ASSERT(batchIdx < surfParamsList.size());  // batchsize

    NvDsFrameMeta* frameMeta = frameMetaList[batchIdx];
    guint frame_width = frameMeta->source_frame_width;
    guint frame_height = frameMeta->source_frame_height;

    // scale
    float netW = NET_INPUT_WIDTH;
    float netH = NET_INPUT_HEIGHT;
    float scale = std::min(netW / frame_width, netH / frame_height);
    float pad_x = (netW - frame_width * scale) / 2;
    float pad_y = (netH - frame_height * scale) / 2;

    for (const auto& obj : detectObjs) {
        NvDsObjectMeta* objMeta = nvds_acquire_obj_meta_from_pool(batchMeta);
        objMeta->unique_component_id = unique_id;
        objMeta->confidence = obj.detectionConfidence;

        /* This is an untracked object. Set tracking_id to -1. */
        objMeta->object_id = UNTRACKED_OBJECT_ID;
        objMeta->class_id = obj.classId;

        float left   = (obj.left   - pad_x) / scale;
        float top    = (obj.top    - pad_y) / scale;
        float width  = obj.width  / scale;
        float height = obj.height / scale;

        left = std::max(0.f, std::min(left, (float)frame_width  - 1.f));
        top  = std::max(0.f, std::min(top,  (float)frame_height - 1.f));
        width = std::min(width, (float)(frame_width - left));
        height = std::min(height, (float)(frame_height - top));

        NvOSD_RectParams& rect_params = objMeta->rect_params;
        NvOSD_TextParams& text_params = objMeta->text_params;

        rect_params.left = left;
        rect_params.top = top;
        rect_params.width = width;
        rect_params.height = height;

        /* Border of width 3. */
        rect_params.border_width = 3;
        rect_params.has_bg_color = 0;
        rect_params.border_color = (NvOSD_ColorParams){1, 0, 0, 1};

        /* display_text requires heap allocated memory. */
        if (obj.classId < kClassLabels.size()) {
            text_params.display_text = g_strdup(kClassLabels[obj.classId].c_str());
            strncpy(objMeta->obj_label, kClassLabels[obj.classId].c_str(), MAX_LABEL_SIZE - 1);
            objMeta->obj_label[MAX_LABEL_SIZE - 1] = 0;
        }
        /* Display text above the left top corner of the object. */
        text_params.x_offset = rect_params.left;
        text_params.y_offset = rect_params.top > 10 ? rect_params.top - 10 : rect_params.top;
        /* Set black background for the text. */
        text_params.set_bg_clr = 1;
        text_params.text_bg_clr = (NvOSD_ColorParams){0, 0, 0, 1};
        /* Font face, size and color. */
        text_params.font_params.font_name = (gchar*)"Serif";
        text_params.font_params.font_size = 11;
        text_params.font_params.font_color = (NvOSD_ColorParams){1, 1, 1, 1};

        nvds_acquire_meta_lock(batchMeta);
        nvds_add_obj_meta_to_frame(frameMeta, objMeta, nullptr);
        frameMeta->bInferDone = TRUE;
        nvds_release_meta_lock(batchMeta);
    }

    return NVDSINFER_SUCCESS;
}

std::vector<NvDsInferObjectDetectionInfo>
YoloV8CustomPostprocessor::applyNMS(
    const std::vector<NvDsInferObjectDetectionInfo>& inputObjs)
{
    std::vector<NvDsInferObjectDetectionInfo> nmsFiltered;

    std::vector<NvDsInferObjectDetectionInfo> sortedObjs = inputObjs;
    std::sort(sortedObjs.begin(), sortedObjs.end(),
              [](const auto& a, const auto& b) {
                  return a.detectionConfidence > b.detectionConfidence;
              });

    std::vector<bool> suppressed(sortedObjs.size(), false);

    for (size_t i = 0; i < sortedObjs.size(); ++i) {
        if (suppressed[i]) continue;
        const auto& curr = sortedObjs[i];
        nmsFiltered.push_back(curr);

        for (size_t j = i + 1; j < sortedObjs.size(); ++j) {
            if (suppressed[j]) continue;
            if (curr.classId == sortedObjs[j].classId &&
                IoU(curr, sortedObjs[j]) > NMS_IOU_THRESH) {
                suppressed[j] = true;
            }
        }
    }

    return nmsFiltered;
}
