/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "nvdsinfer_custom_impl.h"

#include "utils.h"

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

static NvDsInferParseObjectInfo
convertBBox(const float& bx1, const float& by1, const float& bx2, const float& by2, const uint& netW, const uint& netH)
{
  NvDsInferParseObjectInfo b;

  float x1 = bx1;
  float y1 = by1;
  float x2 = bx2;
  float y2 = by2;

  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  return b;
}

static void
addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, const uint& netW, const uint& netH,
    const int maxIndex, const float maxProb, std::vector<NvDsInferParseObjectInfo>& binfo)
{
  NvDsInferParseObjectInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);

  if (bbi.width < 1 || bbi.height < 1) {
    return;
  }

  bbi.detectionConfidence = maxProb;
  bbi.classId = maxIndex;
  binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeTensorYolo(const float* output, const uint& outputSize, const uint& netW, const uint& netH,
    const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;

  for (uint b = 0; b < outputSize; ++b) {
    float maxProb = output[b * 6 + 4];
    int maxIndex = (int) output[b * 6 + 5];

    if (maxProb < preclusterThreshold[maxIndex]) {
      continue;
    }

    float bx1 = output[b * 6 + 0];
    float by1 = output[b * 6 + 1];
    float bx2 = output[b * 6 + 2];
    float by2 = output[b * 6 + 3];

    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, binfo);
  }

  return binfo;
}

/**
 * @brief Decodes the output tensors of a YOLO custom model with separate layers for class scores and bounding boxes.
 *
 * This function is intended for the YOLO custom model where:
 * 
 *   - "scores" layer [numBoxes x numClasses]: contains the class scores for each detected bbox.
 *     - The predicted class for each bbox is computed as: argmax(scores[i])
 *     - The predicted class confidence is computed as: max(scores[i])
 *
 *   - "bboxes" layer [numBoxes x 4]: contains each bbox represented as:
 *     - rxc: center X
 *     - ryc: center Y
 *     - rw: width
 *     - rh: height
 *
 *     - Upper-left corner bbox:    ( (rxc - rw/2) * input_width,  (ryc - rh/2) * input_height )
 *     - Bottom-right corner bbox:  ( (rxc + rw/2) * input_width,  (ryc + rh/2) * input_height )
 *
 * Each bounding box is parsed, filtered by a class-specific confidence threshold, and added to the result list.
 *
 * @param scoresPtr Pointer to scores tensor [numBoxes x numClasses].
 * @param bboxesPtr Pointer to bounding boxes tensor [numBoxes x 4], in (rxc, ryc, rw, rh) format.
 * @param numBoxes Number of bounding boxes.
 * @param numClasses Number of classes.
 * @param netW Width of the input network, used to scale bbox coordinates.
 * @param netH Height of the input network, used to scale bbox coordinates.
 * @param preclusterThreshold Vector of per-class thresholds to filter low-confidence detections.
 *
 * @return Vector of valid detections parsed into NvDsInferParseObjectInfo format.
 */
static std::vector<NvDsInferParseObjectInfo>
decodeTensorCustomYoloDynamic(const float* scoresPtr, const float* bboxesPtr,
                       int numBoxes, int numClasses,
                       const uint& netW, const uint& netH,
                       const std::vector<float>& preclusterThreshold)
{
    std::vector<NvDsInferParseObjectInfo> binfo;

    for (int i = 0; i < numBoxes; ++i) {
        // Get class scores for the i bbox
        const float* scoreData = scoresPtr + i * numClasses;

        // Get bbox data in (rxc, ryc, rw, rh) format
        const float* bboxData = bboxesPtr + i * 4;

        // Find class with highest probability (argmax)
        int maxIndex = 0;
        float maxProb = scoreData[0];
        for (int c = 1; c < numClasses; ++c) {
            if (scoreData[c] > maxProb) {
                maxProb = scoreData[c];
                maxIndex = c;
            }
        }

        // Apply class-specific threshold
        if (maxProb < preclusterThreshold[maxIndex]) continue;

        // Extract and convert relative bbox to absolute corner format
        float rxc = bboxData[0];
        float ryc = bboxData[1];
        float rw  = bboxData[2];
        float rh  = bboxData[3];

        float x1 = (rxc - rw / 2.f) * netW;
        float y1 = (ryc - rh / 2.f) * netH;
        float x2 = (rxc + rw / 2.f) * netW;
        float y2 = (ryc + rh / 2.f) * netH;

        addBBoxProposal(x1, y1, x2, y2, netW, netH, maxIndex, maxProb, binfo);

    }

    return binfo;
}

static bool
NvDsInferParseCustomYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> objects;

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  const uint outputSize = output.inferDims.d[0];

  std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorYolo((const float*) (output.buffer), outputSize,
      networkInfo.width, networkInfo.height, detectionParams.perClassPreclusterThreshold);

  objects.insert(objects.end(), outObjs.begin(), outObjs.end());

  objectList = objects;

  return true;
}

/**
 * @brief Parses the output of a custom YOLO model and separate output layers for scores and 
 *        bounding boxes.
 *
 * This function is used for a custom yolo model with two outputs:
 *   - A score tensor with shape [num_boxes, num_classes]
 *   - A bounding box tensor with shape [num_boxes, 4], where each box is in (cx, cy, w, h) format
 *
 * It identifies the most probable class for each box, checks if the confidence exceeds the
 * class-specific threshold, and constructs bounding boxes accordingly.
 *
 * @param outputLayersInfo  Vector containing output layer information (scores and bboxes).
 * @param networkInfo       Network dimensions used to scale bounding boxes.
 * @param detectionParams   Contains per-class pre-cluster thresholds.
 * @param objectList        Output vector where parsed objects will be stored.
 * 
 * @return true if parsing succeeds, false otherwise.
 */
static bool
NvDsInferParseCustomYoloDynamic(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Ensure there are at least two output layers: one for scores and one for bounding boxes
    if (outputLayersInfo.size() < 2) {
        std::cerr << "ERROR: Expected at least 2 output layers: scores and bboxes." << std::endl;
        return false;
    }

    // Get the score and bounding box layers
    const NvDsInferLayerInfo& scores = outputLayersInfo[0];
    const NvDsInferLayerInfo& bboxes = outputLayersInfo[1];

    std::vector<NvDsInferParseObjectInfo> objects;

    // Extract tensor dimensions
    const auto& dimsScores = scores.inferDims;  //  [8400, 80]
    const auto& dimsBboxes = bboxes.inferDims;  //  [8400, 4]

    // Number of boxes and classes inferred from score tensor shape
    int numBoxes = dimsScores.d[0];             //  8400
    int numClasses = dimsScores.d[1];           //  80

    // Decode both tensors into object bounding boxes
    std::vector<NvDsInferParseObjectInfo> outObjs = decodeTensorCustomYoloDynamic(
        (const float*) (scores.buffer),
        (const float*) (bboxes.buffer),
        numBoxes, numClasses,
        networkInfo.width, networkInfo.height,
        detectionParams.perClassPreclusterThreshold);

    objects.insert(objects.end(), outObjs.begin(), outObjs.end());

    objectList = objects;

    return true;
}

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);

/**
 * @brief Externally exposed entry point for DeepStream to use the dynamic YOLO custom parser.
 *
 * This function is registered via `parse-bbox-func-name=NvDsInferParseYoloDynamic` in the configuration file.
 * It simply delegates the parsing task to the internal `NvDsInferParseCustomYoloDynamic` function,
 * which is designed to handle YOLO custom models with:
 *   - Dynamic batch sizes,
 *   - Separate outputs for scores and bounding boxes,
 *   - Bounding boxes in (rxc, ryc, rw, rh) format.
 *
 * @param outputLayersInfo Vector containing output layers.
 * @param networkInfo Network input dimensions.
 * @param detectionParams Contains confidence thresholds per class.
 * @param objectList Output list of parsed object detections.
 *
 * @return true if parsing is successful, false otherwise.
 */
extern "C" bool
NvDsInferParseYoloDynamic(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloDynamic(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloDynamic);
