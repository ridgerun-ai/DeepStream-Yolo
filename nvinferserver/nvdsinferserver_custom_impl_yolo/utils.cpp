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

#include "utils.h"

#include <iomanip>
#include <algorithm>
#include <cassert>
#include <experimental/filesystem>

float
clamp(const float val, const float minVal, const float maxVal)
{
  assert(minVal <= maxVal);
  return std::min(maxVal, std::max(minVal, val));
}

bool
fileExists(const std::string fileName, bool verbose)
{
  if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(fileName))) {
    if (verbose) {
      std::cout << "\nFile does not exist: " << fileName << std::endl;
    }
    return false;
  }
  return true;
}

float IoU(const NvDsInferObjectDetectionInfo& a, const NvDsInferObjectDetectionInfo& b) {
    float x1 = std::max(a.left, b.left);
    float y1 = std::max(a.top, b.top);
    float x2 = std::min(a.left + a.width, b.left + b.width);
    float y2 = std::min(a.top + a.height, b.top + b.height);

    float interArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float unionArea = a.width * a.height + b.width * b.height - interArea;

    return unionArea > 0 ? interArea / unionArea : 0.0f;
}