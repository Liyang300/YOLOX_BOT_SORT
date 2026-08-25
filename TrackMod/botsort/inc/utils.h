#pragma once

#include "DataType.h"

/**
 * @brief Calculate the cosine distance between two feature vectors
 * 
 * @param x Feature vector 1
 * @param y Feature vector 2
 * @return float Cosine distance (1 - cosine similarity)
 */
inline float cosine_distance(const std::unique_ptr<FeatureVector> &x,
                             const std::shared_ptr<FeatureVector> &y)
{
    return 1.0f - (x->dot(*y) / (x->norm() * y->norm() + 1e-5f));
}


/**
 * @brief Calculate the euclidean distance between two feature vectors
 * 
 * @param x Feature vector 1
 * @param y Feature vector 2
 * @return float Euclidean distance
 */
inline float euclidean_distance(const std::unique_ptr<FeatureVector> &x,
                                const std::shared_ptr<FeatureVector> &y)
{
    return (x->transpose() - y->transpose()).norm();
}


/**
 * @brief Calculate the intersection over union (IoU) between two bounding boxes
 * 
 * @param tlwh_a Bounding box 1 in the format (top left x, top left y, width, height)
 * @param tlwh_b Bounding box 2 in the format (top left x, top left y, width, height)
 * @return float IoU
 */
inline float iou(const std::vector<float> &tlwh_a,
                 const std::vector<float> &tlwh_b)
{
     // 计算交集矩形
    float x1 = std::max(tlwh_a[0], tlwh_b[0]);
    float y1 = std::max(tlwh_a[1], tlwh_b[1]);
    float x2 = std::min(tlwh_a[0] + tlwh_a[2], tlwh_b[0] + tlwh_b[2]);
    float y2 = std::min(tlwh_a[1] + tlwh_a[3], tlwh_b[1] + tlwh_b[3]);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;

    // 计算各自面积
    float area_a = tlwh_a[2] * tlwh_a[3];
    float area_b = tlwh_b[2] * tlwh_b[3];
    float union_area = area_a + area_b - inter;

    // 防止除零
    if (union_area <= 0) return 0.0f;

    return inter / union_area;
}

double lapjv(CostMatrix &cost, std::vector<int> &rowsol,
             std::vector<int> &colsol, bool extend_cost = false,
             float cost_limit = std::numeric_limits<float>::max(),
             bool return_cost = true);