#include "KalmanFilter.h"

#include <eigen3/Eigen/Cholesky>

namespace bot_kalman
{
KalmanFilter::KalmanFilter(double dt)
    : _std_weight_position(1.0 / 5), _std_weight_velocity(1.0 / 8)
{

    _init_kf_matrices(dt);
}

void KalmanFilter::_init_kf_matrices(double dt)
{
    // This is a 4x8 matrix that maps the 8-dimensional state space vector [x, y, w, h, vx, vy, vw, vh]
    // to the 4-dimensional measurement space vector [x, y, w, h]
    _measurement_matrix.setIdentity();

    // This is a 8x8 matrix that defines the state transition function.
    // It maps the current state space vector to the next state space vector.
    _state_transition_matrix.setIdentity();
    for (Eigen::Index i = 0; i < 4; i++)
    {
        _state_transition_matrix(i, i + 4) = static_cast<float>(dt);
    }
}

KFDataStateSpace KalmanFilter::init(const DetVec &measurement) const
{
    constexpr float init_velocity = 0.0;

    KFStateSpaceVec mean_state_space;
    mean_state_space.head<4>() = measurement.head<4>();
    mean_state_space.tail<4>().setConstant(init_velocity);


    float w = measurement(2), h = measurement(3);
    KFStateSpaceVec std_dev;
    std_dev.head<4>() =
            2 * _std_weight_position * (Eigen::Vector4f(w, h, w, h).array());
    std_dev.tail<4>() =
            10 * _std_weight_velocity * (Eigen::Vector4f(w, h, w, h).array());

    KFStateSpaceMatrix covariance =
            std_dev.array().square().matrix().asDiagonal();
    return {mean_state_space, covariance};
}

void KalmanFilter::predict(KFStateSpaceVec &mean,
                           KFStateSpaceMatrix &covariance)
{
    Eigen::VectorXf std_combined;
    std_combined.resize(KALMAN_STATE_SPACE_DIM);
    std_combined << mean(2), mean(3), mean(2), mean(3), mean(2), mean(3),
            mean(2), mean(3);
    std_combined.head<4>().array() *= _std_weight_position;
    std_combined.tail<4>().array() *= _std_weight_velocity;
    KFStateSpaceMatrix motion_cov =
            std_combined.array().square().matrix().asDiagonal();

    mean = _state_transition_matrix.lazyProduct(mean.transpose());
    covariance = (_state_transition_matrix * covariance)
                         .lazyProduct(_state_transition_matrix.transpose()) +
                 motion_cov;
}

KFDataMeasurementSpace
KalmanFilter::project(const KFStateSpaceVec &mean,
                      const KFStateSpaceMatrix &covariance) const
{
//     KFMeasSpaceVec innovation_cov =
//             (_std_weight_position *
//              Eigen::Vector4f(mean(2), mean(3), mean(2), mean(3)))
//                     .array()
//                     .square()
//                     .matrix();


// 原代码（统一噪声）
    // KFMeasSpaceVec innovation_cov =
    //         (_std_weight_position *
    //          Eigen::Vector4f(mean(2), mean(3), mean(2), mean(3)))
    //                 .array()
    //                 .square()
    //                 .matrix();

    // 修改为：位置使用标准噪声，尺寸使用放大 5~10 倍的噪声
    float size_noise_factor = 8.0;  // 可调，建议 5~10
    Eigen::Array4f weight_vec;
        weight_vec << _std_weight_position,
                _std_weight_position,
                _std_weight_position * size_noise_factor,
                _std_weight_position * size_noise_factor;

    KFMeasSpaceVec innovation_cov =
        (weight_vec * Eigen::Vector4f(mean(2), mean(3), mean(2), mean(3)).array())
                .square()
                .matrix();

    //或者
//     Eigen::Vector4f weight_vec;
//     weight_vec << _std_weight_position,
//               _std_weight_position,
//               _std_weight_position * size_noise_factor,
//               _std_weight_position * size_noise_factor;

//      KFMeasSpaceVec innovation_cov =
//     (weight_vec.cwiseProduct(Eigen::Vector4f(mean(2), mean(3), mean(2), mean(3))))
//         .array()
//         .square()
//         .matrix();





    KFMeasSpaceMatrix innovation_cov_diag = innovation_cov.asDiagonal();

    KFMeasSpaceVec mean_projected =
            _measurement_matrix.lazyProduct(mean.transpose());
    KFMeasSpaceMatrix covariance_projected =
            (_measurement_matrix * covariance)
                    .lazyProduct(_measurement_matrix.transpose()) +
            innovation_cov_diag;
    return {mean_projected, covariance_projected};
}

KFDataStateSpace KalmanFilter::update(const KFStateSpaceVec &mean,
                                      const KFStateSpaceMatrix &covariance,
                                      const DetVec &measurement)
{
    KFDataMeasurementSpace projected = project(mean, covariance);
    KFMeasSpaceVec projected_mean = projected.first;
    KFMeasSpaceMatrix projected_covariance = projected.second;

    Eigen::Matrix<float, KALMAN_MEASUREMENT_SPACE_DIM, KALMAN_STATE_SPACE_DIM>
            B = (covariance * _measurement_matrix.transpose()).transpose();
    Eigen::Matrix<float, KALMAN_STATE_SPACE_DIM, KALMAN_MEASUREMENT_SPACE_DIM>
            kalman_gain = (projected_covariance.llt().solve(B)).transpose();
    Eigen::Matrix<float, 1, KALMAN_MEASUREMENT_SPACE_DIM> innovation =
            measurement - projected_mean;

    KFStateSpaceVec mean_updated = mean + innovation * kalman_gain.transpose();
    KFStateSpaceMatrix covariance_updated =
            covariance -
            kalman_gain * projected_covariance * kalman_gain.transpose();
    return {mean_updated, covariance_updated};
}


Eigen::Matrix<float, 1, Eigen::Dynamic> KalmanFilter::gating_distance(
        const KFStateSpaceVec &mean, const KFStateSpaceMatrix &covariance,
        const std::vector<DetVec> &measurements, bool only_position) const
{
    KFDataMeasurementSpace projected = this->project(mean, covariance);
    KFMeasSpaceVec projected_mean = projected.first;
    KFMeasSpaceMatrix projected_covariance = projected.second;

    if (only_position)
    {
        projected_mean.tail<4>().setZero();
        projected_covariance.bottomRightCorner<2, 2>().setZero();
    }

    Eigen::LLT<Eigen::MatrixXf> lltOfProjectedCovariance(projected_covariance);
    Eigen::Matrix<float, 1, Eigen::Dynamic> mahalanobis_distances(
            measurements.size());
    mahalanobis_distances.setZero();

    for (Eigen::Index i = 0; i < measurements.size(); i++)
    {
        Eigen::VectorXf diff = measurements[i] - projected_mean;
        // Solve for y in Ly = diff using forward substitution, more efficient than computing the inverse
        Eigen::VectorXf y = lltOfProjectedCovariance.matrixL().solve(diff);
        // Mahalanobis distance is the norm of y
        mahalanobis_distances(i) = y.squaredNorm();
    }

    return mahalanobis_distances;
}
}// namespace bot_kalman