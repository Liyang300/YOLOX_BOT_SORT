#include "BoTSORT.h"

#include <optional>
#include <unordered_set>

#include <opencv2/imgproc.hpp>

#include "DataType.h"
#include "INIReader.h"
#include "matching.h"
#include "profiler.h"

namespace
{
template<typename T>
bool requires_load(const Config<T> &config)
{
    return std::holds_alternative<std::string>(config) &&
           !std::get<std::string>(config).empty();
}

template<typename T>
bool not_empty(const Config<T> &config)
{
    bool has_config = !std::holds_alternative<std::monostate>(config);
    bool is_non_empty = !(std::holds_alternative<std::string>(config) &&
                          std::get<std::string>(config).empty());

    return has_config && is_non_empty;
}

template<typename T>
T fetch_config(const Config<T> &config,
               std::function<T(const std::string &)> loader)
{
    if (std::holds_alternative<T>(config))
    {
        return std::get<T>(config);
    }

    if (requires_load(config))
    {
        return loader(std::get<std::string>(config));
    }

    throw std::runtime_error("Config is empty");
}
}// namespace

BoTSORT::BoTSORT(const Config<TrackerParams> &tracker_config,
                 const Config<GMC_Params> &gmc_config)
{
    auto tracker_params = fetch_config<TrackerParams>(
            tracker_config, TrackerParams::load_config);
    _load_params_from_config(tracker_params);

    // Tracker module
    _frame_id = 0;
    _buffer_size = static_cast<uint8_t>(_frame_rate / 30.0 * _track_buffer);
    _max_time_lost = _buffer_size;
    _kalman_filter = std::make_unique<KalmanFilter>(
            static_cast<double>(1.0 / _frame_rate));


    // Re-ID module, load visual feature extractor here
    // if (_reid_enabled && not_empty(reid_config) &&
    //     reid_onnx_model_path.size() > 0)
    // {
    //     auto reid_params =
    //             fetch_config<ReIDParams>(reid_config, ReIDParams::load_config);
    //     _reid_model =
    //             std::make_unique<ReIDModel>(reid_params, reid_onnx_model_path);
    // }
    // else
    // {
    //     std::cout << "Re-ID module disabled" << std::endl;
    //     _reid_enabled = false;
    // }

    // Global motion compensation module
    if (_gmc_enabled && not_empty(gmc_config))
    {
        auto gmc_params = fetch_config<
                GMC_Params>(gmc_config, [this](const std::string &config_path) {
            return GMC_Params::load_config(
                    GlobalMotionCompensation::GMC_method_map[_gmc_method_name],
                    config_path);
        });
        _gmc_algo = std::make_unique<GlobalMotionCompensation>(gmc_params);
    }
    else
    {
        std::cout << "GMC disabled" << std::endl;
        _gmc_enabled = false;
    }
}


std::vector<std::shared_ptr<Track>>
BoTSORT::track(const std::vector<Detection> &detections, const cv::Mat &frame)
{
    std::cout << "All detections confidence: ";
    for (const auto& det : detections) {
        std::cout << det.confidence << " ";
    }
    std::cout << std::endl;
    std::cout << "_track_high_thresh = " << _track_high_thresh << std::endl;

    PROFILE_FUNCTION();
    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////
    // For all detections, extract features, create tracks and classify on the segregate of confidence
    _frame_id++;
    std::vector<std::shared_ptr<Track>> activated_tracks, refind_tracks;
    std::vector<std::shared_ptr<Track>> detections_high_conf,
            detections_low_conf;
    detections_low_conf.reserve(detections.size()),
            detections_high_conf.reserve(detections.size());

    if (!detections.empty())
    {
        //std::cout << "Creating track objects for detections..." << std::endl;
        for (Detection &detection:
             const_cast<std::vector<Detection> &>(detections))
        {
            detection.bbox_tlwh.x = std::max(0.0f, detection.bbox_tlwh.x);
            detection.bbox_tlwh.y = std::max(0.0f, detection.bbox_tlwh.y);
            detection.bbox_tlwh.width =
                    std::min(static_cast<float>(frame.cols - 1),
                             detection.bbox_tlwh.width);
            detection.bbox_tlwh.height =
                    std::min(static_cast<float>(frame.rows - 1),
                             detection.bbox_tlwh.height);

            std::shared_ptr<Track> tracklet;
            std::vector<float> tlwh = {
                    detection.bbox_tlwh.x, detection.bbox_tlwh.y,
                    detection.bbox_tlwh.width, detection.bbox_tlwh.height};

            if (detection.confidence > _track_low_thresh)
            {
                // if (_reid_enabled)
                // {
                //     FeatureVector embedding =
                //             _extract_features(frame, detection.bbox_tlwh);
                //     tracklet = std::make_shared<Track>(
                //             tlwh, detection.confidence, detection.class_id,
                //             embedding);
                // }
                // else
                //     tracklet = std::make_shared<Track>(
                //             tlwh, detection.confidence, detection.class_id);

                
                tracklet = std::make_shared<Track>(tlwh, detection.confidence, detection.class_id);
                if (!tracklet) { std::cerr << "ERROR: tracklet is null!" << std::endl; exit(1); }
                

                if (detection.confidence >= _track_high_thresh)
                    detections_high_conf.push_back(tracklet);
                else
                    detections_low_conf.push_back(tracklet);
            }
        }
    }

  
    // ---- 分离未确认和已跟踪轨迹 ----
   
    // Segregate tracks in unconfirmed and tracked tracks
    std::vector<std::shared_ptr<Track>> unconfirmed_tracks, tracked_tracks;
    for (const std::shared_ptr<Track> &track: _tracked_tracks)
    {
        if (!track->is_activated)
        {
            unconfirmed_tracks.push_back(track);
        }
        else
        {
            tracked_tracks.push_back(track);
        }
    }
    ////////////////// CREATE TRACK OBJECT FOR ALL THE DETECTIONS //////////////////


    ////////////////// Apply KF predict and GMC before running association algorithm //////////////////
    // Merge currently tracked tracks and lost tracks

    // ---- 合并轨迹池 ----
   
    std::vector<std::shared_ptr<Track>> tracks_pool;
    tracks_pool = _merge_track_lists(tracked_tracks, _lost_tracks);

    // Predict the location of the tracks with KF (even for lost tracks)

    Track::multi_predict(tracks_pool, *_kalman_filter);



    // Estimate camera motion and apply camera motion compensation
    if (_gmc_enabled)
    {
        HomographyMatrix H = _gmc_algo->apply(frame, detections);
        Track::multi_gmc(tracks_pool, H);
        Track::multi_gmc(unconfirmed_tracks, H);
    }
    ////////////////// Apply KF predict and GMC before running association algorithm //////////////////


    ////////////////// ASSOCIATION ALGORITHM STARTS HERE //////////////////
    ////////////////// First association, with high score detection boxes //////////////////
    // Find IoU distance between all tracked tracks and high confidence detections
    CostMatrix iou_dists, raw_emd_dist, iou_dists_mask_1st_association,
            emd_dist_mask_1st_association;

//=============================================================================
    if (!tracks_pool.empty() && !detections.empty()) 
    {
        auto track_tlwh = tracks_pool[0]->get_tlwh();
        auto det_tlwh = detections[0].bbox_tlwh;
        std::cout << "Track tlwh: " << track_tlwh[0] << ", " << track_tlwh[1] << ", " 
                << track_tlwh[2] << ", " << track_tlwh[3] << std::endl;
        std::cout << "Det tlwh: " << det_tlwh.x << ", " << det_tlwh.y << ", " 
                << det_tlwh.width << ", " << det_tlwh.height << std::endl;
    }



    // 第一次关联
    std::tie(iou_dists, iou_dists_mask_1st_association) = 
        iou_distance(tracks_pool, detections_high_conf, _proximity_thresh);
    

    
    fuse_score(iou_dists, detections_high_conf);
    

    
    fuse_motion(*_kalman_filter, iou_dists, tracks_pool, detections_high_conf, _lambda, true);

    std::cout << "iou_dists matrix (rows=" << iou_dists.rows() << ", cols=" << iou_dists.cols() << "):\n";
    for (int i = 0; i < iou_dists.rows(); ++i) {
        for (int j = 0; j < iou_dists.cols(); ++j) {
            std::cout << iou_dists(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "match_thresh = " << _match_thresh << std::endl;
    std::cout << "Calling linear_assignment..." << std::endl;

    std::cout << "[First Association] Calling linear_assignment..." << std::endl;
    AssociationData first_associations = linear_assignment(iou_dists, _match_thresh);
    std::cout << "[First Association] matches=" << first_associations.matches.size() << std::endl;
    
//===================================================================================
   

    // Update the tracks with the associated detections
    
    for (const auto &match : first_associations.matches)
    {
        
        const std::shared_ptr<Track> &track = tracks_pool[match.first];
        const std::shared_ptr<Track> &detection = detections_high_conf[match.second];

        if (track->state == TrackState::Tracked)
        {
            track->update(*_kalman_filter, *detection, _frame_id);
            activated_tracks.push_back(track);
        }
        else
        {
            track->re_activate(*_kalman_filter, *detection, _frame_id, false);
            refind_tracks.push_back(track);
        }
    }
    

    // ==================== 第二次关联（低置信度检测框） ====================
    
    std::vector<std::shared_ptr<Track>> unmatched_tracks_after_1st_association;
    for (int track_idx : first_associations.unmatched_track_indices)
    {
        const std::shared_ptr<Track> &track = tracks_pool[track_idx];
        if (track->state == TrackState::Tracked)
        {
            unmatched_tracks_after_1st_association.push_back(track);
        }
    }
   

    CostMatrix iou_dists_second;
    iou_dists_second = iou_distance(unmatched_tracks_after_1st_association,
                                    detections_low_conf);
  
    std::cout << "[Second Association] Calling linear_assignment..." << std::endl;
    AssociationData second_associations = linear_assignment(iou_dists_second, 0.5);
    std::cout << "[Second Association] matches=" << second_associations.matches.size() << std::endl;
   
    for (const auto &match : second_associations.matches)
    {
       
        const std::shared_ptr<Track> &track = unmatched_tracks_after_1st_association[match.first];
        const std::shared_ptr<Track> &detection = detections_low_conf[match.second];

        if (track->state == TrackState::Tracked)
        {
            track->update(*_kalman_filter, *detection, _frame_id);
            activated_tracks.push_back(track);
        }
        else
        {
            track->re_activate(*_kalman_filter, *detection, _frame_id, false);
            refind_tracks.push_back(track);
        }
    }
   
    // 标记未匹配的轨迹为丢失
    std::vector<std::shared_ptr<Track>> lost_tracks;
    for (int unmatched_track_index : second_associations.unmatched_track_indices)
    {
        const std::shared_ptr<Track> &track = unmatched_tracks_after_1st_association[unmatched_track_index];
        if (track->state != TrackState::Lost)
        {
            track->mark_lost();
            lost_tracks.push_back(track);
        }
    }
   

    // ==================== 未确认轨迹关联 ====================
   
    std::vector<std::shared_ptr<Track>> unmatched_detections_after_1st_association;
    for (int detection_idx : first_associations.unmatched_det_indices)
    {
        const std::shared_ptr<Track> &detection = detections_high_conf[detection_idx];
        unmatched_detections_after_1st_association.push_back(detection);
    }
    

    CostMatrix iou_dists_unconfirmed, iou_dists_mask_unconfirmed;
    std::tie(iou_dists_unconfirmed, iou_dists_mask_unconfirmed) =
        iou_distance(unconfirmed_tracks, unmatched_detections_after_1st_association, _proximity_thresh);
    fuse_score(iou_dists_unconfirmed, unmatched_detections_after_1st_association);
    // 注意：此处原版有融合运动，但已经去掉了 ReID，可以保留 fuse_motion 或省略
   
    std::cout << "[Unconfirmed Association] Calling linear_assignment..." << std::endl;
    AssociationData unconfirmed_associations = linear_assignment(iou_dists_unconfirmed, 0.7);
    std::cout << "[Unconfirmed Association] matches=" << unconfirmed_associations.matches.size() << std::endl;
   
    for (const auto &match : unconfirmed_associations.matches)
    {
     
        const std::shared_ptr<Track> &track = unconfirmed_tracks[match.first];
        const std::shared_ptr<Track> &detection = unmatched_detections_after_1st_association[match.second];
        track->update(*_kalman_filter, *detection, _frame_id);
        activated_tracks.push_back(track);
    }

    std::vector<std::shared_ptr<Track>> removed_tracks;
    for (int unmatched_track_index : unconfirmed_associations.unmatched_track_indices)
    {
        const std::shared_ptr<Track> &track = unconfirmed_tracks[unmatched_track_index];
        track->mark_removed();
        removed_tracks.push_back(track);
    }
    

    // ==================== 初始化新轨迹 ====================
   
    std::vector<std::shared_ptr<Track>> unmatched_high_conf_detections;
    for (int detection_idx: unconfirmed_associations.unmatched_det_indices)
    {
        const std::shared_ptr<Track> &detection =
                unmatched_detections_after_1st_association[detection_idx];
        unmatched_high_conf_detections.push_back(detection);
    }

    // Initialize new tracks for the high confidence detections left after all the associations
    for (const std::shared_ptr<Track> &detection:
         unmatched_high_conf_detections)
    {
        if (detection->get_score() >= _new_track_thresh)
        {
            detection->activate(*_kalman_filter, _frame_id);
            activated_tracks.push_back(detection);
        }
    }
    
    // ==================== 更新丢失轨迹状态 ====================
  
    for (const std::shared_ptr<Track> &track : _lost_tracks)
    {
        if (_frame_id - track->end_frame() > _max_time_lost)
        {
            track->mark_removed();
            removed_tracks.push_back(track);
        }
    }
   

    // ==================== 清理跟踪列表 ====================
   
    std::vector<std::shared_ptr<Track>> updated_tracked_tracks;
    for (const std::shared_ptr<Track> &_tracked_track : _tracked_tracks)
    {
        if (_tracked_track->state == TrackState::Tracked)
        {
            updated_tracked_tracks.push_back(_tracked_track);
        }
    }
    _tracked_tracks = _merge_track_lists(updated_tracked_tracks, activated_tracks);
    _tracked_tracks = _merge_track_lists(_tracked_tracks, refind_tracks);

    _lost_tracks = _merge_track_lists(_lost_tracks, lost_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, _tracked_tracks);
    _lost_tracks = _remove_from_list(_lost_tracks, removed_tracks);

    std::vector<std::shared_ptr<Track>> tracked_tracks_cleaned,
            lost_tracks_cleaned;
    _remove_duplicate_tracks(tracked_tracks_cleaned, lost_tracks_cleaned,
                             _tracked_tracks, _lost_tracks);
    _tracked_tracks = tracked_tracks_cleaned;
    _lost_tracks = lost_tracks_cleaned;
  

    // ==================== 输出跟踪结果 ====================
    std::vector<std::shared_ptr<Track>> output_tracks;
    for (const std::shared_ptr<Track> &track : _tracked_tracks)
    {
        if (track->is_activated)
        {
            output_tracks.push_back(track);
        }
    }
  
    return output_tracks;
}

// FeatureVector BoTSORT::_extract_features(const cv::Mat &frame,
//                                          const cv::Rect_<float> &bbox_tlwh)
// {
//     cv::Mat patch = frame(bbox_tlwh);
//     return _reid_model->extract_features(patch);
// }


std::vector<std::shared_ptr<Track>>
BoTSORT::_merge_track_lists(std::vector<std::shared_ptr<Track>> &tracks_list_a,
                            std::vector<std::shared_ptr<Track>> &tracks_list_b)
{
    std::map<int, bool> exists;
    std::vector<std::shared_ptr<Track>> merged_tracks_list;

    for (const std::shared_ptr<Track> &track: tracks_list_a)
    {
        exists[track->track_id] = true;
        merged_tracks_list.push_back(track);
    }

    for (const std::shared_ptr<Track> &track: tracks_list_b)
    {
        if (exists.find(track->track_id) == exists.end())
        {
            exists[track->track_id] = true;
            merged_tracks_list.push_back(track);
        }
    }

    return merged_tracks_list;
}


std::vector<std::shared_ptr<Track>> BoTSORT::_remove_from_list(
        std::vector<std::shared_ptr<Track>> &tracks_list,
        std::vector<std::shared_ptr<Track>> &tracks_to_remove)
{
    std::map<int, bool> exists;
    std::vector<std::shared_ptr<Track>> new_tracks_list;

    for (const std::shared_ptr<Track> &track: tracks_to_remove)
    {
        exists[track->track_id] = true;
    }

    for (const std::shared_ptr<Track> &track: tracks_list)
    {
        if (exists.find(track->track_id) == exists.end())
        {
            new_tracks_list.push_back(track);
        }
    }

    return new_tracks_list;
}


void BoTSORT::_remove_duplicate_tracks(
        std::vector<std::shared_ptr<Track>> &result_tracks_a,
        std::vector<std::shared_ptr<Track>> &result_tracks_b,
        std::vector<std::shared_ptr<Track>> &tracks_list_a,
        std::vector<std::shared_ptr<Track>> &tracks_list_b)
{
    CostMatrix iou_dists = iou_distance(tracks_list_a, tracks_list_b);

    std::unordered_set<size_t> dup_a, dup_b;
    for (Eigen::Index i = 0; i < iou_dists.rows(); i++)
    {
        for (Eigen::Index j = 0; j < iou_dists.cols(); j++)
        {
            if (iou_dists(i, j) < 0.15)
            {
                int time_a = static_cast<int>(tracks_list_a[i]->frame_id -
                                              tracks_list_a[i]->start_frame);
                int time_b = static_cast<int>(tracks_list_b[j]->frame_id -
                                              tracks_list_b[j]->start_frame);

                // We make an assumption that the longer trajectory is the correct one
                if (time_a > time_b)
                {
                    dup_b.insert(
                            j);// In list b, track with index j is a duplicate
                }
                else
                {
                    dup_a.insert(
                            i);// In list a, track with index i is a duplicate
                }
            }
        }
    }

    // Remove duplicates from the lists
    for (size_t i = 0; i < tracks_list_a.size(); i++)
    {
        if (dup_a.find(i) == dup_a.end())
        {
            result_tracks_a.push_back(tracks_list_a[i]);
        }
    }

    for (size_t i = 0; i < tracks_list_b.size(); i++)
    {
        if (dup_b.find(i) == dup_b.end())
        {
            result_tracks_b.push_back(tracks_list_b[i]);
        }
    }
}

void BoTSORT::_load_params_from_config(const TrackerParams &config)
{
    _reid_enabled = config.reid_enabled;
    _gmc_enabled = config.gmc_enabled;
    _track_high_thresh = config.track_high_thresh;
    _track_low_thresh = config.track_low_thresh;
    _new_track_thresh = config.new_track_thresh;
    _track_buffer = config.track_buffer;
    _match_thresh = config.match_thresh;
    _proximity_thresh = config.proximity_thresh;
    _appearance_thresh = config.appearance_thresh;
    _gmc_method_name = config.gmc_method_name;
    _frame_rate = config.frame_rate;
    _lambda = config.lambda;
}