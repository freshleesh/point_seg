#pragma once

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include <string>
#include <vector>

namespace point_seg
{

struct SegmentationOutput
{
  struct Detection
  {
    int class_id{0};   // zero-based class id
    float confidence{0.0F};
    cv::Rect box;
  };

  cv::Mat class_mask;  // CV_16UC1, class id + 1 (0 = background)
  cv::Mat conf_mask;   // CV_32FC1
  std::vector<Detection> detections;
};

class Inference
{
public:
  Inference(
    const std::string & model_path,
    const std::string & device,
    const cv::Size & input_shape,
    float conf_threshold,
    float iou_threshold);

  SegmentationOutput Run(const cv::Mat & frame);

private:
  struct Candidate
  {
    int class_id;
    float confidence;
    cv::Rect box;
    std::vector<float> mask_coeff;
  };

  static float sigmoid(float x);
  static std::string normalize_device(const std::string & device);

  void initialize_model(const std::string & model_path, const cv::Size & input_shape, const std::string & device);
  std::vector<Candidate> decode_candidates(const ov::Tensor & pred_tensor, int nm, int input_w, int input_h) const;

  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;

  int input_w_{640};
  int input_h_{640};
  int num_masks_{32};
  int proto_h_{160};
  int proto_w_{160};

  float conf_threshold_{0.25F};
  float iou_threshold_{0.45F};
};

}  // namespace point_seg
