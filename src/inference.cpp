#include "point_seg/inference.hpp"

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace point_seg
{

Inference::Inference(
  const std::string & model_path,
  const std::string & device,
  const cv::Size & input_shape,
  float conf_threshold,
  float iou_threshold)
: conf_threshold_(conf_threshold), iou_threshold_(iou_threshold)
{
  initialize_model(model_path, input_shape, normalize_device(device));
}

float Inference::sigmoid(float x)
{
  return 1.0F / (1.0F + std::exp(-x));
}

std::string Inference::normalize_device(const std::string & device)
{
  std::string lower = device;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "auto") {
    return "AUTO";
  }
  if (lower.find("gpu") != std::string::npos) {
    return "GPU";
  }
  if (lower.find("npu") != std::string::npos) {
    return "NPU";
  }
  if (lower.find("cpu") != std::string::npos) {
    return "CPU";
  }
  return device;
}

void Inference::initialize_model(
  const std::string & model_path, const cv::Size & input_shape, const std::string & device)
{
  ov::Core core;
  std::string resolved_model_path = model_path;
  const std::filesystem::path p(model_path);
  if (std::filesystem::is_directory(p)) {
    std::filesystem::path xml_path;
    for (const auto & entry : std::filesystem::directory_iterator(p)) {
      if (entry.is_regular_file() && entry.path().extension() == ".xml") {
        xml_path = entry.path();
        break;
      }
    }
    if (xml_path.empty()) {
      throw std::runtime_error("No .xml model file found in directory: " + model_path);
    }
    resolved_model_path = xml_path.string();
  }

  std::shared_ptr<ov::Model> model = core.read_model(resolved_model_path);

  if (model->is_dynamic()) {
    model->reshape({{1, 3, input_shape.height, input_shape.width}});
  }

  compiled_model_ = core.compile_model(model, device);
  infer_request_ = compiled_model_.create_infer_request();

  const auto input_shape_vec = compiled_model_.input().get_shape();
  input_h_ = static_cast<int>(input_shape_vec.at(2));
  input_w_ = static_cast<int>(input_shape_vec.at(3));

  const auto output1_shape = compiled_model_.output(1).get_shape();
  num_masks_ = static_cast<int>(output1_shape.at(1));
  proto_h_ = static_cast<int>(output1_shape.at(2));
  proto_w_ = static_cast<int>(output1_shape.at(3));
}

std::vector<Inference::Candidate> Inference::decode_candidates(
  const ov::Tensor & pred_tensor, int nm, int input_w, int input_h) const
{
  std::vector<Candidate> candidates;
  const auto shape = pred_tensor.get_shape();
  if (shape.size() != 3) {
    return candidates;
  }

  const bool pred_major = shape[1] > shape[2];  // e.g. [1,300,38]
  const int n_pred = pred_major ? static_cast<int>(shape[1]) : static_cast<int>(shape[2]);
  const int n_attr = pred_major ? static_cast<int>(shape[2]) : static_cast<int>(shape[1]);
  const bool end2end_seg = (n_attr == (nm + 6));  // [x1,y1,x2,y2,score,cls,mask...]
  const int n_cls = end2end_seg ? 0 : (n_attr - 4 - nm);
  if (n_cls <= 0) {
    if (!end2end_seg) {
      return candidates;
    }
  }

  const float * data = pred_tensor.data<const float>();
  auto val = [data, n_attr, n_pred, pred_major](int pred_idx, int attr_idx) -> float {
      if (pred_major) {
        return data[pred_idx * n_attr + attr_idx];
      }
      return data[attr_idx * n_pred + pred_idx];
    };

  candidates.reserve(n_pred / 4);
  for (int i = 0; i < n_pred; ++i) {
    if (end2end_seg) {
      const float score_raw = val(i, 4);
      // End2end export here provides confidence already in probability-like scale.
      const float score = std::clamp(score_raw, 0.0F, 1.0F);
      if (score < conf_threshold_) {
        continue;
      }

      const int cls0 = static_cast<int>(std::lround(val(i, 5)));
      if (cls0 < 0 || cls0 > 1000) {
        continue;
      }

      float x1 = val(i, 0);
      float y1 = val(i, 1);
      float x2 = val(i, 2);
      float y2 = val(i, 3);
      if (x2 < x1) {
        std::swap(x1, x2);
      }
      if (y2 < y1) {
        std::swap(y1, y2);
      }

      int ix1 = std::clamp(static_cast<int>(std::floor(x1)), 0, input_w - 1);
      int iy1 = std::clamp(static_cast<int>(std::floor(y1)), 0, input_h - 1);
      int ix2 = std::clamp(static_cast<int>(std::ceil(x2)), 0, input_w - 1);
      int iy2 = std::clamp(static_cast<int>(std::ceil(y2)), 0, input_h - 1);
      int bw = std::max(1, ix2 - ix1 + 1);
      int bh = std::max(1, iy2 - iy1 + 1);

      Candidate cand;
      cand.class_id = cls0 + 1;  // reserve 0 for background
      cand.confidence = score;
      cand.box = cv::Rect(ix1, iy1, bw, bh);
      cand.mask_coeff.resize(static_cast<size_t>(nm));
      for (int m = 0; m < nm; ++m) {
        cand.mask_coeff[static_cast<size_t>(m)] = val(i, 6 + m);
      }
      candidates.push_back(std::move(cand));
      continue;
    }

    int best_cls = -1;
    float best_score = 0.0F;
    for (int c = 0; c < n_cls; ++c) {
      const float score = std::clamp(val(i, 4 + c), 0.0F, 1.0F);
      if (score > best_score) {
        best_score = score;
        best_cls = c;
      }
    }
    if (best_cls < 0 || best_score < conf_threshold_) {
      continue;
    }

    const float cx = val(i, 0);
    const float cy = val(i, 1);
    const float w = val(i, 2);
    const float h = val(i, 3);

    int x0 = static_cast<int>(std::round(cx - w * 0.5F));
    int y0 = static_cast<int>(std::round(cy - h * 0.5F));
    int bw = static_cast<int>(std::round(w));
    int bh = static_cast<int>(std::round(h));
    x0 = std::clamp(x0, 0, input_w - 1);
    y0 = std::clamp(y0, 0, input_h - 1);
    bw = std::clamp(bw, 1, input_w - x0);
    bh = std::clamp(bh, 1, input_h - y0);

    Candidate cand;
    cand.class_id = best_cls + 1;  // reserve 0 for background
    cand.confidence = best_score;
    cand.box = cv::Rect(x0, y0, bw, bh);
    cand.mask_coeff.resize(static_cast<size_t>(nm));
    for (int m = 0; m < nm; ++m) {
      cand.mask_coeff[static_cast<size_t>(m)] = val(i, 4 + n_cls + m);
    }
    candidates.push_back(std::move(cand));
  }

  if (candidates.empty()) {
    return candidates;
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  boxes.reserve(candidates.size());
  scores.reserve(candidates.size());
  for (const auto & c : candidates) {
    boxes.push_back(c.box);
    scores.push_back(c.confidence);
  }

  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, iou_threshold_, keep);

  std::vector<Candidate> filtered;
  filtered.reserve(keep.size());
  for (const int idx : keep) {
    filtered.push_back(std::move(candidates[static_cast<size_t>(idx)]));
  }
  std::sort(
    filtered.begin(), filtered.end(),
    [](const Candidate & a, const Candidate & b) {return a.confidence > b.confidence;});
  return filtered;
}

SegmentationOutput Inference::Run(const cv::Mat & frame)
{
  if (frame.empty()) {
    throw std::runtime_error("Inference input frame is empty");
  }

  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_AREA);
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  cv::Mat f32;
  rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

  std::vector<cv::Mat> chw(3);
  for (auto & c : chw) {
    c = cv::Mat(input_h_, input_w_, CV_32FC1);
  }
  cv::split(f32, chw);
  std::vector<float> input_chw(static_cast<size_t>(3 * input_h_ * input_w_));
  const size_t plane = static_cast<size_t>(input_h_ * input_w_);
  std::memcpy(input_chw.data() + 0 * plane, chw[0].data, plane * sizeof(float));
  std::memcpy(input_chw.data() + 1 * plane, chw[1].data, plane * sizeof(float));
  std::memcpy(input_chw.data() + 2 * plane, chw[2].data, plane * sizeof(float));

  ov::Tensor input_tensor = infer_request_.get_input_tensor();
  const size_t bytes_to_copy = input_chw.size() * sizeof(float);
  std::memcpy(input_tensor.data<float>(), input_chw.data(), bytes_to_copy);
  infer_request_.set_input_tensor(input_tensor);
  infer_request_.infer();

  const ov::Tensor pred = infer_request_.get_output_tensor(0);
  const ov::Tensor proto = infer_request_.get_output_tensor(1);
  const auto candidates = decode_candidates(pred, num_masks_, input_w_, input_h_);

  SegmentationOutput out;
  out.class_mask = cv::Mat::zeros(frame.rows, frame.cols, CV_16UC1);
  out.conf_mask = cv::Mat::zeros(frame.rows, frame.cols, CV_32FC1);
  if (candidates.empty()) {
    return out;
  }

  const float sx = static_cast<float>(frame.cols) / static_cast<float>(input_w_);
  const float sy = static_cast<float>(frame.rows) / static_cast<float>(input_h_);
  out.detections.reserve(candidates.size());
  for (const auto & cand : candidates) {
    SegmentationOutput::Detection d;
    d.class_id = std::max(0, cand.class_id - 1);
    d.confidence = cand.confidence;
    const int x = std::clamp(static_cast<int>(std::lround(cand.box.x * sx)), 0, frame.cols - 1);
    const int y = std::clamp(static_cast<int>(std::lround(cand.box.y * sy)), 0, frame.rows - 1);
    const int w = std::max(1, std::min(frame.cols - x, static_cast<int>(std::lround(cand.box.width * sx))));
    const int h = std::max(1, std::min(frame.rows - y, static_cast<int>(std::lround(cand.box.height * sy))));
    d.box = cv::Rect(x, y, w, h);
    out.detections.push_back(std::move(d));
  }

  const float * proto_data = proto.data<const float>();
  const int proto_plane = proto_h_ * proto_w_;

  for (const auto & cand : candidates) {
    cv::Mat mask_small(proto_h_, proto_w_, CV_32FC1, cv::Scalar(0.0F));
    for (int y = 0; y < proto_h_; ++y) {
      float * row_ptr = mask_small.ptr<float>(y);
      for (int x = 0; x < proto_w_; ++x) {
        const int idx = y * proto_w_ + x;
        float value = 0.0F;
        for (int m = 0; m < num_masks_; ++m) {
          value += cand.mask_coeff[static_cast<size_t>(m)] * proto_data[m * proto_plane + idx];
        }
        row_ptr[x] = sigmoid(value);
      }
    }

    cv::Mat mask_input;
    cv::resize(mask_small, mask_input, cv::Size(input_w_, input_h_), 0, 0, cv::INTER_LINEAR);

    cv::Mat mask_bin;
    cv::threshold(mask_input, mask_bin, 0.5, 255.0, cv::THRESH_BINARY);
    mask_bin.convertTo(mask_bin, CV_8UC1);

    cv::Mat crop = cv::Mat::zeros(mask_bin.size(), CV_8UC1);
    crop(cand.box).setTo(255);
    cv::bitwise_and(mask_bin, crop, mask_bin);

    cv::Mat mask_orig;
    cv::resize(mask_bin, mask_orig, cv::Size(frame.cols, frame.rows), 0, 0, cv::INTER_NEAREST);

    for (int y = 0; y < frame.rows; ++y) {
      const uint8_t * mptr = mask_orig.ptr<uint8_t>(y);
      uint16_t * cptr = out.class_mask.ptr<uint16_t>(y);
      float * fptr = out.conf_mask.ptr<float>(y);
      for (int x = 0; x < frame.cols; ++x) {
        if (mptr[x] > 0 && cand.confidence > fptr[x]) {
          cptr[x] = static_cast<uint16_t>(cand.class_id);
          fptr[x] = cand.confidence;
        }
      }
    }
  }

  return out;
}

}  // namespace point_seg
