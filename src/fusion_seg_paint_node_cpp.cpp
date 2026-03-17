#include "point_seg/inference.hpp"

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <yaml-cpp/yaml.h>
#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace point_seg
{

class FusionSegPaintNodeCpp : public rclcpp::Node
{
public:
  FusionSegPaintNodeCpp()
  : Node("point_seg_node")
  {
    declare_parameter("image_topic", "/image_raw");
    declare_parameter("lidar_topic", "/livox/lidar");
    declare_parameter("painted_cloud_topic", "/yolo_seg/painted_cloud");
    declare_parameter("projected_image_topic", "/yolo_seg/projected_image");
    declare_parameter("camera_intrinsic_yaml", "");
    declare_parameter("camera_extrinsic_yaml", "");
    declare_parameter("model_path", "/home/nuc14/work/model/yolo26n-seg_openvino_model");
    declare_parameter("device", "auto");
    declare_parameter("conf", 0.25);
    declare_parameter("iou", 0.45);
    declare_parameter("publish_projected_image", true);
    declare_parameter("overlay_segmentation_mask", false);
    declare_parameter("overlay_segmentation_alpha", 0.35);
    declare_parameter("profile_timing", true);
    declare_parameter("passthrough_x_min", 0.0);
    declare_parameter("passthrough_x_max", 120.0);
    declare_parameter("passthrough_y_abs_max", 60.0);
    declare_parameter("passthrough_z_min", -5.0);
    declare_parameter("passthrough_z_max", 5.0);
    declare_parameter("voxel_leaf_size", 0.10);
    declare_parameter("inference_width", 640);
    declare_parameter("inference_height", 640);

    image_topic_ = get_parameter("image_topic").as_string();
    lidar_topic_ = get_parameter("lidar_topic").as_string();
    painted_cloud_topic_ = get_parameter("painted_cloud_topic").as_string();
    projected_image_topic_ = get_parameter("projected_image_topic").as_string();
    intrinsic_yaml_ = get_parameter("camera_intrinsic_yaml").as_string();
    extrinsic_yaml_ = get_parameter("camera_extrinsic_yaml").as_string();
    model_path_ = get_parameter("model_path").as_string();
    device_ = get_parameter("device").as_string();
    conf_thr_ = static_cast<float>(get_parameter("conf").as_double());
    iou_thr_ = static_cast<float>(get_parameter("iou").as_double());
    publish_projected_image_ = get_parameter("publish_projected_image").as_bool();
    overlay_segmentation_mask_ = get_parameter("overlay_segmentation_mask").as_bool();
    overlay_segmentation_alpha_ = static_cast<float>(get_parameter("overlay_segmentation_alpha").as_double());
    profile_timing_ = get_parameter("profile_timing").as_bool();
    x_min_ = static_cast<float>(get_parameter("passthrough_x_min").as_double());
    x_max_ = static_cast<float>(get_parameter("passthrough_x_max").as_double());
    y_abs_max_ = static_cast<float>(get_parameter("passthrough_y_abs_max").as_double());
    z_min_ = static_cast<float>(get_parameter("passthrough_z_min").as_double());
    z_max_ = static_cast<float>(get_parameter("passthrough_z_max").as_double());
    voxel_leaf_size_ = static_cast<float>(get_parameter("voxel_leaf_size").as_double());
    const int inference_width = get_parameter("inference_width").as_int();
    const int inference_height = get_parameter("inference_height").as_int();

    load_camera_intrinsics(intrinsic_yaml_);
    load_extrinsics(extrinsic_yaml_);
    load_class_names_from_metadata(model_path_);

    inference_ = std::make_unique<Inference>(
      model_path_, device_, cv::Size(inference_width, inference_height), conf_thr_, iou_thr_);

    const auto qos = rclcpp::SensorDataQoS().keep_last(10);
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, qos, std::bind(&FusionSegPaintNodeCpp::on_lidar, this, std::placeholders::_1));
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_, qos, std::bind(&FusionSegPaintNodeCpp::on_image, this, std::placeholders::_1));

    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(painted_cloud_topic_, 10);
    projected_img_pub_ = create_publisher<sensor_msgs::msg::Image>(projected_image_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "point_seg_node(C++) ready | image=%s lidar=%s out_cloud=%s out_image=%s model=%s conf=%.3f iou=%.2f",
      image_topic_.c_str(), lidar_topic_.c_str(), painted_cloud_topic_.c_str(),
      projected_image_topic_.c_str(), model_path_.c_str(), conf_thr_, iou_thr_);
  }

private:
  static sensor_msgs::msg::PointField make_field(
    const std::string & name, uint32_t offset, uint8_t datatype, uint32_t count)
  {
    sensor_msgs::msg::PointField f;
    f.name = name;
    f.offset = offset;
    f.datatype = datatype;
    f.count = count;
    return f;
  }

  struct PackedPoint
  {
    float x;
    float y;
    float z;
    float intensity;
    uint16_t class_id;
    float class_conf;
    uint32_t track_id;
    uint8_t painted;
  } __attribute__((packed));

  static_assert(sizeof(PackedPoint) == 27, "PackedPoint must be 27 bytes");

  static Eigen::Matrix4d build_extrinsic_matrix(
    double x, double y, double z, double roll_deg, double pitch_deg, double yaw_deg)
  {
    constexpr double deg2rad = M_PI / 180.0;
    const double r = roll_deg * deg2rad;
    const double p = pitch_deg * deg2rad;
    const double yw = yaw_deg * deg2rad;

    Eigen::Matrix3d Rx, Ry, Rz;
    Rx << 1, 0, 0,
      0, std::cos(r), -std::sin(r),
      0, std::sin(r), std::cos(r);
    Ry << std::cos(p), 0, std::sin(p),
      0, 1, 0,
      -std::sin(p), 0, std::cos(p);
    Rz << std::cos(yw), -std::sin(yw), 0,
      std::sin(yw), std::cos(yw), 0,
      0, 0, 1;

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rz * Ry * Rx;
    T(0, 3) = x;
    T(1, 3) = y;
    T(2, 3) = z;
    return T;
  }

  void load_camera_intrinsics(const std::string & path)
  {
    if (path.empty()) {
      throw std::runtime_error("camera_intrinsic_yaml is empty");
    }
    const auto cfg = YAML::LoadFile(path);
    const auto cam_data = cfg["camera_matrix"]["data"].as<std::vector<double>>();
    if (cam_data.size() != 9) {
      throw std::runtime_error("camera_matrix.data must have 9 elements");
    }
    K_ = cv::Mat::zeros(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
      K_.at<double>(i / 3, i % 3) = cam_data[static_cast<size_t>(i)];
    }
  }

  void load_extrinsics(const std::string & path)
  {
    if (path.empty()) {
      throw std::runtime_error("camera_extrinsic_yaml is empty");
    }
    const auto cfg = YAML::LoadFile(path);
    const auto p = cfg["extrinsic_calibration_by_hand"]["ros__parameters"];
    const double x = p["x"].as<double>();
    const double y = p["y"].as<double>();
    const double z = p["z"].as<double>();
    const double roll = p["roll"].as<double>();
    const double pitch = p["pitch"].as<double>();
    const double yaw = p["yaw"].as<double>();
    T_lidar_to_cam_ = build_extrinsic_matrix(x, y, z, roll, pitch, yaw);
    R_ = T_lidar_to_cam_.block<3, 3>(0, 0);
    t_ = T_lidar_to_cam_.block<3, 1>(0, 3);
  }

  void load_class_names_from_metadata(const std::string & model_path)
  {
    std::filesystem::path p(model_path);
    if (!std::filesystem::is_directory(p)) {
      p = p.parent_path();
    }
    const auto meta = p / "metadata.yaml";
    if (!std::filesystem::exists(meta)) {
      return;
    }

    const auto cfg = YAML::LoadFile(meta.string());
    const auto names = cfg["names"];
    if (!names || !names.IsMap()) {
      return;
    }

    int max_id = -1;
    for (const auto & it : names) {
      max_id = std::max(max_id, it.first.as<int>());
    }
    if (max_id < 0) {
      return;
    }

    class_names_.assign(static_cast<size_t>(max_id + 1), "");
    for (const auto & it : names) {
      const int id = it.first.as<int>();
      if (id >= 0 && static_cast<size_t>(id) < class_names_.size()) {
        class_names_[static_cast<size_t>(id)] = it.second.as<std::string>();
      }
    }
  }

  void on_lidar(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    pcl::PointCloud<pcl::PointXYZI> cloud_in;
    pcl::fromROSMsg(*msg, cloud_in);
    if (cloud_in.empty()) {
      return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>(cloud_in));

    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(filtered);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(x_min_, x_max_);
    pass.filter(*filtered);

    pass.setInputCloud(filtered);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-y_abs_max_, y_abs_max_);
    pass.filter(*filtered);

    pass.setInputCloud(filtered);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_min_, z_max_);
    pass.filter(*filtered);

    if (voxel_leaf_size_ > 0.0F) {
      pcl::VoxelGrid<pcl::PointXYZI> voxel;
      voxel.setInputCloud(filtered);
      voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
      pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>());
      voxel.filter(*downsampled);
      filtered = downsampled;
    }

    std::lock_guard<std::mutex> lk(cloud_mutex_);
    latest_cloud_ = filtered;
    latest_cloud_header_ = msg->header;
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    std_msgs::msg::Header cloud_header;
    {
      std::lock_guard<std::mutex> lk(cloud_mutex_);
      if (!latest_cloud_ || latest_cloud_->empty()) {
        return;
      }
      cloud = latest_cloud_;
      cloud_header = latest_cloud_header_;
    }

    cv::Mat frame;
    try {
      frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "image cv_bridge failed: %s", e.what());
      return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    SegmentationOutput seg;
    try {
      seg = inference_->Run(frame);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "inference failed: %s", e.what());
      return;
    }

    const int h = frame.rows;
    const int w = frame.cols;
    const size_t n = cloud->points.size();
    std::vector<PackedPoint> points(n);
    std::vector<cv::Point> projected;
    projected.reserve(n);
    std::vector<uint8_t> projected_painted;
    projected_painted.reserve(n);

    size_t painted_count = 0;
    for (size_t i = 0; i < n; ++i) {
      const auto & p = cloud->points[i];
      PackedPoint out{};
      out.x = p.x;
      out.y = p.y;
      out.z = p.z;
      out.intensity = p.intensity;
      out.class_id = 0;
      out.class_conf = 0.0F;
      out.track_id = 0;
      out.painted = 0;

      Eigen::Vector3d pl(p.x, p.y, p.z);
      const Eigen::Vector3d pc = R_ * pl + t_;
      const double z = pc.z();
      if (z > 1e-6) {
        const int u = static_cast<int>(std::lround(K_.at<double>(0, 0) * pc.x() / z + K_.at<double>(0, 2)));
        const int v = static_cast<int>(std::lround(K_.at<double>(1, 1) * pc.y() / z + K_.at<double>(1, 2)));
        if (u >= 0 && u < w && v >= 0 && v < h) {
          const uint16_t cid = seg.class_mask.at<uint16_t>(v, u);
          if (cid > 0) {
            out.class_id = cid;
            out.class_conf = seg.conf_mask.at<float>(v, u);
            out.painted = 1;
            ++painted_count;
          }
          projected.emplace_back(u, v);
          projected_painted.push_back(out.painted);
        }
      }
      points[i] = out;
    }

    sensor_msgs::msg::PointCloud2 cloud_out;
    cloud_out.header = cloud_header;
    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(points.size());
    cloud_out.is_bigendian = false;
    cloud_out.is_dense = false;
    cloud_out.point_step = static_cast<uint32_t>(sizeof(PackedPoint));
    cloud_out.row_step = cloud_out.point_step * cloud_out.width;
    cloud_out.fields.clear();
    cloud_out.fields.reserve(8);
    cloud_out.fields.push_back(make_field("x", 0, sensor_msgs::msg::PointField::FLOAT32, 1));
    cloud_out.fields.push_back(make_field("y", 4, sensor_msgs::msg::PointField::FLOAT32, 1));
    cloud_out.fields.push_back(make_field("z", 8, sensor_msgs::msg::PointField::FLOAT32, 1));
    cloud_out.fields.push_back(make_field("intensity", 12, sensor_msgs::msg::PointField::FLOAT32, 1));
    cloud_out.fields.push_back(make_field("class_id", 16, sensor_msgs::msg::PointField::UINT16, 1));
    cloud_out.fields.push_back(make_field("class_conf", 18, sensor_msgs::msg::PointField::FLOAT32, 1));
    cloud_out.fields.push_back(make_field("track_id", 22, sensor_msgs::msg::PointField::UINT32, 1));
    cloud_out.fields.push_back(make_field("painted", 26, sensor_msgs::msg::PointField::UINT8, 1));
    cloud_out.data.resize(points.size() * sizeof(PackedPoint));
    std::memcpy(cloud_out.data.data(), points.data(), cloud_out.data.size());
    cloud_pub_->publish(cloud_out);

    if (publish_projected_image_) {
      cv::Mat overlay = frame.clone();
      for (const auto & det : seg.detections) {
        cv::rectangle(overlay, det.box, cv::Scalar(0, 255, 0), 2);
        std::string name = "cls:" + std::to_string(det.class_id);
        if (det.class_id >= 0 && static_cast<size_t>(det.class_id) < class_names_.size() &&
          !class_names_[static_cast<size_t>(det.class_id)].empty())
        {
          name = class_names_[static_cast<size_t>(det.class_id)];
        }
        std::ostringstream ss;
        ss << name << " " << std::fixed << std::setprecision(2) << det.confidence;
        const std::string text = ss.str();
        int base = 0;
        const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        const int tx = det.box.x;
        const int by = std::max(ts.height + 2, det.box.y - 4);
        cv::rectangle(
          overlay,
          cv::Rect(tx, by - ts.height - 4, ts.width + 6, ts.height + 4),
          cv::Scalar(0, 255, 0),
          cv::FILLED);
        cv::putText(
          overlay, text, cv::Point(tx + 3, by - 2),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
      }
      for (size_t i = 0; i < projected.size(); ++i) {
        const cv::Scalar c = projected_painted[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 255);
        cv::circle(overlay, projected[i], 1, c, cv::FILLED);
      }
      auto out_img = cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg();
      projected_img_pub_->publish(*out_img);
    }

    if (profile_timing_) {
      const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
      const double ratio = static_cast<double>(painted_count) / static_cast<double>(std::max<size_t>(1, n)) * 100.0;
      float det_min = 0.0F;
      float det_max = 0.0F;
      if (!seg.detections.empty()) {
        det_min = seg.detections.front().confidence;
        det_max = seg.detections.front().confidence;
        for (const auto & d : seg.detections) {
          det_min = std::min(det_min, d.confidence);
          det_max = std::max(det_max, d.confidence);
        }
      }
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "infer+paint=%ldms det=%zu conf=[%.3f,%.3f] painted=%zu/%zu (%.2f%%)",
        dt_ms, seg.detections.size(), det_min, det_max, painted_count, n, ratio);
    }
  }

  std::string image_topic_;
  std::string lidar_topic_;
  std::string painted_cloud_topic_;
  std::string projected_image_topic_;
  std::string intrinsic_yaml_;
  std::string extrinsic_yaml_;
  std::string model_path_;
  std::string device_;

  float conf_thr_{0.25F};
  float iou_thr_{0.45F};
  float x_min_{0.0F};
  float x_max_{120.0F};
  float y_abs_max_{60.0F};
  float z_min_{-5.0F};
  float z_max_{5.0F};
  float voxel_leaf_size_{0.1F};
  float overlay_segmentation_alpha_{0.35F};
  bool publish_projected_image_{true};
  bool overlay_segmentation_mask_{false};
  bool profile_timing_{true};
  std::vector<std::string> class_names_;

  cv::Mat K_;
  Eigen::Matrix4d T_lidar_to_cam_{Eigen::Matrix4d::Identity()};
  Eigen::Matrix3d R_{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t_{Eigen::Vector3d::Zero()};

  std::unique_ptr<Inference> inference_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr projected_img_pub_;

  std::mutex cloud_mutex_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr latest_cloud_;
  std_msgs::msg::Header latest_cloud_header_;
};

}  // namespace point_seg

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<point_seg::FusionSegPaintNodeCpp>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
