#include "colmap/scene/database_file.h"

#include "colmap/feature/types.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/rig.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/endian.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/string.h"
#include "colmap/util/types.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <fstream>
#include <sstream>

#include <Eigen/Geometry>

namespace colmap {
namespace fs = std::filesystem;

// Helper functions to convert between keypoints/matches and blobs
FeatureKeypointsBlob FeatureKeypointsToBlob(const FeatureKeypoints& keypoints) {
  const FeatureKeypointsBlob::Index kNumCols = 6;
  FeatureKeypointsBlob blob(keypoints.size(), kNumCols);
  for (size_t i = 0; i < keypoints.size(); ++i) {
    blob(i, 0) = keypoints[i].x;
    blob(i, 1) = keypoints[i].y;
    blob(i, 2) = keypoints[i].a11;
    blob(i, 3) = keypoints[i].a12;
    blob(i, 4) = keypoints[i].a21;
    blob(i, 5) = keypoints[i].a22;
  }
  return blob;
}

FeatureKeypoints FeatureKeypointsFromBlob(const FeatureKeypointsBlob& blob) {
  FeatureKeypoints keypoints(static_cast<size_t>(blob.rows()));
  if (blob.cols() == 2) {
    for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
      keypoints[i] = FeatureKeypoint(blob(i, 0), blob(i, 1));
    }
  } else if (blob.cols() == 4) {
    for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
      keypoints[i] =
          FeatureKeypoint(blob(i, 0), blob(i, 1), blob(i, 2), blob(i, 3));
    }
  } else if (blob.cols() == 6) {
    for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
      keypoints[i] = FeatureKeypoint(blob(i, 0),
                                     blob(i, 1),
                                     blob(i, 2),
                                     blob(i, 3),
                                     blob(i, 4),
                                     blob(i, 5));
    }
  } else {
    LOG(FATAL_THROW) << "Keypoint format not supported";
  }
  return keypoints;
}

FeatureMatchesBlob FeatureMatchesToBlob(const FeatureMatches& matches) {
  const FeatureMatchesBlob::Index kNumCols = 2;
  FeatureMatchesBlob blob(matches.size(), kNumCols);
  for (size_t i = 0; i < matches.size(); ++i) {
    blob(i, 0) = matches[i].point2D_idx1;
    blob(i, 1) = matches[i].point2D_idx2;
  }
  return blob;
}

FeatureMatches FeatureMatchesFromBlob(const FeatureMatchesBlob& blob) {
  THROW_CHECK_EQ(blob.cols(), 2);
  FeatureMatches matches(static_cast<size_t>(blob.rows()));
  for (FeatureMatchesBlob::Index i = 0; i < blob.rows(); ++i) {
    matches[i].point2D_idx1 = blob(i, 0);
    matches[i].point2D_idx2 = blob(i, 1);
  }
  return matches;
}

namespace fs = std::filesystem;


FileDatabase::FileDatabase(const std::string& root_dir) {
  root_dir_ = fs::absolute(root_dir);

  // Create directory structure
  rigs_dir_ = root_dir_ / "rigs";
  cameras_dir_ = root_dir_ / "cameras";
  frames_dir_ = root_dir_ / "frames";
  images_dir_ = root_dir_ / "images";
  pose_priors_dir_ = root_dir_ / "pose_priors";
  features_dir_ = root_dir_ / "features";
  matches_dir_ = root_dir_ / "matches";
  two_view_geometries_dir_ = root_dir_ / "two_view_geometries";

  fs::create_directories(rigs_dir_);
  fs::create_directories(cameras_dir_);
  fs::create_directories(frames_dir_);
  fs::create_directories(images_dir_);
  fs::create_directories(pose_priors_dir_);
  fs::create_directories(features_dir_);
  fs::create_directories(matches_dir_);
  fs::create_directories(two_view_geometries_dir_);
}

FileDatabase::~FileDatabase() {
  Close();
}

void FileDatabase::Close() {
  // Clear caches
  cameras_cache_.clear();
  images_cache_.clear();
  matches_cache_.clear();
  two_view_geometry_cache_.clear();
  cache_loaded_ = false;
}

// Helper methods for file paths
std::filesystem::path FileDatabase::GetRigsPath() const {
  return rigs_dir_;
}

std::filesystem::path FileDatabase::GetCamerasPath() const {
  return cameras_dir_;
}

std::filesystem::path FileDatabase::GetFramesPath() const {
  return frames_dir_;
}

std::filesystem::path FileDatabase::GetImagesPath() const {
  return images_dir_;
}

std::filesystem::path FileDatabase::GetPosePriorsPath() const {
  return pose_priors_dir_;
}

std::filesystem::path FileDatabase::GetKeypointsPath(image_t image_id) const {
  return features_dir_ / (std::to_string(image_id) + "_keypoints.bin");
}

std::filesystem::path FileDatabase::GetDescriptorsPath(image_t image_id) const {
  return features_dir_ / (std::to_string(image_id) + "_descriptors.bin");
}

std::filesystem::path FileDatabase::GetMatchesPath(image_t image_id1,
                                                  image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  return matches_dir_ / (std::to_string(pair_id) + "_matches.json");
}

std::filesystem::path FileDatabase::GetTwoViewGeometryPath(image_t image_id1,
                                                          image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  return two_view_geometries_dir_ / (std::to_string(pair_id) + "_twoview.bin");
}

// Helper methods for caching
void FileDatabase::LoadCache() const {
  if (cache_loaded_) return;
  
  // Load all cameras
  for (const auto& entry : fs::directory_iterator(cameras_dir_)) {
    if (entry.path().extension() == ".bin") {
      const std::string filename = entry.path().stem().string();
      try {
        const camera_t camera_id = std::stoi(filename);
        cameras_cache_[camera_id] = ReadCamera(camera_id);
      } catch (...) {
        // Skip invalid filenames
      }
    }
  }
  
  // Load all images
  for (const auto& entry : fs::directory_iterator(images_dir_)) {
    if (entry.path().extension() == ".json") {
      // Parse image_id from filename like "image_123.json"
      const std::string stem = entry.path().stem().string();
      if (stem.substr(0, 6) == "image_") {
        try {
          const std::string id_str = stem.substr(6);
          const image_t image_id = std::stoi(id_str);
          images_cache_[image_id] = ReadImage(image_id);
        } catch (...) {
          // Skip invalid filenames
        }
      }
    }
  }
  
  cache_loaded_ = true;
}

void FileDatabase::SaveCache() const {
  // Nothing to save since we don't maintain persistent cache
}

// Existence checks
bool FileDatabase::ExistsRig(rig_t rig_id) const {
  const std::string rig_filename = std::to_string(rig_id) + ".json";
  return fs::exists(rigs_dir_ / rig_filename);
}

bool FileDatabase::ExistsCamera(camera_t camera_id) const {
  const std::string camera_filename = std::to_string(camera_id) + ".bin";
  return fs::exists(cameras_dir_ / camera_filename);
}

bool FileDatabase::ExistsFrame(frame_t frame_id) const {
  const std::string frame_filename = std::to_string(frame_id) + ".json";
  return fs::exists(frames_dir_ / frame_filename);
}

bool FileDatabase::ExistsImage(image_t image_id) const {
  const std::string image_filename = "image_" + std::to_string(image_id) + ".json";
  return fs::exists(images_dir_ / image_filename);
}

bool FileDatabase::ExistsImageWithName(const std::string& name) const {
  // Look through all image files to find the name
  for (const auto& entry : fs::directory_iterator(images_dir_)) {
    if (entry.path().extension() == ".json") {
      try {
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(entry.path().string(), pt);
        if (pt.count("name") > 0 && pt.get<std::string>("name") == name) {
          return true;
        }
      } catch (...) {
        // Skip files that are not valid JSON
      }
    }
  }
  return false;
}

bool FileDatabase::ExistsPosePrior(pose_prior_t pose_prior_id,
                                  bool /*is_deprecated_image_prior*/) const {
  const std::string pose_prior_filename = std::to_string(pose_prior_id) + ".json";
  return fs::exists(pose_priors_dir_ / pose_prior_filename);
}

bool FileDatabase::ExistsKeypoints(image_t image_id) const {
  return fs::exists(GetKeypointsPath(image_id));
}

bool FileDatabase::ExistsDescriptors(image_t image_id) const {
  return fs::exists(GetDescriptorsPath(image_id));
}

bool FileDatabase::ExistsMatches(image_t image_id1, image_t image_id2) const {
  return fs::exists(GetMatchesPath(image_id1, image_id2));
}

bool FileDatabase::ExistsTwoViewGeometry(image_t image_id1, image_t image_id2) const {
  return fs::exists(GetTwoViewGeometryPath(image_id1, image_id2));
}

// Count methods
size_t FileDatabase::NumRigs() const {
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator(rigs_dir_)) {
    if (entry.path().extension() == ".json") {
      count++;
    }
  }
  return count;
}

size_t FileDatabase::NumCameras() const {
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator(cameras_dir_)) {
    if (entry.path().extension() == ".bin") {
      count++;
    }
  }
  return count;
}

size_t FileDatabase::NumFrames() const {
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator(frames_dir_)) {
    if (entry.path().extension() == ".json") {
      count++;
    }
  }
  return count;
}

size_t FileDatabase::NumImages() const {
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator(images_dir_)) {
    if (entry.path().extension() == ".json") {
      count++;
    }
  }
  return count;
}

size_t FileDatabase::NumPosePriors() const {
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator(pose_priors_dir_)) {
    if (entry.path().extension() == ".json") {
      count++;
    }
  }
  return count;
}

size_t FileDatabase::NumKeypoints() const {
  size_t total = 0;
  for (const auto& entry : fs::directory_iterator(features_dir_)) {
    if (entry.path().extension() == ".bin" && 
        entry.path().string().find("_keypoints") != std::string::npos) {
      std::ifstream file(entry.path(), std::ios::binary);
      if (file.is_open()) {
        // Calculate number of keypoints based on file size
        file.seekg(0, std::ios::end);
        const size_t file_size = file.tellg();
        // Assuming each keypoint is stored as 6 floats (x, y, a11, a12, a21, a22)
        total += file_size / (6 * sizeof(float));
      }
    }
  }
  return total;
}

size_t FileDatabase::MaxNumKeypoints() const {
  size_t max_count = 0;
  for (const auto& entry : fs::directory_iterator(features_dir_)) {
    if (entry.path().extension() == ".bin" && 
        entry.path().string().find("_keypoints") != std::string::npos) {
      std::ifstream file(entry.path(), std::ios::binary);
      if (file.is_open()) {
        file.seekg(0, std::ios::end);
        const size_t file_size = file.tellg();
        const size_t count = file_size / (6 * sizeof(float));
        if (count > max_count) {
          max_count = count;
        }
      }
    }
  }
  return max_count;
}

size_t FileDatabase::NumKeypointsForImage(image_t image_id) const {
  const auto keypoints_path = GetKeypointsPath(image_id);
  if (!fs::exists(keypoints_path)) {
    return 0;
  }
  
  std::ifstream file(keypoints_path, std::ios::binary);
  if (!file.is_open()) {
    return 0;
  }
  
  file.seekg(0, std::ios::end);
  const size_t file_size = file.tellg();
  return file_size / (6 * sizeof(float));  // 6 floats per keypoint
}

size_t FileDatabase::NumDescriptors() const {
  size_t total = 0;
  for (const auto& entry : fs::directory_iterator(features_dir_)) {
    if (entry.path().extension() == ".bin" && 
        entry.path().string().find("_descriptors") != std::string::npos) {
      std::ifstream file(entry.path(), std::ios::binary);
      if (file.is_open()) {
        file.seekg(0, std::ios::end);
        const size_t file_size = file.tellg();
        // Assuming each descriptor is 128 bytes (SIFT) or similar
        // We need to determine descriptor size from the first descriptor
        if (file_size > 0) {
          // For now, assume 128-byte descriptors
          total += file_size / 128;
        }
      }
    }
  }
  return total;
}

size_t FileDatabase::MaxNumDescriptors() const {
  size_t max_count = 0;
  for (const auto& entry : fs::directory_iterator(features_dir_)) {
    if (entry.path().extension() == ".bin" && 
        entry.path().string().find("_descriptors") != std::string::npos) {
      std::ifstream file(entry.path(), std::ios::binary);
      if (file.is_open()) {
        file.seekg(0, std::ios::end);
        const size_t file_size = file.tellg();
        const size_t count = file_size / 128;  // Assume 128-byte descriptors
        if (count > max_count) {
          max_count = count;
        }
      }
    }
  }
  return max_count;
}

size_t FileDatabase::NumDescriptorsForImage(image_t image_id) const {
  const auto descriptors_path = GetDescriptorsPath(image_id);
  if (!fs::exists(descriptors_path)) {
    return 0;
  }
  
  std::ifstream file(descriptors_path, std::ios::binary);
  if (!file.is_open()) {
    return 0;
  }
  
  file.seekg(0, std::ios::end);
  const size_t file_size = file.tellg();
  return file_size / 128;  // Assume 128-byte descriptors
}

size_t FileDatabase::NumMatches() const {
  size_t total = 0;
  for (const auto& entry : fs::directory_iterator(matches_dir_)) {
    if (entry.path().extension() == ".json") {
      try {
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(entry.path().string(), pt);
        if (pt.count("matches") > 0) {
          total += pt.get_child("matches").size();
        }
      } catch (...) {
        // Skip invalid JSON files
      }
    }
  }
  return total;
}

size_t FileDatabase::NumInlierMatches() const {
  size_t total = 0;
  for (const auto& entry : fs::directory_iterator(two_view_geometries_dir_)) {
    if (entry.path().extension() == ".bin") {
      // For binary format, we need to read the file to get the number of inliers
      std::ifstream file(entry.path(), std::ios::binary);
      if (file.is_open()) {
        // Skip to the matches count part of the binary file
        // This is a simplified approach - in practice, we'd need to know the exact format
        // For now, we'll just count the number of two-view geometry files
        total++; // This is a placeholder - actual implementation would read the binary data
      }
    }
  }
  return total;
}

size_t FileDatabase::NumMatchedImagePairs() const {
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator(matches_dir_)) {
    if (entry.path().extension() == ".json") {
      count++;
    }
  }
  return count;
}

size_t FileDatabase::NumVerifiedImagePairs() const {
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator(two_view_geometries_dir_)) {
    if (entry.path().extension() == ".bin") {
      count++;
    }
  }
  return count;
}

// Read methods
Rig FileDatabase::ReadRig(rig_t rig_id) const {
  const std::string rig_filename = std::to_string(rig_id) + ".json";
  const fs::path rig_path = rigs_dir_ / rig_filename;

  if (!fs::exists(rig_path)) {
    return Rig{};  // Return empty rig
  }

  try {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(rig_path.string(), pt);

    Rig rig;
    // Deserialize rig from JSON
    rig.SetRigId(pt.get<rig_t>("rig_id", kInvalidRigId));

    return rig;
  } catch (...) {
    return Rig{};  // Return empty rig if JSON is invalid
  }
}

std::optional<Rig> FileDatabase::ReadRigWithSensor(sensor_t sensor_id) const {
  // Iterate through all rigs to find one with the specified sensor
  for (const auto& entry : fs::directory_iterator(rigs_dir_)) {
    if (entry.path().extension() == ".json") {
      try {
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(entry.path().string(), pt);

        // Check if this rig contains the sensor
        if (pt.count("ref_sensor_id") > 0) {
          const auto& ref_sensor_id_pt = pt.get_child("ref_sensor_id");
          if (ref_sensor_id_pt.count("id") > 0 && ref_sensor_id_pt.count("type") > 0 &&
              ref_sensor_id_pt.get<uint32_t>("id") == sensor_id.id &&
              static_cast<SensorType>(ref_sensor_id_pt.get<int>("type")) == sensor_id.type) {
            rig_t rig_id = pt.get<rig_t>("rig_id", kInvalidRigId);
            if (rig_id != kInvalidRigId) {
              return ReadRig(rig_id);
            }
          }
        }
      } catch (...) {
        // Skip invalid JSON files
      }
    }
  }
  return std::nullopt;
}

std::vector<Rig> FileDatabase::ReadAllRigs() const {
  std::vector<Rig> rigs;
  for (const auto& entry : fs::directory_iterator(rigs_dir_)) {
    if (entry.path().extension() == ".json") {
      const std::string stem = entry.path().stem().string();
      try {
        const rig_t rig_id = std::stoi(stem);
        rigs.push_back(ReadRig(rig_id));
      } catch (...) {
        // Skip invalid filenames
      }
    }
  }
  return rigs;
}

Camera FileDatabase::ReadCamera(camera_t camera_id) const {
  const std::string camera_filename = std::to_string(camera_id) + ".bin";
  const fs::path camera_path = cameras_dir_ / camera_filename;
  
  if (!fs::exists(camera_path)) {
    return Camera{};  // Return empty camera
  }
  
  std::ifstream file(camera_path, std::ios::binary);
  if (!file.is_open()) {
    return Camera{};
  }
  
  Camera camera;
  
  // Read camera_id
  camera.camera_id = ReadBinaryLittleEndian<camera_t>(&file);

  // Read model_id
  camera.model_id = ReadBinaryLittleEndian<CameraModelId>(&file);

  // Read dimensions
  camera.width = ReadBinaryLittleEndian<size_t>(&file);
  camera.height = ReadBinaryLittleEndian<size_t>(&file);

  // Read parameters
  const size_t num_params = ReadBinaryLittleEndian<size_t>(&file);
  camera.params.resize(num_params);
  for (size_t i = 0; i < num_params; ++i) {
    camera.params[i] = ReadBinaryLittleEndian<double>(&file);
  }

  // Read focal length prior flag
  camera.has_prior_focal_length = ReadBinaryLittleEndian<bool>(&file);
  
  return camera;
}

std::vector<Camera> FileDatabase::ReadAllCameras() const {
  std::vector<Camera> cameras;
  for (const auto& entry : fs::directory_iterator(cameras_dir_)) {
    if (entry.path().extension() == ".bin") {
      const std::string stem = entry.path().stem().string();
      try {
        const camera_t camera_id = std::stoi(stem);
        cameras.push_back(ReadCamera(camera_id));
      } catch (...) {
        // Skip invalid filenames
      }
    }
  }
  return cameras;
}

Image FileDatabase::ReadImage(image_t image_id) const {
  const std::string image_filename = "image_" + std::to_string(image_id) + ".json";
  const fs::path image_path = images_dir_ / image_filename;

  if (!fs::exists(image_path)) {
    return Image{};  // Return empty image
  }

  try {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(image_path.string(), pt);

    Image image;
    image.SetImageId(pt.get<image_t>("image_id", kInvalidImageId));
    image.SetName(pt.get<std::string>("name", ""));
    image.SetCameraId(pt.get<camera_t>("camera_id", kInvalidCameraId));

    // Note: Image pose is typically stored in the frame system, not directly in the image
    // For file-based storage, we might need to store width/height separately if needed
    // but they're usually derived from the camera

    if (pt.count("frame_id") > 0) {
      image.SetFrameId(pt.get<frame_t>("frame_id"));
    }

    return image;
  } catch (...) {
    return Image{};  // Return empty image if JSON is invalid
  }
}

std::optional<Image> FileDatabase::ReadImageWithName(const std::string& name) const {
  for (const auto& entry : fs::directory_iterator(images_dir_)) {
    if (entry.path().extension() == ".json") {
      try {
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(entry.path().string(), pt);
        if (pt.count("name") > 0 && pt.get<std::string>("name") == name) {
          Image image;
          image.SetImageId(pt.get<image_t>("image_id", kInvalidImageId));
          image.SetName(pt.get<std::string>("name", ""));
          image.SetCameraId(pt.get<camera_t>("camera_id", kInvalidCameraId));

          // Note: Image pose is typically stored in the frame system, not directly in the image
          // For file-based storage, we might need to store width/height separately if needed
          // but they're usually derived from the camera

          if (pt.count("frame_id") > 0) {
            image.SetFrameId(pt.get<frame_t>("frame_id"));
          }

          return image;
        }
      } catch (...) {
        // Skip invalid JSON files
      }
    }
  }
  return std::nullopt;
}

std::vector<Image> FileDatabase::ReadAllImages() const {
  std::vector<Image> images;
  for (const auto& entry : fs::directory_iterator(images_dir_)) {
    if (entry.path().extension() == ".json") {
      const std::string stem = entry.path().stem().string();
      if (stem.substr(0, 6) == "image_") {
        try {
          const std::string id_str = stem.substr(6);
          const image_t image_id = std::stoi(id_str);
          images.push_back(ReadImage(image_id));
        } catch (...) {
          // Skip invalid filenames
        }
      }
    }
  }
  return images;
}

FeatureKeypointsBlob FileDatabase::ReadKeypointsBlob(image_t image_id) const {
  const auto keypoints_path = GetKeypointsPath(image_id);
  if (!fs::exists(keypoints_path)) {
    return FeatureKeypointsBlob(0, 6);  // Empty blob with 6 columns
  }
  
  std::ifstream file(keypoints_path, std::ios::binary);
  if (!file.is_open()) {
    return FeatureKeypointsBlob(0, 6);
  }
  
  // Determine number of keypoints from file size
  file.seekg(0, std::ios::end);
  const size_t file_size = file.tellg();
  const size_t num_keypoints = file_size / (6 * sizeof(float));
  file.seekg(0, std::ios::beg);
  
  FeatureKeypointsBlob blob(num_keypoints, 6);
  
  for (size_t i = 0; i < num_keypoints; ++i) {
    blob(i, 0) = ReadBinaryLittleEndian<float>(&file);
    blob(i, 1) = ReadBinaryLittleEndian<float>(&file);
    blob(i, 2) = ReadBinaryLittleEndian<float>(&file);
    blob(i, 3) = ReadBinaryLittleEndian<float>(&file);
    blob(i, 4) = ReadBinaryLittleEndian<float>(&file);
    blob(i, 5) = ReadBinaryLittleEndian<float>(&file);
  }
  
  return blob;
}

FeatureKeypoints FileDatabase::ReadKeypoints(image_t image_id) const {
  const FeatureKeypointsBlob blob = ReadKeypointsBlob(image_id);
  FeatureKeypoints keypoints;
  keypoints.reserve(blob.rows());
  
  for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
    FeatureKeypoint kpt;
    kpt.x = blob(i, 0);
    kpt.y = blob(i, 1);
    kpt.a11 = blob(i, 2);
    kpt.a12 = blob(i, 3);
    kpt.a21 = blob(i, 4);
    kpt.a22 = blob(i, 5);
    keypoints.push_back(kpt);
  }
  
  return keypoints;
}

FeatureDescriptors FileDatabase::ReadDescriptors(image_t image_id) const {
  const auto descriptors_path = GetDescriptorsPath(image_id);
  if (!fs::exists(descriptors_path)) {
    return FeatureDescriptors(0, 128);  // Empty descriptors with 128 cols
  }
  
  std::ifstream file(descriptors_path, std::ios::binary);
  if (!file.is_open()) {
    return FeatureDescriptors(0, 128);
  }
  
  // Determine number of descriptors from file size
  file.seekg(0, std::ios::end);
  const size_t file_size = file.tellg();
  const size_t num_descriptors = file_size / (128 * sizeof(uint8_t));
  file.seekg(0, std::ios::beg);
  
  FeatureDescriptors descriptors(num_descriptors, 128);
  
  for (size_t i = 0; i < num_descriptors; ++i) {
    for (int j = 0; j < 128; ++j) {
      descriptors(i, j) = ReadBinaryLittleEndian<uint8_t>(&file);
    }
  }
  
  return descriptors;
}

FeatureMatchesBlob FileDatabase::ReadMatchesBlob(image_t image_id1,
                                                image_t image_id2) const {
  const auto matches_path = GetMatchesPath(image_id1, image_id2);
  if (!fs::exists(matches_path)) {
    return FeatureMatchesBlob(0, 2);  // Empty blob with 2 columns
  }

  try {
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(matches_path.string(), pt);

    if (pt.count("matches") == 0) {
      return FeatureMatchesBlob(0, 2);
    }

    const auto& matches_array = pt.get_child("matches");
    const size_t num_matches = matches_array.size();

    FeatureMatchesBlob blob(num_matches, 2);

    size_t i = 0;
    for (const auto& match_pair : matches_array) {
      auto match_it = match_pair.second.begin();
      blob(i, 0) = match_it->second.get_value<point2D_t>();
      ++match_it;
      blob(i, 1) = match_it->second.get_value<point2D_t>();
      ++i;
    }

    return blob;
  } catch (...) {
    return FeatureMatchesBlob(0, 2);  // Return empty blob if JSON is invalid
  }
}

FeatureMatches FileDatabase::ReadMatches(image_t image_id1, image_t image_id2) const {
  const FeatureMatchesBlob blob = ReadMatchesBlob(image_id1, image_id2);
  FeatureMatches matches;
  matches.reserve(blob.rows());

  for (FeatureMatchesBlob::Index i = 0; i < blob.rows(); ++i) {
    matches.emplace_back(static_cast<point2D_t>(blob(i, 0)),
                         static_cast<point2D_t>(blob(i, 1)));
  }

  return matches;
}

std::vector<std::pair<image_pair_t, FeatureMatchesBlob>> FileDatabase::ReadAllMatchesBlob() const {
  std::vector<std::pair<image_pair_t, FeatureMatchesBlob>> all_matches;

  for (const auto& entry : fs::directory_iterator(matches_dir_)) {
    if (entry.path().extension() == ".json") {
      const std::string stem = entry.path().stem().string();
      try {
        // Extract pair_id from filename like "123456789_matches.json"
        const size_t underscore_pos = stem.find('_');
        if (underscore_pos != std::string::npos) {
          const std::string pair_id_str = stem.substr(0, underscore_pos);
          const image_pair_t pair_id = std::stoull(pair_id_str);

          // Get image IDs from pair_id
          const auto image_pair = PairIdToImagePair(pair_id);

          const FeatureMatchesBlob blob = ReadMatchesBlob(image_pair.first, image_pair.second);
          all_matches.emplace_back(pair_id, blob);
        }
      } catch (...) {
        // Skip invalid filenames
      }
    }
  }

  return all_matches;
}

std::vector<std::pair<image_pair_t, FeatureMatches>> FileDatabase::ReadAllMatches() const {
  std::vector<std::pair<image_pair_t, FeatureMatches>> all_matches;
  
  for (const auto& entry : fs::directory_iterator(matches_dir_)) {
    if (entry.path().extension() == ".json") {
      const std::string stem = entry.path().stem().string();
      try {
        // Extract pair_id from filename like "123456789_matches.json"
        const size_t underscore_pos = stem.find('_');
        if (underscore_pos != std::string::npos) {
          const std::string pair_id_str = stem.substr(0, underscore_pos);
          const image_pair_t pair_id = std::stoull(pair_id_str);
          
          // Get image IDs from pair_id
          const auto image_pair = PairIdToImagePair(pair_id);
          
          const FeatureMatches matches = ReadMatches(image_pair.first, image_pair.second);
          all_matches.emplace_back(pair_id, matches);
        }
      } catch (...) {
        // Skip invalid filenames
      }
    }
  }
  
  return all_matches;
}

std::vector<std::pair<image_pair_t, int>> FileDatabase::ReadNumMatches() const {
  std::vector<std::pair<image_pair_t, int>> num_matches;

  for (const auto& entry : fs::directory_iterator(matches_dir_)) {
    if (entry.path().extension() == ".json") {
      const std::string stem = entry.path().stem().string();
      try {
        // Extract pair_id from filename like "123456789_matches.json"
        const size_t underscore_pos = stem.find('_');
        if (underscore_pos != std::string::npos) {
          const std::string pair_id_str = stem.substr(0, underscore_pos);
          const image_pair_t pair_id = std::stoull(pair_id_str);

          try {
            boost::property_tree::ptree pt;
            boost::property_tree::read_json(entry.path().string(), pt);

            if (pt.count("matches") > 0) {
              num_matches.emplace_back(pair_id, static_cast<int>(pt.get_child("matches").size()));
            }
          } catch (...) {
            // Skip invalid JSON files
          }
        }
      } catch (...) {
        // Skip invalid filenames
      }
    }
  }

  return num_matches;
}

TwoViewGeometry FileDatabase::ReadTwoViewGeometry(image_t image_id1,
                                                 image_t image_id2) const {
  const auto geometry_path = GetTwoViewGeometryPath(image_id1, image_id2);
  if (!fs::exists(geometry_path)) {
    return TwoViewGeometry{};  // Return empty geometry
  }
  
  std::ifstream file(geometry_path, std::ios::binary);
  if (!file.is_open()) {
    return TwoViewGeometry{};
  }
  
  TwoViewGeometry geometry;
  
  // Read configuration
  geometry.config = ReadBinaryLittleEndian<int>(&file);

  // Read fundamental matrix
  for (int i = 0; i < 9; ++i) {
    geometry.F(i / 3, i % 3) = ReadBinaryLittleEndian<double>(&file);
  }

  // Read essential matrix
  for (int i = 0; i < 9; ++i) {
    geometry.E(i / 3, i % 3) = ReadBinaryLittleEndian<double>(&file);
  }

  // Read homography matrix
  for (int i = 0; i < 9; ++i) {
    geometry.H(i / 3, i % 3) = ReadBinaryLittleEndian<double>(&file);
  }

  // Read rotation quaternion (w, x, y, z)
  Eigen::Vector4d qvec;
  for (int i = 0; i < 4; ++i) {
    qvec(i) = ReadBinaryLittleEndian<double>(&file);
  }
  geometry.cam2_from_cam1.rotation = Eigen::Quaterniond(qvec(0), qvec(1), qvec(2), qvec(3));

  // Read translation vector
  for (int i = 0; i < 3; ++i) {
    geometry.cam2_from_cam1.translation(i) = ReadBinaryLittleEndian<double>(&file);
  }

  // Read number of inlier matches
  const size_t num_inliers = ReadBinaryLittleEndian<size_t>(&file);

  // Read inlier matches
  geometry.inlier_matches.reserve(num_inliers);
  for (size_t i = 0; i < num_inliers; ++i) {
    const point2D_t idx1 = ReadBinaryLittleEndian<point2D_t>(&file);
    const point2D_t idx2 = ReadBinaryLittleEndian<point2D_t>(&file);
    geometry.inlier_matches.emplace_back(idx1, idx2);
  }
  
  return geometry;
}

std::vector<std::pair<image_pair_t, TwoViewGeometry>> FileDatabase::ReadTwoViewGeometries() const {
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> geometries;
  
  for (const auto& entry : fs::directory_iterator(two_view_geometries_dir_)) {
    if (entry.path().extension() == ".bin") {
      const std::string stem = entry.path().stem().string();
      try {
        // Extract pair_id from filename like "123456789_twoview.bin"
        const size_t underscore_pos = stem.find('_');
        if (underscore_pos != std::string::npos) {
          const std::string pair_id_str = stem.substr(0, underscore_pos);
          const image_pair_t pair_id = std::stoull(pair_id_str);
          
          // Get image IDs from pair_id
          const auto image_pair = PairIdToImagePair(pair_id);
          
          const TwoViewGeometry geometry = ReadTwoViewGeometry(image_pair.first, image_pair.second);
          geometries.emplace_back(pair_id, geometry);
        }
      } catch (...) {
        // Skip invalid filenames
      }
    }
  }
  
  return geometries;
}

std::vector<std::pair<image_pair_t, int>> FileDatabase::ReadTwoViewGeometryNumInliers() const {
  std::vector<std::pair<image_pair_t, int>> num_inliers;
  
  for (const auto& entry : fs::directory_iterator(two_view_geometries_dir_)) {
    if (entry.path().extension() == ".bin") {
      const std::string stem = entry.path().stem().string();
      try {
        // Extract pair_id from filename like "123456789_twoview.bin"
        const size_t underscore_pos = stem.find('_');
        if (underscore_pos != std::string::npos) {
          const std::string pair_id_str = stem.substr(0, underscore_pos);
          const image_pair_t pair_id = std::stoull(pair_id_str);
          
          // Read the geometry to get number of inliers
          const auto geometry_path = entry.path();
          std::ifstream file(geometry_path, std::ios::binary);
          if (file.is_open()) {
            // Skip to the number of inliers (after F, E, H matrices and rotation/translation)
            file.seekg(sizeof(int) + 9*sizeof(double) + 9*sizeof(double) + 9*sizeof(double) + 4*sizeof(double) + 3*sizeof(double), std::ios::beg);

            const size_t num_inlier_matches = ReadBinaryLittleEndian<size_t>(&file);
            num_inliers.emplace_back(pair_id, static_cast<int>(num_inlier_matches));
          }
        }
      } catch (...) {
        // Skip invalid filenames
      }
    }
  }
  
  return num_inliers;
}

// Write methods
rig_t FileDatabase::WriteRig(const Rig& rig, bool use_rig_id) {
  rig_t rig_id = rig.RigId();
  if (!use_rig_id || rig_id == kInvalidRigId) {
    // Generate a new ID
    rig_id = static_cast<rig_t>(NumRigs() + 1);
  }

  const std::string rig_filename = std::to_string(rig_id) + ".json";
  const fs::path rig_path = rigs_dir_ / rig_filename;

  boost::property_tree::ptree pt;
  pt.put("rig_id", rig_id);
  // Serialize other rig properties as needed

  std::ofstream file(rig_path);
  boost::property_tree::write_json(file, pt);

  return rig_id;
}

camera_t FileDatabase::WriteCamera(const Camera& camera, bool use_camera_id) {
  camera_t camera_id = camera.camera_id;
  if (!use_camera_id || camera_id == kInvalidCameraId) {
    // Generate a new ID
    camera_id = static_cast<camera_t>(NumCameras() + 1);
  }
  
  const std::string camera_filename = std::to_string(camera_id) + ".bin";
  const fs::path camera_path = cameras_dir_ / camera_filename;
  
  std::ofstream file(camera_path, std::ios::binary);
  
  // Write camera_id
  WriteBinaryLittleEndian<camera_t>(&file, camera_id);
  
  // Write model_id
  WriteBinaryLittleEndian<CameraModelId>(&file, camera.model_id);
  
  // Write dimensions
  WriteBinaryLittleEndian<size_t>(&file, camera.width);
  WriteBinaryLittleEndian<size_t>(&file, camera.height);
  
  // Write parameters
  const size_t num_params = camera.params.size();
  WriteBinaryLittleEndian<size_t>(&file, num_params);
  for (size_t i = 0; i < num_params; ++i) {
    WriteBinaryLittleEndian<double>(&file, camera.params[i]);
  }
  
  // Write focal length prior flag
  WriteBinaryLittleEndian<bool>(&file, camera.has_prior_focal_length);
  
  return camera_id;
}

image_t FileDatabase::WriteImage(const Image& image, bool use_image_id) {
  image_t image_id = image.ImageId();
  if (!use_image_id || image_id == kInvalidImageId) {
    // Generate a new ID
    image_id = static_cast<image_t>(NumImages() + 1);
  }

  const std::string image_filename = "image_" + std::to_string(image_id) + ".json";
  const fs::path image_path = images_dir_ / image_filename;

  boost::property_tree::ptree pt;
  pt.put("image_id", image_id);
  pt.put("name", image.Name());
  pt.put("camera_id", image.CameraId());

  // Note: Image pose is typically stored in the frame system, not directly in the image
  // For file-based storage, we might need to store width/height separately if needed
  // but they're usually derived from the camera

  if (image.HasFrameId()) {
    pt.put("frame_id", image.FrameId());
  }

  std::ofstream file(image_path);
  boost::property_tree::write_json(file, pt);

  return image_id;
}

void FileDatabase::WriteKeypoints(image_t image_id,
                                 const FeatureKeypoints& keypoints) {
  WriteKeypoints(image_id, FeatureKeypointsToBlob(keypoints));
}

void FileDatabase::WriteKeypoints(image_t image_id,
                                 const FeatureKeypointsBlob& blob) {
  const auto keypoints_path = GetKeypointsPath(image_id);
  std::ofstream file(keypoints_path, std::ios::binary);
  
  for (FeatureKeypointsBlob::Index i = 0; i < blob.rows(); ++i) {
    for (FeatureKeypointsBlob::Index j = 0; j < blob.cols(); ++j) {
      const float val = static_cast<float>(blob(i, j));
      WriteBinaryLittleEndian<float>(&file, val);
    }
  }
}

void FileDatabase::WriteDescriptors(image_t image_id,
                                   const FeatureDescriptors& descriptors) {
  const auto descriptors_path = GetDescriptorsPath(image_id);
  std::ofstream file(descriptors_path, std::ios::binary);
  
  for (FeatureDescriptors::Index i = 0; i < descriptors.rows(); ++i) {
    for (FeatureDescriptors::Index j = 0; j < descriptors.cols(); ++j) {
      const uint8_t val = static_cast<uint8_t>(descriptors(i, j));
      WriteBinaryLittleEndian<uint8_t>(&file, val);
    }
  }
}

void FileDatabase::WriteMatches(image_t image_id1,
                               image_t image_id2,
                               const FeatureMatches& matches) {
  WriteMatches(image_id1, image_id2, FeatureMatchesToBlob(matches));
}

void FileDatabase::WriteMatches(image_t image_id1,
                               image_t image_id2,
                               const FeatureMatchesBlob& blob) {
  const auto matches_path = GetMatchesPath(image_id1, image_id2);
  std::ofstream file(matches_path);

  boost::property_tree::ptree pt;
  pt.put("image_id1", image_id1);
  pt.put("image_id2", image_id2);
  pt.put("num_matches", static_cast<int>(blob.rows()));

  boost::property_tree::ptree matches_array;
  for (FeatureMatchesBlob::Index i = 0; i < blob.rows(); ++i) {
    boost::property_tree::ptree match_pair;
    boost::property_tree::ptree idx1, idx2;
    idx1.put_value(static_cast<int>(blob(i, 0)));
    idx2.put_value(static_cast<int>(blob(i, 1)));
    match_pair.push_back(std::make_pair("", idx1));
    match_pair.push_back(std::make_pair("", idx2));
    matches_array.push_back(std::make_pair("", match_pair));
  }
  pt.add_child("matches", matches_array);

  boost::property_tree::write_json(file, pt);
}

void FileDatabase::WriteTwoViewGeometry(
    image_t image_id1,
    image_t image_id2,
    const TwoViewGeometry& two_view_geometry) {
  const auto geometry_path = GetTwoViewGeometryPath(image_id1, image_id2);
  std::ofstream file(geometry_path, std::ios::binary);
  
  // Write configuration
  WriteBinaryLittleEndian<int>(&file, two_view_geometry.config);
  
  // Write fundamental matrix
  for (int i = 0; i < 9; ++i) {
    const double val = two_view_geometry.F(i / 3, i % 3);
    WriteBinaryLittleEndian<double>(&file, val);
  }
  
  // Write essential matrix
  for (int i = 0; i < 9; ++i) {
    const double val = two_view_geometry.E(i / 3, i % 3);
    WriteBinaryLittleEndian<double>(&file, val);
  }
  
  // Write homography matrix
  for (int i = 0; i < 9; ++i) {
    const double val = two_view_geometry.H(i / 3, i % 3);
    WriteBinaryLittleEndian<double>(&file, val);
  }
  
  // Write rotation quaternion (w, x, y, z)
  const Eigen::Vector4d qvec(two_view_geometry.cam2_from_cam1.rotation.w(),
                             two_view_geometry.cam2_from_cam1.rotation.x(),
                             two_view_geometry.cam2_from_cam1.rotation.y(),
                             two_view_geometry.cam2_from_cam1.rotation.z());
  for (int i = 0; i < 4; ++i) {
    WriteBinaryLittleEndian<double>(&file, qvec(i));
  }
  
  // Write translation vector
  for (int i = 0; i < 3; ++i) {
    WriteBinaryLittleEndian<double>(&file, two_view_geometry.cam2_from_cam1.translation(i));
  }
  
  // Write number of inlier matches
  const size_t num_inliers = two_view_geometry.inlier_matches.size();
  WriteBinaryLittleEndian<size_t>(&file, num_inliers);
  
  // Write inlier matches
  for (const auto& match : two_view_geometry.inlier_matches) {
    WriteBinaryLittleEndian<point2D_t>(&file, match.point2D_idx1);
    WriteBinaryLittleEndian<point2D_t>(&file, match.point2D_idx2);
  }
}

// Update methods
void FileDatabase::UpdateRig(const Rig& rig) {
  // For file-based storage, update is the same as write
  WriteRig(rig, true);
}

void FileDatabase::UpdateCamera(const Camera& camera) {
  // For file-based storage, update is the same as write
  WriteCamera(camera, true);
}

void FileDatabase::UpdateImage(const Image& image) {
  // For file-based storage, update is the same as write
  WriteImage(image, true);
}

void FileDatabase::UpdateKeypoints(image_t image_id,
                                  const FeatureKeypoints& keypoints) {
  UpdateKeypoints(image_id, FeatureKeypointsToBlob(keypoints));
}

void FileDatabase::UpdateKeypoints(image_t image_id,
                                  const FeatureKeypointsBlob& blob) {
  // For file-based storage, update is the same as write
  WriteKeypoints(image_id, blob);
}

void FileDatabase::UpdateTwoViewGeometry(
    image_t image_id1,
    image_t image_id2,
    const TwoViewGeometry& two_view_geometry) {
  // For file-based storage, update is the same as write
  WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
}

// Delete methods
void FileDatabase::DeleteMatches(image_t image_id1, image_t image_id2) {
  const auto matches_path = GetMatchesPath(image_id1, image_id2);
  if (fs::exists(matches_path)) {
    fs::remove(matches_path);
  }
}

void FileDatabase::DeleteTwoViewGeometry(image_t image_id1, image_t image_id2) {
  const auto geometry_path = GetTwoViewGeometryPath(image_id1, image_id2);
  if (fs::exists(geometry_path)) {
    fs::remove(geometry_path);
  }
}

void FileDatabase::DeleteInlierMatches(image_t image_id1, image_t image_id2) {
  if (!ExistsTwoViewGeometry(image_id1, image_id2)) {
    return;
  }
  
  TwoViewGeometry geometry = ReadTwoViewGeometry(image_id1, image_id2);
  geometry.inlier_matches.clear();
  UpdateTwoViewGeometry(image_id1, image_id2, geometry);
}

// Clear methods
void FileDatabase::ClearAllTables() {
  ClearRigs();
  ClearCameras();
  ClearFrames();
  ClearImages();
  ClearPosePriors();
  ClearDescriptors();
  ClearKeypoints();
  ClearMatches();
  ClearTwoViewGeometries();
}

void FileDatabase::ClearRigs() {
  for (const auto& entry : fs::directory_iterator(rigs_dir_)) {
    fs::remove(entry.path());
  }
}

void FileDatabase::ClearCameras() {
  for (const auto& entry : fs::directory_iterator(cameras_dir_)) {
    fs::remove(entry.path());
  }
}

void FileDatabase::ClearFrames() {
  for (const auto& entry : fs::directory_iterator(frames_dir_)) {
    fs::remove(entry.path());
  }
}

void FileDatabase::ClearImages() {
  for (const auto& entry : fs::directory_iterator(images_dir_)) {
    fs::remove(entry.path());
  }
}

void FileDatabase::ClearPosePriors() {
  for (const auto& entry : fs::directory_iterator(pose_priors_dir_)) {
    fs::remove(entry.path());
  }
}

void FileDatabase::ClearDescriptors() {
  for (const auto& entry : fs::directory_iterator(features_dir_)) {
    if (entry.path().string().find("_descriptors") != std::string::npos) {
      fs::remove(entry.path());
    }
  }
}

void FileDatabase::ClearKeypoints() {
  for (const auto& entry : fs::directory_iterator(features_dir_)) {
    if (entry.path().string().find("_keypoints") != std::string::npos) {
      fs::remove(entry.path());
    }
  }
}

void FileDatabase::ClearMatches() {
  for (const auto& entry : fs::directory_iterator(matches_dir_)) {
    fs::remove(entry.path());
  }
}

void FileDatabase::ClearTwoViewGeometries() {
  for (const auto& entry : fs::directory_iterator(two_view_geometries_dir_)) {
    fs::remove(entry.path());
  }
}

// Transaction methods (no-op for file-based storage)
void FileDatabase::BeginTransaction() const {
  // File-based storage doesn't support transactions, so this is a no-op
}

void FileDatabase::EndTransaction() const {
  // File-based storage doesn't support transactions, so this is a no-op
}

Frame FileDatabase::ReadFrame(frame_t frame_id) const {
  // Placeholder implementation - return empty frame
  return Frame();
}

std::vector<Frame> FileDatabase::ReadAllFrames() const {
  // Placeholder implementation - return empty vector
  return std::vector<Frame>();
}

PosePrior FileDatabase::ReadPosePrior(pose_prior_t pose_prior_id,
                                    bool /*is_deprecated_image_prior*/) const {
  // Placeholder implementation - return empty pose prior
  return PosePrior();
}

std::vector<PosePrior> FileDatabase::ReadAllPosePriors() const {
  // Placeholder implementation - return empty vector
  return std::vector<PosePrior>();
}

frame_t FileDatabase::WriteFrame(const Frame& frame, bool use_frame_id) {
  // Placeholder implementation - return invalid ID
  return kInvalidFrameId;
}

pose_prior_t FileDatabase::WritePosePrior(const PosePrior& pose_prior,
                                        bool use_pose_prior_id) {
  // Placeholder implementation - return invalid ID
  return kInvalidPosePriorId;
}

void FileDatabase::UpdateFrame(const Frame& frame) {
  // Placeholder implementation - do nothing
}

void FileDatabase::UpdatePosePrior(const PosePrior& pose_prior) {
  // Placeholder implementation - do nothing
}

std::shared_ptr<Database> OpenFileDatabase(const std::string& path) {
  // Check if path is a directory (indicating file-based database)
  // or if it has characteristics of a file database setup
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) {
    // If it's a directory, initialize as FileDatabase
    return std::make_shared<FileDatabase>(path);
  }
  // If path is not a directory, let other database implementations handle it
  throw std::runtime_error("FileDatabase factory: Path is not a directory, deferring to other implementations");
}
}  // namespace colmap