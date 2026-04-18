#pragma once

#include "colmap/scene/database.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/types.h"

#include <filesystem>
#include <mutex>
#include <unordered_map>

namespace colmap {

// File-based database implementation that stores data in text (JSON) and binary files
// instead of SQLite, making it easier to debug and distribute across clusters
class FileDatabase : public Database {
 public:
  explicit FileDatabase(const std::string& root_dir);
  ~FileDatabase() override;

  // Open and close database
  void Close() override;

  // Check if entry already exists in database
  bool ExistsRig(rig_t rig_id) const override;
  bool ExistsCamera(camera_t camera_id) const override;
  bool ExistsFrame(frame_t frame_id) const override;
  bool ExistsImage(image_t image_id) const override;
  bool ExistsImageWithName(const std::string& name) const override;
  bool ExistsPosePrior(pose_prior_t pose_prior_id,
                       bool is_deprecated_image_prior = true) const override;
  bool ExistsKeypoints(image_t image_id) const override;
  bool ExistsDescriptors(image_t image_id) const override;
  bool ExistsMatches(image_t image_id1, image_t image_id2) const override;
  bool ExistsTwoViewGeometry(image_t image_id1, image_t image_id2) const override;

  // Number of entries in tables
  size_t NumRigs() const override;
  size_t NumCameras() const override;
  size_t NumFrames() const override;
  size_t NumImages() const override;
  size_t NumPosePriors() const override;
  size_t NumKeypoints() const override;
  size_t MaxNumKeypoints() const override;
  size_t NumKeypointsForImage(image_t image_id) const override;
  size_t NumDescriptors() const override;
  size_t MaxNumDescriptors() const override;
  size_t NumDescriptorsForImage(image_t image_id) const override;
  size_t NumMatches() const override;
  size_t NumInlierMatches() const override;
  size_t NumMatchedImagePairs() const override;
  size_t NumVerifiedImagePairs() const override;

  // Read existing entries
  Rig ReadRig(rig_t rig_id) const override;
  std::optional<Rig> ReadRigWithSensor(sensor_t sensor_id) const override;
  std::vector<Rig> ReadAllRigs() const override;

  Camera ReadCamera(camera_t camera_id) const override;
  std::vector<Camera> ReadAllCameras() const override;

  Frame ReadFrame(frame_t frame_id) const override;
  std::vector<Frame> ReadAllFrames() const override;

  Image ReadImage(image_t image_id) const override;
  std::optional<Image> ReadImageWithName(const std::string& name) const override;
  std::vector<Image> ReadAllImages() const override;

  PosePrior ReadPosePrior(pose_prior_t pose_prior_id,
                          bool is_deprecated_image_prior = true) const override;
  std::vector<PosePrior> ReadAllPosePriors() const override;

  FeatureKeypointsBlob ReadKeypointsBlob(image_t image_id) const override;
  FeatureKeypoints ReadKeypoints(image_t image_id) const override;
  FeatureDescriptors ReadDescriptors(image_t image_id) const override;

  FeatureMatchesBlob ReadMatchesBlob(image_t image_id1,
                                     image_t image_id2) const override;
  FeatureMatches ReadMatches(image_t image_id1, image_t image_id2) const override;
  std::vector<std::pair<image_pair_t, FeatureMatchesBlob>> ReadAllMatchesBlob()
      const override;
  std::vector<std::pair<image_pair_t, FeatureMatches>> ReadAllMatches()
      const override;
  std::vector<std::pair<image_pair_t, int>> ReadNumMatches() const override;

  TwoViewGeometry ReadTwoViewGeometry(image_t image_id1,
                                      image_t image_id2) const override;
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> ReadTwoViewGeometries()
      const override;
  std::vector<std::pair<image_pair_t, int>> ReadTwoViewGeometryNumInliers()
      const override;

  // Write new entries
  rig_t WriteRig(const Rig& rig, bool use_rig_id = false) override;
  camera_t WriteCamera(const Camera& camera, bool use_camera_id = false) override;
  frame_t WriteFrame(const Frame& frame, bool use_frame_id = false) override;
  image_t WriteImage(const Image& image, bool use_image_id = false) override;
  pose_prior_t WritePosePrior(const PosePrior& pose_prior,
                              bool use_pose_prior_id = false) override;

  void WriteKeypoints(image_t image_id,
                      const FeatureKeypoints& keypoints) override;
  void WriteKeypoints(image_t image_id,
                      const FeatureKeypointsBlob& blob) override;
  void WriteDescriptors(image_t image_id,
                        const FeatureDescriptors& descriptors) override;
  void WriteMatches(image_t image_id1,
                    image_t image_id2,
                    const FeatureMatches& matches) override;
  void WriteMatches(image_t image_id1,
                    image_t image_id2,
                    const FeatureMatchesBlob& blob) override;
  void WriteTwoViewGeometry(
      image_t image_id1,
      image_t image_id2,
      const TwoViewGeometry& two_view_geometry) override;

  // Update existing entries
  void UpdateRig(const Rig& rig) override;
  void UpdateCamera(const Camera& camera) override;
  void UpdateFrame(const Frame& frame) override;
  void UpdateImage(const Image& image) override;
  void UpdatePosePrior(const PosePrior& pose_prior) override;
  void UpdateKeypoints(image_t image_id,
                       const FeatureKeypoints& keypoints) override;
  void UpdateKeypoints(image_t image_id,
                       const FeatureKeypointsBlob& blob) override;
  void UpdateTwoViewGeometry(
      image_t image_id1,
      image_t image_id2,
      const TwoViewGeometry& two_view_geometry) override;

  // Delete entries
  void DeleteMatches(image_t image_id1, image_t image_id2) override;
  void DeleteTwoViewGeometry(image_t image_id1, image_t image_id2) override;
  void DeleteInlierMatches(image_t image_id1, image_t image_id2) override;

  // Clear tables
  void ClearAllTables() override;
  void ClearRigs() override;
  void ClearCameras() override;
  void ClearFrames() override;
  void ClearImages() override;
  void ClearPosePriors() override;
  void ClearDescriptors() override;
  void ClearKeypoints() override;
  void ClearMatches() override;
  void ClearTwoViewGeometries() override;

  // Transaction support
  void BeginTransaction() const override;
  void EndTransaction() const override;

 private:
  // Helper methods for file paths
  std::filesystem::path GetRigsPath() const;
  std::filesystem::path GetCamerasPath() const;
  std::filesystem::path GetFramesPath() const;
  std::filesystem::path GetImagesPath() const;
  std::filesystem::path GetPosePriorsPath() const;
  std::filesystem::path GetKeypointsPath(image_t image_id) const;
  std::filesystem::path GetDescriptorsPath(image_t image_id) const;
  std::filesystem::path GetMatchesPath(image_t image_id1, image_t image_id2) const;
  std::filesystem::path GetTwoViewGeometryPath(image_t image_id1, image_t image_id2) const;

  // Helper methods for reading/writing individual records
  void LoadCache() const;
  void SaveCache() const;
  
  // Root directory and subdirectories
  std::filesystem::path root_dir_;
  std::filesystem::path rigs_dir_;
  std::filesystem::path cameras_dir_;
  std::filesystem::path frames_dir_;
  std::filesystem::path images_dir_;
  std::filesystem::path pose_priors_dir_;
  std::filesystem::path features_dir_;
  std::filesystem::path matches_dir_;
  std::filesystem::path two_view_geometries_dir_;

  // Cache for performance
  mutable bool cache_loaded_ = false;
  mutable std::unordered_map<camera_t, Camera> cameras_cache_;
  mutable std::unordered_map<image_t, Image> images_cache_;
  mutable std::unordered_map<image_pair_t, FeatureMatches> matches_cache_;
  mutable std::unordered_map<image_pair_t, TwoViewGeometry> two_view_geometry_cache_;

  // Thread safety
  mutable std::mutex mutex_;
};

std::shared_ptr<Database> OpenFileDatabase(const std::string& path);

}  // namespace colmap