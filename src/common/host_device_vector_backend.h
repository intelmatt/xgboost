/**
 * Copyright 2026, XGBoost contributors
 *
 * @brief Storage backends of HostDeviceVector.
 *
 * `HostDeviceVectorImpl` (host_device_vector.cc) is compiled once by the host compiler and owns
 * the host vector and one backend that refers to it. The CPU backend is always available, the
 * CUDA and SYCL backends are defined in their device TUs and selected by the device type. When
 * the device type changes, the data is synchronized to the host vector and the backend of the
 * new device type takes over. The host vector itself is never replaced, so references returned
 * by `HostVector()` remain valid across `SetDevice()`.
 *
 * This header must not include any CUDA or SYCL headers.
 */
#ifndef XGBOOST_COMMON_HOST_DEVICE_VECTOR_BACKEND_H_
#define XGBOOST_COMMON_HOST_DEVICE_VECTOR_BACKEND_H_

#include <cstddef>  // for size_t
#include <cstdint>  // for int32_t, uint8_t
#include <memory>   // for unique_ptr
#include <vector>   // for vector

#include "xgboost/context.h"             // for DeviceOrd
#include "xgboost/host_device_vector.h"  // for GPUAccess
#include "xgboost/span.h"                // for Span

namespace xgboost::common {
enum class HDVBackendKind : std::uint8_t { kCPU, kCUDA, kSycl };

template <typename T>
class HDVBackend {
 public:
  virtual ~HDVBackend() = default;

  [[nodiscard]] virtual HDVBackendKind Kind() const = 0;
  [[nodiscard]] virtual std::size_t Size() const = 0;
  [[nodiscard]] virtual DeviceOrd Device() const = 0;
  [[nodiscard]] virtual GPUAccess Access() const = 0;

  virtual T* DevicePointer() = 0;
  virtual T const* ConstDevicePointer() = 0;
  virtual Span<T> DeviceSpan() = 0;
  virtual Span<T const> ConstDeviceSpan() = 0;
  virtual std::vector<T>& HostVector() = 0;
  virtual std::vector<T> const& ConstHostVector() = 0;

  virtual void Fill(T v) = 0;
  /** @brief Copy from a backend of the same kind. */
  virtual void Copy(HDVBackend<T>* other) = 0;
  virtual void Copy(Span<T const> other) = 0;
  /** @brief Extend with a backend of the same kind. */
  virtual void Extend(HDVBackend<T>* other) = 0;
  virtual void SetDevice(DeviceOrd device) = 0;
  virtual void Resize(std::size_t new_size) = 0;
  virtual void Resize(std::size_t new_size, T v) = 0;
  /** @brief Make the host vector hold the data, before another backend takes it over. */
  virtual void SyncToHost() = 0;
};

/**
 * Factories of each device backend. The backend stores the data in `h_data` while it is on the
 * host; `h_data` must outlive the backend. `MakeHDV` has the semantics of the HostDeviceVector
 * constructors, `AdoptHostHDV` takes over the data already in `h_data` and sets the device
 * without copying the data to the device.
 */
namespace cuda_impl {
template <typename T>
std::unique_ptr<HDVBackend<T>> MakeHDV(std::vector<T>* h_data, std::size_t size, T v,
                                       DeviceOrd device);
template <typename T>
std::unique_ptr<HDVBackend<T>> MakeHDV(std::vector<T>* h_data, Span<T const> init,
                                       DeviceOrd device);
template <typename T>
std::unique_ptr<HDVBackend<T>> AdoptHostHDV(std::vector<T>* h_data, DeviceOrd device);
}  // namespace cuda_impl

namespace sycl_impl {
template <typename T>
std::unique_ptr<HDVBackend<T>> MakeHDV(std::vector<T>* h_data, std::size_t size, T v,
                                       DeviceOrd device);
template <typename T>
std::unique_ptr<HDVBackend<T>> MakeHDV(std::vector<T>* h_data, Span<T const> init,
                                       DeviceOrd device);
template <typename T>
std::unique_ptr<HDVBackend<T>> AdoptHostHDV(std::vector<T>* h_data, DeviceOrd device);
}  // namespace sycl_impl
}  // namespace xgboost::common

#if defined(__APPLE__) || defined(__EMSCRIPTEN__)
// std::size_t is not the same type as bst_idx_t (std::uint64_t) on these platforms.
#define XGBOOST_HDV_PLATFORM_TYPES(M) M(std::size_t)
#else
#define XGBOOST_HDV_PLATFORM_TYPES(M)
#endif  // defined(__APPLE__) || defined(__EMSCRIPTEN__)

#if defined(XGBOOST_USE_SYCL)
#define XGBOOST_HDV_SYCL_TYPES(M) M(::xgboost::sycl::predictor::Node)
#else
#define XGBOOST_HDV_SYCL_TYPES(M)
#endif  // defined(XGBOOST_USE_SYCL)

/**
 * @brief Apply M to each element type of HostDeviceVector. The front end and all device backends
 * are instantiated for the same list. Requires xgboost/data.h, xgboost/tree_model.h and, with
 * SYCL, plugin/sycl/predictor/node.h.
 */
#define XGBOOST_HDV_FOR_EACH_TYPE(M)                     \
  M(float)                                               \
  M(double)                                              \
  M(::xgboost::GradientPair)                             \
  M(::xgboost::GradientPairPrecise)                      \
  M(::xgboost::GradientPairInt64)                        \
  M(std::int32_t)                                        \
  M(std::uint8_t)                                        \
  M(std::int8_t)                                         \
  M(::xgboost::FeatureType)                              \
  M(::xgboost::Entry)                                    \
  M(::xgboost::bst_idx_t)                                \
  M(std::uint32_t)                                       \
  M(::xgboost::RegTree::Node)                            \
  M(::xgboost::RegTree::CategoricalSplitMatrix::Segment) \
  M(::xgboost::RTreeNodeStat)                            \
  XGBOOST_HDV_PLATFORM_TYPES(M)                          \
  XGBOOST_HDV_SYCL_TYPES(M)

/** @brief Explicitly instantiate the factories of a device backend namespace. */
#define XGBOOST_HDV_INSTANTIATE_FACTORIES(ns, T)                                  \
  template std::unique_ptr<::xgboost::common::HDVBackend<T>> ns::MakeHDV<T>(      \
      std::vector<T>*, std::size_t, T, ::xgboost::DeviceOrd);                     \
  template std::unique_ptr<::xgboost::common::HDVBackend<T>> ns::MakeHDV<T>(      \
      std::vector<T>*, ::xgboost::common::Span<T const>, ::xgboost::DeviceOrd);   \
  template std::unique_ptr<::xgboost::common::HDVBackend<T>> ns::AdoptHostHDV<T>( \
      std::vector<T>*, ::xgboost::DeviceOrd);

#endif  // XGBOOST_COMMON_HOST_DEVICE_VECTOR_BACKEND_H_
