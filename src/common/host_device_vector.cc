/**
 * Copyright 2017-2026, XGBoost contributors
 *
 * @brief Front end of HostDeviceVector and its CPU backend. Compiled by the host compiler in all
 * builds; the CUDA and SYCL backends are selected at run time by the device type.
 */
#include "xgboost/host_device_vector.h"

#include <algorithm>  // for copy, fill
#include <cstddef>    // for size_t
#include <cstdint>    // for int32_t, uint8_t
#include <memory>     // for unique_ptr, make_unique
#include <utility>    // for move
#include <vector>     // for vector

#include "host_device_vector_backend.h"
#include "xgboost/base.h"        // for GradientPair
#include "xgboost/data.h"        // for Entry, FeatureType
#include "xgboost/logging.h"     // for CHECK_EQ
#include "xgboost/tree_model.h"  // for RegTree

#if defined(XGBOOST_USE_SYCL)
#include "../../plugin/sycl/predictor/node.h"  // for Node
#endif                                         // defined(XGBOOST_USE_SYCL)

namespace xgboost {
namespace common {
namespace {
template <typename T>
class HostOnlyBackend : public HDVBackend<T> {
 public:
  explicit HostOnlyBackend(std::vector<T>* h_data) : data_h_(*h_data) {}

  [[nodiscard]] HDVBackendKind Kind() const override { return HDVBackendKind::kCPU; }
  [[nodiscard]] std::size_t Size() const override { return data_h_.size(); }
  [[nodiscard]] DeviceOrd Device() const override { return DeviceOrd::CPU(); }
  [[nodiscard]] GPUAccess Access() const override { return GPUAccess::kNone; }

  T* DevicePointer() override { return nullptr; }
  T const* ConstDevicePointer() override { return nullptr; }
  Span<T> DeviceSpan() override { return {}; }
  Span<T const> ConstDeviceSpan() override { return {}; }
  std::vector<T>& HostVector() override { return data_h_; }
  std::vector<T> const& ConstHostVector() override { return data_h_; }

  void Fill(T v) override { std::fill(data_h_.begin(), data_h_.end(), v); }
  void Copy(HDVBackend<T>* other) override {
    auto const& h_other = other->ConstHostVector();
    CHECK_EQ(Size(), h_other.size());
    std::copy(h_other.cbegin(), h_other.cend(), data_h_.begin());
  }
  void Copy(Span<T const> other) override {
    CHECK_EQ(Size(), other.size());
    std::copy(other.cbegin(), other.cend(), data_h_.begin());
  }
  void Extend(HDVBackend<T>* other) override {
    auto ori_size = Size();
    data_h_.resize(ori_size + other->Size());
    auto const& h_other = other->ConstHostVector();
    std::copy(h_other.cbegin(), h_other.cend(), data_h_.begin() + ori_size);
  }
  void SetDevice(DeviceOrd) override {}
  void Resize(std::size_t new_size) override { data_h_.resize(new_size, T{}); }
  void Resize(std::size_t new_size, T v) override { data_h_.resize(new_size, v); }
  void SyncToHost() override {}

 private:
  std::vector<T>& data_h_;
};

/**
 * @brief The backend kind for a device. Devices without a backend in this build are handled by
 * the CPU kind, which keeps the current backend.
 */
HDVBackendKind KindOf(DeviceOrd device) {
#if defined(XGBOOST_USE_CUDA)
  if (device.IsCUDA()) {
    return HDVBackendKind::kCUDA;
  }
#endif  // defined(XGBOOST_USE_CUDA)
#if defined(XGBOOST_USE_SYCL)
  if (device.IsSycl()) {
    return HDVBackendKind::kSycl;
  }
#endif  // defined(XGBOOST_USE_SYCL)
  (void)device;
  return HDVBackendKind::kCPU;
}

template <typename T>
std::unique_ptr<HDVBackend<T>> MakeBackend(std::vector<T>* h_data, std::size_t size, T v,
                                           DeviceOrd device) {
  switch (KindOf(device)) {
#if defined(XGBOOST_USE_CUDA)
    case HDVBackendKind::kCUDA:
      return cuda_impl::MakeHDV<T>(h_data, size, v, device);
#endif  // defined(XGBOOST_USE_CUDA)
#if defined(XGBOOST_USE_SYCL)
    case HDVBackendKind::kSycl:
      return sycl_impl::MakeHDV<T>(h_data, size, v, device);
#endif  // defined(XGBOOST_USE_SYCL)
    default:
      h_data->resize(size, v);
      return std::make_unique<HostOnlyBackend<T>>(h_data);
  }
}

template <typename T>
std::unique_ptr<HDVBackend<T>> MakeBackend(std::vector<T>* h_data, Span<T const> init,
                                           DeviceOrd device) {
  switch (KindOf(device)) {
#if defined(XGBOOST_USE_CUDA)
    case HDVBackendKind::kCUDA:
      return cuda_impl::MakeHDV<T>(h_data, init, device);
#endif  // defined(XGBOOST_USE_CUDA)
#if defined(XGBOOST_USE_SYCL)
    case HDVBackendKind::kSycl:
      return sycl_impl::MakeHDV<T>(h_data, init, device);
#endif  // defined(XGBOOST_USE_SYCL)
    default:
      h_data->assign(init.cbegin(), init.cend());
      return std::make_unique<HostOnlyBackend<T>>(h_data);
  }
}

template <typename T>
std::unique_ptr<HDVBackend<T>> AdoptHost(std::vector<T>* h_data, DeviceOrd device) {
  switch (KindOf(device)) {
#if defined(XGBOOST_USE_CUDA)
    case HDVBackendKind::kCUDA:
      return cuda_impl::AdoptHostHDV<T>(h_data, device);
#endif  // defined(XGBOOST_USE_CUDA)
#if defined(XGBOOST_USE_SYCL)
    case HDVBackendKind::kSycl:
      return sycl_impl::AdoptHostHDV<T>(h_data, device);
#endif  // defined(XGBOOST_USE_SYCL)
    default:
      return std::make_unique<HostOnlyBackend<T>>(h_data);
  }
}
}  // namespace
}  // namespace common

template <typename T>
class HostDeviceVectorImpl {
 public:
  HostDeviceVectorImpl(std::size_t size, T v, DeviceOrd device)
      : backend_{common::MakeBackend<T>(data_h_.get(), size, v, device)} {}
  HostDeviceVectorImpl(common::Span<T const> init, DeviceOrd device)
      : backend_{common::MakeBackend<T>(data_h_.get(), init, device)} {}
  HostDeviceVectorImpl(HostDeviceVectorImpl<T>&& that)
      : data_h_{std::move(that.data_h_)}, backend_{std::move(that.backend_)} {
    that.data_h_ = std::make_unique<std::vector<T>>();
    that.backend_ = std::make_unique<common::HostOnlyBackend<T>>(that.data_h_.get());
  }

  common::HDVBackend<T>* Backend() { return backend_.get(); }

  void SetDevice(DeviceOrd device) {
    auto kind = common::KindOf(device);
    if (kind == common::HDVBackendKind::kCPU || kind == backend_->Kind()) {
      backend_->SetDevice(device);
      return;
    }
    backend_->SyncToHost();
    backend_.reset();
    backend_ = common::AdoptHost<T>(data_h_.get(), device);
  }

  void Copy(HostDeviceVectorImpl<T>* other) {
    CHECK_EQ(backend_->Size(), other->backend_->Size());
    this->SetDevice(other->backend_->Device());
    if (backend_->Kind() == other->backend_->Kind()) {
      backend_->Copy(other->backend_.get());
      return;
    }
    auto const& h_other = other->backend_->ConstHostVector();
    backend_->Copy(common::Span<T const>{h_other.data(), h_other.size()});
  }

  void Extend(HostDeviceVectorImpl<T>* other) {
    if (backend_->Kind() == other->backend_->Kind()) {
      backend_->Extend(other->backend_.get());
      return;
    }
    auto const& h_other = other->backend_->ConstHostVector();
    auto& h_vec = backend_->HostVector();
    h_vec.insert(h_vec.end(), h_other.cbegin(), h_other.cend());
  }

 private:
  // Heap-allocated so that references from HostVector() survive moves and backend changes.
  std::unique_ptr<std::vector<T>> data_h_{std::make_unique<std::vector<T>>()};
  std::unique_ptr<common::HDVBackend<T>> backend_;
};

template <typename T>
HostDeviceVector<T>::HostDeviceVector(size_t size, T v, DeviceOrd device)
    : impl_(new HostDeviceVectorImpl<T>(size, v, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(std::initializer_list<T> init, DeviceOrd device)
    : impl_(new HostDeviceVectorImpl<T>(common::Span<T const>{init.begin(), init.size()}, device)) {
}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(const std::vector<T>& init, DeviceOrd device)
    : impl_(new HostDeviceVectorImpl<T>(common::Span<T const>{init.data(), init.size()}, device)) {}

template <typename T>
HostDeviceVector<T>::HostDeviceVector(HostDeviceVector<T>&& that)
    : impl_(new HostDeviceVectorImpl<T>(std::move(*that.impl_))) {}

template <typename T>
HostDeviceVector<T>& HostDeviceVector<T>::operator=(HostDeviceVector<T>&& that) {
  if (this == &that) {
    return *this;
  }

  std::unique_ptr<HostDeviceVectorImpl<T>> new_impl(
      new HostDeviceVectorImpl<T>(std::move(*that.impl_)));
  delete impl_;
  impl_ = new_impl.release();
  return *this;
}

template <typename T>
HostDeviceVector<T>::~HostDeviceVector() {
  delete impl_;
  impl_ = nullptr;
}

template <typename T>
size_t HostDeviceVector<T>::Size() const {
  return impl_->Backend()->Size();
}

template <typename T>
DeviceOrd HostDeviceVector<T>::Device() const {
  return impl_->Backend()->Device();
}

template <typename T>
T* HostDeviceVector<T>::DevicePointer() {
  return impl_->Backend()->DevicePointer();
}

template <typename T>
const T* HostDeviceVector<T>::ConstDevicePointer() const {
  return impl_->Backend()->ConstDevicePointer();
}

template <typename T>
common::Span<T> HostDeviceVector<T>::DeviceSpan() {
  return impl_->Backend()->DeviceSpan();
}

template <typename T>
common::Span<const T> HostDeviceVector<T>::ConstDeviceSpan() const {
  return impl_->Backend()->ConstDeviceSpan();
}

template <typename T>
std::vector<T>& HostDeviceVector<T>::HostVector() {
  return impl_->Backend()->HostVector();
}

template <typename T>
const std::vector<T>& HostDeviceVector<T>::ConstHostVector() const {
  return impl_->Backend()->ConstHostVector();
}

template <typename T>
void HostDeviceVector<T>::Resize(size_t new_size, T v) {
  impl_->Backend()->Resize(new_size, v);
}

template <typename T>
void HostDeviceVector<T>::Resize(size_t new_size) {
  impl_->Backend()->Resize(new_size);
}

template <typename T>
void HostDeviceVector<T>::Fill(T v) {
  impl_->Backend()->Fill(v);
}

template <typename T>
void HostDeviceVector<T>::Copy(const HostDeviceVector<T>& other) {
  impl_->Copy(other.impl_);
}

template <typename T>
void HostDeviceVector<T>::Copy(const std::vector<T>& other) {
  impl_->Backend()->Copy(common::Span<T const>{other.data(), other.size()});
}

template <typename T>
void HostDeviceVector<T>::Copy(std::initializer_list<T> other) {
  impl_->Backend()->Copy(common::Span<T const>{other.begin(), other.size()});
}

template <typename T>
void HostDeviceVector<T>::Extend(HostDeviceVector const& other) {
  impl_->Extend(other.impl_);
}

template <typename T>
bool HostDeviceVector<T>::HostCanRead() const {
  return impl_->Backend()->Access() <= GPUAccess::kRead;
}

template <typename T>
bool HostDeviceVector<T>::HostCanWrite() const {
  return impl_->Backend()->Access() <= GPUAccess::kNone;
}

template <typename T>
bool HostDeviceVector<T>::DeviceCanRead() const {
  return impl_->Backend()->Access() >= GPUAccess::kRead;
}

template <typename T>
bool HostDeviceVector<T>::DeviceCanWrite() const {
  return impl_->Backend()->Access() >= GPUAccess::kWrite;
}

template <typename T>
GPUAccess HostDeviceVector<T>::DeviceAccess() const {
  return impl_->Backend()->Access();
}

template <typename T>
void HostDeviceVector<T>::SetDevice(DeviceOrd device) const {
  impl_->SetDevice(device);
}

// explicit instantiations are required, as HostDeviceVector isn't header-only
#define XGBOOST_HDV_INSTANTIATE(T) template class HostDeviceVector<T>;
XGBOOST_HDV_FOR_EACH_TYPE(XGBOOST_HDV_INSTANTIATE)
#undef XGBOOST_HDV_INSTANTIATE
}  // namespace xgboost
