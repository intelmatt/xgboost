/**
 * Copyright 2017-2026, XGBoost contributors
 *
 * @brief CUDA backend of HostDeviceVector.
 */
#include <thrust/fill.h>

#include <algorithm>
#include <cstddef>  // for size_t
#include <cstdint>
#include <memory>   // for unique_ptr
#include <utility>  // for move
#include <vector>   // for vector

#include "cuda_stream.h"  // for DefaultStream
#include "device_helpers.cuh"
#include "device_vector.cuh"  // for DeviceUVector
#include "host_device_vector_backend.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/tree_model.h"  // for RegTree

#if defined(XGBOOST_USE_SYCL)
#include "../../plugin/sycl/predictor/node.h"  // for Node
#endif                                         // defined(XGBOOST_USE_SYCL)

namespace xgboost {

// the handler to call instead of cudaSetDevice; only used for testing
static void (*cudaSetDeviceHandler)(int) = nullptr;  // NOLINT

void SetCudaSetDeviceHandler(void (*handler)(int)) { cudaSetDeviceHandler = handler; }

namespace common::cuda_impl {
template <typename T>
class HDVBackendCuda : public HDVBackend<T> {
 public:
  HDVBackendCuda(std::vector<T>* h_data, size_t size, T v, DeviceOrd device)
      : device_(device), data_h_{*h_data} {
    if (device.IsCUDA()) {
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      data_d_->resize(size, v);
    } else {
      data_h_.resize(size, v);
    }
  }

  HDVBackendCuda(std::vector<T>* h_data, Span<T const> init, DeviceOrd device)
      : device_(device), data_h_{*h_data} {
    if (device.IsCUDA()) {
      gpu_access_ = GPUAccess::kWrite;
      LazyResizeDevice(init.size());
      Copy(init);
    } else {
      data_h_.assign(init.cbegin(), init.cend());
    }
  }

  HDVBackendCuda(std::vector<T>* h_data, DeviceOrd device) : data_h_{*h_data} { SetDevice(device); }

  ~HDVBackendCuda() override {
    if (device_.IsCUDA()) {
      SetDevice();
    }
  }

  [[nodiscard]] HDVBackendKind Kind() const override { return HDVBackendKind::kCUDA; }

  [[nodiscard]] size_t Size() const override {
    return HostCanRead() ? data_h_.size() : data_d_ ? data_d_->size() : 0;
  }

  [[nodiscard]] DeviceOrd Device() const override { return device_; }

  T* DevicePointer() override {
    LazySyncDevice(GPUAccess::kWrite);
    return data_d_->data();
  }

  const T* ConstDevicePointer() override {
    LazySyncDevice(GPUAccess::kRead);
    return data_d_->data();
  }

  common::Span<T> DeviceSpan() override {
    LazySyncDevice(GPUAccess::kWrite);
    return {this->DevicePointer(), Size()};
  }

  common::Span<const T> ConstDeviceSpan() override {
    LazySyncDevice(GPUAccess::kRead);
    return {this->ConstDevicePointer(), Size()};
  }

  void Fill(T v) override {  // NOLINT
    if (HostCanWrite()) {
      std::fill(data_h_.begin(), data_h_.end(), v);
    } else {
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      auto s_data = dh::ToSpan(*data_d_);
      dh::LaunchN(data_d_->size(), curt::DefaultStream(),
                  [=] XGBOOST_DEVICE(size_t i) { s_data[i] = v; });
    }
  }

  void Copy(HDVBackend<T>* that) override {
    auto* other = static_cast<HDVBackendCuda<T>*>(that);
    CHECK_EQ(Size(), other->Size());
    SetDevice(other->device_);
    // Data is on host.
    if (HostCanWrite() && other->HostCanWrite()) {
      std::copy(other->data_h_.begin(), other->data_h_.end(), data_h_.begin());
      return;
    }
    SetDevice();
    CopyToDevice(other);
  }

  void Copy(Span<T const> other) override {
    CHECK_EQ(Size(), other.size());
    if (HostCanWrite()) {
      std::copy(other.cbegin(), other.cend(), data_h_.begin());
    } else {
      CopyToDevice(other.data());
    }
  }

  void Extend(HDVBackend<T>* that) override {
    auto* other = static_cast<HDVBackendCuda<T>*>(that);
    auto ori_size = this->Size();
    this->Resize(ori_size + other->Size(), T{});
    if (HostCanWrite() && other->HostCanRead()) {
      auto& h_vec = this->HostVector();
      auto& other_vec = other->HostVector();
      CHECK_EQ(h_vec.size(), ori_size + other->Size());
      std::copy(other_vec.cbegin(), other_vec.cend(), h_vec.begin() + ori_size);
    } else {
      auto ptr = other->ConstDevicePointer();
      SetDevice();
      CHECK_EQ(this->Device(), other->Device());
      dh::safe_cuda(cudaMemcpyAsync(this->DevicePointer() + ori_size, ptr,
                                    other->Size() * sizeof(T), cudaMemcpyDeviceToDevice,
                                    curt::DefaultStream()));
    }
  }

  std::vector<T>& HostVector() override {
    LazySyncHost(GPUAccess::kNone);
    return data_h_;
  }

  const std::vector<T>& ConstHostVector() override {
    LazySyncHost(GPUAccess::kRead);
    return data_h_;
  }

  void SetDevice(DeviceOrd device) override {
    if (device_ == device) {
      return;
    }
    if (device_.IsCUDA()) {
      LazySyncHost(GPUAccess::kNone);
    }

    if (device_.IsCUDA() && device.IsCUDA()) {
      CHECK_EQ(device_.ordinal, device.ordinal)
          << "New device ordinal is different from previous one.";
    }
    device_ = device;
    if (device_.IsCUDA()) {
      LazyResizeDevice(data_h_.size());
    }
  }

  void Resize(std::size_t new_size) override { this->ResizeImpl(new_size); }
  void Resize(std::size_t new_size, T v) override { this->ResizeImpl(new_size, v); }

  void SyncToHost() override { LazySyncHost(GPUAccess::kNone); }

  [[nodiscard]] bool HostCanAccess(GPUAccess access) const { return gpu_access_ <= access; }
  [[nodiscard]] bool HostCanRead() const { return HostCanAccess(GPUAccess::kRead); }
  [[nodiscard]] bool HostCanWrite() const { return HostCanAccess(GPUAccess::kNone); }
  [[nodiscard]] bool DeviceCanAccess(GPUAccess access) const { return gpu_access_ >= access; }
  [[nodiscard]] bool DeviceCanRead() const { return DeviceCanAccess(GPUAccess::kRead); }
  [[nodiscard]] bool DeviceCanWrite() const { return DeviceCanAccess(GPUAccess::kWrite); }
  [[nodiscard]] GPUAccess Access() const override { return gpu_access_; }

 private:
  DeviceOrd device_{DeviceOrd::CPU()};
  std::vector<T>& data_h_;
  std::unique_ptr<dh::DeviceUVector<T>> data_d_{};
  GPUAccess gpu_access_{GPUAccess::kNone};

  template <typename... U>
  void ResizeImpl(std::size_t new_size, U&&... args) {
    if (new_size == Size()) {
      return;
    }
    if ((Size() == 0 && device_.IsCUDA()) || (DeviceCanWrite() && device_.IsCUDA())) {
      // fast on-device resize
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      data_d_->resize(new_size, std::forward<U>(args)...);
    } else {
      // resize on host
      LazySyncHost(GPUAccess::kNone);
      data_h_.resize(new_size, std::forward<U>(args)...);
    }
  }

  void LazySyncHost(GPUAccess access) {
    if (HostCanAccess(access)) {
      return;
    }
    if (HostCanRead()) {
      // data is present, just need to deny access to the device
      gpu_access_ = access;
      return;
    }
    gpu_access_ = access;
    if (data_h_.size() != data_d_->size()) {
      data_h_.resize(data_d_->size());
    }
    SetDevice();
    dh::safe_cuda(cudaMemcpy(data_h_.data(), data_d_->data(), data_d_->size() * sizeof(T),
                             cudaMemcpyDeviceToHost));
  }

  void LazySyncDevice(GPUAccess access) {
    if (DeviceCanAccess(access)) {
      return;
    }
    if (DeviceCanRead()) {
      // deny read to the host
      gpu_access_ = access;
      return;
    }
    // data is on the host
    LazyResizeDevice(data_h_.size());
    SetDevice();
    dh::safe_cuda(cudaMemcpyAsync(data_d_->data(), data_h_.data(), data_d_->size() * sizeof(T),
                                  cudaMemcpyHostToDevice, curt::DefaultStream()));
    gpu_access_ = access;
  }

  void CopyToDevice(HDVBackendCuda* other) {
    if (other->HostCanWrite()) {
      CopyToDevice(other->data_h_.data());
    } else {
      LazyResizeDevice(Size());
      gpu_access_ = GPUAccess::kWrite;
      SetDevice();
      dh::safe_cuda(cudaMemcpyAsync(data_d_->data(), other->data_d_->data(),
                                    data_d_->size() * sizeof(T), cudaMemcpyDefault,
                                    curt::DefaultStream()));
    }
  }

  void CopyToDevice(const T* begin) {
    LazyResizeDevice(Size());
    gpu_access_ = GPUAccess::kWrite;
    SetDevice();
    dh::safe_cuda(cudaMemcpyAsync(data_d_->data(), begin, data_d_->size() * sizeof(T),
                                  cudaMemcpyDefault, curt::DefaultStream()));
  }

  void LazyResizeDevice(size_t new_size) {
    if (data_d_ && new_size == data_d_->size()) {
      return;
    }
    SetDevice();
    data_d_->resize(new_size);
  }

  void SetDevice() {
    CHECK_GE(device_.ordinal, 0);
    if (cudaSetDeviceHandler == nullptr) {
      dh::safe_cuda(cudaSetDevice(device_.ordinal));
    } else {
      (*cudaSetDeviceHandler)(device_.ordinal);
    }

    if (!data_d_) {
      data_d_.reset(new dh::DeviceUVector<T>{});
    }
  }
};

template <typename T>
std::unique_ptr<HDVBackend<T>> MakeHDV(std::vector<T>* h_data, std::size_t size, T v,
                                       DeviceOrd device) {
  return std::make_unique<HDVBackendCuda<T>>(h_data, size, v, device);
}

template <typename T>
std::unique_ptr<HDVBackend<T>> MakeHDV(std::vector<T>* h_data, Span<T const> init,
                                       DeviceOrd device) {
  return std::make_unique<HDVBackendCuda<T>>(h_data, init, device);
}

template <typename T>
std::unique_ptr<HDVBackend<T>> AdoptHostHDV(std::vector<T>* h_data, DeviceOrd device) {
  return std::make_unique<HDVBackendCuda<T>>(h_data, device);
}
}  // namespace common::cuda_impl
}  // namespace xgboost

#define XGBOOST_HDV_INSTANTIATE(T) XGBOOST_HDV_INSTANTIATE_FACTORIES(xgboost::common::cuda_impl, T)
XGBOOST_HDV_FOR_EACH_TYPE(XGBOOST_HDV_INSTANTIATE)
#undef XGBOOST_HDV_INSTANTIATE
