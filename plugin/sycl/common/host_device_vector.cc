/**
 * Copyright 2017-2026, XGBoost contributors
 *
 * @brief SYCL backend of HostDeviceVector.
 */

#ifdef XGBOOST_USE_SYCL

#include <dmlc/registry.h>  // for DMLC_REGISTRY_LINK_TAG

#include <memory>
#include <utility>
#include <vector>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-W#pragma-messages"
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "xgboost/host_device_vector.h"
#pragma GCC diagnostic pop

#include "../../src/common/host_device_vector_backend.h"
#include "../data.h"
#include "../device_manager.h"
#include "../predictor/node.h"

namespace xgboost::common::sycl_impl {
template <typename T>
class HDVBackendSycl : public HDVBackend<T> {
  using DeviceStorage = sycl::USMVector<T, sycl::MemoryType::on_device>;

 public:
  HDVBackendSycl(std::vector<T>* h_data, size_t size, T v, DeviceOrd device)
      : device_(device), data_h_{*h_data} {
    if (device.IsSycl()) {
      device_access_ = GPUAccess::kWrite;
      SetDevice();
      data_d_->Resize(qu_, size, v);
    } else {
      data_h_.resize(size, v);
    }
  }

  HDVBackendSycl(std::vector<T>* h_data, Span<T const> init, DeviceOrd device)
      : device_(device), data_h_{*h_data} {
    if (device.IsSycl()) {
      device_access_ = GPUAccess::kWrite;

      ResizeDevice(init.size());
      Copy(init);
    } else {
      data_h_.assign(init.cbegin(), init.cend());
    }
  }

  HDVBackendSycl(std::vector<T>* h_data, DeviceOrd device) : data_h_{*h_data} { SetDevice(device); }

  [[nodiscard]] HDVBackendKind Kind() const override { return HDVBackendKind::kSycl; }

  std::vector<T>& HostVector() override {
    SyncHost(GPUAccess::kNone);
    return data_h_;
  }

  const std::vector<T>& ConstHostVector() override {
    SyncHost(GPUAccess::kRead);
    return data_h_;
  }

  void SetDevice(DeviceOrd device) override {
    if (device_ == device) {
      return;
    }
    if (device_.IsSycl()) {
      SyncHost(GPUAccess::kNone);
    }

    if (device_.IsSycl() && device.IsSycl()) {
      CHECK_EQ(device_, device) << "New device is different from previous one.";
    }
    device_ = device;
    if (device_.IsSycl()) {
      ResizeDevice(data_h_.size());
    }
  }

  void Resize(std::size_t new_size) override { this->ResizeImpl(new_size); }
  void Resize(std::size_t new_size, T v) override { this->ResizeImpl(new_size, v); }

  void SyncToHost() override { SyncHost(GPUAccess::kNone); }

  void SyncHost(GPUAccess access) {
    if (HostCanAccess(access)) {
      return;
    }
    if (HostCanRead()) {
      // data is present, just need to deny access to the device
      device_access_ = access;
      return;
    }
    device_access_ = access;
    if (data_h_.size() != data_d_->Size()) {
      data_h_.resize(data_d_->Size());
    }
    SetDevice();
    qu_->memcpy(data_h_.data(), data_d_->Data(), data_d_->Size() * sizeof(T)).wait();
  }

  void SyncDevice(GPUAccess access) {
    if (DeviceCanAccess(access)) {
      return;
    }
    if (DeviceCanRead()) {
      device_access_ = access;
      return;
    }
    // data is on the host
    ResizeDevice(data_h_.size());
    SetDevice();
    qu_->memcpy(data_d_->Data(), data_h_.data(), data_d_->Size() * sizeof(T)).wait();
    device_access_ = access;
  }

  bool HostCanAccess(GPUAccess access) const { return device_access_ <= access; }
  bool HostCanRead() const { return HostCanAccess(GPUAccess::kRead); }
  bool HostCanWrite() const { return HostCanAccess(GPUAccess::kNone); }
  bool DeviceCanAccess(GPUAccess access) const { return device_access_ >= access; }
  bool DeviceCanRead() const { return DeviceCanAccess(GPUAccess::kRead); }
  bool DeviceCanWrite() const { return DeviceCanAccess(GPUAccess::kWrite); }
  GPUAccess Access() const override { return device_access_; }

  size_t Size() const override {
    return HostCanRead() ? data_h_.size() : data_d_ ? data_d_->Size() : 0;
  }

  DeviceOrd Device() const override { return device_; }

  T* DevicePointer() override {
    SyncDevice(GPUAccess::kWrite);
    return data_d_->Data();
  }

  const T* ConstDevicePointer() override {
    SyncDevice(GPUAccess::kRead);
    return data_d_->DataConst();
  }

  common::Span<T> DeviceSpan() override {
    SyncDevice(GPUAccess::kWrite);
    return {this->DevicePointer(), Size()};
  }

  common::Span<const T> ConstDeviceSpan() override {
    SyncDevice(GPUAccess::kRead);
    return {this->ConstDevicePointer(), Size()};
  }

  void Fill(T v) override {
    if (HostCanWrite()) {
      std::fill(data_h_.begin(), data_h_.end(), v);
    } else {
      device_access_ = GPUAccess::kWrite;
      SetDevice();
      qu_->fill(data_d_->Data(), v, data_d_->Size()).wait();
    }
  }

  void Copy(HDVBackend<T>* that) override {
    auto* other = static_cast<HDVBackendSycl<T>*>(that);
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
    auto* other = static_cast<HDVBackendSycl<T>*>(that);
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
      qu_->memcpy(this->DevicePointer() + ori_size, ptr, other->Size() * sizeof(T)).wait();
    }
  }

 private:
  template <typename... U>
  void ResizeImpl(size_t new_size, U&&... args) {
    if (new_size == Size()) {
      return;
    }
    if ((Size() == 0 && device_.IsSycl()) || (DeviceCanWrite() && device_.IsSycl())) {
      // fast on-device resize
      device_access_ = GPUAccess::kWrite;
      SetDevice();
      data_d_->Resize(qu_, new_size, std::forward<U>(args)...);
    } else {
      // resize on host
      SyncHost(GPUAccess::kNone);
      data_h_.resize(new_size, std::forward<U>(args)...);
    }
  }

  void ResizeDevice(size_t new_size) {
    if (data_d_ && new_size == data_d_->Size()) {
      return;
    }
    SetDevice();
    data_d_->Resize(qu_, new_size);
  }

  void SetDevice() {
    if (!qu_) {
      qu_ = device_manager_.GetQueue(device_);
    }
    if (!data_d_) {
      data_d_.reset(new DeviceStorage());
    }
  }

  void CopyToDevice(HDVBackendSycl* other) {
    if (other->HostCanWrite()) {
      CopyToDevice(other->data_h_.data());
    } else {
      ResizeDevice(Size());
      device_access_ = GPUAccess::kWrite;
      SetDevice();
      qu_->memcpy(data_d_->Data(), other->data_d_->Data(), data_d_->Size() * sizeof(T)).wait();
    }
  }

  void CopyToDevice(const T* begin) {
    data_d_->ResizeNoCopy(qu_, Size());
    qu_->memcpy(data_d_->Data(), begin, data_d_->Size() * sizeof(T)).wait();
    device_access_ = GPUAccess::kWrite;
  }

  sycl::DeviceManager device_manager_;
  ::sycl::queue* qu_ = nullptr;
  DeviceOrd device_{DeviceOrd::CPU()};
  std::vector<T>& data_h_;
  std::unique_ptr<DeviceStorage> data_d_{};
  GPUAccess device_access_{GPUAccess::kNone};
};

template <typename T>
std::unique_ptr<HDVBackend<T>> MakeHDV(std::vector<T>* h_data, std::size_t size, T v,
                                       DeviceOrd device) {
  return std::make_unique<HDVBackendSycl<T>>(h_data, size, v, device);
}

template <typename T>
std::unique_ptr<HDVBackend<T>> MakeHDV(std::vector<T>* h_data, Span<T const> init,
                                       DeviceOrd device) {
  return std::make_unique<HDVBackendSycl<T>>(h_data, init, device);
}

template <typename T>
std::unique_ptr<HDVBackend<T>> AdoptHostHDV(std::vector<T>* h_data, DeviceOrd device) {
  return std::make_unique<HDVBackendSycl<T>>(h_data, device);
}
}  // namespace xgboost::common::sycl_impl

#define XGBOOST_HDV_INSTANTIATE(T) XGBOOST_HDV_INSTANTIATE_FACTORIES(xgboost::common::sycl_impl, T)
XGBOOST_HDV_FOR_EACH_TYPE(XGBOOST_HDV_INSTANTIATE)
#undef XGBOOST_HDV_INSTANTIATE

// Prediction initialization references this translation unit through the HostDeviceVector
// backend factories. Retain the plugin's predictor and kernel registrations when linking a static
// archive.
namespace xgboost::sycl::predictor {
DMLC_REGISTRY_LINK_TAG(predictor_sycl);
}  // namespace xgboost::sycl::predictor

#endif  // XGBOOST_USE_SYCL
