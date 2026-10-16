/**
 * Copyright 2026, XGBoost contributors
 *
 * @brief Moving HostDeviceVector data between the CPU and the device backends of this build.
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>             // for DeviceOrd
#include <xgboost/host_device_vector.h>  // for HostDeviceVector

#include <cstddef>  // for size_t
#include <numeric>  // for iota
#include <vector>   // for vector

#if defined(XGBOOST_USE_CUDA)
#include "../../../src/common/cuda_rt_utils.h"  // for AllVisibleGPUs
#endif                                          // defined(XGBOOST_USE_CUDA)

namespace xgboost::common {
namespace {
std::vector<DeviceOrd> DeviceBackends() {
  std::vector<DeviceOrd> devices;
#if defined(XGBOOST_USE_CUDA)
  if (curt::AllVisibleGPUs() > 0) {
    devices.push_back(DeviceOrd::CUDA(0));
  }
#endif  // defined(XGBOOST_USE_CUDA)
#if defined(XGBOOST_USE_SYCL)
  devices.push_back(DeviceOrd::SyclDefault());
#endif  // defined(XGBOOST_USE_SYCL)
  return devices;
}

std::vector<int> Iota(std::size_t n, int first) {
  std::vector<int> h(n);
  std::iota(h.begin(), h.end(), first);
  return h;
}

// The host has the only valid copy of the data.
void CheckHostOnly(HostDeviceVector<int> const& v) {
  ASSERT_EQ(v.DeviceAccess(), GPUAccess::kNone);
  ASSERT_TRUE(v.HostCanRead());
  ASSERT_TRUE(v.HostCanWrite());
  ASSERT_FALSE(v.DeviceCanRead());
  ASSERT_FALSE(v.DeviceCanWrite());
}

// Device memory may be host memory (SYCL CPU device), so check the access state as well.
void CheckDeviceWrite(HostDeviceVector<int> const& v) {
  ASSERT_EQ(v.DeviceAccess(), GPUAccess::kWrite);
  ASSERT_FALSE(v.HostCanRead());
  ASSERT_FALSE(v.HostCanWrite());
  ASSERT_TRUE(v.DeviceCanRead());
  ASSERT_TRUE(v.DeviceCanWrite());
}

// Give the device the only valid copy of the data.
void MoveToDevice(DeviceOrd device, HostDeviceVector<int>* v) {
  v->SetDevice(device);
  ASSERT_EQ(v->Device(), device);
  ASSERT_NE(v->DevicePointer(), nullptr);
  CheckDeviceWrite(*v);
}
}  // namespace

TEST(HostDeviceVectorBackends, CpuDeviceCpu) {
  auto devices = DeviceBackends();
  if (devices.empty()) {
    GTEST_SKIP() << "No device backend available.";
  }
  std::size_t n = 17;
  for (auto device : devices) {
    HostDeviceVector<int> v{Iota(n, 3)};
    CheckHostOnly(v);
    ASSERT_EQ(v.Device(), DeviceOrd::CPU());
    ASSERT_EQ(v.DevicePointer(), nullptr);

    // Setting the device does not move the data yet.
    v.SetDevice(device);
    ASSERT_EQ(v.Device(), device);
    CheckHostOnly(v);

    // Read on the device: both sides can read.
    ASSERT_NE(v.ConstDevicePointer(), nullptr);
    ASSERT_EQ(v.DeviceAccess(), GPUAccess::kRead);
    ASSERT_TRUE(v.HostCanRead());
    ASSERT_FALSE(v.HostCanWrite());

    // Write on the device: the host has no access.
    ASSERT_NE(v.DevicePointer(), nullptr);
    CheckDeviceWrite(v);
    ASSERT_EQ(v.Size(), n);

    v.SetDevice(DeviceOrd::CPU());
    ASSERT_EQ(v.Device(), DeviceOrd::CPU());
    CheckHostOnly(v);
    ASSERT_EQ(v.ConstHostVector(), Iota(n, 3));
  }
}

TEST(HostDeviceVectorBackends, DeviceToDevice) {
  auto devices = DeviceBackends();
  if (devices.size() < 2) {
    GTEST_SKIP() << "Requires two device backends.";
  }
  std::size_t n = 33;
  for (auto src : devices) {
    for (auto dst : devices) {
      if (src == dst) {
        continue;
      }
      HostDeviceVector<int> v{Iota(n, 1)};
      MoveToDevice(src, &v);
      // The data is moved to the host and handed over to the other backend.
      v.SetDevice(dst);
      ASSERT_EQ(v.Device(), dst);
      CheckHostOnly(v);
      ASSERT_EQ(v.Size(), n);
      MoveToDevice(dst, &v);
      MoveToDevice(src, &v);
      ASSERT_EQ(v.ConstHostVector(), Iota(n, 1));
    }
  }
}

TEST(HostDeviceVectorBackends, ConstructOnDevice) {
  auto devices = DeviceBackends();
  if (devices.empty()) {
    GTEST_SKIP() << "No device backend available.";
  }
  std::size_t n = 9;
  for (auto device : devices) {
    HostDeviceVector<int> v{Iota(n, 2), device};
    ASSERT_EQ(v.Device(), device);
    CheckDeviceWrite(v);
    v.SetDevice(DeviceOrd::CPU());
    ASSERT_EQ(v.ConstHostVector(), Iota(n, 2));

    HostDeviceVector<int> w(n, 5, device);
    ASSERT_EQ(w.Device(), device);
    CheckDeviceWrite(w);
    ASSERT_EQ(w.ConstHostVector(), std::vector<int>(n, 5));

    HostDeviceVector<int> x{{4, 5, 6}, device};
    CheckDeviceWrite(x);
    ASSERT_EQ(x.ConstHostVector(), Iota(3, 4));
  }
}

TEST(HostDeviceVectorBackends, Copy) {
  auto devices = DeviceBackends();
  if (devices.empty()) {
    GTEST_SKIP() << "No device backend available.";
  }
  std::size_t n = 21;
  devices.push_back(DeviceOrd::CPU());
  for (auto src : devices) {
    for (auto dst : devices) {
      HostDeviceVector<int> from{Iota(n, 7)};
      if (!src.IsCPU()) {
        MoveToDevice(src, &from);
      }
      HostDeviceVector<int> to(n, 0);
      if (!dst.IsCPU()) {
        MoveToDevice(dst, &to);
      }
      // Copy takes the device of the source.
      to.Copy(from);
      ASSERT_EQ(to.Device(), from.Device());
      ASSERT_EQ(to.ConstHostVector(), Iota(n, 7));
      ASSERT_EQ(from.ConstHostVector(), Iota(n, 7));
    }
  }
}

TEST(HostDeviceVectorBackends, Extend) {
  auto devices = DeviceBackends();
  if (devices.empty()) {
    GTEST_SKIP() << "No device backend available.";
  }
  std::size_t n = 12, m = 5;
  devices.push_back(DeviceOrd::CPU());
  for (auto lhs : devices) {
    for (auto rhs : devices) {
      HostDeviceVector<int> a{Iota(n, 0)};
      if (!lhs.IsCPU()) {
        MoveToDevice(lhs, &a);
      }
      HostDeviceVector<int> b{Iota(m, static_cast<int>(n))};
      if (!rhs.IsCPU()) {
        MoveToDevice(rhs, &b);
      }
      a.Extend(b);
      ASSERT_EQ(a.Size(), n + m);
      ASSERT_EQ(a.Device(), lhs);
      ASSERT_EQ(a.ConstHostVector(), Iota(n + m, 0));
      ASSERT_EQ(b.ConstHostVector(), Iota(m, static_cast<int>(n)));
    }
  }
}

TEST(HostDeviceVectorBackends, ResizeAfterMigration) {
  auto devices = DeviceBackends();
  if (devices.empty()) {
    GTEST_SKIP() << "No device backend available.";
  }
  std::size_t n = 10;
  for (auto device : devices) {
    HostDeviceVector<int> v{Iota(n, 0)};
    v.SetDevice(device);
    v.Resize(2 * n, -1);
    ASSERT_EQ(v.Size(), 2 * n);
    MoveToDevice(device, &v);
    // On-device resize keeps the device access.
    v.Resize(3 * n, -2);
    ASSERT_EQ(v.Size(), 3 * n);
    CheckDeviceWrite(v);

    v.SetDevice(DeviceOrd::CPU());
    CheckHostOnly(v);
    auto const& h = v.ConstHostVector();
    for (std::size_t i = 0; i < h.size(); ++i) {
      int expected = i < n ? static_cast<int>(i) : (i < 2 * n ? -1 : -2);
      ASSERT_EQ(h[i], expected) << i;
    }
  }
}

TEST(HostDeviceVectorBackends, Move) {
  auto devices = DeviceBackends();
  if (devices.empty()) {
    GTEST_SKIP() << "No device backend available.";
  }
  std::size_t n = 6;
  for (auto device : devices) {
    HostDeviceVector<int> v{Iota(n, 1)};
    MoveToDevice(device, &v);
    HostDeviceVector<int> w{std::move(v)};
    ASSERT_EQ(w.Device(), device);
    CheckDeviceWrite(w);
    ASSERT_EQ(w.ConstHostVector(), Iota(n, 1));
    // The moved-from vector is empty and usable.
    ASSERT_TRUE(v.Empty());  // NOLINT
    v.Resize(2, 3);
    ASSERT_EQ(v.ConstHostVector(), std::vector<int>(2, 3));
  }
}

TEST(HostDeviceVectorBackends, HostVectorReference) {
  auto devices = DeviceBackends();
  if (devices.empty()) {
    GTEST_SKIP() << "No device backend available.";
  }
  for (auto device : devices) {
    HostDeviceVector<int> v;
    auto& h_v = v.HostVector();
    h_v = Iota(5, 1);
    v.SetDevice(device);
    v.DeviceSpan();
    // Callers keep references across SetDevice, e.g. RandomDataGenerator::GenerateCSR.
    ASSERT_EQ(h_v.size(), std::size_t{5});
    ASSERT_EQ(&v.HostVector(), &h_v);
    ASSERT_EQ(h_v, Iota(5, 1));
  }
}
}  // namespace xgboost::common
