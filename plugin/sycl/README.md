<!--
******************************************************************************
* Copyright by Contributors 2017-2023
*******************************************************************************/-->

# SYCL-based Algorithm for Tree Construction
This plugin adds support of SYCL programming model for prediction algorithms to XGBoost.

## Usage
Specify the 'device' parameter as described in the table below to offload model training and inference on SYCL device.

### Algorithms
| device | Description |
| --- | --- |
sycl | use default sycl device  |
sycl:gpu | use default sycl gpu  |
sycl:cpu | use default sycl cpu  |
sycl:gpu:N | use sycl gpu number N |
sycl:cpu:N | use sycl cpu number N |

Python example:
```python
param['device'] = 'sycl:gpu:0'
```
Note: 'sycl:cpu' devices have full functional support but can't provide good enough performance. We recommend use 'sycl:cpu' devices only for test purposes.
Note: if device is specified to be 'sycl', device type will be automatically chosen. In case the system has both sycl GPU and sycl CPU, GPU will on use.

## Dependencies
To build and use the plugin, install [Intel® oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html).
See also [Intel® oneAPI Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2024-0/overview.html).

## Build
From the ``xgboost`` directory, run:

```bash
$ cmake -B build -S . -DPLUGIN_SYCL=ON
$ cmake --build build -j
```

Only the plugin sources are compiled by the SYCL compiler. The rest of XGBoost is compiled and
linked by the regular C++ compiler (``CMAKE_CXX_COMPILER``). CMake 3.19 or newer is required.

| CMake option | Default | Description |
| --- | --- | --- |
| ``SYCL_CXX_COMPILER`` | ``icpx`` | SYCL compiler, a name in ``PATH`` or a full path. |
| ``SYCL_CXX_FLAGS`` | empty | Additional flags for the SYCL compiler only, e.g. ``--gcc-toolchain=<path>``. |

The SYCL runtime library (``libsycl``) is looked up next to the compiler (``<bin>/../lib``) and
becomes a load-time dependency of ``libxgboost``. Intel compiler runtime libraries are not linked
(``-no-intel-lib``). SYCL devices additionally need the oneAPI Unified Runtime adapters and their
dependencies at run time; ``sycl-ls --verbose`` shows why an adapter fails to load.
