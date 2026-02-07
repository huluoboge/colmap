#pragma once

#include <OpenImageIO/imageio.h>
#ifndef OIIO_MAKE_VERSION
#define OIIO_MAKE_VERSION(major, minor, patch) \
  ((major) * 10000 + (minor) * 100 + (patch))
#endif
namespace colmap {

// Declaration of the thread-safe, one-time initialization function.
void EnsureOpenImageIOInitialized();

}  // namespace colmap
