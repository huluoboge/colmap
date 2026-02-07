# OpenImageIOConfig.cmake - Configuration file for OpenImageIO

# 计算安装前缀
get_filename_component(OpenImageIO_CURRENT_CONFIG_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(OpenImageIO_INSTALL_PREFIX "${OpenImageIO_CURRENT_CONFIG_DIR}/../../../" ABSOLUTE)

# 设置版本信息（从pkg-config获取）
set(OpenImageIO_VERSION "2.1.12")

# 设置包含目录
set(OpenImageIO_INCLUDE_DIRS "/usr/include")

# 设置库文件
set(OpenImageIO_LIBRARIES 
    "/usr/lib/x86_64-linux-gnu/libOpenImageIO.so"
    "/usr/lib/x86_64-linux-gnu/libOpenImageIO_Util.so"
)

# 创建导入的目标
if(NOT TARGET OpenImageIO::OpenImageIO)
    add_library(OpenImageIO::OpenImageIO UNKNOWN IMPORTED)
    set_target_properties(OpenImageIO::OpenImageIO PROPERTIES
        IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libOpenImageIO.so"
        INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
        INTERFACE_LINK_LIBRARIES "OpenImageIO::OpenImageIO_Util"
    )
endif()

if(NOT TARGET OpenImageIO::OpenImageIO_Util)
    add_library(OpenImageIO::OpenImageIO_Util UNKNOWN IMPORTED)
    set_target_properties(OpenImageIO::OpenImageIO_Util PROPERTIES
        IMPORTED_LOCATION "/usr/lib/x86_64-linux-gnu/libOpenImageIO_Util.so"
        INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
    )
endif()

# 兼容性设置
set(OPENIMAGEIO_FOUND TRUE)
set(OPENIMAGEIO_INCLUDE_DIRS ${OpenImageIO_INCLUDE_DIRS})
set(OPENIMAGEIO_LIBRARIES ${OpenImageIO_LIBRARIES})

# 检查库文件是否存在
foreach(_lib_path ${OpenImageIO_LIBRARIES})
    if(NOT EXISTS ${_lib_path})
        message(WARNING "OpenImageIO library not found: ${_lib_path}")
    endif()
endforeach()

# 检查包含目录是否存在
if(NOT EXISTS "/usr/include/OpenImageIO")
    message(WARNING "OpenImageIO include directory not found: /usr/include/OpenImageIO")
endif()

message(STATUS "Found OpenImageIO: ${OpenImageIO_VERSION}")