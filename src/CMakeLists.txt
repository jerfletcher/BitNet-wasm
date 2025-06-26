# Reference 3rdparty BitNet and llama.cpp code directly
set(GGML_HEADERS_BITNET 
    ../3rdparty/BitNet/include/ggml-bitnet.h
    ../3rdparty/BitNet/include/bitnet-lut-kernels.h
)
set(GGML_SOURCES_BITNET 
    ../3rdparty/BitNet/src/ggml-bitnet-mad.cpp
    ../3rdparty/BitNet/src/ggml-bitnet-lut.cpp
)

# Include directories for all 3rdparty code
include_directories(../3rdparty/BitNet/include)
include_directories(../3rdparty/BitNet/3rdparty/llama.cpp/ggml/include)
include_directories(../3rdparty/BitNet/3rdparty/llama.cpp/include)
include_directories(../3rdparty/BitNet/3rdparty/llama.cpp/common)

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for Bitnet.cpp compilation")
endif()
