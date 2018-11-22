include_directories(../external/ProbImage ../external/ )

find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ${OpenMP_CXX_FLAGS}")
endif()

add_library(color_to_label color_to_label.cpp)

add_library(file_storage file_storage.cpp)
target_link_libraries(file_storage probimage color_to_label ${OpenCV_LIBS})

#add_library(test test.cpp)
#target_link_libraries(test densecrf)

# Executables
# inference.cpp
add_executable(inference inference.cpp)
target_link_libraries(inference densecrf file_storage densecrf_utils tree_utils)

#add_executable(inference_tree inference_tree.cpp)
#target_link_libraries(inference_tree densecrf file_storage densecrf_utils tree_utils)

#add_executable(inference_denoising inference_denoising.cpp)
#target_link_libraries(inference_denoising densecrf file_storage densecrf_utils)

#add_executable(inference_stereo inference_stereo.cpp)
#target_link_libraries(inference_stereo densecrf file_storage densecrf_utils)

#add_executable(inference_sparse_stereo inference_sparse_stereo.cpp)
#target_link_libraries(inference_sparse_stereo densecrf file_storage densecrf_utils)

#add_executable(inference_sparse inference_sparse.cpp)
#target_link_libraries(inference_sparse sparsecrf densecrf file_storage densecrf_utils tree_utils)

#add_executable(inference_sparse_synthetic_tree inference_sparse_synthetic_tree.cpp)
#target_link_libraries(inference_sparse_synthetic_tree sparsecrf densecrf file_storage densecrf_utils tree_utils)

#add_executable(inference_sparse_synthetic inference_sparse_synthetic.cpp)
#target_link_libraries(inference_sparse_synthetic sparsecrf densecrf file_storage densecrf_utils tree_utils)

#add_executable(inference_sparse_stereo inference_sparse_stereo.cpp)
#target_link_libraries(inference_sparse_stereo densecrf file_storage densecrf_utils)

#add_executable(inference_temp inference_temp.cpp)
#target_link_libraries(inference_temp densecrf file_storage densecrf_utils)

#add_executable(gradientDescent gradientDescentTest.cpp)
#target_link_libraries(gradientDescent densecrf file_storage densecrf_utils)

# cv-script.cpp 
#add_executable(msrc_inference cv-script.cpp)
#target_link_libraries(msrc_inference densecrf file_storage densecrf_utils)

# toyExample.cpp
#add_executable(toyExample toyExample.cpp)
#target_link_libraries(toyExample densecrf file_storage densecrf_utils)

# trial.cpp
#add_executable(trial trial.cpp)
#target_link_libraries(trial densecrf file_storage densecrf_utils)

# trial_downsampleimage.cpp
#add_executable(trial_downsampleimage trial_downsampleimage.cpp)
#target_link_libraries(trial_downsampleimage densecrf file_storage densecrf_utils)
