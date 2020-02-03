This code uses C++ 2011 features, so you may need to pass the flag `-std=c++11` to the nvcc compiler, e.g.:

  nvcc -std=c++11 main.cu

Also, when generating timing results, you should enable all optimizations by passing the flag `-O3` to the nvcc compiler, e.g.:

  nvcc -std=c++11 -O3 main.cu
