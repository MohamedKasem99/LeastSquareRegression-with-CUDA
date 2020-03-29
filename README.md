# LeastSquareRegression-with-CUDA
### compile using this command:
```nvcc kernel.cu -o mykernel -gencode arch=compute_35,code=sm_35 -lcublas```
 This will generate code for any gpu with compute capability 3.5 and will link the cublas library 
 To know your Nvidia GPU compute capabilities, visit this website: 
 https://developer.nvidia.com/cuda-gpus

 This code has been compiled and tested on Nvidia GPU GT-920M and an x86 Intel CPU.
 Solves for **[x1,x2,x3, ...] in [A1*x1 = y1, A2*x2 = y2, A3*x3 = y3, ...]** . This code solves all of the equations in parallel
 To specify the batch size change mybatch variable in BackSlashOp() function below to implement piecewise modelling in a 
 least squares system.
