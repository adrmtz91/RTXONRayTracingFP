# Real-Time Ray Tracing with CUDA - RTXONRayTracingFP

## <font size="5">Team RTX ON Project Proposal</font>

**Team Members**: Steven Ryan Leonido, Jenny Quan, Adrian Martinez

**Team Name**: RTX ON

**Proposed Project**: Real Time Ray Tracing on GPU with OptiX acceleration

**Required Libraries and Frameworks**: C++, CUDA, OptiX, RT Cores

**Potential Risks or Technical Problems/Challenges**: 
The potential challenges for this project included learning several new topics, such as item rendering and creation, ray tracing algorithm creation, the OptiX library for algorithm acceleration, and real-time ray tracing. Our team was also not familiar with graphics computing and was working under the assumption that OptiX would function well with CUDA.

**Outline of Project**: 
1. Research rendering, ray tracing, and object creation
2. Apply ray tracing algorithms to create better-rendered images
3. Apply GPU and OptiX ray tracing acceleration
4. Create real-time ray tracing of the environment with an interactive camera

## <font size="5">Introduction</font>

The project implements a CUDA-based ray tracer aimed at the parallel processing power of GPUs to accelerate the rendering process. The key components include device functions and kernels: ray_color, which calculates the color of a ray; render, which performs the rendering by calculating the color for each pixel; create_world, which initializes the world with the object (sphere); free_world, which deallocates memory for the object; and init_rand_state_kernel, which initializes random states for each pixel. 

The main function initializes necessary data structures, allocates memory on the GPU using cudaMalloc, launches the CUDA kernels, handles memory transfers between the host and device using cudaMemcpy, outputs the rendered image, and frees allocated memory. Moreover, initialization and memory allocation involve setting up the frame buffer and random states, while the world creation is handled by the create_world kernel. 

The random states are initialized with the init_rand_state_kernel kernel. The render kernel, launched with a grid of blocks, computes the color for each pixel by determining pixel coordinates and generating rays, with anti-aliasing achieved through multiple samples. After rendering, the frame buffer is copied back to host memory, and the image is saved as a PPM file. Cleanup includes deallocating world objects with the free_world kernel and freeing GPU memory with cudaFree, with CUDA events being destroyed using cudaEventDestroy.

## <font size="5">Outcome</font>

Although the initial proposal planned for the use of OptiX for algorithm acceleration, we successfully implemented the project using CUDA instead and achieved good results. The project demonstrates the capability of CUDA to handle parallel processing for real-time ray tracing, providing efficient and high-quality rendered images without the need for OptiX.

## <font size="5">Performance Comparison</font>

The render times for both the C++ implementation and the CUDA implementation are as follows:

- **Render Time (C++):** 4604.66 milliseconds
- **Render Time (CUDA):** 157.959 milliseconds
  
This significant reduction in render time demonstrates the effectiveness of using CUDA for accelerating the ray tracing process.

## <font size="5">Acknowledgments</font>

This project is based on the "Ray Tracing in One Weekend" series by Peter Shirley. The idea was to transform and optimize the ray tracing code utilizing CUDA for parallel processing and accelerated rendering.

**Citation**:

Shirley, P. (2016). Ray Tracing in One Weekend. Retrieved from Ray Tracing in One Weekend

## <font size="5">Installation</font>

1. **Clone the repository**:
    ```sh
    git clone https://github.com/adrmtz91/RTXONRayTracingFP.git
    cd RTXONRayTracingFP
    ```

2. **Install dependencies**:
    Ensure you have Nvidia's CUDA Toolkit and a compatible compiler installed on your system.
    Magick - For converting .ppm to .png files for better compatibility.

3. **Build the project**:
    For CUDA:
    ```sh
    make rt_cuda
    ```

    For C++:
    ```sh
    make rt_cpu
    ```

4. **Clean build files**:
    ```sh
    make clean
    ```

## <font size="5">Usage</font>

To render the scene using the CUDA implementation:
    ```sh
    make render_cuda
    ```
To render the scene using the C++ implementation:
    ```sh
    make render_cpu
    ```
The output images will be saved as output_cuda.png and output_cpu.png respectively.

## <font size="5">Results</font>
Below are the images rendered using the CUDA and C++ implementations:

<font size="4">CUDA Rendered Image</font>

![CUDA Rendered Image](images/output_cuda.png)

<font size="4">C++ Rendered Image</font>

![C++ Rendered Image](images/output_cpu.png)

As seen in the images, the CUDA implementation provides comparable quality while significantly reducing the render time.
