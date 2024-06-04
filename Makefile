CXX = g++
CXX_FLAGS = 
NVXX = nvcc

.PHONY: main cuda render clean

main: rt_cpu
cuda: rt_cuda

rt_cpu:
	$(CXX) cpu/main.cpp -o rt_cpu $(CXX_FLAGS)

rt_cuda:
	$(NVXX) cuda/main.cu -o rt_cuda

render_cpu: rt_cpu
	./rt_cpu > output.ppm
	magick output.ppm output_cpu.png

render_cuda: rt_cuda
	./rt_cuda
	magick output.ppm output_cuda.png

clean: 
	rm -f rt_cpu rt_cuda output.ppm output_cpu.png output_cuda.png