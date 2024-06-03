CXX = g++
CXX_FLAGS = 

.PHONY: main

main: rt_cpu

rt_cpu:
	$(CXX) cpu/main.cpp -o rt_cpu $(CXX_FLAGS)

render_cpu: rt_cpu
	./rt_cpu > output.ppm
	magick output.ppm output_cpu.png

clean:
	rm -f rt_cpu output.ppm output_cpu.png