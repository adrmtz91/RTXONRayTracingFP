all: main.cpp
	g++ -o program main.cpp

clean:
	rm -f program