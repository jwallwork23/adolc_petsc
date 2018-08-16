all: clean exec


SRC=f1.cpp
OBJ=f1.o

%.o:%.cpp
	g++ -std=c++11 -I${ADOLC_HOME}/build/include $^ -c -o$@

exec: $(OBJ)
	g++ -std=c++11 $(OBJ) -L${ADOLC_HOME}/build/ -ladolc -o exec

clean:
	rm -Rf *.o exec *~ *.tap
