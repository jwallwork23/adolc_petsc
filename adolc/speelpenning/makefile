all: clean exec


SRC=speelpenning.cpp
OBJ=speelpenning.o

#speelpenning.o:speelpenning.cpp
%.o:%.cpp
	g++ -std=c++11 -I${ADOLC_HOME}/build/include $^ -c -o$@

exec: $(OBJ)
	g++ -std=c++11 $(OBJ) -L${ADOLC_HOME}/build/ -ladolc -o exec

clean:
	rm -Rf *.o exec *~ *.tap
