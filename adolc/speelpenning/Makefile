all: clean exec


SRC=speelpenning.cpp
OBJ=speelpenning.o

#speelpenning.o:speelpenning.cpp
%.o:%.cpp
	g++ -std=c++11 -I/homes/jwallwork/software/adol-c/ADOL-C-2.6.0/build/include $^ -c -o$@

exec: $(OBJ)
	g++ -std=c++11 $(OBJ) -L/homes/jwallwork/software/adol-c/ADOL-C-2.6.0/build/ -ladolc -o exec 

clean:
	rm -Rf *.o exec *~ *.tap
