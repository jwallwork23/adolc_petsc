ALL: clean exec

CXXFLAGS	= -std=c++11
INCLUDES	= -I${ADOLC_HOME}/build/include
LIB		= -L${ADOLC_HOME}/build/ -ladolc
CC		= g++
CLEANFILES	= *.o exec *~ *.tap

SRC=speelpenning.cpp
OBJ=speelpenning.o

%.o:%.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) $^ -c -o$@

exec: $(OBJ)
	$(CC) $(CXXFLAGS) $(OBJ) $(LIB) -o $@

clean:
	rm -Rf $(CLEANFILES)
