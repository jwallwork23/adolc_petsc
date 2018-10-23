ALL: clean exec

CXXFLAGS	= -std=c++11
INCLUDES	= -I${ADOLC_BUILDDIR}/include
LIB		= -L${ADOLC_BUILDDIR}/lib64 -ladolc
CC		= g++
CLEANFILES	= *.o exec *~ *.tap

SRC=speelpenning.cpp
OBJ=speelpenning.o

%.o:%.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) $^ -c -o $@

exec: $(OBJ)
	$(CC) $(CXXFLAGS) $(OBJ) $(LIB) -o $@

clean:
	rm -Rf $(CLEANFILES)
