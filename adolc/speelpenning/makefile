ALL: clean exec

CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/ADOL-C/include
LIB		= -L${ADOLC_BUILDDIR}/lib64 -ladolc
CC		= g++
CLEANFILES	= *.o exec *~ *.tap

SRC=speelpenning.cpp
OBJ=speelpenning.o

%.o:%.cpp
	$(CC) $(CXXFLAGS) $^ -c -o $@

exec: $(OBJ)
	$(CC) -o $@ $^ $(LIB)

clean:
	rm -Rf $(CLEANFILES)
