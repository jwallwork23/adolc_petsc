ALL: clean exec

CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/include -L${USER_LIB} -lboost_system
LIB 		= -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
CC		= g++
CLEANFILES	= *.o exec *~ *.tap

SRC=speelpenning.cpp
OBJ=speelpenning.o

%.o:%.cpp
	$(CC) $(CXXFLAGS) $^ -c -o $@

exec: $(OBJ)
	$(CC) $^ $(LIB) -o $@

clean:
	rm -Rf $(CLEANFILES)
