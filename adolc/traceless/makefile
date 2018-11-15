ALL: clean basic

CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/include 
LIB 		= -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
LIB		+= -L${USER_LIB} -lboost_system
CC		= g++
CLEANFILES	= *.o basic vector *~ *.tap

%.o: %.cpp
	$(CC) $(CXXFLAGS) $^ -c -o $@

basic: basic.o
	$(CC) $^ -o $@ $(LIB)

clean:
	rm -Rf $(CLEANFILES)
