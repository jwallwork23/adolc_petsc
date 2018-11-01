ALL: clean traceless

CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/include -L${USER_LIB} -lboost_system
LIB 		= -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
CC		= g++
CLEANFILES	= *.o traceless *~ *.tap

%.o: %.cpp
	$(CC) $(CXXFLAGS) $^ -c -o $@

traceless: traceless.o
	$(CC) $^ -o $@ $(LIB)

clean:
	rm -Rf $(CLEANFILES)
