ALL: clean speelpenning

CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/include
LIB 		= -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
LIB		+= -L${USER_LIB} -lboost_system
CC		= g++
CLEANFILES	= *.o speelpenning traceless *~ *.tap

%.o: %.cpp
	$(CC) $(CXXFLAGS) $^ -c -o $@

speelpenning: speelpenning.o
	$(CC) $^ -o $@ $(LIB)

traceless: traceless.o
	$(CC) $^ -o $@ $(LIB)

exec: $(OBJ)
	$(CC) -o $@ $^ $(LIB)

clean:
	rm -Rf $(CLEANFILES)
