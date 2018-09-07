COLPACK_INSTALL_PATH = ${COLPACK_HOME}/lib/libColPack.so

SRC=adolc_example.cpp
OBJ=adolc_example.o
EXE=exec

# compiler
COMPILE	= g++

#compile flags
CCFLAGS	= -Wall -fopenmp -O3 -std=c++11

# link flags
LDFLAGS	= -Wall -fopenmp -O3 -std=c++11 -L${ADOLC_HOME}/build -ladolc -ldl ${COLPACK_INSTALL_PATH}

INCLUDES= -I./
INCLUDES+= -I${ADOLC_HOME}/build/include/adolc
INCLUDES+= -I${COLPACK_HOME}/include/ColPack
INCLUDES+= -I${COLPACK_HOME}/GraphColoring
INCLUDES+= -I${COLPACK_HOME}/BipartiteGraphBicoloring
INCLUDES+= -I${COLPACK_HOME}/BipartiteGraphPartialColoring
INCLUDES+= -I${COLPACK_HOME}/Utilities


all: clean $(EXE)

%.o:%.cpp
	$(COMPILE) $(INCLUDES) $(CCFLAGS) -c $< -o$@

$(EXE): $(OBJ)
	$(COMPILE) $^ $(INCLUDES) $(LDFLAGS) -o $@

clean:
	rm -Rf $(EXE) $(OBJ)
