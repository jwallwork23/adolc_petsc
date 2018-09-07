# make file for using installed library
# author xin cheng
# usage: chage the following two variable accordingly
COLPACK_INSTALL_PATH = ${COLPACK_HOME}/lib/libColPack.so

SRC = $(wildcard *.cpp)
OBJ = $(SRC:%.cpp=%.o)
EXE = exec

# compiler
COMPILE = g++      # gnu

# compile flags
CCFLAGS = -Wall -fopenmp -O3 -std=c++11

# link flags
LDFLAGS = -Wall -fopenmp -O3 -std=c++11 -ldl ${COLPACK_INSTALL_PATH}

INCLUDES = -I./
INCLUDES+= -I${COLPACK_HOME}/include/ColPack
INCLUDES+= -I${COLPACK_HOME}/GraphColoring
INCLUDES+= -I${COLPACK_HOME}/BipartiteGraphBicoloring
INCLUDES+= -I${COLPACK_HOME}/BipartiteGraphPartialColoring
INCLUDES+= -I${COLPACK_HOME}/Utilities


all: clean $(EXE)

%.o : %.cpp
	$(COMPILE) $(INCLUDES) $(CCFLAGS) -c $< -o $@

$(EXE): $(OBJ)
	$(COMPILE) $^ $(INCLUDES) $(LDFLAGS)  -o $@

clean:
	rm -f $(OBJ) $(EXE)

