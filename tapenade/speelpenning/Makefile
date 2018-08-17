all: clean exec


# SRC=speelpenning.c
# OBJ=speelpenning.o
#
#
# # Establish which c files to generate objects from
# %.o:%.c
# 	gcc $^ -c -o$@

# # Define object for execution
# exec: $(OBJ)
#	gcc $(OBJ) -o exec

# Compile base file and take differentiate forward
exec:
	gcc -c -o product.o product.c
	tapenade -tangent -head "product(x)\(y)" product.c
	gcc -c -o product_d.o product_d.c
	gcc -c -o speelpenning.o speelpenning.c
	gcc -o exec product.o product_d.o speelpenning.o
	rm *.o

# Define cleaning operator for various misc file types
clean:
	rm -Rf *~ *.msg *_d.c *_d.h exec
