MAIN = main
CLEAN = clean
OBJS = Libs/CSP/*.o
CC = g++
CFLAGS = -g -O3 -Wall -fopenmp
RM = rm -f

all: $(CLEAN) $(MAIN)

$(MAIN): $()$(MAIN).cpp
	$(CC) $(CFLAGS) -o $(MAIN) $(MAIN).cpp $(OBJS)

$(CLEAN):
	$(RM) $(MAIN) *.o
