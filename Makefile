EXEC = raytracing \
       raytracing-sse \
       raytracing-avx

GIT_HOOKS := .git/hooks/pre-commit
.PHONY: all
all: $(GIT_HOOKS) $(EXEC)

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo

CC ?= gcc
CFLAGS_SSE = \
	-msse2 -D SSE 
CFLAGS_AVX = \
	-mavx -D AVX 
CFLAGS_COMM = \
	-std=gnu99 -Wall -O0 -g -fopenmp 
LDFLAGS = \
	-lm -fopenmp

ifeq ($(strip $(PROFILE)),1)
PROF_FLAGS = -pg
CFLAGS += $(PROF_FLAGS)
LDFLAGS += $(PROF_FLAGS) 
endif

OBJS := \
	objects.o \
	raytracing.o \
	main.o \
        raytracing-sse.o\
        main-sse.o\
	raytracing-avx.o\
        main-avx.o  

raytracing-avx.o:raytracing-avx.c
	$(CC) $(CFLAGS_COMM) $(CFLAGS_AVX)-c -o $@ $<
raytracing-sse.o:raytracing-sse.c
	$(CC) $(CFLAGS_COMM) $(CFLAGS_SSE)-c -o $@ $<
raytracing.o:raytracing.c
	$(CC) $(CFLAGS_COMM) -c -o $@ $<

raytracing:main.o raytracing.o objects.o 
	$(CC) $(CFLAGS_COMM) -o $@ $^ $(LDFLAGS)

raytracing-sse:main.o raytracing-sse.o objects.o
	$(CC) $(CFLAGS_COMM) $(CFLAGS_SSE) -o $@ $^ $(LDFLAGS)

raytracing-avx:main.o raytracing-avx.o objects.o
	$(CC) $(CFLAGS_COMM) $(CFLAGS_AVX) -o $@ $^ $(LDFLAGS)


main.o: use-models.h
use-models.h: models.inc Makefile
	@echo '#include "models.inc"' > use-models.h
	@egrep "^(light|sphere|rectangular) " models.inc | \
	    sed -e 's/^light /append_light/g' \
	        -e 's/light[0-9]/(\&&, \&lights);/g' \
	        -e 's/^sphere /append_sphere/g' \
	        -e 's/sphere[0-9]/(\&&, \&spheres);/g' \
	        -e 's/^rectangular /append_rectangular/g' \
	        -e 's/rectangular[0-9]/(\&&, \&rectangulars);/g' \
	        -e 's/ = {//g' >> use-models.h

check: $(EXEC)
	@./$(EXEC) && diff -u baseline.ppm out.ppm || (echo Fail; exit)
	@echo "Verified OK"

clean:
	$(RM) $(EXEC) $(OBJS) use-models.h \
		out.ppm gmon.out
