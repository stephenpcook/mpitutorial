EXECS=avg all_avg avgv
MPICC?=mpicc

all: ${EXECS}

avg: avg.c
	${MPICC} -o avg avg.c

all_avg: all_avg.c
	${MPICC} -o all_avg all_avg.c

avgv: avgv.c
	${MPICC} -o avgv avgv.c

clean:
	rm -f ${EXECS}
