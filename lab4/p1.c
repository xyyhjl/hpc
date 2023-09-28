#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
double GetTime() {
    struct timeval t;
    int rc = gettimeofday(&t, NULL);
    assert(rc == 0);
    return (double) t.tv_sec + (double) t.tv_usec/1e6;
}
void Spin(int howlong) {
    double t = GetTime();
    while ((GetTime() - t) < (double) howlong)
        ; // do nothing in loop
}
int main(int argc, char *argv[])
{
    printf("hello world (pid:%d)\n", (int) getpid());
    int rc = fork();
    if (rc < 0) {
        // fork failed; exit
        fprintf(stderr, "fork failed\n");
        exit(1);
    } else if (rc == 0) {
        // child (new process)
	double s1 = GetTime();
        Spin(5);
	double f1 = GetTime();
	printf("hello, I am child (pid:%d)\nduration:%lf\n", (int) getpid(),f1-s1);
    } else {
        // parent goes down this path (original process)
	double s2 = GetTime();
        Spin(5);
	double f2 = GetTime();
        printf("hello, I am parent of %d (pid:%d)\nduration:%lf\n",
	       rc, (int) getpid(),f2-s2);
    }
    return 0;
}
