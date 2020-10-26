#ifndef TIMER_HPP
#define TIMER_HPP

#include <stdio.h>
#include <time.h>


#ifdef _WIN32
#include <windows.h>
struct Timer {
    LARGE_INTEGER t0;
    LARGE_INTEGER t1;
    LARGE_INTEGER freq;
};

void timer_init(struct Timer* t) {
    QueryPerformanceFrequency(&t->freq);
}

void timer_start(struct Timer* t) {
    QueryPerformanceCounter(&t->t0);
}

void timer_stop(struct Timer* t) {
    QueryPerformanceCounter(&t->t1);
}

double timer_get(struct Timer* t) {
    return (double)(t->t1.QuadPart - t->t0.QuadPart) / t->freq.QuadPart;
}
#endif


#ifndef _WIN32
struct Timer {
    clock_t t0;
    clock_t t1;
};

void timer_init(struct Timer* t) {}

void timer_start(struct Timer* t) {
    t->t0 = clock();
}

void timer_stop(struct Timer* t) {
    t->t1 = clock();
}

double timer_get(struct Timer* t) {
    return (double)(t->t1 - t->t0) / (double)(CLOCKS_PER_SEC);
}
#endif


void timer_print(struct Timer* t) {
    printf("%fs\n", timer_get(t));
}

#endif
