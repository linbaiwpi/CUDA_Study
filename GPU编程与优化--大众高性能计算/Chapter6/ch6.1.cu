#include "ch6.h"
// #include <iostream>

void vector_add_serial(DATATYPE* a, DATATYPE* b, DATATYPE* c, int n) {
    for (int i=0; i<n; i++) {
        c[i] = a[i] + b[i];
        // std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }
}
