#include "psana_test/adler.h"
#include <iostream>

uint8_t buffer[1024];

int main() {
  std::cout << "adler: " << psana_test_adler32(buffer, 1024) << std::endl;
  return 0;
}
