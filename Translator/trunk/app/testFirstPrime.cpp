#include "Translator/firstPrimeGreaterOrEqualTo.h"
#include <iostream>

using namespace std;

int main() {
  static unsigned testVals[] = {0,1,2,3,4,32, 359,10000,104728, 104729,104730,43453,9999999999};
  for (unsigned i = 0; i < sizeof(testVals)/sizeof(unsigned); ++i) {
    cout << "firstPrimeGreaterOrEqualTo " << testVals[i] << " is " << firstPrimeGreaterOrEqualTo(testVals[i]) << endl;
  }
  return 0;
}
