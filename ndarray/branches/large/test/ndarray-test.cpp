
#include <iostream>
#include <algorithm>
#include <iterator>

#include "ndarray/ndarray.h"

using namespace std;

int data[24] = { 
    0,1,2,3,4,5,6,7,8,9,
    10,11,12,13,14,15,16,17,18,19,
    20,21,22,23
};


int f2(const ndarray<int,3> arr, int i, int j, int k) 
{
  unsigned index[3] = {i, j, k};
  int val = arr.at(index);
  return val;
}

int f3(const ndarray<int,3> arr, int i, int j, int k)
{
  int val = arr[i][j][k];
  return val;
}


void f5(const ndarray<int,3> arr3)
{
  ndarray<int,2> arr2 = arr3[0];
  cout << "data2: ";
  copy(arr2.begin(), arr2.end(), ostream_iterator<int>(cout, " "));
  cout << "\n";
  arr2 = arr3[1];
  cout << "data2: ";
  copy(arr2.begin(), arr2.end(), ostream_iterator<int>(cout, " "));
  cout << "\n";
}



void f1() 
{
  unsigned dims[3] = {2,3,4};
  ndarray<int,3> arr(data, dims);

  cout << arr;

  for (unsigned i = 0; i != dims[0]; ++ i) {
    for (unsigned j = 0; j != dims[1]; ++ j) {
      for (unsigned k = 0; k != dims[2]; ++ k) {
        int val2 = f2(arr, i, j, k);
        int val3 = f3(arr, i, j, k);
        cout << "[" << i << ", " << j << ", " << k << "]: "
            << "val2 = " << val2 << " val3 = " << val3 << '\n';
      }
    }
  }


  // test iterators
  cout << "data: ";
  copy(arr.begin(), arr.end(), ostream_iterator<int>(cout, " "));
  cout << "\n";
  cout << "reverse: ";
  copy(arr.rbegin(), arr.rend(), ostream_iterator<int>(cout, " "));
  cout << "\n";

  f5(arr);
}


int main()
{
  f1();
}
