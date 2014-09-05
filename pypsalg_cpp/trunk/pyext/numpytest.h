
#include <pyext/NdarrayCvt.h>
#include <iostream>

struct numpytest
{
  void printArray(ndtype array) {
    std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "HELLO FROM PRINT ARRAY !!" << std::endl;
    std::cout << "Number of elements " << array.size() << std::endl;
    std::cout << "Address of underlying array " << array.data() << std::endl;
    ndtype::iterator iter;
    for (iter = array.begin(); iter != array.end(); iter++) {
      std::cout << *iter << " ";
    }
    std::cout << std::endl;

    //std::cout << "NOW DOING BAD THING ---> DELETE THE NDARRAY" << std::endl;
    //std::cout << "DOES THIS AFFECT THE UNDERLYING NUMPY ARRAY ?" <<std::endl; 
    //delete array.data();

    std::cout << "Leaving printArray" << std::endl;
    std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
    return;
  }

  ndtype outArray() {
    std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "outArray called" << std::endl;

    //    float* data = new float[10];
    ndtype::shape_t shape[1] = {10};
    // ==> TRY NDARRAY's INTERNAL MEMORY ALLOCATION
    ndtype array(shape);

    for (int i=0; i<10; i++) {
      //data[i] = i;
      array[i] = i;
    }

    //    ndtype array(data,shape);
    
    
    std::cout << "Contents of array" << std::endl;
    for (int i=0; i<10; i++) {
      std::cout << array[i] << std::endl;      
    }

    //std::cout << "DELETING ORIGINAL ARRAY" << std::endl;
    //delete data;

    std::cout << "Returning array" << std::endl;
    std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
    return array;
  }

  ndtype calcmean(ndtype array) {
    std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "My version of mean calculation" << std::endl;
    std::cout << "Sum up contents of array" << std::endl;
        
    float count = 0.0;
    float sum = 0.0;
    ndtype::iterator iter;
    for (iter = array.begin(); iter != array.end(); iter++) {
      sum += *iter;
      count++;
      std::cout << count << " " << *iter << " " << sum << std::endl;
    }

    float mean = sum / count;
    std::cout << "Mean: " << mean << std::endl;

    ndtype::shape_t shape[1] = {1};    
    //    float* data = new float[1];
    //    data[0] = mean;

    // TRY NDARRAY'S INTERNAL MEMORY ALLOCATION
    //    ndtype outarray(data, shape);
    ndtype outarray(shape);            
    outarray[0] = mean;

    //    std::cout << "DELETING ARRAY THAT HOLDS VALUE OF MEAN" << std::endl;
    //    delete data;
    std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
    return outarray;
  }


};
