
#include <pyext/NdarrayCvt.h>
#include <iostream>



struct numpytest
{
  void printArray(ndarray<float,1> array) {
    std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "HELLO FROM PRINT ARRAY !!" << std::endl;
    std::cout << "Number of elements " << array.size() << std::endl;
    std::cout << "Address of underlying array " << array.data() << std::endl;
    ndarray<float,1>::iterator iter;
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

  ndarray<float,1> outArray() {
    std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "outArray called" << std::endl;

    //    float* data = new float[10];
    ndarray<float,1>::shape_t shape[1] = {10};
    // ==> TRY NDARRAY's INTERNAL MEMORY ALLOCATION
    ndarray<float,1> array(shape);

    for (int i=0; i<10; i++) {
      //data[i] = i;
      array[i] = i;
    }

    //    ndarray<float,1> array(data,shape);
    
    
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

  ndarray<float,1> calcmean(ndarray<float,1> array) {
    std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
    std::cout << "My version of mean calculation" << std::endl;
    std::cout << "Sum up contents of array" << std::endl;
        
    float count = 0.0;
    float sum = 0.0;
    ndarray<float,1>::iterator iter;
    for (iter = array.begin(); iter != array.end(); iter++) {
      sum += *iter;
      count++;
      std::cout << count << " " << *iter << " " << sum << std::endl;
    }

    float mean = sum / count;
    std::cout << "Mean: " << mean << std::endl;

    ndarray<float,1>::shape_t shape[1] = {1};    
    //    float* data = new float[1];
    //    data[0] = mean;

    // TRY NDARRAY'S INTERNAL MEMORY ALLOCATION
    //    ndarray<float,1> outarray(data, shape);
    ndarray<float,1> outarray(shape);            
    outarray[0] = mean;

    //    std::cout << "DELETING ARRAY THAT HOLDS VALUE OF MEAN" << std::endl;
    //    delete data;
    std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;
    return outarray;
  }

  void printArray2D(ndarray<float,2> array, double number1, float number2=100.0) {
    std::cout << "Print 2D float array" << std::endl;
    std::cout << array << std::endl;
    std::cout << "Number1:" << number1 << std::endl;
    std::cout << "Number2:" << number2 << std::endl;
    return;
  }

  ndarray<double,2> find_edges(const ndarray<const double,1>& wf,
			       double baseline_value,
			       double threshold_value,
			       double fraction,
			       double deadtime,
			       bool   leading_edge)  {

    std::cout << "ANKUSH'S FIND EDGES CALLED!!" << std::endl;
    //    baseline_value += 1.0;
    std::cout << wf << std::endl;
    std::cout << baseline_value << std::endl;
    std::cout << threshold_value << std::endl;
    std::cout << fraction << std::endl;
    std::cout << deadtime << std::endl;
    std::cout << leading_edge << std::endl;
    
    ndarray<double,2> outarray = make_ndarray<double>(2,2);
    

    std::cout << outarray << std::endl;
    return outarray;
  }


};


class psalg_ankush {
 public:
    static ndarray<double,2> find_edges(const ndarray<const double,1>& wf,
					double baseline_value,
					double threshold_value,
					double fraction,
					double deadtime,
					bool   leading_edge)  {

    std::cout << "PSALG_ANKUSH'S FIND EDGES CALLED!!" << std::endl;
    std::cout << wf << std::endl;
    std::cout << baseline_value << std::endl;
    std::cout << threshold_value << std::endl;
    std::cout << fraction << std::endl;
    std::cout << deadtime << std::endl;
    std::cout << leading_edge << std::endl;
    
    ndarray<double,2> outarray = make_ndarray<double>(2,2);
    
    std::cout << outarray << std::endl;
    return outarray;
  }



};
