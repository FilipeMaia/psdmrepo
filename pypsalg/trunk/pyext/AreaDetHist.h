#ifndef PYPSALG_AREADETHIST_H
#define PYPSALG_AREADETHIST_H

#include "ndarray/ndarray.h"
#include <boost/shared_ptr.hpp>


typedef uint16_t data_t;

class AreaDetHist {
 public:

  AreaDetHist (ndarray<double,3> calib_data,int,int);
  virtual ~AreaDetHist () ;
  
  ndarray<uint32_t,2> getHist();
  ndarray<uint32_t,2> update(ndarray<double,3> calib_data, int findIsolated, double minAduGap);

  ndarray<uint32_t,2> histogram; // per-pixel histogram (Size x histLength)
  //ndarray<uint32_t,2> photonMap; // per-pixel ADU to photons (Size x histLength)

 private:
  void _fillHistogram(ndarray<double,3> calib_data, ndarray<uint32_t, 2> histogram);
  void _insertHistElement(double x, int pixelInd, ndarray<uint32_t, 2> histogram);
  int _valid_min;
  int _valid_max;
  unsigned _histLength;

};

#endif // PYPSALG_AREADETHIST_H
