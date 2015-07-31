#ifndef PYPSALG_AREADETHIST_H
#define PYPSALG_AREADETHIST_H

#include "ndarray/ndarray.h"
#include <boost/shared_ptr.hpp>
#include <stdint.h>

namespace pypsalg {

class AreaDetHist {
 public:

  AreaDetHist (ndarray<double,3> calib_data,int,int,
               bool findIsolated, double minAduGap);
  virtual ~AreaDetHist ();
  
  ndarray<uint32_t,4> get();

  void update(ndarray<double,3> calib_data);
  void update2x2(ndarray<double,3> calib_data);

 private:
  void _fillHistogram(ndarray<double,3> calib_data);
  void _insertHistElement(double x, int i, int j, int k);
  ndarray<uint32_t,4> _histogram;

  int _valid_min;
  int _valid_max;
  unsigned _histLength;
  bool _findIsolated;
  double _minAduGap;
  unsigned int _segs, _rows, _cols, _numPixPerSeg;

};

} // namespace pypsalg

#endif // PYPSALG_AREADETHIST_H
