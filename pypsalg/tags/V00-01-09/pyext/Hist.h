#ifndef PYPSALG_HIST_H
#define PYPSALG_HIST_H

#include "ndarray/ndarray.h"

namespace pypsalg {

class HistAxis
{
  public:
  HistAxis(unsigned nbins, double low, double high) :
    _low(low),
    _high(high),
    _binsize((high-low)/(nbins + 1.)),
    _nbins(nbins)
  {}

  int bin(double val)
  {
    return (int) floor((val-_low)/_binsize);
  }

  unsigned nbins()
  {
    return _nbins;
  }

  private:
  double _low;
  double _high;
  double _binsize;
  unsigned _nbins;
};

class Hist1D
{
 public:
  Hist1D(unsigned nbins, double low, double high);
  ndarray<double, 1> get();
  void fill(double val, double weight);
  void fill(ndarray<double, 1> vals, double weight);
  void fill(ndarray<double, 1> vals, ndarray<double, 1> weights);

 private:
  HistAxis axis;
  ndarray<double, 1> data;
};

class Hist2D
{
 public:
  Hist2D(unsigned nbinsx, double xlow, double xhigh,
         unsigned nbinsy, double ylow, double yhigh);
  ndarray<double, 2> get();
  void fill(double xval, double yval, double weight=1.);
  void fill(ndarray<double, 2> xyvals, double weight);
  void fill(ndarray<double, 2> xyWeightVals);

 private:
  HistAxis _xaxis;
  HistAxis _yaxis;
  ndarray<double, 2> _data;
};

}

#endif
