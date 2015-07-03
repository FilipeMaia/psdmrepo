#include <math.h>
#include <cstring>
#include "Hist.h"

namespace pypsalg {

Hist1D::Hist1D(unsigned nbins, double low, double high) :
  axis(nbins, low, high)
{
  data = make_ndarray<double>(nbins);
  std::memset(data.data(), 0, data.size()*sizeof(double));
}

ndarray<double, 1> Hist1D::get()
{
  return data;
}

void Hist1D::fill(double val, double weight)
{
  int bin = axis.bin(val);
  if(bin >= 0 && bin < (int) axis.nbins()) {
    data[bin] += weight;
  }
}

void Hist1D::fill(ndarray<double, 1> vals, double weight)
{
  const ndarray<double,1>::shape_t* shape = vals.shape();
  for (unsigned i=0; i<shape[0]; i++) {
    fill(vals[i],weight);
  }
}

void Hist1D::fill(ndarray<double, 1> vals, ndarray<double, 1> weights)
{
  const ndarray<double,1>::shape_t* shapevals = vals.shape();
  const ndarray<double,1>::shape_t* shapeweights = weights.shape();
  assert (shapevals[0]==shapeweights[0]);
  for (unsigned i=0; i<shapevals[0]; i++) {
    fill(vals[i],weights[i]);
  }
}

void Hist1D::fill(ndarray<double, 2> valsWeights)
{
  const ndarray<double,2>::shape_t* shape = valsWeights.shape();
  assert(shape[1]==2); // x-coordinate plus weight
  for (unsigned i=0; i<shape[0]; i++) {
    fill(valsWeights[i][0],valsWeights[i][1]);
  }
}

Hist2D::Hist2D(unsigned nbinsx, double xlow, double xhigh,
               unsigned nbinsy, double ylow, double yhigh) :
  _xaxis(nbinsx, xlow, xhigh),
  _yaxis(nbinsy, ylow, yhigh)
{
  _data = make_ndarray<double>(nbinsx, nbinsy);
  std::memset(_data.data(), 0, _data.size()*sizeof(double));
}

ndarray<double, 2> Hist2D::get()
{
  return _data;
}

void Hist2D::fill(double xval, double yval, double weight)
{
  int binx = _xaxis.bin(xval);
  int biny = _yaxis.bin(yval);
  if(binx >= 0 && binx < (int) _xaxis.nbins() &&
     biny >= 0 && biny < (int) _yaxis.nbins()) {
    _data[binx][biny] += weight;
  }
}

void Hist2D::fill(ndarray<double, 2> xyvals, double weight)
{
  const ndarray<double,1>::shape_t* shape = xyvals.shape();
  assert(shape[1]==2); // 2 columns of x-y coordinates
  for (unsigned i=0; i<shape[0]; i++) {
    fill(xyvals[i][0],xyvals[i][1],weight);
  }
}

void Hist2D::fill(ndarray<double, 2> xyWeightVals)
{
  const ndarray<double,1>::shape_t* shape = xyWeightVals.shape();
  assert(shape[1]==3); // 2 columns of x-y coordinates plus weights
  for (unsigned i=0; i<shape[0]; i++) {
    fill(xyWeightVals[i][0],xyWeightVals[i][1],xyWeightVals[i][2]);
  }
}

void Hist2D::fill(ndarray<double, 1> xvals, ndarray<double, 1> yvals, ndarray<double, 1> weights)
{
  const ndarray<double,1>::shape_t* xshape = xvals.shape();
  const ndarray<double,1>::shape_t* yshape = yvals.shape();
  const ndarray<double,1>::shape_t* wshape = weights.shape();
  assert(xshape[0]==yshape[0]);
  assert(xshape[0]==wshape[0]);
  for (unsigned i=0; i<xshape[0]; i++) {
    fill(xvals[i],yvals[i],weights[i]);
  }
}

void Hist2D::fill(ndarray<double, 1> xvals, ndarray<double, 1> yvals, double weight)
{
  const ndarray<double,1>::shape_t* xshape = xvals.shape();
  const ndarray<double,1>::shape_t* yshape = yvals.shape();
  assert(xshape[0]==yshape[0]);
  for (unsigned i=0; i<xshape[0]; i++) {
    fill(xvals[i],yvals[i],weight);
  }
}

}
