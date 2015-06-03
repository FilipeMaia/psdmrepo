#include <math.h>
#include <vector>
#include "AreaDetHist.h"
#include "MsgLogger/MsgLogger.h"

namespace pypsalg {

// Constructor
AreaDetHist::AreaDetHist (ndarray<double,3> calib_data, int valid_min,
                          int valid_max, bool findIsolated, double minAduGap) :
  _valid_min(valid_min),_valid_max(valid_max),
  _findIsolated(findIsolated),_minAduGap(minAduGap)
{
  _histLength = (_valid_max-_valid_min+1)+2; // extra two pixels for underflow/overflow
  const unsigned int *arrayShape = calib_data.shape();  
  _segs = arrayShape[0];
  _rows = arrayShape[1];
  _cols = arrayShape[2];
  _numPixPerSeg = _rows*_cols;
  _histogram = make_ndarray<uint32_t>(calib_data.size(),_histLength); // doesn't guarantee zeros
  for (ndarray<uint32_t,2>::iterator p = _histogram.begin(); p != _histogram.end(); p++) {
    *p = 0;
  }
}

// Destructor
AreaDetHist::~AreaDetHist () {}

// Returns histogram
ndarray<uint32_t, 2> AreaDetHist::get() {return _histogram;}

// Fills histogram in a standard way using under/overflow stored at the first/last elements.
void AreaDetHist::_fillHistogram(ndarray<double,3> calib_data) {
  int pixelInd = 0;
  int val;
  // fill histogram
  for (ndarray<double,3>::iterator p = calib_data.begin();
       p != calib_data.end(); p++) {
    val = (int) round(*p);   
    if ( val >= _valid_min && val <= _valid_max ) { // in range
      _histogram[ pixelInd ][ val-_valid_min+1 ] += 1;
    } else if ( val > _valid_max ) { // too large
      _histogram[ pixelInd ][ _histLength-1 ] += 1;
    } else { // too small
      _histogram[ pixelInd ][ 0 ] += 1;
    }
    pixelInd++;
  }
}

// Calculates whether val is greater than its neighbors by at least minAduGap
int isIsolated(double val, double minAduGap, std::vector<double> *neighbors) {
	int result = 1;
	std::vector<double>::iterator p;
	for (p = neighbors->begin(); p != neighbors->end(); p++) {
		if (val-minAduGap < *p) {
			result = 0;
			break;
		}
	}
	return result;
}

// Get pixel index given pixel position (seg,row,col)
unsigned int getPixelIndex(const unsigned int numPixPerSeg, const unsigned int Cols, unsigned int seg, unsigned int row, unsigned int col) {
    return seg*numPixPerSeg + row*Cols + col;
}

// Increment counter on histogram
void AreaDetHist::_insertHistElement(double x, int pixelInd) {
	int val = (int) round(x);   
	if ( val >= _valid_min && val <= _valid_max ) { // in range
		_histogram[ pixelInd ][ val-_valid_min+1 ] += 1;
	} else if ( val > _valid_max ) { // too large
		_histogram[ pixelInd ][ _histLength-1 ] += 1;
	} else { // too small
		_histogram[ pixelInd ][ 0 ] += 1;
	}
}

// Update the histograms given calib_data
void AreaDetHist::update(ndarray<double,3> calib_data)
{
  if (_findIsolated) {
	double val = 0;
    int result = 0;
	int pixelInd = 0;
	unsigned int j = 0;
	unsigned int k = 0;

	// corner of detector (3 pixel neighborhood)
	int numNeighbors = 3;
    std::vector<double> neighbors(numNeighbors);
	for (unsigned int i = 0; i < _segs; i++) {
		// Upper left corner
		j = 0;
		k = 0;
		val = calib_data[i][j][k];
		neighbors[0] =	calib_data[i][j+1][k];
		neighbors[1] =	calib_data[i][j][k+1];
		neighbors[2] =	calib_data[i][j+1][k+1];
		result = isIsolated(val, _minAduGap, &neighbors);
		if (result) {
			pixelInd = getPixelIndex(_numPixPerSeg, _cols, i, j, k);
			_insertHistElement(val, pixelInd);
		}

		// Upper right corner
		k = _cols-1;
		val = calib_data[i][j][k];
		neighbors[0] = calib_data[i][j+1][k];
		neighbors[1] = calib_data[i][j][k-1];
		neighbors[2] = calib_data[i][j+1][k-1];
		result = isIsolated(val, _minAduGap, &neighbors);
		if (result) {
			pixelInd = getPixelIndex(_numPixPerSeg, _cols, i, j, k);
			_insertHistElement(val, pixelInd);
		}

		// Lower left corner
		j = _rows-1;
		k = 0;
		val = calib_data[i][j][k];
		neighbors[0] = calib_data[i][j][k+1];
		neighbors[1] = calib_data[i][j-1][k];
		neighbors[2] = calib_data[i][j-1][k+1];
		result = isIsolated(val, _minAduGap, &neighbors);
		if (result) {
			pixelInd = getPixelIndex(_numPixPerSeg, _cols, i, j, k);
			_insertHistElement(val, pixelInd);
		}

		// Lower right corner
		k = _cols-1;
		val = calib_data[i][j][k];
		neighbors[0] = calib_data[i][j][k-1];
		neighbors[1] = calib_data[i][j-1][k];
		neighbors[2] = calib_data[i][j-1][k-1];
		result = isIsolated(val, _minAduGap, &neighbors);
		if (result) {
			pixelInd = getPixelIndex(_numPixPerSeg, _cols, i, j, k);
			_insertHistElement(val, pixelInd);
		}
	}

	// side of detector (5 pixel neighborhood)
	numNeighbors = 5;
    neighbors.resize(numNeighbors);
	for (unsigned int i = 0; i < _segs; i++) {
		// Upper edge
		j = 0;
		for (unsigned int t = 1; t < _cols-1; t++) {
			val = calib_data[i][j][t];
			neighbors[0] = calib_data[i][j][t-1];
			neighbors[1] = calib_data[i][j][t+1];
			neighbors[2] = calib_data[i][j+1][t-1];
			neighbors[3] = calib_data[i][j+1][t];
			neighbors[4] = calib_data[i][j+1][t+1];
			result = isIsolated(val, _minAduGap, &neighbors);
			if (result) {
				pixelInd = getPixelIndex(_numPixPerSeg, _cols, i, j, t);
				_insertHistElement(val, pixelInd);
			}
		}

		// Lower edge
		j = _rows-1;
		for (unsigned int t = 1; t < _cols-1; t++) {
			val = calib_data[i][j][t];
			neighbors[0] = calib_data[i][j][t-1];
			neighbors[1] = calib_data[i][j][t+1];
			neighbors[2] = calib_data[i][j-1][t-1];
			neighbors[3] = calib_data[i][j-1][t];
			neighbors[4] = calib_data[i][j-1][t+1];
			result = isIsolated(val, _minAduGap, &neighbors);
			if (result) {
				pixelInd = getPixelIndex(_numPixPerSeg, _cols, i, j, t);
				_insertHistElement(val, pixelInd);
			}
		}

		// Left edge
		k = 0;
		for (unsigned int t = 1; t < _rows-1; t++) {
			val = calib_data[i][t][k];
			neighbors[0] = calib_data[i][t+1][k];
			neighbors[1] = calib_data[i][t-1][k];
			neighbors[2] = calib_data[i][t+1][k+1];
			neighbors[3] = calib_data[i][t][k+1];
			neighbors[4] = calib_data[i][t+1][k+1];
			result = isIsolated(val, _minAduGap, &neighbors);
			if (result) {
				pixelInd = getPixelIndex(_numPixPerSeg, _cols, i, t, k);
				_insertHistElement(val, pixelInd);
			}
		}

		// Right edge
		k = _cols-1;
		for (unsigned int t = 1; t < _rows-1; t++) {
			val = calib_data[i][t][k];
			neighbors[0] = calib_data[i][t+1][k];
			neighbors[1] = calib_data[i][t-1][k];
			neighbors[2] = calib_data[i][t+1][k-1];
			neighbors[3] = calib_data[i][t][k-1];
			neighbors[4] = calib_data[i][t+1][k-1];
			result = isIsolated(val, _minAduGap, &neighbors);
			if (result) {
				pixelInd = getPixelIndex(_numPixPerSeg, _cols, i, t, k);
				_insertHistElement(val, pixelInd);
			}
		}
	}

	// non-edge of detector (8 pixel neighborhood)
	numNeighbors = 8;
    neighbors.resize(numNeighbors);
	for (unsigned int i = 0; i < _segs; i++) {
	for (unsigned int j = 1; j < _rows-1; j++) {
	for (unsigned int k = 1; k < _cols-1; k++) {
		val = calib_data[i][j][k];
		neighbors[0] = calib_data[i][j-1][k-1];
		neighbors[1] = calib_data[i][j-1][k];
		neighbors[2] = calib_data[i][j-1][k+1];
		neighbors[3] = calib_data[i][j][k-1];
		neighbors[4] = calib_data[i][j][k+1];
		neighbors[5] = calib_data[i][j+1][k-1];
		neighbors[6] = calib_data[i][j+1][k];
		neighbors[7] = calib_data[i][j+1][k+1];
		result = isIsolated(val, _minAduGap, &neighbors);
		if (result) {
			pixelInd = getPixelIndex(_numPixPerSeg, _cols, i, j, k);
			_insertHistElement(val, pixelInd);
		}
	}
	}
	}
  } else {
	_fillHistogram(calib_data);
  }
}

} // namespace pypsalg
