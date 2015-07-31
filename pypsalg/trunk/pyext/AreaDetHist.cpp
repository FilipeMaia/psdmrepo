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
  // Important note: cspad and cspad2x2 have different ordering of segs,rows,cols.
  // cspad shape: (segs,rows,cols)
  // cspad2x2 shape: (rows,cols,segs)
  // But dethist.py converts the cspad2x2's shape into (segs,rows,cols)
  const unsigned int *arrayShape = calib_data.shape();  
  _segs = arrayShape[0]; 
  _rows = arrayShape[1];
  _cols = arrayShape[2];
  _numPixPerSeg = _rows*_cols;
  _histogram = make_ndarray<uint32_t>(_segs,_rows,_cols,_histLength);
  for (ndarray<uint32_t,4>::iterator p = _histogram.begin(); p != _histogram.end(); p++) {
    *p = 0;
  }
}

// Destructor
AreaDetHist::~AreaDetHist () {}

// Returns histogram
ndarray<uint32_t, 4> AreaDetHist::get() {return _histogram;}

// Fills histogram in a standard way using under/overflow stored at the first/last elements.
void AreaDetHist::_fillHistogram(ndarray<double,3> calib_data) {
  int val;
  // fill histogram
  for (unsigned int i = 0; i < _segs; i++) {
  for (unsigned int j = 0; j < _rows; j++) {
  for (unsigned int k = 0; k < _cols; k++) {
    //val = (int) round(*p);
    val = (int) round(calib_data[i][j][k]);   
    if ( val >= _valid_min && val <= _valid_max ) { // in range
      _histogram[i][j][k][ val-_valid_min+1 ] += 1;
    } else if ( val > _valid_max ) { // too large
      _histogram[i][j][k][ _histLength-1 ] += 1;
    } else { // too small
      _histogram[i][j][k][ 0 ] += 1;
    }
    //pixelInd++;
  }
  }
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
void AreaDetHist::_insertHistElement(double x, int i, int j, int k) {
	int val = (int) round(x);   
	if ( val >= _valid_min && val <= _valid_max ) { // in range
		_histogram[i][j][k][ val-_valid_min+1 ] += 1;
	} else if ( val > _valid_max ) { // too large
		_histogram[i][j][k][ _histLength-1 ] += 1;
	} else { // too small
		_histogram[i][j][k][ 0 ] += 1;
	}
}

// Update the histograms given calib_data
void AreaDetHist::update(ndarray<double,3> calib_data)
{
  if (_findIsolated) {
	double val = 0;
    int result = 0;
	//int pixelInd = 0;
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
			// no need to get pixel index
			_insertHistElement(val, i, j, k);
		}

		// Upper right corner
		k = _cols-1;
		val = calib_data[i][j][k];
		neighbors[0] = calib_data[i][j+1][k];
		neighbors[1] = calib_data[i][j][k-1];
		neighbors[2] = calib_data[i][j+1][k-1];
		result = isIsolated(val, _minAduGap, &neighbors);
		if (result) {
			// no need to get pixel index
			_insertHistElement(val, i, j, k);
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
			// no need to get pixel index
			_insertHistElement(val, i, j, k);
		}

		// Lower right corner
		k = _cols-1;
		val = calib_data[i][j][k];
		neighbors[0] = calib_data[i][j][k-1];
		neighbors[1] = calib_data[i][j-1][k];
		neighbors[2] = calib_data[i][j-1][k-1];
		result = isIsolated(val, _minAduGap, &neighbors);
		if (result) {
			// no need to get pixel index
			_insertHistElement(val, i, j, k);
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
				// no need to get pixel index
				_insertHistElement(val, i, j, t);
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
				// no need to get pixel index
				_insertHistElement(val, i, j, t);
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
				// no need to get pixel index
				_insertHistElement(val, i, t, k);
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
				// no need to get pixel index
				_insertHistElement(val, i, t, k);
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
			// no need to get pixel index
			_insertHistElement(val, i, j, k);
		}
	}
	}
	}
  } else {
	_fillHistogram(calib_data);
  }
}

} // namespace pypsalg
