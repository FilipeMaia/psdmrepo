#include <math.h>
#include <vector>
#include "AreaDetHist.h"
#include "MsgLogger/MsgLogger.h"

AreaDetHist::AreaDetHist (ndarray<double,3> calib_data, int valid_min, int valid_max) : _valid_min(valid_min),_valid_max(valid_max)
{
  _histLength = (_valid_max-_valid_min+1)+2; // extra two pixels for underflow/overflow
  histogram = make_ndarray<uint32_t>(calib_data.size(),_histLength); // doesn't guarantee zeros
  for (ndarray<uint32_t,2>::iterator p = histogram.begin(); p != histogram.end(); p++) {
    *p = 0;
  }
}

AreaDetHist::~AreaDetHist ()
{
}

ndarray<uint32_t, 2> AreaDetHist::getHist()
{
  return histogram;
}

void AreaDetHist::_fillHistogram(ndarray<double,3> calib_data, ndarray<uint32_t, 2> histogram) {
  int pixelInd = 0;
  int val;
  // fill histogram
  for (ndarray<double,3>::iterator p = calib_data.begin();
       p != calib_data.end(); p++) {
    val = (int) round(*p);   
    if ( val >= _valid_min && val <= _valid_max ) { // in range
      histogram[ pixelInd ][ val-_valid_min+1 ] += 1;
    } else if ( val > _valid_max ) { // too large
      histogram[ pixelInd ][ _histLength-1 ] += 1;
    } else { // too small
      histogram[ pixelInd ][ 0 ] += 1;
    }
    pixelInd++;
  }
}

int isIsolated(double val, std::vector<double> *neighbors) {
	int result = 1;
	std::vector<double>::iterator p;
	for (p = neighbors->begin(); p != neighbors->end(); p++) {
		if (val < *p) {
			result = 0;
			break;
		}
	}
	return result;
}

void AreaDetHist::_insertHistElement(double x, int pixelInd, ndarray<uint32_t, 2> histogram) {
	int val = (int) round(x);   
	if ( val >= _valid_min && val <= _valid_max ) { // in range
		histogram[ pixelInd ][ val-_valid_min+1 ] += 1;
	} else if ( val > _valid_max ) { // too large
		histogram[ pixelInd ][ _histLength-1 ] += 1;
	} else { // too small
		histogram[ pixelInd ][ 0 ] += 1;
	}
}

ndarray<uint32_t, 2> AreaDetHist::update(ndarray<double,3> calib_data, int findIsolated=1, double minAduGap = 0.)
{
  if (findIsolated) {
	const unsigned int *arrayShape = calib_data.shape();
    
	const unsigned int Segs = arrayShape[0];
	const unsigned int Rows = arrayShape[1];
	const unsigned int Cols = arrayShape[2];
//std::cout << "shape:" << Segs << "," << Rows << "," << Cols << std::endl;
	// corner of detector (3 pixel neighborhood)
	double val = 0;
    int result = 0;
	int pixelInd = 0;
	unsigned int j = 0;
	unsigned int k = 0;

	int numNeighbors = 3;
    std::vector<double> neighbors(numNeighbors);
	for (unsigned int i = 0; i < Segs; i++) {
		// Upper left corner
		j = 0;
		k = 0;
		val = calib_data[i][j][k];
		neighbors[0] =	calib_data[i][j+1][k];
		neighbors[1] =	calib_data[i][j][k+1];
		neighbors[2] =	calib_data[i][j+1][k+1];		
		result = isIsolated(val-minAduGap, &neighbors);
//std::cout << "isIsolated: " << result << "," << val << "," << neighbors[0] << "," << neighbors[1] << "," << neighbors[2] << std::endl;
		if (result) {
			_insertHistElement(val, pixelInd, histogram);
		}
		pixelInd++;

		// Upper right corner
		k = Cols-1;
		val = calib_data[i][j][k];
		neighbors[0] = calib_data[i][j+1][k];
		neighbors[1] = calib_data[i][j][k-1];
		neighbors[2] = calib_data[i][j+1][k-1];
		result = isIsolated(val, &neighbors);
		if (result) {
			_insertHistElement(val-minAduGap, pixelInd, histogram);
		}
		pixelInd++;

		// Lower left corner
		j = Rows-1;
		k = 0;
		val = calib_data[i][j][k];
		neighbors[0] = calib_data[i][j][k+1];
		neighbors[1] = calib_data[i][j-1][k];
		neighbors[2] = calib_data[i][j-1][k+1];
		result = isIsolated(val, &neighbors);
		if (result) {
			_insertHistElement(val-minAduGap, pixelInd, histogram);
		}
		pixelInd++;

		// Lower right corner
		k = Cols-1;
		val = calib_data[i][j][k];
		neighbors[0] = calib_data[i][j][k-1];
		neighbors[1] = calib_data[i][j-1][k];
		neighbors[2] = calib_data[i][j-1][k-1];
		result = isIsolated(val, &neighbors);
		if (result) {
			_insertHistElement(val-minAduGap, pixelInd, histogram);
		}
		pixelInd++;
	}


	numNeighbors = 5;
    neighbors.resize(numNeighbors);
	// side of detector (5 pixel neighborhood)
	for (unsigned int i = 0; i < Segs; i++) {
		// Upper edge
		j = 0;
		for (unsigned int t = 1; t < Cols-1; t++) {
			val = calib_data[i][j][t];
			neighbors[0] = calib_data[i][j][t-1];
			neighbors[1] = calib_data[i][j][t+1];
			neighbors[2] = calib_data[i][j+1][t-1];
			neighbors[3] = calib_data[i][j+1][t];
			neighbors[4] = calib_data[i][j+1][t+1];
			result = isIsolated(val, &neighbors);
			if (result) {
				_insertHistElement(val-minAduGap, pixelInd, histogram);
			}
			pixelInd++;
		}

		// Lower edge
		j = Rows-1;
		for (unsigned int t = 1; t < Cols-1; t++) {
			val = calib_data[i][j][t];
			neighbors[0] = calib_data[i][j][t-1];
			neighbors[1] = calib_data[i][j][t+1];
			neighbors[2] = calib_data[i][j-1][t-1];
			neighbors[3] = calib_data[i][j-1][t];
			neighbors[4] = calib_data[i][j-1][t+1];
			result = isIsolated(val, &neighbors);
			if (result) {
				_insertHistElement(val-minAduGap, pixelInd, histogram);
			}
			pixelInd++;
		}

		// Left edge
		k = 0;
		for (unsigned int t = 1; t < Rows-1; t++) {
			val = calib_data[i][t][k];
			neighbors[0] = calib_data[i][t+1][k];
			neighbors[1] = calib_data[i][t-1][k];
			neighbors[2] = calib_data[i][t+1][k+1];
			neighbors[3] = calib_data[i][t][k+1];
			neighbors[4] = calib_data[i][t+1][k+1];
			result = isIsolated(val, &neighbors);
			if (result) {
				_insertHistElement(val-minAduGap, pixelInd, histogram);
			}
			pixelInd++;
		}

		// Right edge
		k = Cols-1;
		for (unsigned int t = 1; t < Rows-1; t++) {
			val = calib_data[i][t][k];
			neighbors[0] = calib_data[i][t+1][k];
			neighbors[1] = calib_data[i][t-1][k];
			neighbors[2] = calib_data[i][t+1][k-1];
			neighbors[3] = calib_data[i][t][k-1];
			neighbors[4] = calib_data[i][t+1][k-1];
			result = isIsolated(val, &neighbors);
			if (result) {
				_insertHistElement(val-minAduGap, pixelInd, histogram);
			}
			pixelInd++;
		}
	}

	// non-edge of detector (8 pixel neighborhood)
	numNeighbors = 8;
    neighbors.resize(numNeighbors);
	for (unsigned int i = 0; i < Segs; i++) {
	for (unsigned int j = 1; j < Rows-1; j++) {
	for (unsigned int k = 1; k < Cols-1; k++) {
		val = calib_data[i][j][k];
		neighbors[0] = calib_data[i][j-1][k-1];
		neighbors[1] = calib_data[i][j-1][k];
		neighbors[2] = calib_data[i][j-1][k+1];
		neighbors[3] = calib_data[i][j][k-1];
		neighbors[4] = calib_data[i][j][k+1];
		neighbors[5] = calib_data[i][j+1][k-1];
		neighbors[6] = calib_data[i][j+1][k];
		neighbors[7] = calib_data[i][j+1][k+1];
		result = isIsolated(val, &neighbors);
		if (result) {
			_insertHistElement(val-minAduGap, pixelInd, histogram);
		}
		pixelInd++;
	}
	}
	}
  } else {
	_fillHistogram(calib_data, histogram);
  }

  return histogram;
}


