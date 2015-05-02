//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class GlobalMethods...
//
// Author:
//      Matt Weaver
//
//------------------------------------------------------------------------

#include "ImgAlgos/GlobalHitFinder.h"

using namespace std;
using namespace ImgAlgos;

namespace ImgAlgos {

void count_hits(const ndarray<const unsigned,2>& input,
                unsigned threshold,
                ndarray<unsigned,2>& output)
{
  for(unsigned j=1; j<input.shape()[0]-1; j++)
    for(unsigned k=1; k<input.shape()[1]-1; k++) {
      unsigned v = input[j][k];
      if (v > threshold &&
          v > input[j-1][k-1] &&
          v > input[j-1][k] &&
          v > input[j-1][k+1] &&
          v > input[j][k-1] &&
          v > input[j][k+1] &&
          v > input[j+1][k-1] &&
          v > input[j+1][k] &&
          v > input[j+1][k+1])
        output[j][k]++;
    }
}

void sum_hits(const ndarray<const unsigned,2>& input,
              unsigned threshold,
              unsigned offset,
              ndarray<unsigned,2>& output)
{
  for(unsigned j=1; j<input.shape()[0]-1; j++)
    for(unsigned k=1; k<input.shape()[1]-1; k++) {
      unsigned v = input[j][k];
      if (v > threshold &&
          v > input[j-1][k-1] &&
          v > input[j-1][k] &&
          v > input[j-1][k+1] &&
          v > input[j][k-1] &&
          v > input[j][k+1] &&
          v > input[j+1][k-1] &&
          v > input[j+1][k] &&
          v > input[j+1][k+1])
        output[j][k] += v-offset;
    }
}

void count_excess(const ndarray<const unsigned,2>& input,
                  unsigned threshold,
                  ndarray<unsigned,2>& output)
{
  for(unsigned j=0; j<input.shape()[0]; j++)
    for(unsigned k=0; k<input.shape()[1]; k++) {
      unsigned v = input[j][k];
      if (v > threshold)
        output[j][k]++;
    }
}

void sum_excess(const ndarray<const unsigned,2>& input,
                unsigned threshold,
                unsigned offset,
                ndarray<unsigned,2>& output)
{
  for(unsigned j=0; j<input.shape()[0]; j++)
    for(unsigned k=0; k<input.shape()[1]; k++) {
      unsigned v = input[j][k];
      if (v > threshold)
        output[j][k] += v-offset;
    }
}

void count_hits(const ndarray<const unsigned,2>& input,
                const ndarray<const unsigned,2>& threshold,
                ndarray<unsigned,2>& output)
{
  for(unsigned j=1; j<input.shape()[0]-1; j++)
    for(unsigned k=1; k<input.shape()[1]-1; k++) {
      unsigned v = input[j][k];
      if (v > threshold[j][k] &&
          v > input[j-1][k-1] &&
          v > input[j-1][k] &&
          v > input[j-1][k+1] &&
          v > input[j][k-1] &&
          v > input[j][k+1] &&
          v > input[j+1][k-1] &&
          v > input[j+1][k] &&
          v > input[j+1][k+1])
        output[j][k]++;
    }
}

void sum_hits(const ndarray<const unsigned,2>& input,
              const ndarray<const unsigned,2>& threshold,
              unsigned offset,
              ndarray<unsigned,2>& output)
{
  for(unsigned j=1; j<input.shape()[0]-1; j++)
    for(unsigned k=1; k<input.shape()[1]-1; k++) {
      unsigned v = input[j][k];
      if (v > threshold[j][k] &&
          v > input[j-1][k-1] &&
          v > input[j-1][k] &&
          v > input[j-1][k+1] &&
          v > input[j][k-1] &&
          v > input[j][k+1] &&
          v > input[j+1][k-1] &&
          v > input[j+1][k] &&
          v > input[j+1][k+1])
        output[j][k] += v-offset;
    }
}

void count_excess(const ndarray<const unsigned,2>& input,
                  const ndarray<const unsigned,2>& threshold,
                  ndarray<unsigned,2>& output)
{
  for(unsigned j=0; j<input.shape()[0]; j++)
    for(unsigned k=0; k<input.shape()[1]; k++) {
      unsigned v = input[j][k];
      if (v > threshold[j][k])
        output[j][k]++;
    }
}

void sum_excess(const ndarray<const unsigned,2>& input,
                const ndarray<const unsigned,2>& threshold,
                unsigned offset,
                ndarray<unsigned,2>& output)
{
  for(unsigned j=0; j<input.shape()[0]; j++)
    for(unsigned k=0; k<input.shape()[1]; k++) {
      unsigned v = input[j][k];
      if (v > threshold[j][k])
        output[j][k] += v-offset;
    }
}

//--------------------
//--------------------
} // namespace ImgAlgos
