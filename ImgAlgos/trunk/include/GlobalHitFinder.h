#ifndef IMGALGOS_GLOBALHITFINDER_H
#define IMGALGOS_GLOBALHITFINDER_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class ImgParametersV1.
//
//      For 2d image parameters like pedestals, background, gain factor, and mask.
//
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
//#include <vector>
//#include <fstream>   // ofstream
//#include <iomanip>   // for setw, setfill
//#include <sstream>   // for stringstream
//#include <iostream>

#include "ndarray/ndarray.h"

namespace ImgAlgos {

using namespace std;


/**
 *  GlobalHitFinder contains global methods for hit finding algorithms.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id: GlobalHitFinder.h 2013-10-06$
 *
 *  @author Matt Weaver
 */

//--------------------

void count_hits(const ndarray<const unsigned,2>& input,
                unsigned threshold,
                ndarray<unsigned,2>& output);

void sum_hits(const ndarray<const unsigned,2>& input,
              unsigned threshold,
              unsigned offset,
              ndarray<unsigned,2>& output);

void count_excess(const ndarray<const unsigned,2>& input,
                  unsigned threshold,
                  ndarray<unsigned,2>& output);

void sum_excess(const ndarray<const unsigned,2>& input,
                unsigned threshold,
                unsigned offset,
                ndarray<unsigned,2>& output);

void count_hits(const ndarray<const unsigned,2>& input,
                const ndarray<const unsigned,2>& threshold,
                ndarray<unsigned,2>& output);

void sum_hits(const ndarray<const unsigned,2>& input,
              const ndarray<const unsigned,2>& threshold,
              unsigned offset,
              ndarray<unsigned,2>& output);

void count_excess(const ndarray<const unsigned,2>& input,
                  const ndarray<const unsigned,2>& threshold,
                  ndarray<unsigned,2>& output);

void sum_excess(const ndarray<const unsigned,2>& input,
                const ndarray<const unsigned,2>& threshold,
                unsigned offset,
                ndarray<unsigned,2>& output);

//--------------------
//--------------------
//--------------------

} // namespace ImgAlgos

#endif // IMGALGOS_GLOBALHITFINDER_H
