#ifndef PSHIST_HMANAGER_H
#define PSHIST_HMANAGER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class HManager.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <string>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSHist/Axis.h"
#include "PSHist/H1.h"
#include "PSHist/H2.h"
#include "PSHist/Profile.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHist {

/**
 *  @defgroup PSHist PSHist package
 *  
 *  @brief Package defining interfaces for histogramming services.
 *  
 *  This package contains interfaces (abstract classes) for 
 *  histogramming services used by psana framework. 
 */
  
  
/**
 *  @ingroup PSHist
 *  
 *  @brief Interface for histogram/tuple manager class.
 *
 *  HManager is an empty base class which holds information about ntuples/histograms. 
 *  The main reason is to be able to create and hold new histograms or/and ntuples
 *  without knowing without knowing what the underlying system is. For example,
 *  it might be root, hbook, hyppo etc.
 *  
 *  Usage:
 *  @code
 *  #include "RootHist/RootHManager.h"
 *  #include "PSHist/HManager.h"
 *  #include "PSHist/Axis.h"
 *  #include "PSHist/H1.h"
 *  #include "PSHist/H2.h"
 *  #include "PSHist/Profile.h"
 *  @endcode
 *  
 *  1. Create a HManager with specific constructor (root for example):
 *
 *  @code
 *       PSHist::HManager *hMan = new RootHist::RootHManager("my-output-file.root", "RECREATE");
 *  @endcode
 *
 *  2. Create histograms
 *
 *  @code
 *       PSHist::H1 *pHis1f = hMan->hist1f("His1 float  title",100,0.,1.);
 *       PSHist::H2 *pHis2d = hMan->hist2d("His2 double title",100,0.,1.,100,0.,1.);
 *  @endcode
 *
 *
 *  3. Fill histograms
 *
 *  @code
 *       pHis1f->fill(x,[weight]);        // once per event
 *       pHis2d->fill(x,y,[weight]);
 *  @endcode
 *
 *
 *  4. Write the data into a file:
 *
 *  @code
 *       hMan->write();                   // at the end of job
 *       delete hMan;
 *  @endcode
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see H1
 *  @see H2
 *  @see Profile
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class HManager : boost::noncopyable {
public:

  // Destructor
  virtual ~HManager () {}

  // 1-d histograms

  /**
   *  @brief Create new 1-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional histogram with same-width bins. 
   *  Internal storage of the created histogram will consists of 32-bit 
   *  integer per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] nbins  Number of bins.
   *  @param[in] xlow   Low edge of the first bin.
   *  @param[in] xhigh  High edge of the last bin.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H1* hist1i(const std::string& name, const std::string& title, 
      int nbins, double xlow, double xhigh) = 0;

  /**
   *  @brief Create new 1-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional histogram with same-width bins. 
   *  Internal storage of the created histogram will consists of 32-bit 
   *  floating number per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] nbins  Number of bins.
   *  @param[in] xlow   Low edge of the first bin.
   *  @param[in] xhigh  High edge of the last bin.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H1* hist1f(const std::string& name, const std::string& title, 
      int nbins, double xlow, double xhigh) = 0;

  /**
   *  @brief Create new 1-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional histogram with same-width bins. 
   *  Internal storage of the created histogram will consists of 64-bit 
   *  floating number per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] nbins  Number of bins.
   *  @param[in] xlow   Low edge of the first bin.
   *  @param[in] xhigh  High edge of the last bin.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H1* hist1d(const std::string& name, const std::string& title, 
      int nbins, double xlow, double xhigh) = 0;

  /**
   *  @brief Create new 1-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional histogram with variable-width bins. 
   *  Internal storage of the created histogram will consists of 32-bit 
   *  integer per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] nbins  Number of bins.
   *  @param[in] xbinedges Array of the histogram edges, size of the array 
   *                    is @c nbins+1, it should contain ordered values for
   *                    low edges of all bins plus high edge of last bin. 
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H1* hist1i(const std::string& name, const std::string& title, 
      int nbins, const double *xbinedges) = 0;
  
  /**
   *  @brief Create new 1-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional histogram with variable-width bins. 
   *  Internal storage of the created histogram will consists of 32-bit 
   *  floating number per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] nbins  Number of bins.
   *  @param[in] xbinedges Array of the histogram edges, size of the array 
   *                    is @c nbins+1, it should contain ordered values for
   *                    low edges of all bins plus high edge of last bin. 
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H1* hist1f(const std::string& name, const std::string& title, 
      int nbins, const double *xbinedges) = 0;
  
  /**
   *  @brief Create new 1-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional histogram with variable-width bins. 
   *  Internal storage of the created histogram will consists of 64-bit 
   *  floating number per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] nbins  Number of bins.
   *  @param[in] xbinedges Array of the histogram edges, size of the array 
   *                    is @c nbins+1, it should contain ordered values for
   *                    low edges of all bins plus high edge of last bin. 
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H1* hist1d(const std::string& name, const std::string& title, 
      int nbins, const double *xbinedges) = 0;

  /**
   *  @brief Create new 1-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional histogram, number of bins and 
   *  their edges are determined by separate object (Axis class). 
   *  Internal storage of the created histogram will consists of 32-bit 
   *  integer per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] axis   Axis definition.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H1* hist1i(const std::string& name, const std::string& title, 
      const PSHist::Axis& axis) = 0;
  
  /**
   *  @brief Create new 1-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional histogram, number of bins and 
   *  their edges are determined by separate object (Axis class). 
   *  Internal storage of the created histogram will consists of 32-bit 
   *  floating number per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] axis   Axis definition.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H1* hist1f(const std::string& name, const std::string& title, 
      const PSHist::Axis& axis) = 0;
  
  /**
   *  @brief Create new 1-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional histogram, number of bins and 
   *  their edges are determined by separate object (Axis class). 
   *  Internal storage of the created histogram will consists of 64-bit 
   *  floating number per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] axis   Axis definition.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H1* hist1d(const std::string& name, const std::string& title, 
      const PSHist::Axis& axis) = 0;

  // 2-d histograms

  /**
   *  @brief Create new 2-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 2-dimensional histogram, number of bins and 
   *  their edges are determined by separate objects (Axis class). 
   *  Internal storage of the created histogram will consists of 32-bit 
   *  integer per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] xaxis  X axis definition.
   *  @param[in] yaxis  Y axis definition.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H2* hist2i(const std::string& name, const std::string& title, 
      const PSHist::Axis& xaxis, const PSHist::Axis& yaxis ) = 0;

  /**
   *  @brief Create new 2-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 2-dimensional histogram, number of bins and 
   *  their edges are determined by separate objects (Axis class). 
   *  Internal storage of the created histogram will consists of 32-bit 
   *  floating number per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] xaxis  X axis definition.
   *  @param[in] yaxis  Y axis definition.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H2* hist2f(const std::string& name, const std::string& title, 
      const PSHist::Axis& xaxis, const PSHist::Axis& yaxis ) = 0;

  /**
   *  @brief Create new 2-dimensional histogram and return pointer to it.
   *  
   *  This method creates new 2-dimensional histogram, number of bins and 
   *  their edges are determined by separate objects (Axis class). 
   *  Internal storage of the created histogram will consists of 64-bit 
   *  floating number per histogram bin. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] xaxis  X axis definition.
   *  @param[in] yaxis  Y axis definition.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual H2* hist2d(const std::string& name, const std::string& title, 
      const PSHist::Axis& xaxis, const PSHist::Axis& yaxis ) = 0;

  // 1-d profile histograms

  /**
   *  @brief Create new 1-dimensional profile histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional profile histogram with same-width bins. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  Option string determines what value is returned for the bin error,  
   *  possible values are "" (default) for error-of-mean and "s" 
   *  for standard deviation.
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] nbinsx Number of bins.
   *  @param[in] xlow   Low edge of the first bin.
   *  @param[in] xhigh  High edge of the last bin.
   *  @param[in] option Option string.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual Profile* prof1(const std::string& name, const std::string& title, 
      int nbinsx, double xlow, double xhigh, const std::string& option="") = 0;
  
  /**
   *  @brief Create new 1-dimensional profile histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional profile histogram with variable-width bins. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  Option string determines what value is returned for the bin error,  
   *  possible values are "" (default) for error-of-mean and "s" 
   *  for standard deviation.
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] nbins  Number of bins.
   *  @param[in] xbinedges Array of the histogram edges, size of the array 
   *                    is @c nbins+1, it should contain ordered values for
   *                    low edges of all bins plus high edge of last bin. 
   *  @param[in] option Option string.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual Profile* prof1(const std::string& name, const std::string& title, 
      int nbins, const double *xbinedges, const std::string& option="") = 0;
  
  /**
   *  @brief Create new 1-dimensional profile histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional profile histogram, number of bins and 
   *  their edges are determined by separate object (Axis class). 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  Option string determines what value is returned for the bin error,  
   *  possible values are "" (default) for error-of-mean and "s" 
   *  for standard deviation.
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] axis   X axis definition.
   *  @param[in] option Option string.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual Profile* prof1(const std::string& name, const std::string& title, 
      const PSHist::Axis& axis, const std::string& option="") = 0;

  /**
   *  @brief Create new 1-dimensional profile histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional profile histogram with same-width bins. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  Option string determines what value is returned for the bin error,  
   *  possible values are "" (default) for error-of-mean and "s" 
   *  for standard deviation. 
   *  Values of y ouside of range (ylow-yhigh) will be ignored.
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] nbinsx Number of bins.
   *  @param[in] xlow   Low edge of the first bin.
   *  @param[in] xhigh  High edge of the last bin.
   *  @param[in] ylow   Lowest possible value for Y values.
   *  @param[in] yhigh  Highest possible value for Y values.
   *  @param[in] option Option string.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual Profile* prof1(const std::string& name, const std::string& title, 
      int nbinsx, double xlow, double xhigh, double ylow, double yhigh, 
      const std::string& option="") = 0;
  
  /**
   *  @brief Create new 1-dimensional profile histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional profile histogram with variable-width bins. 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  Option string determines what value is returned for the bin error,  
   *  possible values are "" (default) for error-of-mean and "s" 
   *  for standard deviation.
   *  Values of y ouside of range (ylow-yhigh) will be ignored.
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] nbins  Number of bins.
   *  @param[in] xbinedges Array of the histogram edges, size of the array 
   *                    is @c nbins+1, it should contain ordered values for
   *                    low edges of all bins plus high edge of last bin. 
   *  @param[in] ylow   Lowest possible value for Y values.
   *  @param[in] yhigh  Highest possible value for Y values.
   *  @param[in] option Option string.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual Profile* prof1(const std::string& name, const std::string& title, 
      int nbins, const double *xbinedges, double ylow, double yhigh, 
      const std::string& option="") = 0;
  
  /**
   *  @brief Create new 1-dimensional profile histogram and return pointer to it.
   *  
   *  This method creates new 1-dimensional profile histogram, number of bins and 
   *  their edges are determined by separate object (Axis class). 
   *  The name of the histogram must be unique, otherwise exception is thrown. 
   *  Option string determines what value is returned for the bin error,  
   *  possible values are "" (default) for error-of-mean and "s" 
   *  for standard deviation.
   *  Values of y ouside of range (ylow-yhigh) will be ignored.
   *  
   *  <b>Returned pointer should never be deleted by client code.</b>
   *  
   *  @param[in] name   Histogram name, unique string.
   *  @param[in] title  Title of the histogram, arbitrary string.
   *  @param[in] axis   X axis definition.
   *  @param[in] ylow   Lowest possible value for Y values.
   *  @param[in] yhigh  Highest possible value for Y values.
   *  @param[in] option Option string.
   *  @return Pointer to a newly created histogram, do not delete.
   *  
   *  @throw ExceptionDuplicateName thrown if histogram or tuple with 
   *                                identical name exists already
   */
  virtual Profile* prof1(const std::string& name, const std::string& title, 
      const PSHist::Axis& axis, double ylow, double yhigh, 
      const std::string& option="") = 0;

  /**
   *  @brief Store all booked histograms and tuples to a permanent storage.
   *  
   *  This method should be called once before deleting manager object.
   *  
   *  @throw ExceptionStore thrown if write operation fails.
   */
  virtual void write() = 0;

};

} // namespace PSHist

#endif // PSHIST_HMANAGER_H
