#ifndef PSHDF5INPUT_MULTIMERGE_H
#define PSHDF5INPUT_MULTIMERGE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MultiMerge.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>
#include <iterator>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSHdf5Input {

/**
 *  @brief Implementation of the merge function for a number of iterators.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename Iter>
class MultiMerge  {
public:

  typedef typename std::iterator_traits<Iter>::value_type value_type;
  
  // Default constructor
  MultiMerge () {}

  // reserve space for N iterators
  void reserve(size_t n);
  
  // add one more iterator
  void add(Iter begin, Iter end);
  
  // Return the next combined value, return value is false if no more iterations left.
  template <typename OutIter>
  bool next(OutIter result);
  
  // Returns number of iterators to be merged
  size_t size() const { return m_iters.size(); }
  
protected:

  // advance specified iterator, update m_data
  void advance(size_t i);
  
private:

  std::vector< std::pair<Iter,Iter> > m_iters;   ///< Set of begin/end iterators
  std::vector< std::pair<value_type,bool> > m_data;  ///< List of current data objects from each iterator

};

// reserve space for N iterators
template <typename Iter>
void
MultiMerge<Iter>::reserve(size_t n)
{
  m_iters.reserve(n);
  m_data.reserve(n);
}

// add one more iterator
template <typename Iter>
void 
MultiMerge<Iter>::add(Iter begin, Iter end)
{
  m_iters.push_back(std::pair<Iter,Iter>(begin, end));
  m_data.push_back(std::pair<value_type,bool>(value_type(), false));
  advance(m_iters.size()-1);
}

// advance specified iterator, update m_data
template <typename Iter>
void 
MultiMerge<Iter>::advance(size_t i)
{
  std::pair<Iter,Iter>& iters = m_iters[i];
  if (iters.first != iters.second) {
    m_data[i].first = *iters.first;
    m_data[i].second = true;
    ++ iters.first;
  } else {
    m_data[i].first = value_type();
    m_data[i].second = false;
  }
}

// Return the next combined value, return value is false if no more iterations left.
template <typename Iter>
template <typename OutIter>
bool 
MultiMerge<Iter>::next(OutIter result)
{
  unsigned niter = m_data.size();
  
  // find data with the minimum "value"
  int minidx = -1;
  for (unsigned i = 0; i < niter; ++ i) {
    if (m_data[i].second and (minidx < 0 or m_data[i].first < m_data[minidx].first)) {
      minidx = i;
    }
  }

  // if minidx is still negative means all iterators are done
  if (minidx < 0) return false;

  // select those data items which compare exactly
  for (unsigned i = 0; i < niter; ++ i) {
    if (m_data[i].second and not (m_data[minidx].first < m_data[i].first)) {
      // copy the data to output sink
      *result = m_data[i].first;
      ++ result;
      // move corresponding iterator forward, but do not advance the one we 
      // use for comparison, we'll advance it later
      if (i != unsigned(minidx)) advance(i);
    }
  }
  // it is now safe to advance it
  advance(minidx);
  
  return true;
}

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_MULTIMERGE_H
