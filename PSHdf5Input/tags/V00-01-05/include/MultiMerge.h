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
#include <algorithm>
#include <utility>

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
 *  @ingroup PSHdf5Input
 *
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

  // enum values selected so that std::pair<Status, value_type>(Present, val1)
  // is always less than std::pair<Status, value_type>(Missing, val2) for any val1 and val2.
  enum Status { Present = 0, Missing = 1 };

  std::vector< std::pair<Iter,Iter> > m_iters;   ///< Set of begin/end iterators
  std::vector<std::pair<Status, value_type> > m_data;  ///< List of current data objects from each iterator

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
  m_iters.push_back(std::pair<Iter, Iter>(begin, end));
  m_data.push_back(std::pair<Status, value_type>(Missing, value_type()));
  advance(m_iters.size()-1);
}

// advance specified iterator, update m_data
template <typename Iter>
void 
MultiMerge<Iter>::advance(size_t i)
{
  std::pair<Iter,Iter>& iters = m_iters[i];
  if (iters.first != iters.second) {
    m_data[i].first = Present;
    m_data[i].second = *iters.first;
    ++ iters.first;
  } else {
    m_data[i].first = Missing;
    m_data[i].second = value_type();
  }
}

// Return the next combined value, return value is false if no more iterations left.
template <typename Iter>
template <typename OutIter>
bool 
MultiMerge<Iter>::next(OutIter result)
{
  // find data with the minimum value
  typename std::vector<std::pair<Status, value_type> >::const_iterator minIter = 
          std::min_element(m_data.begin(), m_data.end());

  // if minidx is still negative means all iterators are done
  if (minIter == m_data.end() or minIter->first == Missing) return false;

  // minimum value
  value_type minval = minIter->second;

  // select those data items which compare exactly
  unsigned niter = m_data.size();
  for (unsigned i = 0; i < niter; ++ i) {
    // there may be multiple items in one stream, get them all
    using namespace std::rel_ops;
    while (m_data[i].first == Present and m_data[i].second <= minval) {
      // copy the data to output sink
      *result = m_data[i].second;
      ++ result;
      // move corresponding iterator forward
      advance(i);
    }
  }
  
  return true;
}

} // namespace PSHdf5Input

#endif // PSHDF5INPUT_MULTIMERGE_H
