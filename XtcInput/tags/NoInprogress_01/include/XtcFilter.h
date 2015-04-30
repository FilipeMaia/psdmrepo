#ifndef XTCINPUT_XTCFILTER_H
#define XTCINPUT_XTCFILTER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFilter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Dgram.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Class that filters content of the XTC containers
 *
 *  This is a class template, template argument is a functor
 *  which takes XTC object as input and returns true or false
 *  to either keep or discard given XTC.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename Filter>
class XtcFilter  {
public:

  /**
   *  Constructor takes instance of the Filter functor. Note that
   *  copy of this instance will be made.
   *
   *  @param[in] filter  Instance of a functor which makes filtering decisions
   *  @param[in] keepEmptyCont If false (default) then discard container objects that are
   *      empty after filtering.
   *  @param[in] keepEmptyDgram If false (default) then discard datagrams that are
   *      empty after filtering.
   *  @param[in] keepAny If false (default) then all TypeId::Any objects are discarded,
   *      otherwise they are passed to filter functor.
   */
  XtcFilter(const Filter& filter, bool keepEmptyCont = false, bool keepEmptyDgram = false, bool keepAny = false)
    : m_filter(filter)
    , m_keepEmptyCont(keepEmptyCont)
    , m_keepEmptyDgram(keepEmptyDgram)
    , m_keepAny(keepAny)
  {}

  /**
   *  @brief Filter method for XTCs
   *
   *  This method does actual filtering job. Note also that for some types
   *  of damage it may need to skip damaged data if the structure  of XTC
   *  cannot be recovered. This happens independently of content-based
   *  filtering. The size of the output buffer must be big enough to fit
   *  the data,  output data cannot be larger than input XTC.
   *
   *  @param[in] input   XTC container object
   *  @param[out] output Buffer for output data
   *  @return  Number of bytes in the output buffer
   *
   */
  size_t filter(const Pds::Xtc* input, char* output);

  /**
   *  @brief Filter method for whole datagrams
   *
   *  This method does actual filtering job. Note also that for some types
   *  of damage it may need to skip damaged data if the structure  of XTC
   *  cannot be recovered. This happens independently of content-based
   *  filtering. The size of the output buffer must be big enough to fit
   *  the data,  output data cannot be larger than input datagram.
   *
   *  @param[in] input   Datagram object
   *  @param[out] output Buffer for output data
   *  @return  Number of bytes in the output buffer
   *
   */
  size_t filter(const Pds::Dgram* input, char* output);

protected:

private:

  Filter m_filter;
  bool m_keepEmptyCont;
  bool m_keepEmptyDgram;
  bool m_keepAny;

};


template <typename Filter>
size_t
XtcFilter<Filter>::filter(const Pds::Xtc* input, char* output)
{
  if (input->contains.id() == Pds::TypeId::Any and not m_keepAny) {
    return 0;
  }

  // for some type of damage it's dangerous to look inside
  // this code follows what is found in pdsdata/xtc/XtcIterator.cc
  if ( input->damage.value() & (1 << Pds::Damage::IncompleteContribution) ) {
    return 0;
  }

  if (input->contains.id() == Pds::TypeId::Id_Xtc) {

    // copy the header, it will be discarded later if not needed
    char* const hdrPos = output;
    const size_t hdrSize = sizeof(Pds::Xtc);
    const char* src = (const char*)input;
    std::copy(src, src+hdrSize, output);
    output += hdrSize;

    // loop over all sub-objects
    size_t copied = 0;
    if (input->sizeofPayload()) {

      const Pds::Xtc* payload = (const Pds::Xtc*)input->payload();
      size_t psize = 0;
      while (true) {

        // recursively copy it
        size_t n = this->filter(payload, output);
        copied += n;
        output += n;

        // advance
        psize += payload->extent;
        if (psize >= unsigned(input->sizeofPayload())) break;
        payload = payload->next();
      }
    }

    if (copied > 0 or m_keepEmptyCont) {
      // some data has been copied, update XTC extent size
      ((Pds::Xtc*)(hdrPos))->extent = hdrSize + copied;
      return hdrSize + copied;
    }

  } else {

    // call filter for decision
    if (m_filter(input)) {
      // copy the data
      size_t size = input->extent;
      const char* src = (const char*)input;
      std::copy(src, src+size, output);
      return size;
    }

  }

  // nothing has been copied
  return 0;
}

template <typename Filter>
size_t
XtcFilter<Filter>::filter(const Pds::Dgram* input, char* output)
{
  // copy datagram header
  const size_t hdrSize = sizeof(Pds::Dgram) - sizeof(Pds::Xtc);
  const char* src = (const char*)input;
  std::copy(src, src+hdrSize, output);
  output += hdrSize;

  // call XTC copy method
  size_t copied = this->filter(&input->xtc, output);

  if (copied > 0) {

    return hdrSize + copied;

  } else if (m_keepEmptyDgram) {

    // copy XTC header too but set its payload size to 0
    const char* src = ((const char*)input) + hdrSize;
    std::copy(src, src+sizeof(Pds::Xtc), output);
    ((Pds::Xtc*)(output))->extent = sizeof(Pds::Xtc);

    return sizeof(Pds::Dgram);

  } else {

    return 0;

  }
}

} // namespace XtcInput

#endif // XTCINPUT_XTCFILTER_H
