#ifndef TRANSLATOR_HDFWRITERGENERIC_H
#define TRANSLATOR_HDFWRITERGENERIC_H

#include <map>
#include <vector>
#include <string>

#include "hdf5/hdf5.h"

#include "ErrSvc/Issue.h"

#include "Translator/DataSetMeta.h"
#include "Translator/DataSetCreationProperties.h"

namespace Translator {


class HdfWriterGeneric {

 public:
  /**
   *  @ingroup HdfWriterGeneric
   *  
   *  @brief Class that writes data to hdf5 datasets. Data description is defined outside the class.
   * 
   * Manages a hdf5 datasets that looks like a 1D array of whatever data types the client code provides.
   * The client code provides the following:
   *    hdf5 group Id's for where the datasets are to be written
   *    One or more sets of the following, for each dataset to be managed in a given group id:
   *      a hdf5 type Id for the type in the dataset  
   *      name of the dataset
   *      a DatasetCreationProperties object (which includes a boost smart pointer to a ChunkPolicy instance
   *
   * The client code owns the groupId's and typeId's.  HdfWriterGeneric will not call H5close on these
   * resources.  It will remember the groupId and typeId's, so behavior is unpredictable if the 
   * resources are closed before HdfWriterGeneric is finished using them.
   *
   * The client code references a dataset via the original groupId used to create it, and the
   * dataset name, or a dataset index.  HdfWriterGeneric will make any number of distinct datasets
   * in the given group, and they are referenced by a one up counter.  For example:
   *
   *  hid_t file = H5Fcreate("myfile.h5", ... )
   *  hid_t group = H5Gcreate(file,"mygroup",...)
   *  hid_t typeA = H5Tcreate( ... )
   *  hid_t typeB = H5Tcreate( ... )
   *  DataSetCreationProperties dsetProp = ...
   *  HdfWriterGeneric writer;
   *  size_t APos = writer.createUnlimitedSizeDataset(group,"datasetA",typeA,dsetProp);  // APos == 0
   *  size_t BPos = writer.createUnlimitedSizeDataset(group,"datasetB",typeB,dsetProp);  // BPos == 1
   *  writer.append(group,APos,AdataBuffer)
   *  writer.append(group,BPos,BdataBuffer)
   *  writer.append(group,"A",AdataBuffer)  // slower, but can look up dataset by name
   *  writer.append(group,"B",BdataBuffer)
   *  writer.store_at(group,0,Apos,AdataBuffer)  // you can overwrite previous records
   *  writer.store_at(group,1,Apos,AdataBuffer)  
   *  writer.closeDatasets(group)  // closes both the hdf5 datasets for "datasetA" and "datasetB"
   *
   *  The client code must close group, typeA and typeB.
   *
   *  The file structure will now be:
   *   \file\mygroup\datasetA
   *   \file\mygroup\datasetB
   * 
   * append is the most unsafe function to use.  The client code is responsible for providing a 
   * void pointer that contains valid data for one entry of the type for the dataset.  
   *
   * Exceptions: several exceptions are thrown, all derived from ErrSvc::Issue.  Most involve
   *   errors with hdf5 library calls and are serious. The exceptions are:
   *
   *  DataSpaceException - serious error closing the hdf5 dataspace id's that 
   *                       describe the in memory or file layout.
   * 
   *  PropertyListException - serious error making property list, with calls to H5Pset_chunk, shuffle, etc
   *
   *  DataSetException - serious error creating os closing dataset
   *
   *  WriteException - error with H5DF write calls, or duplicate dataset names given for the group.
   *
   *  GroupMapException - not neccessarily serious, the group could not be found in the internal
   *                      map. No dataset was created for this group.  Other cases involve a bad 
   *                      dataset index.
   *                      
   *  @version \$Id:
   *
   *  @author David Schneider
   */

  HdfWriterGeneric();
  ~HdfWriterGeneric();

  size_t createUnlimitedSizeDataset(hid_t groupId,
                                    const std::string & dsetName,
                                    hid_t h5type,
                                    const DataSetCreationProperties & dsetCreateProp);
  size_t createFixedSizeDataset(hid_t groupId,
                                const std::string & dsetName,
                                hid_t h5type,
                                hsize_t fixedSize);
  
  void append(hid_t groupId, size_t idx, const void * data) { store_at(groupId,-1,idx,data); };
  void append(hid_t groupId, const std::string & dsetName, const void * data);

  void store_at(hid_t groupId, long storeIndex, size_t dsetIndex, const void * data);
  void store_at(hid_t groupId, long storeIndex, const std::string & dsetName, const void * data);

  void closeDatasets(hid_t groupId);

 private:
  size_t getDatasetIndex(hid_t groupId, std::string dsetName);
  static const int m_rankOne = 1;

  // maps group Id's to a list of datasets created for that group.
  std::map<hid_t, std::vector<DataSetMeta> > m_datasetMap;
  hid_t m_singleTransferDataSpaceIdMemory;
  hid_t m_unlimitedDataSpaceIdForFile;

 public:
  class PropertyListException : public ErrSvc::Issue {
  public: PropertyListException(const ErrSvc::Context &ctx, const std::string &what) : ErrSvc::Issue(ctx, what) {}
 };

 class DataSpaceException : public ErrSvc::Issue {
 public: DataSpaceException(const ErrSvc::Context &ctx, const std::string &what) : ErrSvc::Issue(ctx,what) {}
 };

 class DataSetException : public ErrSvc::Issue {
 public: DataSetException (const ErrSvc::Context &ctx, const std::string &what) : ErrSvc::Issue(ctx,what) {}
 };
 
 class GroupMapException : public ErrSvc::Issue {
 public: GroupMapException(const ErrSvc::Context &ctx, const std::string &what) : ErrSvc::Issue(ctx,what) {}
 };

 class WriteException : public ErrSvc::Issue {
 public: WriteException(const ErrSvc::Context &ctx, const std::string &what) : ErrSvc::Issue(ctx,what) {}
 };
 
};


} // namespace

#endif 
