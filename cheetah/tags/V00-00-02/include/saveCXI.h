#pragma once

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "hdf5/hdf5.h"
#include <stdlib.h>
#include <string.h>
#include <typeinfo>
#include <vector>


#include "cheetah/detectorObject.h"
#include "cheetah/cheetahGlobal.h"
#include "cheetah/cheetahEvent.h"
#include "cheetah/cheetahmodules.h"
#include "cheetah/median.h"


namespace CXI{
	const char* ATTR_NAME_NUM_EVENTS = "numEvents";

	const int IgnoreOverflow = 1;
	const int IgnoreTruncate = 2;
	const int IgnorePrecision = 4;
	const int IgnoreNAN = 8;
	class Node{
	public:
		enum Type{Dataset, Group, Link};
		Node(const char * filename, bool swmr, int _ignore_flags){
			ignoreConversionExceptions = _ignore_flags;
			parent = NULL;
			type = Group;
			name = std::string("/");
			hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
			if(fapl_id < 0 || H5Pset_fclose_degree(fapl_id, H5F_CLOSE_STRONG) < 0){
				ERROR("Cannot set file access properties.\n");
			}
			if(swmr){
#ifdef H5F_ACC_SWMR_WRITE
				if(H5Pset_libver_bounds(fapl_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST) < 0){
					ERROR("Cannot set file access properties.\n");
				}
#else
				ERROR("Cannot write in SWMR mode, HDF5 library does not support it.\n");
#endif
			}

			id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
			if( id<0 ) {ERROR("Cannot create file.\n");}
			stackCounter = 0;
		}
		Node(std::string s, hid_t oid, Node * p, Type t,  int _ignore_flags){
			name = s;
			parent = p;
			id = oid;
			type = t;
			ignoreConversionExceptions = _ignore_flags;
		}
		Node & operator [](std::string s){
			if(children.find(s) != children.end()){
				return *children.find(s)->second;
			}else{
				ERROR("Could not find child.\n");
				return *this;
			}
		}
		hid_t hid(){
			return id;
			
		}
		// Destructor
		~Node(){
			for(Iter it = children.begin(); it != children.end(); it++) {
				delete it->second;
			}
		}
		/*
		  The base name of the class should be used.
		  For example "entry" if you want to create "entry_N"
		*/
		Node * addClass(const char * s);
		Node * createGroup(const char * s);
		Node * createGroup(const char * prefix, int n);
		Node * createLink(const char * s, std::string target);
		Node * addDatasetLink(const char * s, std::string target);
		
		/*
		  The base name of the class should be used.
		  For example "entry" if you want to create "entry_N"
		*/
		Node * addClassLink(const char * s, std::string target);
		/* 
		   To create a stack pass length = H5S_UNLIMITED
		   To create a string dataset pass dataType = H5T_NATIVE_CHAR and width as maximum string size.
		 */
		Node * createDataset(const char * s, hid_t dataType, hsize_t width = 0, hsize_t height = 0, hsize_t length = 0, hsize_t stackSize = 0, int chunkSize = 0, int heightChunkSize = 0, const char * userAxis = NULL);
		Node * createStack(const char * s, hid_t dataType, hsize_t width = 0, hsize_t height = 0, hsize_t length = 0, hsize_t stackSize = H5S_UNLIMITED, int chunkSize = 0, int heightChunkSize = 0, const char * userAxis = NULL){
			return createDataset(s,dataType,width,height,length,stackSize,chunkSize,heightChunkSize,userAxis);
		}
		template<class T>
			void write(T * data, int stackSlice = -1, int sliceSize = 0, bool varibleSliceSize = false);
		
		void closeAll();
		void openAll();
		std::string path();
		Node & child(std::string prefix, int n);
		void trimAll(int stackSize = -1);
		uint getStackSlice();

		std::string name;
	private:
		Node * addNode(const char * s, hid_t oid, Type t);
		void addStackAttributes(hid_t dataset, int ndims, const char * userAxis);
		hid_t writeNumEvents(hid_t dataset, int stackSlice);
		std::string nextKey(const char * s);
		template <class T>
			hid_t get_datatype(const T * foo);

		typedef std::map<std::string, Node *>::iterator Iter;
		std::map<std::string,Node *> children;
		Node * parent;
		hid_t id;
		Type type;
		/*  This counter defines where in the file each image is stored.
		 *  It is atomically incremented by each thread */
		uint stackCounter;
		int ignoreConversionExceptions;
	};

	const int version = 140;
	const int thumbnailScale = 8;
	// The preferred chunk size for 2D stacks is 16 MBytes
	const int chunkSize2D = 16777216;
	// The preferred chunk size for 1D stacks is 1 MByte
	const int chunkSize1D = 1048576;
	// chunk sizes for peak list, assumes most images have less
	// than 4096/sizeof(float) = 1024 peaks
	const int peaksChunkSize[2] = {4194304, 4096};	
	// The preferred size of charracter (data_type, experimental_identifier,...)
	const int stringSize = 128;
}




