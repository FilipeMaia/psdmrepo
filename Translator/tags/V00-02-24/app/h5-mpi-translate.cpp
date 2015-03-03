#include <iostream>
#include <string>
#include <vector>
#include "openmpi/mpi.h"
#include "Translator/H5MpiTranslateApp.h"

using namespace Translator;

// local namespace
namespace {

  /*
   * packs the command line arguments argc argv into the output argument packedArgs
   * if argc==2 and argv[]={"arg1","arg2"} then
   * packedArgs will be "arg1\0arg2\0"
   */
void packArgs(int argc,char *argv[],std::vector<char> &packedArgs) {
  int n = 0;
  for (int i = 0; i < argc; ++i) {
    n += strlen(argv[i]);
    n += 1;
  }
  packedArgs.resize(n);
  int p = 0;
  for (int i = 0; i < argc; ++i) {
    strcpy(&packedArgs[p],argv[i]);
    p += strlen(argv[i])+1;
  }
}

  /*
   * unpack packedArgs into a list of command line arguments.
   * if packedArgs is "arg1\0arg2\0"  then the output argument argv will be
   * "arg1", "arg2"
   */
void unpackArgs(const std::vector<char> &packedArgs, std::vector<std::string> &argv) {
  argv.clear();
  unsigned i = 0;
  unsigned  n = packedArgs.size();
  while (i < n) {
    argv.push_back(std::string(&packedArgs[i]));
    i += argv.back().size();
    i += 1;
  }
}

  /*---------------------
   MPI does not guarentee all processes receive the same command line arguments. 
   This function takes the command arguments. rank 0 broadcasts to all other processes.
   At the end, all ranks will have rank 0 command line args in the output var args.
   ----------------- */
void broadCastArgs(int argc, char * argv[], std::vector<std::string> &args) {
  int worldRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  std::vector<char> packedArgs;
  int argBufferSize;
  if (worldRank == 0) {
    // rank 0 packs the arguments in a buffer and sets the buffer length
    packArgs(argc, argv, packedArgs);
    argBufferSize = packedArgs.size();
  }
  // the buffer length is broadcast from rank 0 to all other jobs
  MPI_Bcast(&argBufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (worldRank != 0) {
    // all other jobs allocate space for the packed arguments they will receive
    packedArgs.resize(argBufferSize);
  }
  // rank 0 broadcasts the arguments to all jobs
  MPI_Bcast(&packedArgs[0], argBufferSize, MPI_CHAR, 0, MPI_COMM_WORLD);
  // all jobs unpack the arguments into the output argument
  unpackArgs(packedArgs, args);
}

} // local namespace


////////////////////////////////////////////////////////
//  main
//  
// This is a slight modification of the APPUTILS_MAIN macro from the AppUtils package.
// MPI does not guarantee that command line arguments are valid until MPI init is called.
// So we modify the APPUTILS_MAIN macro to initialize and finalize MPI before/after 
// running the app.

int main(int argc, char *argv[], char *env[]) {
  MPI_Init(&argc, &argv);
  std::vector<std::string> args;
  broadCastArgs(argc, argv, args);
  int retValue = -1;
  try {
    H5MpiTranslateApp app(argv[0], env);
    retValue = app.run(args);
  } catch (std::exception &e) {
    std::cerr << "Standard exceptoin caught: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception caught" << std::endl;
  }
  MPI_Finalize();
  return retValue;
}
