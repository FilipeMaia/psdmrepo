import argparse

import psana_test.liveModeSimLib as liveModeLib

programDescription='''
Driver program to simulate live mode.
'''

programDescriptionEpilog='''
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=programDescription, 
                                     epilog=programDescriptionEpilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--inprogress_ext',type=str, help="extension while file is in progress", default=".inprogress")
    parser.add_argument('--start_delays',type=str, help="delays for each stream", default="0-255:0.0")
    parser.add_argument('--mb_per_writes',type=str, help="number of megabytes to write with each stream", default="0-255:1.0")
    parser.add_argument('--max_mbs',type=str, help="maximum number of megabytes to write for each stream. < 0 is all.", default="0-255:-1.0")
    parser.add_argument('--delays_between_writes',type=str, help="seconds to sleep between writes for each stream", default="0-255:.2")
    parser.add_argument('--force',action='store_true', help="force overwrite of dest files", default=False)
    parser.add_argument('-v','--verbose',action='store_true', help="verbose", default=False)
    parser.add_argument('-r','--run',type=int, help="run number", default=None)
    parser.add_argument('srcdir', type=str, help="source directory")
    parser.add_argument('destdir', type=str, help="destination directory")
    
    args = parser.parse_args()
    assert args.run is not None, "you must supply a run with the -r or --run option"
    liveModeLib.simLiveMode(args.inprogress_ext, args.run, args.srcdir, args.destdir, args.start_delays, args.mb_per_writes, 
                            args.max_mbs, args.delays_between_writes, args.force, args.verbose)

    
