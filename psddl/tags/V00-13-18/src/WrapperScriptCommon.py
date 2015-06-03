import os
import sys
import glob
import argparse
import copy

'''functions common to the wrapper scripts
'''

def getPackageTags(ddlFile):
    '''identify the @package XXX lines and return all such
    XXX - not as robust as parsing with HdlReader, but that requires
    additional initialization - the include directories, here we just want
    to know what @package declarations are in this file
    '''
    packageTags = set()
    for ln in file(ddlFile):
        ln = ln.split('//')[0]
        if ln.find('@package')>=0:
            packageTag = ln.split('@package')[1].split()[0]
            packageTags.add(packageTag)
    return list(packageTags)
    
def checkForDataDirWithOnlyDDL(psanaPkg):
    '''Checks that the package is part of the release and has a data directory.
    Checks that all files in the data directory are .ddl files, warns of any other
    files.
    '''
    pkgData = '%s/data' % psanaPkg
    dataPkg = 'data/%s'  % psanaPkg
    assert os.path.exists(psanaPkg) < "%s directory not found. Work from test release. do addpkg %s" % (psanaPkg, psanaPkg)
    assert os.path.exists(pkgData) < "package %s doesn't contain a data directory" % psanaPkg
    assert os.path.exists(dataPkg), "The %s directory was not found. Do scons" % dataPkg
    dataFiles = glob.glob(os.path.join(dataPkg,'*'))
    nonDDLFiles = [os.path.basename(fname) for fname in dataFiles if os.path.splitext(fname)[1] != '.ddl']
    filesToWarnAbout = [fname for fname in nonDDLFiles if (len(fname)>0) and not (fname[0] in ['#','~'])]
    if len(filesToWarnAbout) > 0:
        for fname in filesToWarnAbout:
            sys.stderr.write("Warning: non-ddl file in the %s directory: %s\n" % (dataPkg, os.path.basename(fname)))
            sys.stderr.flush()

def getPackageNameFromDDLFileName(pkg, ddlfile):
    '''Returns the ddl package (related to the data type declared) from the ddlfilename, for the
    givesn pkg (software package in a release). The convention changes for h5schema ddl files vs
    normal psddldata files
    '''
    assert ddlfile.endswith('.ddl')
    if pkg == 'psddldata':
        return os.path.splitext(os.path.basename(ddlfile))[0]
    elif pkg == 'psddl_hdf2psana':
        return os.path.splitext(os.path.basename(ddlfile))[0].split('-h5')[0]
    else:
        raise ValueError("unknown pkg=%s for DDL" % pkg)

def getDDLfilenames(pkg, exclude=None, include=None, verbose=False):
    '''Gets the name of the ddl files for the pkg.
    ARGS:
      pkg  package name. Looks for data/pkg/*.ddl  from current dir
      exclude  optionally exclude some packages by their lowercase basename for the ddl file
      include  optionally filter the packages by their lowercase basename for the ddl file
      verbose  True to get more output
    '''
    assert exclude is None or include is None, "can't specify both an include and exclude set"

    if exclude is None:
        exclude = []

    allDDLFiles = [ddl for ddl in glob.glob(os.path.join('data',pkg,'*.ddl')) if \
                   (len(ddl)>0) and not (ddl[0] in ['#','~'])]
   
    ddlFiles = [fname for fname in allDDLFiles if \
                getPackageNameFromDDLFileName(pkg, fname) not in exclude]

    if pkg == 'psddldata':
        # all packages should have an entry in psddldata, not the case with psddl_hdf2psana
        assert len(allDDLFiles) - len(ddlFiles) == len(exclude), "Excluded fewer packages than " + \
        ("specified.\nSome entries in exclude argument must be wrong.\nexclude=%r\nddl files=%r" % \
        (exclude, allDDLFiles))

    if include is not None:
        ddlFiles = [fname for fname in allDDLFiles if \
                    getPackageNameFromDDLFileName(pkg, fname) in include]
        assert len(ddlFiles)==len(include), "Problem with include argument. It must list packages not found.\ninclude=%r\nallDDLFiles=%r" % (include, allDDLFiles)
                    
    if verbose:
        excludedFiles = set(allDDLFiles).difference(set(ddlFiles))
        for fname in excludedFiles:
            sys.stdout.write("excluding file: %s\n" % fname)
            sys.stdout.flush()
    return ddlFiles

programDescriptionEpilog = '''
Note, do not use the --devel switch to generate code for production releases.
'''


def standardWrapper(description, epilog=programDescriptionEpilog, 
                    defaultExclude=('xtc',), includeHdf=False, includeDecl=False):
    '''standard initialization for a ddl wrapper. This sets up an argparse parser and
    parses the command line arguments. Then it finds the ddl packages and returns a 
    command start and dict for a ddl backend wrapper.
    ARGS:
      description  - program description, for argparse
      epilog       - program epilog, for argparse - defaults to warning about devel switch
      defaultExclude - default list of packages to exclude
      includeHdf     - set to True to get schema files
      includeDecl    - set to True to get @package declarations in files

    RET:
      startcmd, verbose, pkgdict
    
      where
       
      startcmd: is either 'psddlc ' or 'psddlc -D ' depending on whether or not the devel switch
      was given
    
      verbose - if received verbose switch

      pkgdict: is a 2D dict, level one is file package names from the psddldata/data package.
      i.e, keys like 'timetool', 'oceanoptics'

      The next level is a dict with things like
      {'ddlfile':'data/psddldata/timetool.ddl'}
    
      That is ddlfile is a key to the filename relative to the current release directory
    
      If includeHdf and includeDecl are true, you will also get:
      {'ddlfile':'data/psddldata/timetool.ddl',
       'h5schema':'data/psddl_hdf2psana/timetool-h5.ddl',
       'decl':'TimeTool'}
     
      The latter is from the @package declaration within timetool.ddl
    '''
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-d', '--devel', action='store_true', help="include DDL types with the DEVEL tag", default=False)
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose output for script (not backend)", default=False)
    parser.add_argument('-i', '--include', type=str, help="explicitly provid the DDL packages to include as a comma separated list", default=None)
    parser.add_argument('-x', '--exclude', type=str, help="explicitly set the DDL packages to exclude as a comma separated list", default=None)
    args = parser.parse_args()

    exclude = list(copy.deepcopy(defaultExclude))
    if args.exclude:
        oldExclude = exclude
        exclude = [pkg.strip() for pkg in args.exclude.split(',')]
        sys.stdout.write("INFO: replacing standard exclude=%r with %r\n" % (oldExclude, exclude))

    include = args.include
    if include is not None:
        include = [pkg.strip() for pkg in args.include.split(',')]
        sys.stdout.write("INFO: including only these packages: %r\n" % include)
        for pkg in exclude:
            if pkg in include:
                sys.stderr.write("WARNING: include overrides exclude, including package: %s" % pkg)
        exclude = None

    checkForDataDirWithOnlyDDL('psddldata')
    ddlFiles = getDDLfilenames('psddldata', exclude=exclude, include=include, verbose=args.verbose)

    pkgdict = {}
    for ddlFile in ddlFiles:
        basename = os.path.basename(ddlFile)
        pkg = os.path.splitext(basename)[0]
        assert pkg not in pkgdict, "ddlFile=%s conflicts with pkgdict[%s]=%r" % (ddlFile, pkg, pkgdict[pkg])
        pkgdict[pkg]={'ddlfile':ddlFile}

    if includeHdf:
        checkForDataDirWithOnlyDDL('psddl_hdf2psana')
        hdf5schemaDdlFiles = getDDLfilenames('psddl_hdf2psana', exclude=exclude, include=include, verbose=args.verbose)

        for ddlFile in hdf5schemaDdlFiles:
            basename = os.path.basename(ddlFile)
            pkgh5 = os.path.splitext(basename)[0]
            assert pkgh5.endswith('-h5'), "ddl file in psddl_hdf2psana does not end with -h5.ddl: %s" % pkgh5
            pkg = pkgh5.split('-h5')[0]
            assert pkg in pkgdict, "psddl_hdf2psana ddl file exists for a package that doesn't exist in psddldata: %s" % ddlFile
            pkgdict[pkg]['h5schema']=ddlFile
        for pkg,subdict in pkgdict.iteritems():
            if 'h5schema' not in subdict:
                subdict['h5schema']=''

    if includeDecl:
        for subdict in pkgdict.values():
            ddlFile = subdict['ddlfile']
            pkgtags = getPackageTags(ddlFile)
            assert len(pkgtags)==1, "file %s has more than one package tag: %r" % (ddlFile, pkgtags)
            subdict['decl']=pkgtags[0]

    psddlCmdStart = 'psddlc '
    if args.devel:
        psddlCmdStart += '-D '

    sys.stdout.flush()
    sys.stderr.flush()
    return  psddlCmdStart, args.verbose, pkgdict
