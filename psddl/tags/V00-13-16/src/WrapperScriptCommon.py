import os
import sys
import glob

'''functions common to the wrapper scripts
'''

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
        
def getDDLfilenames(pkg, exclude=None, verbose=False):
    if exclude is None:
        exclude = []
    allDDLFiles = [ddl for ddl in glob.glob(os.path.join('data',pkg,'*.ddl')) if \
                   (len(ddl)>0) and not (ddl[0] in ['#','~'])]
   
    ddlFiles = [fname for fname in allDDLFiles if \
                os.path.splitext(os.path.basename(fname))[0] not in exclude]
    if verbose:
        excludedFiles = set(allDDLFiles).difference(set(ddlFiles))
        for fname in excludedFiles:
            sys.stdout.write("excluding file: %s\n" % fname)
        sys.stdout.flush()
    return ddlFiles

