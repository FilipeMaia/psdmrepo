"""SCons.Tool.rpm_spec

Tool-specific initialization for rpm_spec builder.

AUTHORS:
 - Andy Salnikov

"""

import os
import time
import sys
from string import Template

import SCons
from SCons.Builder import Builder
from SCons.Action import Action

from SConsTools.trace import *
from SConsTools.scons_functions import *

_spec = Template(r"""
%define pkg      $package
%define version  $version
%define release  1psdm

Source0:    %{pkg}-%{version}.tar.gz
Source99:   python_version.spec.inc
%include %{S:99}
Source100:  sit_env.spec.inc
%include %{S:100}

# everything will go under SIT_ROOT
%define prefix   %{sit_root}
# installation directory for the package
%define instdir  %{prefix}/sw/releases/$sit_release

Name:       %{pkg}-%{version}
Version:    %{version}
Release:    %{release}

$requires

Vendor:     SLAC/LCLS
License:    Unknown
Group:      PSDM Analysis Software
URL:        https://confluence.slac.stanford.edu/display/PCDS/Data+Analysis
Packager:   PSDM Librarian

BuildRoot:  %{_tmppath}/%{pkg}-%{version}
Prefix:     %{prefix}
AutoReq:    0 
AutoReqProv: 0 

Summary:    PSDM software release $sit_release

%description
PSDM software release $sit_release.

# =============== Scripts =====================

%prep
%setup -q -n $sit_release

%build
. %{sit_root}/bin/sit_setup.sh
scons
scons test
scons doc
find . -wholename ./arch -prune -o -wholename ./build -prune -o -wholename ./__FileList__ -o -print | sed 's#^\.#%{instdir}#' > __FileList__ 

%install
. %{sit_root}/bin/sit_setup.sh
scons install DESTDIR=%{buildroot}/%{instdir}

%clean
rm -rf %{buildroot}

%files -f __FileList__

# ================== Platform-specific subpackage ==================

%package -n %{pkg}-%{version}-%{sit_arch}

Requires: %{pkg}-%{version}
Group:    PSDM Analysis Software
Prefix:   %{prefix}
Autoreq:  0 
Summary:  PSDM software release $sit_release, platform-specific files.

%description -n %{pkg}-%{version}-%{sit_arch}
PSDM software release $sit_release, platform-specific files.

%files -n %{pkg}-%{version}-%{sit_arch}
%{instdir}/arch/%{sit_arch}

# ================= ChangeLog =========================

%changelog

* $date PSDM Librarian <noname@slac.stanford.edu> $version-1psdm
- Auto-generated spec file
""")


def _fmtList(lst):
    return '[' + ','.join(map(str, target)) + ']'

class _makeRpmSpec:

    def __call__(self, target, source, env) :
        """Target should be a single file, no source is needed"""
        if len(target) != 1 : fail("unexpected number of targets for RpmSpec: " + str(target))
        if len(source) != 0 : fail("unexpected number of sources for RpmSpec: " + str(source))

        target = str(target[0])
        trace("Executing RpmSpec `%s'" % (target,), "makeRpmSpec", 3)

        # may need to make a directory for target
        targetdir = os.path.normpath(os.path.dirname(target))
        if not os.path.isdir(targetdir): 
            os.makedirs(targetdir)

        # make release name and version number
        sit_release = env['SIT_RELEASE']
        trace("RpmSpec sit_release: %s" % (sit_release,), "makeRpmSpec", 4)
        pkg_ver = sit_release.rsplit('-', 1)
        if len(pkg_ver) == 1:
            package = "psdm-release"
            version = pkg_ver[0]
        else:
            package = "psdm-release-" + pkg_ver[0]
            version = pkg_ver[1]
        trace("RpmSpec package: %s, version %s" % (package, version), "makeRpmSpec", 4)

        # build requirements list
        requires = ["psdm-root"]
        for pkg, pkginfos in env['EXT_PACKAGE_INFO'].iteritems():
            trace("package %s, pkginfos %s" % (pkg, pkginfos), "makeRpmSpec", 4)
            for pkginfo in pkginfos:
                dep = '-'.join(pkginfo)
                requires.append(dep)

        # add also scons, need to guess version name and python version
        pkg = "scons-%s-python%d.%d" % (SCons.__version__, sys.version_info[0], sys.version_info[1])
        requires.append(pkg)
        
        requires = ["Requires:      " + req for req in requires] + \
            ["BuildRequires: " + req for req in requires]
        requires = '\n'.join(requires)
        trace("requires: %s" % (requires), "makeRpmSpec", 4)

        # make time string
        date = time.strftime("%a %b %d %Y", time.localtime())
        
        # substitute everything in a template
        subs = dict(sit_release=sit_release, package=package, version=version, 
                requires=requires, date=date)
        spec = _spec.substitute(subs)
        
        # write it out
        open(target, "w").write(spec)

    def strfunction(self, target, source, env):
        try :
            return "Creating RPM SPEC file: `" + str(target[0]) + "'"
        except :
            return 'RpmSpec(' + _fmtlist(target) + ')'

def create_builder(env):
    try:
        builder = env['BUILDERS']['RpmSpec']
    except KeyError:
        builder = SCons.Builder.Builder(action=_makeRpmSpec())
        env['BUILDERS']['RpmSpec'] = builder

    return builder

def generate(env):
    """Add Builders and construction variables for making RPM SPEC file."""

    # Create the PythonExtension builder
    create_builder(env)

    trace("Initialized rpm_spec tool", "rpm_spec", 2)

def exists(env):
    return True
