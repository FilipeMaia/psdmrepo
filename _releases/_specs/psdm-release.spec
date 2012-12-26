#
# SPEC file for building offline release, it builds two architectures at a
# time (opt and dbg) 
#
# It needs a macro which have to be defined when executed:
#   relname - full name of the release (ana-0.X.Y)
#

%define relname @RELNAME@

%define reltype  %(echo %{relname} | sed 's/\\(.*-\\)*\\(.*\\)/\\1/')
%define version  %(echo %{relname} | sed 's/\\(.*-\\)*\\(.*\\)/\\2/')

%define pkg      psdm-release
%define release  1psdm

Source0:    %{pkg}-%{relname}.tar.gz
Source99:   python_version.spec.inc
%include %{S:99}
Source100:  sit_env.spec.inc
%include %{S:100}
Source101:  release-find-requires.sh

# everything will go under SIT_ROOT
%define prefix   %{sit_root}
# installation directory for the package
%define sit_reldir %{prefix}/sw/releases
%define instdir  %{sit_reldir}/%{relname}

Name:       %{pkg}-%{relname}
Version:    %{version}
Release:    %{release}

# all other dependencies are defined at build time
Requires:      psdm-root
Requires:      psdm-data

Vendor:     SLAC/LCLS
License:    Unknown
Group:      PSDM Analysis Software
URL:        https://confluence.slac.stanford.edu/display/PCDS/Data+Analysis
Packager:   PSDM Librarian

BuildRoot:  %{_tmppath}/%{pkg}-%{relname}
Prefix:     %{prefix}

%define _use_internal_dependency_generator 0
%define __find_requires %{S:101}
%define __find_provides %{nil}

Summary:    PSDM software release %{relname}

%description
PSDM software release %{relname}.

# =============== Scripts =====================

%prep
%setup -q -n %{relname}

%build
. %{sit_root}/bin/sit_setup.sh -a %{sit_arch_base}-opt
scons PKG_LIST_FILE=arch/__pkg_list__.%{sit_arch_base}-opt all test SConsTools.pkg_list doc
. %{sit_root}/bin/sit_setup.sh -a %{sit_arch_base}-dbg
scons PKG_LIST_FILE=arch/__pkg_list__.%{sit_arch_base}-dbg all test SConsTools.pkg_list
find . -maxdepth 1 -wholename . -o -wholename ./arch -o -wholename ./build \
    -o -wholename ./.sconsign.dblite -o -print | \
    sed 's#^\.#%{instdir}#' > %{_tmppath}/%{pkg}-%{relname}.filelist

%install
. %{sit_root}/bin/sit_setup.sh -a %{sit_arch_base}-opt
scons install DESTDIR=%{buildroot}/%{instdir}

%clean
rm -rf %{buildroot}
rm %{_tmppath}/%{pkg}-%{relname}.filelist

%files -f %{_tmppath}/%{pkg}-%{relname}.filelist
%dir %{instdir}
%dir %{instdir}/arch

# ================== Platform-specific subpackage ==================

%package -n %{pkg}-%{relname}-%{sit_arch_base}-opt

Requires: %{pkg}-%{relname}
Group:    PSDM Analysis Software
Prefix:   %{prefix}

Summary:  PSDM software release %{relname}, platform-specific files.

%description -n %{pkg}-%{relname}-%{sit_arch_base}-opt
PSDM software release %{relname}, platform-specific files for optimized builds.

%files -n %{pkg}-%{relname}-%{sit_arch_base}-opt
%{instdir}/arch/%{sit_arch_base}-opt
%ghost %{instdir}/arch/__pkg_list__.%{sit_arch_base}-opt

%post -n %{pkg}-%{relname}-%{sit_arch_base}-opt
if [ -n "$SIT_ROOT" -a "$SIT_ROOT" != "/reg/g/psdm" ] ; then
    archdir="$SIT_ROOT/sw/releases/%{relname}/arch/%{sit_arch_base}-opt"
    for dir in bin lib geninc python; do
        # find all symlinks that point to /reg/g/psdm/ and redirect them to new location
        find "$archdir/$dir" -lname '/reg/g/psdm/*' | while read link ; do
            newtrgt=`readlink -n "$link" | sed -n "s%/reg/g/psdm/%$SIT_ROOT/%p"`
            test -n "$newtrgt" && ln -sfT "$newtrgt" "$link"
        done
    done 
    # update shebang line in scripts
    find "$archdir/bin" -type f | while read f ; do
        head -1 "$f" | egrep -q '^#!/reg/g/psdm/' && sed -i "1s%#!/reg/g/psdm/%#!$SIT_ROOT/%" "$f"
    done
    true
fi

# ================== Platform-specific subpackage ==================

%package -n %{pkg}-%{relname}-%{sit_arch_base}-dbg

Requires: %{pkg}-%{relname}
Group:    PSDM Analysis Software
Prefix:   %{prefix}

Summary:  PSDM software release %{relname}, platform-specific files.

%description -n %{pkg}-%{relname}-%{sit_arch_base}-dbg
PSDM software release %{relname}, platform-specific files for debug builds.

%files -n %{pkg}-%{relname}-%{sit_arch_base}-dbg
%{instdir}/arch/%{sit_arch_base}-dbg
%ghost %{instdir}/arch/__pkg_list__.%{sit_arch_base}-dbg

%post -n %{pkg}-%{relname}-%{sit_arch_base}-dbg
if [ -n "$SIT_ROOT" -a "$SIT_ROOT" != "/reg/g/psdm" ] ; then
    archdir="$SIT_ROOT/sw/releases/%{relname}/arch/%{sit_arch_base}-dbg"
    for dir in bin lib geninc python; do
        # find all symlinks that point to /reg/g/psdm/ and redirect them to new location
        find $archdir/$dir -lname '/reg/g/psdm/*' | while read link ; do
            newtrgt=`readlink -n "$link" | sed -n "s%/reg/g/psdm/%$SIT_ROOT/%p"`
            test -n "$newtrgt" && ln -sfT "$newtrgt" "$link"
        done
    done 
    # update shebang line in scripts
    find "$archdir/bin" -type f | while read f ; do
        head -1 "$f" | egrep -q '^#!/reg/g/psdm/' && sed -i "1s%#!/reg/g/psdm/%#!$SIT_ROOT/%" "$f"
    done
    true
fi

# ================== "Latest" subpackage ==================

%package -n %{pkg}-%{reltype}latest-%{sit_arch_os}

Requires: %{pkg}-%{relname}-%{sit_arch_base}-opt
Group:    PSDM Analysis Software
Prefix:   %{prefix}

Summary:  Pseudo-package used to install latest release.

%description -n %{pkg}-%{reltype}latest-%{sit_arch_os}
This package is used to install/upgrade to the latest release, it has a
dependecy on %{pkg}-%{relname}-%{sit_arch_base}-opt package which will get
updated or installed when you gen new version of this package.

%files -n %{pkg}-%{reltype}latest-%{sit_arch_os}

# ================= ChangeLog =========================

%changelog

* Tue Dec 25 2012 Andy Salnikov <salnikov@slac.stanford.edu> 0.0.0-1psdm
- added post-install scripts to relocate symlinks in arch/ directory

* Wed Dec 05 2012 Andy Salnikov <salnikov@slac.stanford.edu> 0.0.0-1psdm
- removed current link entirely
- changed rpm name from -current- to -latest-

* Tue Dec 04 2012 Andy Salnikov <salnikov@slac.stanford.edu> 0.0.0-1psdm
- initial release
- no version numbers defined for this file, it will be used for multiple
  releases
