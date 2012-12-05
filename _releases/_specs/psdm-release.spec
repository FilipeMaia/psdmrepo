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
ln -sfT %{relname} %{buildroot}/%{sit_reldir}/%{reltype}current-%{sit_arch_os}

%clean
rm -rf %{buildroot}
rm %{_tmppath}/%{pkg}-%{relname}.filelist

%files -f %{_tmppath}/%{pkg}-%{relname}.filelist

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

# ================== "Current" subpackage ==================

%package -n %{pkg}-%{reltype}current-%{sit_arch_os}

Requires: %{pkg}-%{relname}-%{sit_arch_base}-opt
Group:    PSDM Analysis Software
Prefix:   %{prefix}

Summary:  Current link for PSDM software release %{relname}.

%description -n %{pkg}-%{reltype}current-%{sit_arch_os}
Current link for PSDM software release %{relname}.

%files -n %{pkg}-%{reltype}current-%{sit_arch_os}
%{sit_reldir}/%{reltype}current-%{sit_arch_os}

# ================= ChangeLog =========================

%changelog

* Tue Dec 04 2012 Andy Salnikov <salnikov@slac.stanford.edu> 0.0.0-1psdm
- initial release
- no version numbers defined for this file, it will be used for multiple
  releases
