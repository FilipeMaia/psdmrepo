#!/bin/sh
#
# Script which prepares remote site/machine for installation of PSDM sw
#

sit_root=${1:?root directory is not specified}

rpm_version=4.4.2.3
rpm_release=3psdm
apt_version=0.5.15lorg3.94a
apt_release=5psdm

# ====================================================================

set -e

# messaging
err() { echo $* 1>&2 ; exit 1 ; }
warn() { echo $* 1>&2 ; }

# no relocation for now
test "$sit_root" = "/reg/g/psdm" || err "Relocation not supported, use /reg/g/psdm"

# guess OS, copied from sit_setup.uss
  os=`uname -s`
  if [ "$os" = "Linux" -a -r /etc/redhat-release ] ; then
    rh=`cat /etc/redhat-release | tr ' ' _`
    case $rh in
      Red_Hat_Enterprise_Linux_*_release_5.*)   os=rhel5 ;;
      Red_Hat_Enterprise_Linux_*_release_6.*)   os=rhel6 ;;
      CentOS_release_5.*)                       os=rhel5 ;;
      CentOS_release_6.*)                       os=rhel6 ;;
      Scientific_Linux_*_release_3.*)           os=slc3 ;;
      Scientific_Linux_*_release_4.*)           os=slc4 ;;
    esac
  elif [ "$os" = "Linux" -a -r /etc/lsb-release ] ; then
    . /etc/lsb-release
    if [ "$DISTRIB_ID" = "Ubuntu" ] ; then
      os=`echo ubu${DISTRIB_RELEASE} | cut -d. -f1`
    fi
  fi
  test -n "$os" || err "Failed to determine OS type"

# say we do 64-bit for now nly
proc="x86_64"

# make dist directory
echo "... Create $sit_root/sw/dist directory"
mkdir -p "$sit_root/sw/dist"

# copy rpm tar
echo "... Download RPM"
curl http://pswww.slac.stanford.edu/psdm-repo/rpm-$os-$proc-$rpm_version.tar.gz | \
	tar -C "$sit_root/sw/dist/" -z -x -f -

# make .rpmmacros
echo "... Create \$HOME/.rpmmacros"
if [ -f "$HOME/.rpmmacros" ] ; then
	warn "\$HOME/.rpmmacros exists, renaming to \$HOME/.rpmmacros.dist"
	mv "$HOME/.rpmmacros" "$HOME/.rpmmacros.dist"
fi
cat >"$HOME/.rpmmacros" << EOD
%_topdir $sit_root/sw/dist/rpms
%_dbpath $sit_root/sw/dist/rpmdb
%_tmppath $sit_root/sw/dist/rpms/tmp
%_dbapi                 4
%_dbapi_rebuild         4
%debug_package %{nil}
EOD

echo "... Creating RPM database directory"
mkdir -p $sit_root/sw/dist/rpmdb
mkdir -p $sit_root/sw/dist/rpms/tmp

# install RPM and APT
echo "... Installing RPM"
$sit_root/sw/dist/apt-rpm/$os-$proc/bin/rpm -ivh http://pswww.slac.stanford.edu/psdm-repo/$os/$proc/rpm-$os-$proc-$rpm_version-$rpm_release.$proc.rpm
echo "... Installing APT"
$sit_root/sw/dist/apt-rpm/$os-$proc/bin/rpm -ivh http://pswww.slac.stanford.edu/psdm-repo/$os/$proc/apt-rpm-$apt_version-$apt_release.$proc.rpm
$sit_root/sw/dist/apt-rpm/$os-$proc/bin/rpm -ivh http://pswww.slac.stanford.edu/psdm-repo/$os/$proc/apt-rpm-$os-$proc-$apt_version-$apt_release.$proc.rpm

echo
echo "Site setup finished. Please add these lines to .bashrc:"
echo "============================================================================"
echo "export PATH=$sit_root/sw/dist/apt-rpm/$os-$proc/bin:\$PATH"
echo "export APT_CONFIG=$sit_root/sw/dist/apt-rpm/$os-$proc/etc/apt/apt.conf"
echo "============================================================================"
echo "(or equivalent to your .cshrc)"
