#
#  Determine Python version to use for the build. Python version can be specified on 
#  the command line using --with pythonXY option. If the option is not defined
#  then call default python interpreter (found in $PATH) to ask its version. In 
#  any case we set "python" and assign it a string "pythonX.Y" (like python2.4).
#  Additonally it also defines macros "pythonXY" with the value of 1.
#

%{?_with_python24:%define python python2.4}
%{?_with_python25:%define python python2.5}
%{?_with_python26:%define python python2.6}
%{?_with_python27:%define python python2.7}
# there should be no more 2.x version after 2.7
%{?_with_python30:%define python python3.0}
%{?_with_python31:%define python python3.1}
%{?_with_python32:%define python python3.2}
%{?_with_python33:%define python python3.3}
%{?_with_python34:%define python python3.4}

# as default python for its version
%{!?python:%define python %(python -c 'import sys; print "python%d.%d"%tuple(sys.version_info[:2])')}

%define python24 0
%define python25 0
%define python26 0
%define python27 0
%define python30 0
%define python31 0
%define python32 0
%define python33 0
%define python34 0
%if "%{python}" == "python2.4"
%define python24 1
%endif
%if "%{python}" == "python2.5"
%define python25 1
%endif
%if "%{python}" == "python2.6"
%define python26 1
%endif
%if "%{python}" == "python2.7"
%define python27 1
%endif
%if "%{python}" == "python3.0"
%define python30 1
%endif
%if "%{python}" == "python3.1"
%define python31 1
%endif
%if "%{python}" == "python3.2"
%define python32 1
%endif
%if "%{python}" == "python3.3"
%define python33 1
%endif
%if "%{python}" == "python3.4"
%define python34 1
%endif

# define location of platfom-specific python site-package directory relative to the prefix,
# this will contain something like lib/python2.7/site-packages or lib64/python2.4/site-packages
%define python_sitepkgdir %(%{python} -c 'from distutils import sysconfig; print sysconfig.get_python_lib(1,0,"")')

# determine path to actual python executable (resolve symlinks),
# this path is needed when building python extensions with distutils
%define python_exe %(%{python} -c 'import sys, os; print os.path.realpath(sys.executable)')

