#
# Check that SIT_ROOT and SIT_ARCH envvars are set and define
# few spec variables from them:
#   sit_arch := $SIT_ARCH
#   prefix   := $SIT_ROOT
# 

%define sit_arch %(echo $SIT_ARCH)
%define sit_root %(echo $SIT_ROOT)

%if "%{sit_arch}" == ""
%{error:$SIT_ARCH is not defined}
%endif
%if "%{sit_root}" == ""
%{error:$SIT_ROOT is not defined}
%endif

%define sit_arch_proc %(echo $SIT_ARCH | cut -d- -f1)
%define sit_arch_os   %(echo $SIT_ARCH | cut -d- -f2)
%define sit_arch_comp %(echo $SIT_ARCH | cut -d- -f3)
%define sit_arch_opt  %(echo $SIT_ARCH | cut -d- -f4)
%define sit_arch_base %(echo $SIT_ARCH | cut -d- -f-3)
