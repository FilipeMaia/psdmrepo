#!/bin/sh

# if there is a __pkg_list__.* file(s) in the list then read its
# contents and make a package list out of it.
# Every line of the file should have following format (as produced by 
# scons SConsTools.pkg_list):
#      <rel-pkg-name>$<pkg-name1>%<pkg-name2>%<pkg-name3>...
# By default all <pkg-nameN> parts will be joined with a dash,
# but more complex logic may be needed in some cases.

fgrep "/__pkg_list__." | xargs cat | cut -d'$' -f2 | tr '%' '-'
