#
#   $Id$
#
#   Description:
#     Makefile for building pazlib shared library outside of standard PSDM releases
#

# This name is used by "tar" target to specify tar file name
# and top-level directory inside tar file
PACKAGE = pazlib

# Change these if your zlib is installed in non-default location
ZLIBINC = 
ZLIBLIB = 

# usually no need to change these
CC = gcc
CFLAGS = -O3 -pthread -fPIC $(addprefix -I,$(ZLIBINC))
LDLIBS = $(addprefix -L,$(ZLIBLIB)) -lz
LDFLAGS = -shared

# source file for library
SOURCES = adler32.c pacompress.c

# files that go into tar
DISTFILES = LICENSE INSTALL Makefile ../doc/README ../doc/ChangeLog ../src/adler32.c ../src/pacompress.c

.PHONY: tar $(PACKAGE).tar.gz all

all: libpazlib.so

libpazlib.so: $(SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@  $^ $(LOADLIBES) $(LDLIBS)

tar: $(PACKAGE).tar.gz

$(PACKAGE).tar.gz: $(DISTFILES)
	@tmpdir=/tmp/pazlib-$$$$/; mkdir -p $$tmpdir/$(PACKAGE); \
	cp -p $^ $$tmpdir/$(PACKAGE); \
	tar -c -C $$tmpdir -v -f $@ $(PACKAGE)
