#
# make file for building/installing ndarray package outside oflline release.
# This package is headers-only and it does not duild any libraries, it just
# copies all necessary header files to a destination specified in DESTDIR. 
#

.PHONY: all install help
DEFAULT: all

help:
	@echo "Possible targets:"
	@echo '    all      - (default) builds everyhting'
	@echo '    install  - installs everything, define DESTDIR variable to specify destination,'
	@echo '               $$DESTDIR/ndarray directory will be created with headers in it'
	@echo '    help     - print this information'

all:
	@echo "all is done"

install:
	@if [ -z "$(DESTDIR)" ]; then echo "DESTDIR is not defined"; exit 2; fi
	@install -d "$(DESTDIR)/ndarray"
	@install -m 644 -t "$(DESTDIR)/ndarray" include/*.h
