#
# make file for building/installing psddl_pdsdata package outside oflline release.
#

.PHONY: all install clean help
DEFAULT: all

SHLIB := psddl_pdsdata.so

SRCS := $(wildcard src/*.cpp)
OBJS := $(patsubst src/%.cpp,build/%.o,$(SRCS))

help:
	@echo "Possible targets:"
	@echo '    all      - (default) builds everyhting, needs few variable to be set:'
	@echo '               PDSDATAHDR - location of pdsdata package, directory containing pdsdata/'
	@echo '               PDSDATALIB - location of pdsdata libraries'
	@echo '               NDARRAY - location of ndarray headers, directory containing ndarray/'
	@echo '               BOOST - location of boost header files'
	@echo '    install  - installs everything, define DESTDIR variable to specify destination,'
	@echo '               files fill be copied to $$DESTDIR/psddl_pdsdata and $$DESTDIR/lib'
	@echo '    help     - print this information'

all: build/$(SHLIB)

build/$(SHLIB): $(OBJS)
	@if [ -z "$(PDSDATALIB)" ]; then echo "PDSDATALIB is not defined"; exit 2; fi
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -shared $(OBJS) -L$(PDSDATALIB) -lxtcdata -o $@

build/%.o: src/%.cpp build/include/psddl_pdsdata
	@if [ -z "$(NDARRAY)" ]; then echo "NDARRAY is not defined"; exit 2; fi
	@if [ -z "$(PDSDATAHDR)" ]; then echo "PDSDATAHDR is not defined"; exit 2; fi
	@if [ -z "$(BOOST)" ]; then echo "BOOST is not defined"; exit 2; fi
	$(CXX) $(CPPFLAGS) -Ibuild/include -I$(PDSDATAHDR) -I$(NDARRAY) -I$(BOOST)/include $(CXXFLAGS) -fPIC -o $@ -c $<

build/include/psddl_pdsdata: build/include
	@test -e $@ || ln -s ../../include $@

build/include:
	@mkdir -p build/include

install: build/$(SHLIB)
	@if [ -z "$(DESTDIR)" ]; then echo "DESTDIR is not defined"; exit 2; fi
	@install -d "$(DESTDIR)/psddl_pdsdata" "$(DESTDIR)/lib"
	@install -m 644 -t "$(DESTDIR)/psddl_pdsdata" include/*.h
	@install -t "$(DESTDIR)/lib" build/$(SHLIB)

clean:
	@rm -rf build
