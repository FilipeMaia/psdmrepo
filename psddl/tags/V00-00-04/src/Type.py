#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Type...
#
#------------------------------------------------------------------------

"""DDL class representing a type (class).

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging
import types

#---------------------------------
#  Imports of base class module --
#---------------------------------
from psddl.Namespace import Namespace

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.ExprVal import ExprVal
from psddl.Method import Method

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class Type ( Namespace ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, **kw ) :

        parent = kw.get('package')
        Namespace.__init__(self, name, parent)
        
        self.name = name
        self.version = kw.get('version')
        self.type_id = kw.get('type_id')
        self.levels = kw.get('levels')
        self.comment = kw.get('comment')
        self.package = parent
        self.size = kw.get('size')
        self.align = kw.get('align')
        self.pack = kw.get('pack')
        if self.pack : self.pack = int(self.pack)
        self.included = kw.get('included')
        self.base = kw.get('base')
        self.tags = kw.get('tags', {}).copy()

        self.xtcConfig = []

        self.repeat = None
                
        self.ctors = []   # constructors

    @property
    def basic(self):
        return 'basic' in self.tags

    @property
    def external(self):
        return 'external' in self.tags

    @property
    def value_type(self):
        return 'value-type' in self.tags

    def fullName(self, lang=None, topNs=None):
        if self.external: topNs = None
        return Namespace.fullName(self, lang, topNs)

    def fullName(self, lang=None, topNs=None):
        if self.external: topNs = None
        sep = {'C++' : '::'}.get(lang, '.')
        name = self.name
        if lang == 'C++' and 'c++-name' in self.tags: name = self.tags['c++-name']
        if self._parent: 
            parent = self._parent.fullName(lang, topNs)
            if parent: 
                name = parent + sep + name
            elif topNs:
                name = topNs + sep + name
        return name
    
    def __str__(self):
        
        return "<Type(%s)>" % self.__dict__

    def __repr__(self):
        
        return "<Type(%s)>" % self.name

    def calcOffsets(self):
        """Calculate offsets for all members of the type"""

        logging.debug("_calcOffsets: type=%s", self)

        offset = ExprVal(0)
        if self.base : offset = self.base.size
        maxalign = 1
        for attr in self.attributes():

            logging.debug("_calcOffsets: offset=%s attr=%s", offset, attr)

            align = attr.align()
            if align : maxalign = max(maxalign, align)

            if attr.offset is None:
                
                # need to calculate offset for this attribute
            
                if type(offset.value) == types.IntType:
                
                    # no explicit offset - use implicit but check alignment
                    align = attr.align()
                    if align is None:
                        logging.warning('unknown alignment for %s %s.%s', attr.type.name, self.name, attr.name)
                    else :
                        if self.pack: align = min(align, self.pack)
                        if offset.value % align != 0:
                            logging.error('unaligned attribute %s %s.%s', attr.type.name, self.name, attr.name)
                            logging.error('implicit offset = %s, alignment = %s', offset, align)
                            logging.error('use pack="N" or add padding attributes')
                            raise TypeError('%s.%s unaligned attribute' % (self.name, attr.name))
    
                    attr.offset = offset

                else:
    
                    # attribute has no offset defined, current offset is an expression
                    # no way now to evaluate expression and check it's alignment, so we 
                    # just accept 
                    attr.offset = offset

            else:

                # attribute already has an offset defined

                if type(attr.offset) is types.IntType and type(offset.value) is types.IntType:
                    
                    if attr.offset < offset.value :
                        logging.error('offset specification mismatch for %s.%s', self.name, attr.name)
                        logging.error('implicit offset = %s, explicit offset = %s', offset, attr.offset)
                        raise TypeError('%s.%s offset mismatch' % (self.name, attr.name))
                    elif attr.offset > offset.value :
                        # need padding
                        pad = attr.offset - offset
                        logging.error('extra padding needed before %s.%s', self.name, attr.name)
                        raise TypeError('%s.%s extra padding needed' % (self.name, attr.name))
                    else:
                        # this is what we expect
                        pass

                else:
                    
                    # at least one of them is an expression, currently there is no way 
                    # to verify that two expressions are the same
                    
                    logging.warning('%s.%s has pre-defined offset, make sure it has right value', self.name, attr.name)
                    logging.warning('pre-defined offset = %s, computed offset = %s', attr.offset, offset)

                    # safer to reset offset to a pre-defined value
                    offset = ExprVal(attr.offset)

            # move to a next attribute
            offset = offset+attr.sizeBytes()

        if self.pack: maxalign = min(maxalign, self.pack)
        logging.debug("_calcOffsets: type=%s size=%s align=%s", repr(self), offset, maxalign)
        self.align = maxalign
        
        if self.size:
            # size was already pre-defined
            if type(self.size) is types.IntType and type(offset.value) is types.IntType:
                if self.size != offset.value :
                    logging.error('object size mismatch for %s', self.name)
                    logging.error('implicit size = %d, explicit size = %d', offset, self.size)
                    raise TypeError('%s size mismatch' % self.name)
            else:
                logging.warning('%s has pre-defined size, make sure it has right value', self.name)
                logging.warning('pre-defined size = %s, computed size = %s', self.size, offset)
        else:
            
            # set it to calculate value
            self.size = offset

        if 'no-sizeof' not in self.tags:
            self._genSizeof()

    def _genSizeof(self):

        # generate public _sizeof method

        if self.size.value is None :
            
            # special case for unknown size
            expr = "~uint32_t(0)"
            needCfg = False
            static = True
            logging.debug("_genSizeof: expr=%s", expr)
            
        else :

            expr = ExprVal(0, self)
            if self.base :
                expr = ExprVal(self.base.fullName('C++')+"::_sizeof()", self)
            for attr in self.attributes():
                meth = attr.type.lookup('_sizeof', Method)
                if not meth:
                    size = ExprVal(attr.type.size, self)
                else:
                    cfg = ''
                    if meth.expr.get('C++',"").find("{xtc-config}") >= 0: cfg = 'cfg'
                    if meth.expr.get('C++',"").find("{self}") >= 0:
                        if attr.isfixed():
                            size = ExprVal("{self}."+attr.name+"._sizeof(%s)"%cfg, self)
                        else:
                            size = ExprVal("{self}."+attr.accessor.name+"()._sizeof(%s)"%cfg, self)
                    else:
                        size = ExprVal(attr.type.fullName('C++')+"::_sizeof(%s)"%cfg, self)

                if attr.dimensions: 
                    size = str(size)
                    for dim in attr.dimensions.dims:
                        size += '*(%s)' % dim
                    size = ExprVal(size, self)
                expr += size
            expr = str(expr)
            logging.debug("_genSizeof: expr=%s", expr)

            needCfg = expr.find('{xtc-config}') >= 0
            static = expr.find('{self}') < 0

        type = self.lookup('uint32_t', Type)
        tags = dict(inline=None)
        meth = Method('_sizeof', type=type, parent=self,
                      expr={"C++": expr}, tags=tags, static=static)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
