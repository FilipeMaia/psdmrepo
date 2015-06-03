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

def _hasconfig(str):
    return '{xtc-config}' in str or '@config' in str

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
        self.location = kw.get('location')
        self.base = kw.get('base')
        self.tags = kw.get('tags', {}).copy()

        self.xtcConfig = kw.get('xtcConfig', [])

        self.ctors = []   # constructors

        self.h5schemas = []    # list of hdf schemas for this type.

    @property
    def basic(self):
        '''
        "basic" tag is used to specify types as basic. Basic type has no 
        attributes and has all other good properties - pre-defined size and
        alignment. Typically only standard types (uint32_t and friends) are
        defined as basic, user-defined types should not be defined as basic.
        '''
        return 'basic' in self.tags

    @property
    def variable(self):
        """ variable means instances may have different size """
        if self.base and self.base.variable: return True
        for attr in self.attributes():
            if attr.stor_type.variable: return True
            if attr.shape: 
                for dim in attr.shape.dims:
                    if '{self}' in str(dim): return True
                    if '@self' in str(dim): return True
        return False

    @property
    def external(self):
        '''
        "external" tag is used to specify external types for which there 
        is no code generated at all and they are expected to be implemented 
        by some external means (providing C++ implementation in a separate
        file for example). Type is external either if it is marked external 
        or it lives in a package which is marked external.
        '''
        if 'external' in self.tags : return True
        if self.parent: return self.parent.external
        return False

    @property
    def value_type(self):
        ''' 
        "value-type" tag is used to specify value types. Value type is a 
        type which is a good C++ type (i.e. can be expressed in pure C++), 
        can be copied and is relatively cheap to copy. For value type we 
        will not generate abstract types with virtual methods but concrete 
        C++ classes with data member and no virtual methods.  
        '''  
        return 'value-type' in self.tags

    def fullName(self, lang=None, topNs=None):
        if self.external: topNs = None
        sep = {'C++' : '::'}.get(lang, '.')
        name = self.name
        if lang == 'C++' and 'c++-name' in self.tags: name = self.tags['c++-name']
        if self.parent: 
            parent = self.parent.fullName(lang, topNs)
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
                    # no way now to evaluate expression and check its alignment, so we 
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
        logging.debug("_calcOffsets: type=%r offset=%s align=%s", self, offset, maxalign)
        self.align = maxalign
        
        # adjust for a padding after last element
        align = ExprVal(self.align)
        if self.align: offset = ( (offset + align - ExprVal(1)) / align ) * align
        
        logging.debug('_calcOffsets: type=%r size = %s', self, self.size)
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
            logging.debug('_calcOffsets: type=%r computed size = %s', self, self.size)

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
                
                if attr.stor_type.variable and attr.shape:
                    logging.warning("Cannot generate _sizeof for type "+self.fullName('C++'))
                    return
                
                meth = attr.stor_type.lookup('_sizeof', Method)
                if not meth:
                    
                    size = ExprVal(attr.stor_type.size, self)
                    if not size.isconst():
                        # attribute of complex type without _sizeof?
                        logging.warning("Cannot generate _sizeof for type "+self.fullName('C++'))
                        return
                else:
                    cfg = ''
                    code = meth.expr.get('C++',"")
                    if _hasconfig(code): cfg = 'cfg'
                    if '{self}' in code or '@self' in code:
                        if attr.isfixed():
                            size = ExprVal("@self."+attr.name+"._sizeof(%s)"%cfg, self)
                        else:
                            size = ExprVal("@self."+attr.accessor.name+"()._sizeof(%s)"%cfg, self)
                    else:
                        size = ExprVal(attr.stor_type.fullName('C++')+"::_sizeof(%s)"%cfg, self)

                if attr.shape: 
                    size = str(size)
                    for dim in attr.shape.dims:
                        size += '*(%s)' % dim
                    size = ExprVal(size, self)
                expr += size

            # adjust for a padding after last element
            align = ExprVal(self.align)
            if self.align: expr = ( (expr + align - ExprVal(1)) / align ) * align

            expr = str(expr)
            logging.debug("_genSizeof: expr=%s", expr)

            needCfg = _hasconfig(expr)
            static = '{self}' not in expr and '@self' not in expr

        type = self.lookup('uint32_t', Type)
        tags = dict(inline=None)
        meth = Method('_sizeof', type=type, parent=self,
                      expr={"C++": expr}, tags=tags, static=static)

    def h5schema(self, schema_version):
        '''find a schema for a given type and schema version'''
        schemas = [sch for sch in self.h5schemas if sch.version == schema_version] + [None]
        return schemas[0]

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
