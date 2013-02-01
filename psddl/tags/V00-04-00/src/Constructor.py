#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Constructor...
#
#------------------------------------------------------------------------

"""Class describing type's constructor.


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

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl.Attribute import Attribute

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class Constructor ( object ) :
    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, parent, **kw ) :
        
        self.parent = parent

        self._args = kw.get('args', [])
        self.attr_init = kw.get('attr_init', {})
        self.comment = kw.get('comment')
        self.access = kw.get('access', "public")
        self.tags = kw.get('tags', {}).copy()
        
        self.parent.ctors.append(self)

        self._cargs = None

    #-------------------
    #  Public methods --
    #-------------------

    @property
    def args(self):
        '''
        Get the list of constructor arguments.
        
        Returns the lsit of triplets, each triplet has 
        '''
        
        # find attribute for a given name
        def name2attr(name, type):
            
            if not name: return None
            
            # try attribute name
            attr = type.localName(name)
            if isinstance(attr, Attribute): return attr
            
            # look for bitfield names also
            for attr in type.attributes():
                for bf in attr.bitfields:
                    if bf.name == name: return bf
                    
            raise ValueError('No attribute or bitfield with name %s defined in type %s' % (name, type.name))


        if self._cargs is not None: return self._cargs
        
        # build a list of arguments to ctor
        if not self._args :
            
            self._cargs = []
            
            if 'auto' in self.tags:
                # make one argument per type attribute
                for attr in self.parent.attributes():
                    if attr.bitfields:
                        for bf in attr.bitfields:
                            if bf.accessor:
                                name = "arg_bf_"+bf.name
                                btype = bf.type
                                dest = bf
                                self._cargs.append((name, btype, dest))
                    else:
                        name = "arg_"+attr.name
                        atype = attr.type
                        dest = attr
                        self._cargs.append((name, atype, dest))
        
        else:
            
            # convert destination names to attribute objects
            self._cargs = [(name, atype, name2attr(dest, self.parent)) for name, atype, dest in self._args]

        return self._cargs
 

    def __str__(self):
        return "<%s(%s)>" % (self.parent.name, self.args)

    def __repr__(self):
        return "<%s(%s)>" % (self.parent.name, self.__dict__)

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
