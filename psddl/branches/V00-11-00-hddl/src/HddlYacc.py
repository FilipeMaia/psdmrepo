#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module HddlYacc...
#
#------------------------------------------------------------------------

"""Syntax definition for HDDL parser.
This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andy Salnikov
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
from psddl import yacc
from psddl.Package import Package
from psddl.HddlLex import HddlLex

#----------------------------------
# Local non-exported definitions --
#----------------------------------

class QID(object):
    '''class representing quailified identifier like A.B.C'''
    def __init__(self, identifier):
        self.id = [identifier]
    def append(self, identifier):
        self.id.append(identifier)
    def __str__(self):
        return '.'.join(self.id)
    def __repr__(self):
        return '.'.join(self.id)


#------------------------
# Exported definitions --
#------------------------


#---------------------
#  Class definition --
#---------------------
class HddlYacc ( object ) :

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, model=None, **kwargs):
        
        self.model = model or Package('')        
        self.parser = yacc.yacc(module=self, **kwargs)

    def parse(self, input, name):

        self.name = name

        # make lexer
        lexer = HddlLex(name=name)

        lexer.lexer.lineno = 1
        tree = self.parser.parse(input=input, lexer=lexer.lexer, tracking=1)
        
        return tree

    def _makepos(self, p, n=0, m=-1):
        ''' capture position as 2-tuple ((fromline, toline), (fromlexpos, tolexpos)) '''
        pline = p.linespan(n)
        plex = p.lexspan(n)
        if m > n:
            pline = (pline[0], p.linespan(m)[1])
            plex = (plex[0], p.lexspan(m)[1])
        return pline, plex
            

    tokens = [t for t in HddlLex.tokens if t not in ('COMMENT', 'COMMENTCPP')] 

    precedence = (
        ('left', '|'),
        ('left', '^'),
        ('left', '&'),
        ('left', 'LSHIFT', 'RSHIFT'),
        ('left', '+', '-'),
        ('left', '*', '/', '%'),
        ('right', 'UPLUS', 'UMINUS', 'UCOMPL'),  # unary plus minus and complement
    )

    # this is the starting rule
    def p_input(self, p):
        ''' input : includes declaration_seq
        '''
        p[0] = dict(includes=p[1], declarations=p[2], name=self.name, tags=[])
   
    # quilaified identifier: set of identifiers separated by dots, makes QID object
    def p_qidentifier(self, p):
        ''' qidentifier : qidentifier '.' IDENTIFIER
                        | IDENTIFIER
        '''
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        else:
            p[0] = QID(p[1])


    # returns the list of includes each item in the list is a dict with items:
    #   1. name: the name of the include
    #   2. tags: list of tags, each tag is a dict with items:
    #        1. name: name of tag,
    #        2. args: either None or tuple of tag arguments
    #        3. pos: (file, lineno) of the statement
    #   3. pos: (file, lineno) of the statement
    def p_includes(self, p):
        ''' includes : includes INCLUDE STRING tags ';'
                     | empty
        '''
        if len(p) == 2:
            p[0] = []
        else:
            p[0] = p[1] + [dict(name=p[3], tags=p[4], pos=self._makepos(p, 2, 5))]

    # makes list of tags (tag is a 2-tuple)
    def p_tags(self, p):
        ''' tags : tags L2BRACKET taglist R2BRACKET
                 | tags DOCSTRING
                 | empty
        '''
        if len(p) == 2:
            p[0] = []
        elif len(p) == 3:
            p[0] = p[1] + [dict(name="doc", args=(p[2],), pos=self._makepos(p, 2))]
        else:
            p[0] = p[1] + p[3]

    # makes list of tags (tag is a 2-tuple)
    def p_taglist(self, p):
        ''' taglist : taglist ',' tag
                    | tag
        '''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        else:
            p[0] = [p[1]]

    # makes dict with items:
    #    name:  name of tag
    #    args: either None or tuple of tag arguments
    #    pos: (file, lineno) of the statement
    def p_tag(self, p):
        ''' tag : IDENTIFIER '(' tagargs ')'
                | IDENTIFIER
        '''
        if len(p) == 2:
            p[0] = dict(name=p[1], args=None, pos=self._makepos(p))
        else:
            p[0] = dict(name=p[1], args=p[3], pos=self._makepos(p))

    # makes tuple of args
    def p_tagargs(self, p):
        ''' tagargs : tagargs ',' tagarg
                    | tagarg
        '''
        if len(p) == 2:
            p[0] = (p[1],)
        else:
            p[0] = p[1] + (p[3],)

    # just return whatever object it is (number, string or QID)
    def p_tagarg(self, p):
        ''' tagarg : NUMBER
                | qidentifier
                | STRING
        '''
        p[0] = p[1]

    # declaration_seq returns list of dicts
    def p_declaration_seq(self, p):
        ''' declaration_seq : declaration_seq package
                            | declaration_seq type
                            | declaration_seq enum
                            | declaration_seq constdecl
                            | declaration_seq h5schema
                            | empty
        '''
        if len(p) == 2:
            p[0] = []
        else:
            p[0] = p[1] + [p[2]]

    # constant declaration, returns dict with keys:
    #    decl:   "const"
    #    name:   name of the const
    #    type:   type name of the const
    #    value:  expression 
    #    pos:    declaration location 
    def p_const(self, p):
        ''' constdecl : tags CONST IDENTIFIER IDENTIFIER '=' const_expr tags ';'
        '''
        tags = p[1] + p[7]
        p[0] = dict(decl="const", name=p[4], type=p[3], value=p[6], tags=tags, pos=self._makepos(p))

    # constant expression, which is basically a simple math with numbers and identifiers and does 
    # not involve function calls
    def p_const_expr_binary(self, p):
        ''' const_expr : const_expr '+' const_expr
                       | const_expr '-' const_expr
                       | const_expr '*' const_expr
                       | const_expr '/' const_expr
                       | const_expr '%' const_expr
                       | const_expr LSHIFT const_expr
                       | const_expr RSHIFT const_expr
                       | const_expr '&' const_expr
                       | const_expr '^' const_expr
                       | const_expr '|' const_expr
        '''
        p[0] = dict(decl="const_expr", left=p[1], right=p[3], op=p[2], pos=self._makepos(p))

    # unary expressions
    def p_const_expr_unary(self, p):
        ''' const_expr : '+' const_expr %prec UPLUS
                       | '-' const_expr %prec UMINUS
                       | '~' const_expr %prec UCOMPL
        '''
        p[0] = dict(decl="const_expr", left=None, right=p[2], op=p[1], pos=self._makepos(p))
        
    # simple expression is a number of qualified identifier (identifier is supposed to be a constant)
    def p_const_expr_simple(self, p):
        ''' const_expr : qidentifier
                       | NUMBER
        '''
        p[0] = dict(decl="const_expr", left=None, right=p[1], op=None, pos=self._makepos(p))

    # simple expression is a number of qualified identifier (identifier is supposed to be a constant)
    def p_const_expr_group(self, p):
        ''' const_expr : '(' const_expr ')'
        '''
        p[0] = dict(decl="const_expr", left=None, right=p[2], op='(', pos=self._makepos(p))

    # enum declaration, returns dict with keys:
    #    decl:   "enum"
    #    name:   name of the enum
    #    type:   name of the base type or None
    #    constants:  list of enum constant definitions
    #    pos:    declaration location 
    def p_enum(self, p):
        ''' enum : tags ENUM IDENTIFIER tags '{' enum_constants '}'
                 | tags ENUM IDENTIFIER tags '{' enum_constants ',' '}'
                 | tags ENUM IDENTIFIER '(' IDENTIFIER ')' tags '{' enum_constants '}'
                 | tags ENUM IDENTIFIER '(' IDENTIFIER ')' tags '{' enum_constants ',' '}'
        '''
        if len(p) <= 9:
            tags = p[1] + p[4]
            p[0] = dict(decl="enum", name=p[3], type=None, constants=p[6], tags=tags, pos=self._makepos(p))
        else:
            tags = p[1] + p[7]
            p[0] = dict(decl="enum", name=p[3], type=p[5], constants=p[9], tags=tags, pos=self._makepos(p))

    def p_enum_constants(self, p):
        ''' enum_constants : enum_constants ',' enum_const
                           | enum_const
        '''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    # one enum constant, returns dict(name=..., value=..., pos), value may be None
    def p_enum_const(self, p):
        ''' enum_const : tags IDENTIFIER '=' NUMBER tags
                       | tags IDENTIFIER tags
        '''
        if len(p) == 4:
            tags = p[1] + p[3]
            p[0] = dict(name=p[2], value=None, tags=tags, pos=self._makepos(p))
        else:
            tags = p[1] + p[5]
            p[0] = dict(name=p[2], value=p[4], tags=tags, pos=self._makepos(p))


    # package declaration will make a dict with the keys:
    #    decl: 'package'
    #    name: package name
    #    declarations: list of enclosed declarations
    #    tags: list of associated tags
    #    pos: location of a declaration
    def p_package(self, p):
        ''' package : tags PACKAGE IDENTIFIER tags '{' declaration_seq '}'
        '''
        tags = p[1] + p[4]
        p[0] = dict(decl="package", name=p[3], declarations=p[6], tags=tags, pos=self._makepos(p))

    # type declaration, returns dict with the keys:
    #    decl: 'type'
    #    name: type name
    #    base: base type (QID) or None
    #    declarations: list of enclosed declarations
    #    tags: list of associated tags
    #    pos: location of a declaration
    def p_type(self, p):
        ''' type : tags TYPE IDENTIFIER tags '{' member_declaration_seq '}'
        '''
        tags = p[1] + p[4]
        p[0] = dict(decl="type", name=p[3], base=None, declarations=p[6], tags=tags, pos=self._makepos(p))

    def p_type_base(self, p):
        ''' type : tags TYPE IDENTIFIER '(' qidentifier ')' tags '{' member_declaration_seq '}'
        '''
        tags = p[1] + p[7]
        p[0] = dict(decl="type", name=p[3], base=p[5], declarations=p[9], tags=tags, pos=self._makepos(p))

    # member_declaration_seq returns list of dicts
    def p_member_declaration_seq(self, p):
        ''' member_declaration_seq : member_declaration_seq enum
                                   | member_declaration_seq constdecl
                                   | member_declaration_seq member
                                   | member_declaration_seq method
                                   | member_declaration_seq ctor
                                   | empty
        '''
        if len(p) == 2:
            p[0] = []
        else:
            p[0] = p[1] + [p[2]]
    
    # declaration of a type member returns a dict with keys:
    #    decl:       "member"
    #    name:       member name
    #    type:       type name (QID)
    #    shape:      list oof dimension expressions or None
    #    method:     name of accessor method or none
    #    bitfields:  list of bitfield definitions or None 
    #    tags:       list of associated tags
    #    pos:        location of a declaration
    def p_member(self, p):
        ''' member : tags qidentifier IDENTIFIER tags ';'
        '''
        tags = p[1] + p[4]
        p[0] = dict(decl='member', name=p[3], type=p[2], shape=None, method=None, bitfields=None, tags=tags, pos=self._makepos(p))
    
    def p_member_meth(self, p):
        ''' member : tags qidentifier IDENTIFIER RARROW IDENTIFIER tags ';'
        '''
        tags = p[1] + p[6]
        p[0] = dict(decl='member', name=p[3], type=p[2], shape=None, method=p[5], bitfields=None, tags=tags, pos=self._makepos(p))
    
    def p_member_arr(self, p):
        ''' member : tags qidentifier IDENTIFIER arr_shape tags ';'
        '''
        tags = p[1] + p[5]
        p[0] = dict(decl='member', name=p[3], type=p[2], shape=p[4], method=None, bitfields=None, tags=tags, pos=self._makepos(p))
    
    def p_member_arr_meth(self, p):
        ''' member : tags qidentifier IDENTIFIER arr_shape RARROW IDENTIFIER tags ';'
        '''
        tags = p[1] + p[7]
        p[0] = dict(decl='member', name=p[3], type=p[2], shape=p[4], method=p[6], bitfields=None, tags=tags, pos=self._makepos(p))

    def p_member_bf(self, p):
        ''' member : tags qidentifier IDENTIFIER tags '{' bitfields '}'
        '''
        tags = p[1] + p[4]
        p[0] = dict(decl='member', name=p[3], type=p[2], shape=None, method=None, bitfields=p[6], tags=tags, pos=self._makepos(p))
    
    def p_member_bf_meth(self, p):
        ''' member : tags qidentifier IDENTIFIER RARROW IDENTIFIER tags '{' bitfields '}'
        '''
        tags = p[1] + p[6]
        p[0] = dict(decl='member', name=p[3], type=p[2], shape=None, method=p[5], bitfields=p[8], tags=tags, pos=self._makepos(p))
    
    # makes the list of bitfield definitions
    def p_bitfields(self, p):
        ''' bitfields : bitfields bitfield
                      | bitfield
        '''
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = [p[1]]
        
    # bitfield declaration, makes a dict with the keys:
    #    decl:    'bitfield'
    #    name:
    #    type:    type (QID) ir None
    #    size:    number of bits
    #    method:  name of the access method or none
    #    tags:       list of associated tags
    #    pos:        location of a declaration
    def p_bitfield(self, p):
        ''' bitfield : tags qidentifier IDENTIFIER ':' NUMBER tags ';'
        '''
        tags = p[1] + p[6]
        p[0] = dict(decl='bitfield', name=p[3], type=p[2], size=p[5], method=None, tags=tags, pos=self._makepos(p))
        
    def p_bitfield_meth(self, p):
        ''' bitfield : tags qidentifier IDENTIFIER ':' NUMBER RARROW IDENTIFIER tags ';'
        '''
        tags = p[1] + p[8]
        p[0] = dict(decl='bitfield', name=p[3], type=p[2], size=p[5], method=p[7], tags=tags, pos=self._makepos(p))
        
    # array shape, returns the list of expressions (strings)
    def p_arr_shape(self, p):
        ''' arr_shape : arr_shape SIZE_EXPR
                      | SIZE_EXPR
        '''
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = [p[1]]
    
    # convenience terminal stuff, returns None
    def p_empty(self, p):
        'empty :'
        pass

    # declaration of a method returns a dict with keys:
    #    decl:       "method"
    #    name:       method name
    #    type:       type name (QID)
    #    rank:       rank of returned data
    #    args:       list of arguments
    #    bodies:     list of bodies
    #    tags:       list of associated tags
    #    pos:        location of a declaration



    def p_meth_head(self, p):
        ''' method_head : tags qidentifier meth_name '(' method_args ')' tags
        '''
        tags = p[1] + p[7]
        p[0] = dict(decl='method', name=p[3], type=p[2], rank=0, args=p[5], tags=tags, bodies=None, pos=self._makepos(p))
    
    def p_meth_head_rank(self, p):
        ''' method_head : tags qidentifier rank IDENTIFIER '(' method_args ')' tags
        '''
        tags = p[1] + p[8]
        p[0] = dict(decl='method', name=p[4], type=p[2], rank=p[3], args=p[6], tags=tags, bodies=None, pos=self._makepos(p))
    
    # method name
    def p_meth_name(self, p):
        ''' meth_name : IDENTIFIER
                      | OPERATOR LE
                      | OPERATOR GE
                      | OPERATOR EQ
                      | OPERATOR '='
                      | OPERATOR '<'
                      | OPERATOR '>'
        '''
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[1] + p[2]
    
    # method with no bodies, supposedly has [[external]] tags
    def p_method_ext(self, p):
        ''' method : method_head ';'
        '''
        p[0] = p[1]
        p[0]['pos'] = self._makepos(p)

    # method with one or more bodies
    def p_method_body(self, p):
        ''' method : method_head codeblocks
        '''
        p[0] = p[1]
        p[0]['bodies'] = p[2]
        p[0]['pos'] = self._makepos(p)


    # method arguments makes list of argumetns
    def p_method_args(self, p):
        ''' method_args : method_args ',' method_arg
                        | method_arg
                        | 
        '''
        if len(p) == 4:
            p[0] = p[1] + [p[3]]
        elif len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = []

    # method argument, makes a dict with keys:
    #    name:
    #    type:   QID
    #    rank:
    def p_method_arg(self, p):
        ''' method_arg : qidentifier IDENTIFIER
                       | qidentifier rank IDENTIFIER
        '''
        if len(p) == 3:
            p[0] = dict(name=p[2], type=p[1], rank=0)
        else:
            p[0] = dict(name=p[3], type=p[1], rank=p[2])


    # rank is just a rank number, specified as '[][][]' === 3
    def p_rank(self, p):
        ''' rank : rank LRBRACKET
                 | LRBRACKET
        '''
        if len(p) == 3:
            p[0] = p[1] + 1
        else:
            p[0] = 1
            
    # list of method bodies, tags that preceed first CODEBLOCK are consumed by method
    # makes a list of dicts, each dict has keys:
    #    tags:  list of tags
    #    code:  string
    def p_codeblocks(self, p):
        ''' codeblocks : codeblocks ',' tags CODEBLOCK
                       | CODEBLOCK
        '''
        if len(p) == 2:
            p[0] = [dict(tags=[], code=p[1])]
        else:
            p[0] = p[1] + [dict(tags=p[3], code=p[4])]

    # constructor declaration
    #
    def p_ctor(self, p):
        ''' ctor : tags CTOR '(' ctor_args ')' ctor_inits tags ';'
        '''
        tags = p[1] + p[7]
        p[0] = dict(decl='ctor', args=p[4], inits=p[6], tags=tags, pos=self._makepos(p))

    # makes a tuple of constructor arguments
    def p_ctor_args(self, p):
        ''' ctor_args : ctor_args ',' ctor_arg
                      | ctor_arg
                      |
        '''
        if len(p) == 1:
            p[0] = tuple()
        elif len(p) == 2:
            p[0] = (p[1],)
        else:
            p[0] = p[1] + (p[3],)
    
    # single constructor arg, makes a dict with keys:
    #    1. name:  argument name
    #    2. type:  argument type or None
    #    3. rank:  argument rank or None
    #    4. dest:  destination name or None
    #    4. tags:  list of tags
    def p_ctor_arg(self, p):
        ''' ctor_arg : method_arg tags
                     | method_arg RARROW IDENTIFIER tags
                     | IDENTIFIER RARROW IDENTIFIER tags
        '''
        if len(p) == 3:
            p[0] = p[1]
            p[0]['dest'] = None
            p[0]['tags'] = p[2]
        elif isinstance(p[1], dict):
            p[0] = p[1]
            p[0]['dest'] = p[3]
            p[0]['tags'] = p[4]
        else:
            p[0] = dict(name=p[1], type=None, rank=0, dest=p[3], tags=p[4])

    # makes a list of constructor initializers
    def p_ctor_inits(self, p):
        ''' ctor_inits : ctor_inits ',' ctor_init
                       | ctor_init
                       |
        '''
        if len(p) == 1:
            p[0] = []
        elif len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]
    
    # constructor initializer, makes dict with keys:
    #    dest: name of the destination (attribute or bitfield)
    #    expt: expression (see const_expr)
    def p_ctor_init(self, p):
        ''' ctor_init : IDENTIFIER '(' const_expr ')'
        '''
        p[0] = dict(dest=p[1], expr=p[3])


    # hdf5 schema declaration, makes a dict with the keys:
    #    name:     schema (type) name
    #    declarations: list of declarations
    #    tags:     list of tags
    def p_h5schema(self, p):
        '''  h5schema : tags H5SCHEMA IDENTIFIER tags '{' schema_items '}'
        '''
        tags = p[1] + p[4]
        p[0] = dict(decl='h5schema', name=p[3], declarations=p[6], tags=tags, pos=self._makepos(p))

    def p_schema_items(self, p):
        ''' schema_items : schema_items dataset
                         | schema_items enum_remap
                         | empty
        '''
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = []

    # dataset declaration, makes a dict with the keys:
    #    name:         dataset name
    #    type:         type name (QID) or None
    #    attributes:   list of attributes or None
    #    tags:         list of tags
    def p_dataset(self, p):
        ''' dataset : tags H5DS IDENTIFIER ds_shape tags ';'
                    | tags H5DS qidentifier IDENTIFIER ds_shape tags ';'
        '''
        if len(p) == 7:
            tags = p[1] + p[5]
            p[0] = dict(decl='h5ds', name=p[3], type=None, tags=tags, shape=p[4], attributes=None, pos=self._makepos(p))
        elif len(p) == 8:
            tags = p[1] + p[6]
            p[0] = dict(decl='h5ds', name=p[4], type=p[3], tags=tags, shape=p[5], attributes=None, pos=self._makepos(p))

    def p_dataset_a(self, p):
        ''' dataset : tags H5DS IDENTIFIER tags '{' attributes '}'
        '''
        tags = p[1] + p[4]
        p[0] = dict(decl='h5ds', name=p[3], type=None, tags=tags, shape=None, attributes=p[6], pos=self._makepos(p))

    # dataset shape, can return
    #    1. None if there is nothing specified
    #    2. Number if rank specified ([][][] -> 3)
    def p_ds_shape(self, p):
        ''' ds_shape :
                     | rank
        '''
        if len(p) == 1:
            p[0] = None
        else:
            p[0] = p[1]

    def p_attributes(self, p):
        ''' attributes : attributes attribute
                     | empty
        '''
        if len(p) == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = []

    # aqttribute declaration makes a dict with keys:
    #    name:   attribute name
    #    type:   type (QID) or None
    #    tags:   list of tags
    def p_attribute(self, p):
        ''' attribute : tags H5ATTR IDENTIFIER attr_shape tags ';'
                      | tags H5ATTR qidentifier IDENTIFIER attr_shape tags ';'
        '''
        if len(p) == 7:
            tags = p[1] + p[5]
            p[0] = dict(decl='h5attr', name=p[3], type=None, shape=p[4], tags=tags, pos=self._makepos(p))
        else:
            tags = p[1] + p[6]
            p[0] = dict(decl='h5attr', name=p[4], type=p[3], shape=p[5], tags=tags, pos=self._makepos(p))

    # attribute shape, can return
    #    1. None if there is nothing specified
    #    2. Number if rank specified ([][][] -> 3)
    #    3. List of dimension expression strings ([10][10] -> ['10', '10'])
    def p_attr_shape(self, p):
        ''' attr_shape :
                       | rank
                       | arr_shape
        '''
        if len(p) == 1:
            p[0] = None
        else:
            p[0] = p[1]


    # enum remapping declaration returns dict with keys:
    #    name:   enum name
    #    remaps: list of remappings
    def p_enum_remap(self, p):
        ''' enum_remap : tags ENUM IDENTIFIER tags '{' enum_remaps '}'
                       | tags ENUM IDENTIFIER tags '{' enum_remaps ',' '}'
        '''
        tags = p[1] + p[4]
        p[0] = dict(decl='enum_remap', name=p[3], remaps=p[6], tags=tags, pos=self._makepos(p))

    def p_enum_remaps(self, p):
        ''' enum_remaps : enum_remaps ',' enum_map
                        | enum_map
        '''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    # one enum remapping, returns dict(from=..., to=..., pos)
    def p_enum_map(self, p):
        ''' enum_map : IDENTIFIER RARROW IDENTIFIER
        '''
        p[0] = {'from': p[1], 'to': p[3], 'pos': self._makepos(p)}


    # ---------- end of all grammar rules ----------

    # Error rule for syntax errors
    def p_error(self, p):
        if p is None:
            raise SyntaxError("{0}: EOF reached while expecting further input".format(self.name))
        else:
            raise SyntaxError("{0}:{1}: Syntax error near or at '{2}'".format(self.name, p.lineno, p.value))


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
