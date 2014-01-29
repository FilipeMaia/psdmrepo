#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module HddlLex...
#
#------------------------------------------------------------------------

"""Module which defines lexer for HDDL.

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
import re

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from psddl import lex

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

class HddlLex(object):

    # instantiate lexer
    def __init__(self, **kwargs):
        
        kw = dict(reflags=re.DOTALL)
        kw.update(kwargs)
        kw['reflags'] = kw['reflags'] | re.DOTALL

        self.name = kw.get('name', '')
        if 'name' in kw: del kw['name']
        
        self.lexer = lex.lex(module=self, **kw)
        
        self.comments = []  # list of (lexpos, lineno, comment) tuples

    literals = ".,;{}[]()-+=*/:<>?&%!~|"

    # reserved words in a grammar
    reserved = {
        '@package'  : 'PACKAGE',
        '@type'     : 'TYPE',
        '@enum'     : 'ENUM',
        '@const'    : 'CONST',
        '@include'  : 'INCLUDE',
        '@init'     : 'CTOR',
        '@h5schema' : 'H5SCHEMA',
        '@dataset'  : 'H5DS',
        '@attribute': 'H5ATTR',
        'operator'  : 'OPERATOR',
    }

    # List of token names.
    tokens = (
        'PACKAGE',
        'TYPE',
        'ENUM',
        'CONST',
        'INCLUDE',
        'CTOR',
        'H5SCHEMA',
        'H5DS',
        'H5ATTR',
        'OPERATOR',
        'STRING',
        'L2BRACKET', 'R2BRACKET',
        'LRBRACKET',
        'RARROW',
        'LSHIFT', 'RSHIFT',
        'EQ', 'LE', 'GE',
        'NUMBER',
        'IDENTIFIER',
        'CODEBLOCK',
        'SIZE_EXPR',
        'COMMENT',
        'COMMENTCPP',
        'DOCSTRING',
    )

    # Regular expression rules for simple tokens
    t_RARROW = r'->'
    t_LSHIFT = r'<<'
    t_RSHIFT = r'>>'
    t_EQ = r'=='
    t_LE = r'<='
    t_GE = r'>='
    
    # normal comments are returned after stripping delimiters 
    def t_COMMENTCPP(self, t):
        r'///*([^\n]*)'
        t.value = t.value.lstrip('/')
        t.value = t.value.strip()
        # remember comment position for later matching
        self.comments.append((t.lexpos, t.lexer.lineno, t.value))
        return None

    # normal comments are returned after stripping delimiters 
    def t_COMMENT(self, t):
        r'/\*(.*?)\*/'
        lineno = t.lexer.lineno
        t.lexer.lineno += t.value.count("\n")
        t.value = t.value.strip('/')
        t.value = t.value.strip('*')
        t.value = t.value.strip()
        # remember comment position for later matching
        self.comments.append((t.lexpos, lineno, t.value))
        return None

    # doc string 
    def t_DOCSTRING(self, t):
        r'\#.*?\#'
        t.lexer.lineno += t.value.count("\n")
        t.value = t.value[3:-3]
        return t

    # codeblock is anything between  %{ and %} including newlines
    def t_CODEBLOCK(self, t):
        r'@\{(.*?)@}'
        t.lexer.lineno += t.value.count("\n")
        # strip delimiters
        t.value = t.value[2:-2]
        t.value = t.value.strip(' ')
        return t

    # numbers are converted to integers 
    def t_NUMBER(self, t):
        r'[-]?[ \t]*(0x[0-9a-fA-F]+|\d+)'
        # return pair (int, string)
        t.value = (int(t.value, 0), t.value)
        return t

    # quoted string 
    def t_STRING(self, t):
        r'\"([^\\\n]|(\\(.|\n)))*?\"'
        t.lexer.lineno += t.value.count("\n")
        t.value = t.value[1:-1]
        return t

    def t_L2BRACKET(self, t):
        r'\[\['
        return t
    
    def t_R2BRACKET(self, t):
        r'\]\]'
        return t

    def t_LRBRACKET(self, t):
        r'\[\]'
        return t

    def t_SIZE_EXPR(self, t):
        r'\[(.+?)\]'
        t.value = t.value[1:-1]
        return t

    def t_IDENTIFIER(self, t):
        r'[A-Za-z_@][A-Za-z0-9_]*'
        t.type = HddlLex.reserved.get(t.value, 'IDENTIFIER')
        return t
        
    # Define a rule so we can track line numbers
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    # A string containing ignored characters (spaces and tabs)
    t_ignore = ' \t'

    # Error handling rule
    def t_error(self, t):
        raise SyntaxError("{0}:{1}: Unexpected character '{2}'".format(self.name, t.lexer.lineno, t.value[0]))

    # Test it output
    def test(self, data):
        self.lexer.input(data)
        self.lexer.lineno = 1
        while True:
             tok = self.lexer.token()
             if not tok: break
             print tok

#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
