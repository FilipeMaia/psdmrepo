#!/usr/bin/env python

#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module Template...
#
#------------------------------------------------------------------------
"""
@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$

#----------------------------------

import os
import sys
import zipfile
import xml.etree.ElementTree as ET

#----------------------------------

def convert_xlsx_to_text(ifname, ofname='metrology.txt', print_bits=0):
    """1) unzip xlsx file,
       2) loop over sheets and extract text,
       3) saves extracted text in output file for cspad or files for cspad2x2
       @param ifname      string, input file name
       @param ofname      string, output file name (or basename in case of many files for cspad2x2)
       @param print_bits  int, bit-word for verbosity control; 1-input file name, 2-output file(s) name, 4-text
     """
    arc= zipfile.ZipFile( ifname, "r" )
    #member = arc.getinfo("xl/sharedStrings.xml")
    #arc.extract( member )

    if print_bits & 1 : print 'Input file: %s' % ifname

    list_ofnames = []
    dic_num_txt = {}

    for member in arc.infolist():
        #arc.extract(member)
        fname = member.filename
        #print fname
    
        if fname.startswith("xl/worksheets/sheet") and fname.endswith('.xml'):
            #arc.extract(member)
            #print fname

            npoints, num, txt = get_single_worksheet(arc, fname)

            if npoints == 8 : # This is cspad2x2
                root, ext = os.path.splitext(ofname)          
                fname = '%s-%d%s' % (root, num, ext) 
                if print_bits & 2 : print 'Output file: %s' % fname
                if print_bits & 4 : print txt
                save_textfile(txt, fname)
                list_ofnames.append(fname)

            if npoints == 32 : # This is cspad
                dic_num_txt[num-1] = txt
                #print txt


    if dic_num_txt != {} :
        txt_tot = ''
        for quad in range(4) :
           txt_tot += dic_num_txt[quad] + '\n'

        if print_bits & 2 : print 'Output file: %s' % ofname
        if print_bits & 4 : print txt_tot
        
        save_textfile(txt_tot, ofname)
        list_ofnames.append(ofname)

    return list_ofnames 


#----------------------------------

def get_str_number_from_fname(fname):    # ex: fname = 'xl/worksheets/sheet1.xml'
    """Returns the sheet number as a string"""
    bname = os.path.basename(fname)  # ex: bname = 'sheet1.xml'
    name  = bname.rstrip('.xml')     # ex:  name = 'sheet1'
    str_num = name[5:]
    #print 'fname, bname, name, str_num :', fname, bname, name, str_num
    return str_num

#----------------------------------

def get_single_worksheet(arc, fname):
    """Parses single xlsx sheet and returns (int) number of found points, sheet (int) number, and metrology text.
    """
    str_num = get_str_number_from_fname(fname)
   
    #member = arc.getinfo(fname)
    #arc.extract( member )
    #print 'Open file %s' % fname 
   
    zef = arc.open(fname, 'r')

    #print_open_file(zef)
    #sys.exit ( "End of test" )

    #print zef.read()

    #root = ET.fromstring(zef.read())
    tree = ET.parse ( zef )
    root = tree.getroot()

    prefix = root.tag.rstrip('worksheet') # prefix = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'

    #print 'root.tag    : ', root.tag.rsplit('worksheet') 
    #print 'root.attrib : ', root.attrib 
    #print 'prefix      : ', prefix 

    #for child in root:
    #    print child.tag, child.attrib, child.text

    #for el in root.iter():
    #for el in root.iter(prefix+'v'):
    #for el in root.iter(prefix+'col'):


    txt = ''
    point = None
    
    for el_row in root.iter( prefix+'row' ):

        str_row = el_row.attrib['r']
        #print 'raw ', str_row, el_row.tag, el_row.attrib, el_row.text

        X, Y, Z = None, None, None

        for el_col in el_row:
            if el_col.attrib['r'] == 'A'+str_row : X = find_element_text(el_col, prefix+'v') # el_col.find(prefix+'v').text
            if el_col.attrib['r'] == 'B'+str_row : Y = find_element_text(el_col, prefix+'v') # el_col.find(prefix+'v').text
            if el_col.attrib['r'] == 'C'+str_row : Z = find_element_text(el_col, prefix+'v') # el_col.find(prefix+'v').text

        if X is not None and Y is not None and Z is not None :

            if point is not None and point != 32 :
                point += 1
                #print '    point  X, Y, Z : %2s %10s  %10s  %10s' % (point, X, Y, Z)
                txt += '%2s %7s %7s %7s\n' % (point, X, Y, Z)

            if point is None : point = 0 # skip 1st record, that is a title

    # Add header 

    num = int(str_num)

    if point == 8 : # This is cspad2x2
       txt = 'Point    X       Y       Z\n' + txt

    if point == 32 : # This is cspad
       txt = 'Quad %d \nPoint    X       Y        Z\n' % (num-1) + txt

    return point, num, txt

#----------------------------------

def find_element_text(el, tag) :
    """Return value text for element with given tag or None
    """
    found_el = el.find(tag)
    if found_el is None : return None
    return found_el.text

#----------------------------------

def print_open_file(f) :
    """print per-string content from open file
    """
    for line in f : print line

#----------------------------------

def save_textfile(text, path, mode='w') :
    """Saves text in file specified by path. mode: 'w'-write, 'a'-append 
    """
    #print 'Save file %s' % path
    f=open(path,mode)
    f.write(text)
    f.close() 

#----------------------------------
#----------------------------------
#----------------------------------
#----------------------------------

def get_input_parameters() :
    """Parses command line parameters and returns ifname, ofname
    """
    ifname, ofname = 'metrology.xlsx', 'metrology.txt'

    nargs = len(sys.argv)
    #print 'sys.argv[0]: ', sys.argv[1]
    #print 'nargs: ', nargs

    if   nargs == 1 : return ifname, ofname

    if not os.path.exists(sys.argv[1]) :
        print 'Input file %s DOES NOT EXIST!' % (sys.argv[1])
        sys.exit ()

    if   nargs == 2 : return sys.argv[1], ofname
    elif nargs == 3 : return sys.argv[1], sys.argv[2]

    print 'Command line for %s has a WRONG number of arguments: nargs=%d' % (sys.argv[0], nargs)
    sys.exit ()

#----------------------------------
          
def main() :
    ifname, ofname = get_input_parameters()
    list_ofnames = convert_xlsx_to_text(ifname, ofname, print_bits=7)
    #list_ofnames = convert_xlsx_to_text('metrology.xlsx', 'metrology.txt')

#----------------------------------

if __name__ == "__main__" :
    main()
    sys.exit ( "End of test" )

#----------------------------------

