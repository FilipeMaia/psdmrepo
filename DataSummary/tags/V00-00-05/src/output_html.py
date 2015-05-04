import markup
from markup import oneliner as e
import os, sys
import shutil
import logging

from AppUtils.AppDataPath import AppDataPath

html_dir_finder = AppDataPath('DataSummary/html')
html_ref_dir = html_dir_finder.path()

hutchcolor = {
        'AMO': 'blue',
        'CXI': 'red',
        'MEC': 'goldenrod',
        'SXR': 'black',
        'XCS': 'purple',
        'XPP': 'green',
        }


class report:
    def __init__(self,*args,**kwargs):
        self.output_file = 'report.html'
        self.logger = logging.getLogger(__name__+'.output_html')

        self.output_dir = kwargs.get('output_dir','./')
        self.logger.info('output directory is '+self.output_dir)
        if 'output_dir' in kwargs:
            del kwargs['output_dir']
        cssdir =  os.path.join( self.output_dir, 'css')
        jsdir  =  os.path.join( self.output_dir, 'js')
        if not os.path.isdir( self.output_dir ):
            self.logger.info('output directory doesn"t exit, making it.')
            os.mkdir( self.output_dir )
        if not os.path.isdir( cssdir ):
            os.mkdir( cssdir )
        if not os.path.isdir( jsdir ):
            os.mkdir( jsdir )

        self.logger.info('copying files...')
        shutil.copy( os.path.join( html_ref_dir, 'css', 'bootstrap.min.css' ), os.path.join( cssdir) )
        shutil.copy( os.path.join( html_ref_dir, 'mine.css' ), os.path.join( cssdir) )
        shutil.copy( os.path.join( html_ref_dir, 'jumbotron-narrow.css' ), self.output_dir  )
        shutil.copy( os.path.join( html_ref_dir, 'js', 'bootstrap.min.js' ), os.path.join(jsdir) )
        shutil.copy( os.path.join( html_ref_dir, 'jquery.min.js' ), os.path.join(jsdir) )
        shutil.copy( os.path.join( html_ref_dir, 'toggler.js' ), os.path.join(jsdir) )
        shutil.copy( os.path.join( html_ref_dir, 'sticky.js' ), os.path.join(jsdir) )
        self.logger.info('copying files... done.')

        self.sections = []

        self.logger.info('instantiating html page object')
        self.page = markup.page()
        self.page.twotags.append('nav')
        self.page.init(**kwargs)

        self.page.div(class_='container')
        
        self._build_header(*args)
        self._build_jumbotron(*args)

        self.page.div(class_="row marketing")
        self.page.div(class_="col-md-9")


    def _finish_page(self):
        self.page.div.close()
        # footer here?
        self.page.div.close()
        self.logger.info('finishing html page')
        self.logger.info('output file is '+os.path.join(self.output_dir,self.output_file))


    def _build_header(self,*args,**kwargs):
        self.page.div(class_='header')

        self.page.nav(class_='navigation')

        self.page.ul(class_='nav nav-pills pull-right')
        self.page.li(e.a('Top',href='#top'),class_='active')
        self.page.ul.close()
        self.page.nav.close()
        
        hutch = args[0].split('/')[0]

        self.page.h3('{:} Data Summary '.format(hutch) +e.small('{:}, Run {:0.0f}'.format(*args),style='font-family:monospace;')
                ,class_='text-muted',style='color: {:};'.format( hutchcolor.get(hutch, 'gray') ))

        self.page.div.close()

    def _build_jumbotron(self,*args):
        self.page.a('',id='top',class_='anchor')
        self.page.div(class_='jumbotron')
        self.page.h1(args[0],style='font-family:monospace;')
        self.page.p('Run {:0.0f}'.format(args[1]),class_='lead')
        #self.page.p('Other Information Goes Here. Take advantage of tool tips.',style='font-size: 14px;')
        self.page.div.close()

    def start_block(self,title,id=None,class_=""):
        self.sections.append({})
        self.sections[-1]['title'] = title
        self.sections[-1]['id'] = id
        self.sections[-1]['subsections'] = []
        self.page.a('',id=id,class_='anchor')
        self.page.h3(title,class_="text-center bg-primary "+class_, onclick="toggler('{:}group');".format(id))
        self.page.div(id="{:}group".format(id))
        return
        
    def end_block(self):
        self.page.div.close()
        return

    def start_subblock(self,title,id=None):
        self.page.a('',id=id,class_='anchor')
        self.page.div()
        self.page.h4(title)
        self.sections[-1]['subsections'].append({})
        self.sections[-1]['subsections'][-1]['title'] = title
        self.sections[-1]['subsections'][-1]['id'] = id

        return

    def end_subblock(self):
        self.page.div.close()
        return

    def start_hidden(self,group):
        self.page.p(e.a('Toggle more info',class_='btn btn-sm btn-default',onclick="toggler('{:}extra');".format(group)),class_="text-center")
        self.page.div(id='{:}extra'.format(group),class_="myhidden")
        self.page.div()
        return

    def end_hidden(self):
        self.page.div.close()
        self.page.div.close()
        return

    def mk_nav(self):
        self.page.div(class_='col-md-3 blog-sidebar text-muted')
        self.page.div(id_='sticky-anchor') # sticky nav from here: jsfiddle.net/livibetter/HV9HM
        self.page.div.close()
        self.page.div(class_='nav sidebar-module',id_='sticky')
        self.page.h4('Navigation')
        self.page.ul(class_='list-unstyled',role='navigation',style='font-size:90%')
        for sec in self.sections:
            self.page.li( e.a( sec['title'], href='#{:}'.format(sec['id']) ), style="border-bottom: 1px solid #eee; margin-top:8px;")
            for subsec in sec['subsections']:
                self.page.li( e.a( subsec['title'], href='#{:}'.format(subsec['id']) ) )
        self.page.ul.close()
        self.page.div.close()
        self.page.div.close()

    def add_table(self,table):
        self.page.add( table() )

    def myprint(self,tofile=False):
        if tofile:
            outfile = open( os.path.join( self.output_dir, self.output_file ), 'w' )
            outfile.write(self.page(escape=False))
            outfile.close()
        else:
            print self.page

def mk_table(dd):
    pg = markup.page()
    pg.table(class_='table table-condensed')
    for ii, kk in enumerate(sorted(dd)):
        if ii == 0:
            pg.thead()
            pg.tr()
            pg.th('var')
            for jj in sorted(dd[kk]):
                pg.th(jj)
            pg.tr.close()
            pg.thead.close()
        pg.tr()
        pg.td(kk)
        for jj in sorted(dd[kk]):
            pg.td( '{:0.4f}'.format(dd[kk][jj]))
        pg.tr.close()
    pg.table.close()
    return(pg)


if __name__ == '__main__':
    myout = report('CXIC0114', 37,
            title="CXIC0114 Run 37 Data Summary",
            css=('css/bootstrap.min.css','jumbotron-narrow.css','css/mine.css'),
            script=('js/ie-emulation-modes-warning.js','js/jquery.min.js','js/toggler.js'),
            output_dir='/reg/neh/home/justing/CXI/cxic0114_run37/')

    myout.start_block('Meta Data', id="metadata")
    myout.start_subblock('Data Time Information',id='datatime')
    myout.page.p('Start Time: some time<br/>End Time: some time later<br/>Duration: time time')
    myout.start_subblock('Data Information', id='datainfo')
    myout.end_block()

    myout.start_block('Detector Data', id="detectordata")
    myout.end_block()

    myout.start_block('Analysis', id='analysis')
    myout.end_block()

    # this closes the left column
    myout.page.div.close()

    myout.mk_nav()

    myout._finish_page()
    myout.myprint(tofile=True)
