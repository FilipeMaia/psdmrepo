import event_process
import logging
import os
import time
import output_html
import markup

class build_html(event_process.event_process):
    def __init__(self):
        self.output = event_process.event_process_output()
        self.reducer_rank = 0
        self.logger = logging.getLogger(__name__+'.build_html')
        self.logger.info(__name__)
        return

    def getsize(self,start_path = '.'):
        tt = 0
        nn = 0
        run = 'r{:04.0f}'.format(self.parent.run)
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                if 'xtc' in f.lower() and run in f.lower():
                    fp = os.path.join(dirpath,f)
                    tt += os.path.getsize(fp)
                    nn += 1
        return nn, tt

    def mk_output_html(self,gathered):
        if self.parent.rank != 0:
            self.logger.info( "mk_output_html rank {:} exiting".format(self.parent.rank) )
            return

        instr, exp = self.parent.exp.split('/')
        thisdir = os.path.join( '/reg/d/psdm/',instr.lower(),exp.lower(),'xtc')
        self.logger.info( "counting files in {:}".format(thisdir) )
        nfile, nbytes = self.getsize(start_path=thisdir)

        self.logger.info( "mk_output_html rank {:} continuing".format(self.parent.rank) )
        self.html = output_html.report(  self.parent.exp, self.parent.run, 
            title='{:} Run {:}'.format(self.parent.exp,self.parent.run),
            css=('css/bootstrap.min.css','jumbotron-narrow.css','css/mine.css'),
            script=('js/ie-emulation-modes-warning.js','js/jquery.min.js','js/toggler.js','js/sticky.js'),
            output_dir=self.parent.output_dir)

        self.html.start_block('Meta Data', id="metadata")  ##############################################################
                                                                                                                        #
        self.html.start_subblock('Data Information',id='datatime')  ##############################################      #
        seconds = self.parent.all_times[-1].seconds()-self.parent.all_times[0].seconds()
        minutes,fracseconds = divmod(seconds,60)
        self.html.page.p('Start Time: {:}<br/>End Time: {:}<br/>Duration: {:} seconds ({:02.0f}:{:02.0f})'.format(                   #      #
            time.ctime( self.parent.all_times[0].seconds()) ,                                                    #      #
            time.ctime(self.parent.all_times[-1].seconds()),                                                     #      #
            seconds, minutes,fracseconds) )                           #      #
        self.html.page.p('Total files: {:}<br/>Total bytes: {:} ({:0.1f} GB)<br/>'.format(nfile,nbytes,nbytes/(1000.**3)))
        self.html.end_subblock()                                    ##############################################      #

                                                                                                                        #
        self.html.start_subblock('Processing Information', id='datainfo')      ###################################      #
        self.html.page.p('Report time: {:}'.format(time.ctime()))                                                #      #
        self.html.page.p('Total events processed: {:} of {:}'.format(                                            #      #
            self.parent.shared['total_processed'], len(self.parent.all_times) ) )                                #      #
        self.html.page.p('Total processors: {:}'.format( self.parent.size ) )                                    #      #
        self.html.page.p('Wall Time: {:0.1f} seconds'.format( time.time() - self.parent.start_time ) )           #      #
        self.html.page.p('CPU Time: {:0.1f} seconds (accuracy ~10%)'.format( self.parent.cpu_time ))             #      #
        #self.html.page.p( "CPU time for final step on rank {:} : {:0.1f} seconds".format(                       #      #
            #self.rank, self.finaltime))                                                                         #      #
        self.html.end_subblock()                                               ###################################      #

        self.html.start_subblock('Access the Data',id='accessdata')
        self.html.page.p('Access this data with ipsdata on a psana node:<br/><pre>~koglin/repo/psdata/trunk/src/ipsdata.sh -e {:} -r {:}</pre>'.format(self.parent.exp.split('/')[-1], self.parent.run))
        self.html.end_subblock()                                    ##############################################      #
                                                                                                                        #
                                                                                                                        #
        for thisep in sorted(gathered,key=lambda kk: kk.in_report_title):                                                                                #
            if thisep['in_report'] == 'meta':                                                                     #
                self.logger.info( 'adding '+ repr(thisep)+ ' to meta section')
                ep = thisep['in_report_title'].replace(' ','_')
                self.html.start_subblock(thisep['in_report_title'],id=ep)                ################# a sub block #########    #
                if 'text' in thisep:
                    if isinstance(thisep['text'],list):
                        for p in thisep['text']:
                            self.html.page.p( p )
                    else:
                        self.html.page.p( thisep['text'] )
                if 'table' in thisep:
                    self.html.page.add( output_html.mk_table( thisep['table'] )() )                           #    #
                if 'figures' in thisep:
                    self.html.start_hidden(ep)                                       ######### the hidden part ##     #    #
                    for img in sorted(thisep['figures']):                                                  #     #    #
                        imgsrc = os.path.join(*thisep['figures'][img]['png'].split('/')[-2:])
                        self.html.page.a( markup.oneliner.img(src=imgsrc,style='width:49%;'), 
                                href=imgsrc )        #     #    #
                    self.html.end_hidden()                                           ############################     #    #
                self.html.end_subblock()                                    #######################################    #
                                                                                                                        #
        self.html.end_block()          ##################################################################################

        self.html.start_block('Detector Data', id="detectordata") ################################## a block ########### 
                                                                                                                       #
        for thisep in sorted(gathered,key=lambda kk: kk.in_report_title):                                                                                #
            if thisep['in_report'] == 'detectors':                                                                     #
                self.logger.info( 'adding '+ repr(thisep)+ ' to detectors section')
                ep = thisep['in_report_title'].replace(' ','_')
                self.html.start_subblock(thisep['in_report_title'],id=ep)                ################# a sub block #########    #
                if 'text' in thisep:
                    if isinstance(thisep['text'],list):
                        for p in thisep['text']:
                            self.html.page.p( p )
                    else:
                        self.html.page.p( thisep['text'] )
                if 'table' in thisep:
                    self.html.page.add( output_html.mk_table( thisep['table'] )() )                           #    #
                if 'figures' in thisep:
                    self.html.start_hidden(ep)                                       ######### the hidden part ##     #    #
                    for img in sorted(thisep['figures']):                                               #     #    #
                        imgsrc = os.path.join(*thisep['figures'][img]['png'].split('/')[-2:])
                        self.html.page.a( markup.oneliner.img(src=imgsrc,style='width:49%;'), 
                                href=imgsrc )        #     #    #
                    self.html.end_hidden()                                           ############################ #    #
                self.html.end_subblock()                                    #######################################    #
                                                                                                                       #
                                                                                                                       #
        self.html.end_block()                                      #####################################################

        self.html.start_block('Analysis', id='analysis')
        for thisep in sorted(gathered,key=lambda kk: kk.in_report_title):                                                                                #
            if thisep['in_report'] == 'analysis':                                                                     #
                self.logger.info( 'adding '+ repr(thisep)+ ' to analysis section')
                ep = thisep['in_report_title'].replace(' ','_')
                self.html.start_subblock(thisep['in_report_title'],id=ep)                ################# a sub block #########    #
                if 'text' in thisep:
                    if isinstance(thisep['text'],list):
                        for p in thisep['text']:
                            self.html.page.p( p )
                    else:
                        self.html.page.p( thisep['text'] )
                if 'table' in thisep:
                    self.html.page.add( output_html.mk_table( thisep['table'] )() )                           #    #
                if 'figures' in thisep:
                    self.html.start_hidden(ep)                                       ######### the hidden part ##     #    #
                    for img in sorted(thisep['figures']):                                               #     #    #
                        imgsrc = os.path.join(*thisep['figures'][img]['png'].split('/')[-2:])
                        self.html.page.a( markup.oneliner.img(src=imgsrc,style='width:49%;'), 
                                href=imgsrc )        #     #    #
                    self.html.end_hidden()                                           ############################ #    #
                self.html.end_subblock()                                    #######################################    #
                                                                                                                       #
                                                                                                                       #
        self.html.end_block()                                      #####################################################

        self.logger.info('done adding results')
        self.html.start_block('Logging', id='logging')   ######################################## a block #############
        self.html.page.p('link to LSF log file.')                                                                     #
        logs = ','.join(['<a href="log_{0:}.log">log {0:}</a> '.format(x) for x in xrange(self.parent.size) ])
        self.html.page.p('Subjob log files: {:}'.format(logs) )
        if len(self.parent.previous_versions) > 0:                                                                    #
            self.html.start_subblock('Previous Versions',id='prevver') ###################### a sub block ########    #
            self.html.start_hidden('prevver')                             ############## hidden part ####        #    #
            self.html.page.table(class_='table table-condensed')                                        #        #    #
            self.html.page.thead()                                                                      #        #    #
            self.html.page.tr()                                                                         #        #    #
            self.html.page.td('link')                                                                   #        #    #
            self.html.page.td('date')                                                                   #        #    #
            self.html.page.tr.close()                                                                   #        #    #
            self.html.page.thead.close()                                                                #        #    #
            self.parent.previous_versions.reverse()                                                     #        #    #
            for a,b in self.parent.previous_versions:                                                   #        #    #
                self.html.page.tr()                                                                     #        #    #
                self.html.page.td(a)                                                                    #        #    #
                self.html.page.td(b)                                                                    #        #    #
                self.html.page.tr.close()                                                               #        #    #
            self.html.page.table.close()                                                                #        #    #
            self.html.end_hidden()                                        ###############################        #    #
                                                                                                                 #    #
            self.html.end_subblock()                                   ###########################################    #
        self.html.end_block()                            ##############################################################
        self.logger.info('done adding past analysis runs')

        # this closes the left column
        self.html.page.div.close()

        self.html.mk_nav()

        self.html._finish_page()

        self.logger.info( "mk_output_html rank {:} outputing file...".format(self.parent.rank) )
        self.html.myprint(tofile=True)
        self.logger.info( "mk_output_html rank {:} outputing file... finished".format(self.parent.rank) )

        return
    def make_report(self,gathered):
        #print gathered
        self.mk_output_html(gathered)
        return

    def endJob(self):
        self.parent.gather_output()
        if self.parent.rank == 0:
            self.make_report(self.parent.gathered_output)
        return

