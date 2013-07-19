<?php

require_once 'dataportal/dataportal.inc.php' ;
require_once 'shiftmgr/shiftmgr.inc.php' ;
require_once 'lusitime/lusitime.inc.php' ;

use LusiTime\LusiTime ;
use LusiTime\LusiInterval ;

\DataPortal\ServiceJSON::run_handler('GET', function($SVC) {

    $instr_name       = $SVC->required_str ('instr_name') ;
    $range            = $SVC->required_str ('range') ;
    $range_begin_time = $SVC->optional_time('begin', null) ;
    $range_end_time   = $SVC->optional_time('end',   null) ;

    switch ($range) {
        case 'week' :
            break ;
        case 'month' :
            break ;
        case 'range' :
            break ;
        default :
            $SVC->abort('unknown range requested from the shifts service') ;
    }
    $begin = LusiTime::parse('2013-07-11 09:00:00') ;
    $end   = LusiTime::parse('2013-07-11 21:00:00') ;
    $ival  = new LusiInterval($begin, $end) ;
    $end1  = LusiTime::parse('2013-07-12 09:00:00') ;
    $ival1 = new LusiInterval($end, $end1) ;
    $shifts = array (

        array (
            'id'      => 2 ,
            'begin'   => array('day' => $end ->toStringDay(), 'hm' => $end ->toStringHM(), 'hour' => $end ->hour(), 'minute' => $end ->minute(), 'full' => $end ->toStringShort()) ,
            'end'     => array('day' => $end1->toStringDay(), 'hm' => $end1->toStringHM(), 'hour' => $end1->hour(), 'minute' => $end1->minute(), 'full' => $end1->toStringShort()) ,
            'duration'     => $ival1->toStringHM() ,
            'duration_min' => $ival1->toMinutes() ,
            'stopper' => 20.0 ,
            'door'    => 80.0 ,
            'area'    => array (
                'FEL' => array (
                    'problem'       => 0 ,
                    'time_down_min' => 0 ,
                    'comments'      => ''
                ) ,
                'BMLN' => array (
                    'problem'       => 0 ,
                    'time_down_min' => 0 ,
                    'comments'      => ''
                ) ,
                'CTRL' => array (
                    'problem'       => 1 ,
                    'time_down_min' => 20 ,
                    'comments'      => "EPICS wasn't properly functioning"
                ) ,
                'DAQ' => array (
                    'problem'       => 1 ,
                    'time_down_min' => 10 ,
                    'comments'      => "CSPAD compression was saturating CPU on one of the DAQ machines"
                ) ,
                'LASR' => array (
                    'problem'       => 0 ,
                    'time_down_min' => 0 ,
                    'comments'      => ''
                ) ,
                'HALL' => array (
                    'problem'       => 0 ,
                    'time_down_min' => 0 ,
                    'comments'      => ''
                ) ,
                'OTHR' => array (
                    'problem'       => 1 ,
                    'time_down_min' => 78 ,
                    'comments'      => "Experimentalist swere not paying close attention to what they were doing during the experiment"
                )
            ) ,
            'activity' => array (
                'tuning' => array (
                    'duration_min' => 0 ,
                    'comments'     => ''
                ) ,
                'alignment' => array (
                    'duration_min' => 0 ,
                    'comments'     => ''
                ) ,
                'daq' => array (
                    'duration_min' => 456 ,
                    'comments'     => 'To be calculated automathically (e-log)'
                ) ,
                'access' => array (
                    'duration_min' => 0 ,
                    'comments'     => ''
                ) ,
                'other' => array (
                    'duration_min' => $ival1->toMinutes() - 456 ,
                    'comments'     => ''
                )
            ) ,
            'notes'    => 'Here be general notes on the shift' ,
            'editor'   => 'gapon' ,
            'modified' => LusiTime::now()->toStringShort()
        ) ,

        array (
            'id'       => 1 ,
            'begin'    => array('day' => $begin->toStringDay(), 'hm' => $begin->toStringHM(), 'hour' => $begin->hour(), 'minute' => $begin->minute(), 'full' => $begin->toStringShort()) ,
            'end'      => array('day' => $end  ->toStringDay(), 'hm' => $end  ->toStringHM(), 'hour' => $end  ->hour(), 'minute' => $end  ->minute(), 'full' => $end  ->toStringShort()) ,
            'duration'     => $ival->toStringHM() ,
            'duration_min' => $ival->toMinutes() ,
            'stopper' => 0.0 ,
            'door'    => 0.0 ,
            'area'    => array (
                'FEL' => array (
                    'problem'       => 1 ,
                    'time_down_min' => 10 ,
                    'comments'      => ''
                ) ,
                'BMLN' => array (
                    'problem'       => 1 ,
                    'time_down_min' => 134 ,
                    'comments'      => 'Here be something wrong with beamline'
                ) ,
                'CTRL' => array (
                    'problem'       => 0 ,
                    'time_down_min' => 0 ,
                    'comments'      => ''
                ) ,
                'DAQ' => array (
                    'problem'       => 0 ,
                    'time_down_min' => 0 ,
                    'comments'      => ''
                ) ,
                'LASR' => array (
                    'problem'       => 0 ,
                    'time_down_min' => 0 ,
                    'comments'      => ''
                ) ,
                'HALL' => array (
                    'problem'       => 0 ,
                    'time_down_min' => 0 ,
                    'comments'      => ''
                ) ,
                'OTHR' => array (
                    'problem'       => 0 ,
                    'time_down_min' => 0 ,
                    'comments'      => ''
                )
            ) ,
            'activity' => array (
                'tuning' => array (
                    'duration_min' => 10 ,
                    'comments'     => 'Instrumnent tuneup'
                ) ,
                'alignment' => array (
                    'duration_min' => 0 ,
                    'comments'     => ''
                ) ,
                'daq' => array (
                    'duration_min' => 123 ,
                    'comments'     => 'To be calculated automathically (e-log)'
                ) ,
                'access' => array (
                    'duration_min' => 0 ,
                    'comments'     => ''
                ) ,
                'other' => array (
                    'duration_min' => $ival->toMinutes() - 10 - 123 ,
                    'comments'     => 'Nothing remarkable to be reported'
                )
            ) ,
            'notes'    => 'Here be general notes on the shift' ,
            'editor'   => $SVC->authdb()->authName() ,
            'modified' => LusiTime::now()->toStringShort()
        )
    ) ;
    $SVC->finish(array('shifts' => $shifts));
});

?>
