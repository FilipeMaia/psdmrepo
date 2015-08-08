<?php

/*
 * Report file migration status
 * 
 * For complete documentation see JIRA ticket:
 * https://jira.slac.stanford.edu/browse/PSDH-35
 *
 */
require_once 'dataportal/dataportal.inc.php' ;
require_once 'sysmon/sysmon.inc.php' ;

use \SysMon\SysMonFileMigrDelays ;

/**
 * The option parser is decoupled from the service handler for better
 * code structure.
 *
 * @param type $SVC
 * @return \stdClass
 */
function parse_options ($SVC) {

    $opt = new \stdClass ;

    ////////////////////////////////////////////////////////////////////
    // Parameters representing the general scope of the request.
    //
    // - the experiment-specific scope can be narrowsed down to
    //   a sub-range of runs
    //
    // - the instrument scope (or a lack of that) can be narrowed down
    //   to a time interval when files were created/registered.

    $opt->exper_id = $SVC->optional_int('exper_id', null) ;
    if ($opt->exper_id) {
        $opt->run_range = $SVC->optional_range( 'run_range', null) ;
    } else {
        $opt->instr_name = $SVC->optional_enum('instr_name' ,
                                               $SVC->regdb()->instrument_names() ,
                                               null ,
                                               array('ignore_case' => true, 'convert' => 'toupper')) ;
        $opt->begin_time = $SVC->optional_time_any('begin_time', null) ;
        $opt->end_time   = $SVC->optional_time_any('end_time', null) ;
        $SVC->assert (
            $opt->begin_time && $opt->end_time ?
                $opt->begin_time->less($opt->end_time) :
                true ,
            'invalid interval: [begin_time,end_time)') ;
    }
    $opt->file_type = $SVC->optional_enum (
        'file_type' ,
         array('xtc', 'hdf5') ,
         null ,
         array('ignore_case' => true, 'convert' => 'tolower')) ;

    ////////////////////////////////////////
    // Migration status filtering options

    $opt->min_delay_sec = $SVC->optional_int('min_delay_sec', 0) ;
    if ($opt->min_delay_sec <= 0) $opt->min_delay_sec = null ;

    $opt->include_complete = $SVC->optional_bool('include_complete', false) ;

    /////////////////////////////////////////////////////
    // Options controlling the behavior of the service

    $opt->max_entries = $SVC->optional_int('max_entries', 0) ;
    if ($opt->max_entries <= 0) $opt->max_entries = null ;

    return $opt;
}


/**
 * Service handler
 * 
 * @param type $SVC
 * @return array - to be serialized into an JSON object
 */
function handler ($SVC) {

    $opt = parse_options($SVC) ;

    $fileitr = SysMonFileMigrDelays::iterator ($SVC, $opt) ;

    // Apply an optimal filtering based on the 'min_delay_sec' parameter
    // if any provided.
    //
    // Also apply the optional result set transation based on
    // the 'max_entries' parameter if any provided

    $num_files = 0 ;
    $files = array () ;
    foreach ($fileitr as $f) {
        if ($opt->min_delay_sec &&
            !($f->DSS2FFB ->begin_delay >= $opt->min_delay_sec ||
              $f->DSS2FFB ->end_delay   >= $opt->min_delay_sec ||
              $f->FFB2ANA ->begin_delay >= $opt->min_delay_sec ||
              $f->FFB2ANA ->end_delay   >= $opt->min_delay_sec ||
              $f->ANA2HPSS->begin_delay >= $opt->min_delay_sec ||
              $f->ANA2HPSS->end_delay   >= $opt->min_delay_sec)) continue ;

        if (!$opt->include_complete &&
            $f->ANA2HPSS->status === 'C') continue ;

        if ($opt->max_entries && $num_files >= $opt->max_entries) break ;

        ++$num_files ;
        array_push($files, $f) ;
    }

    return array (
        'opt'         => $opt ,
        'num_files'   => $num_files ,
        'files'       => $files ,
        'experiments' => $fileitr->experiment_names()
    ) ;
}

\DataPortal\ServiceJSON::run_handler ('GET', 'handler') ;

?>
