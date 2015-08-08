<?php

/*
 * Return the list of experiments and relevant file systems used to store
 * data of those experiments in a scope of an instrument.
 * 
 * The following experiments will be excluded from teh report:
 * - the ones for which the data path is not known
 * - the ones which haven't taken any data
 * 
 * AUTHORIZATION: not required
 */
require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $instr_name = $SVC->required_enum('instr_name' ,
                                      $SVC->regdb()->instrument_names() ,
                                      array('ignore_case' => true, 'convert' => 'toupper')) ;

    $fs_name_filter = $SVC->optional_str('fs',   '') ;
    $year_filter    = $SVC->optional_int('year', null) ;

    // Experiments will be grouppped by the year of their last run
    $experiments = array() ;

    // The years of the last run of _all_ experiments of the specified
    // instrument regardless if they're selected or not.
    $years = array() ;

    // File system paths of _all_ experiments of the specified
    // instrument regardless if they're selected or not.
    $filesystems = array() ;

    foreach ($SVC->logbook()->experiments_for_instrument($instr_name) as $exper) {

        // 'NO DATA PATH' FILTER:
        // - Skip experiments for which a data path hasn't been configured.
        //   Those may haven't taken any data yet.

        $param = $exper->regdb_experiment()->find_param_by_name('DATA_PATH') ;
        if (!$param) continue ;
        $fs = is_null($param) ? '' : $param->value() ;
        if ($fs === '') continue ;

        // 'NO DATA TAKEN' FILTER:
        // - Skip experiments which haven't taken any data (runs)

        $run = $exper->find_last_run() ;
        if (!$run) continue ;

        // Populate all filesystem and year -indexed arrays and dictionaries
        // filtering out experiments according to the input parameters.

        $y = $run->begin_time()->year() ;

        if (!in_array($y,  $years))       array_push($years,       $y) ;
        if (!in_array($fs, $filesystems)) array_push($filesystems, $fs) ;

        // 'FILESYSTEM' INPUT PARAMETER FILTER:
        // - Skip experiments which are not configured to store data on
        //   the specified file system (if provided).

        if (($fs_name_filter !== '') && ($fs_name_filter !== $fs)) continue ;

        // 'EXPERIMENT YEAR' INPUT PARAMETER FILTER:
        // - Skip experiments which didn't take their last run in
        //   the specified year (if provided).

        if ($year_filter && ($year_filter !== $y)) continue ;

        // PASSED

        if (!array_key_exists($y, $experiments)) $experiments[$y] = array() ;

        array_push($experiments[$y], array (
            'id'   => $exper->id() ,
            'name' => $exper->name() ,
            'fs'   => $fs ,
            'last_run' => array (
                'num' => $run->num() ,
                'day' => $run->begin_time()->toStringDay()
            )
        )) ;
    }
    sort($filesystems) ;
    rsort($years) ;
    
    // Sort experiments in each year group by the a when
    // the last run of an experiment was taken. Return the list
    // in the reversed order (most recent experiment always comes first).
    foreach ($experiments as $year => $experiments_year) {
        usort(
            $experiments_year ,
            function ($a, $b) {
                // NOTE: the '-' sign will reverse thevorder in which
                //       the elemnets are sorted.
                return -strcmp($a['last_run']['day'], $b['last_run']['day']) ;
            }
        ) ;
        $experiments[$year] = $experiments_year ;
    }

    return array (
        'experiments' => $experiments ,
        'years'       => $years ,
        'filesystems' => $filesystems
    ) ;
}) ;
?>