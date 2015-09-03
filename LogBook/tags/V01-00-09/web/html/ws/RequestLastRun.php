<?php

require_once( 'logbook/logbook.inc.php' );

use LogBook\LogBook;

/* The script will return the last run of the specified experiment. The result
 * is reported as a JSON object. Errors handled by the script are also returnd as
 * JSON objects.
 * 
 * Paameters:
 * 
 *   { <exper_id> | <instr_name> <exper_name> }
 */

require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id   = $SVC->optional_int('exper_id', null) ;
    $instr_name = null ;
    $exper_name = null ;

    if (!$exper_id) {
        $instr_name = strtoupper($SVC->required_str ('instr_name')) ;
        $exper_name =            $SVC->required_str ('exper_name') ;
    }

    $experiment = $SVC->safe_assign (
        $exper_id ?
            $SVC->logbook()->find_experiment_by_id($exper_id) :
            $SVC->logbook()->find_experiment      ($instr_name, $exper_name) ,
        'no such experiemnt found') ;

    $last_run = $experiment->find_last_run() ;
    return array (
        'runs' => is_null($last_run) ?
        array () :
        array (
            array (
                'instr_name'      => $experiment->instrument()->name() ,
                'exper_name'      => $experiment->name() ,
                'exper_id'        => intval($experiment->id()) ,
                'runnum'          => intval($last_run->num()) ,
                'begin_time_unix' => intval($last_run->begin_time()->sec) ,
                'begin_time'      => $last_run->begin_time()->toStringShort() ,
                'end_time_unix'   => is_null($last_run->begin_time()) ? 0  : intval($last_run->begin_time()->sec) ,
                'end_time'        => is_null($last_run->begin_time()) ? '' : $last_run->begin_time()->toStringShort()
            )
        )
    ) ;
}) ;


?>
