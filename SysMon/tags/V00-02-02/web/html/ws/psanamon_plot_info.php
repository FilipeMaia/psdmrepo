<?php

/**
 * This service will return an information record on the specified psanamon
 * plot.
 * 
 * Parameters:
 * 
 *   [<exper_id> | <exper> <instr>] <name> [<attach_data>]
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $exper_id = $SVC->optional_int('exper_id', null) ;
    if (!$exper_id) {
        $exper_name = $SVC->required_str('exper_name') ;
        $instr_name = strtoupper($SVC->required_str('instr_name')) ;
    }
    $name        = $SVC->required_str ('name') ;
    $attach_data = $SVC->optional_flag('attach_data', false) ;
    $attach_data = true ;

    if (!$exper_id) {
        $experiment = $SVC->regdb()->find_experiment($instr_name, $exper_name) ;
        if (!$experiment) $SVC->abort("no experiment found for '{$instr_name}/{$exper_name}'") ;
        $exper_id = $experiment->id() ;
    }
    $plot = $SVC->sysmon()->find_psanamon_plot($exper_id, $name) ;
    if (!$plot) $SVC->abort("no such plot '{$plot}' found for experiment ID {$exper_id}") ;

    $update_time = $plot->update_time() ;
    return array ('plot' => array (
        'id'          => $plot->id() ,
        'exper_id'    => $plot->exper_id() ,
        'name'        => $plot->name() ,
        'type'        => $plot->type() ,
        'descr'       => $plot->descr() ,
        'update_time' => array (
            'dt'      => $update_time->toStringShort() ,
            'time64'  => $update_time->to64()
        ) ,
        'update_uid'  => $plot->update_uid() ,
        'data_size'   => $plot->data_size() ,
        'data'        => base64_encode($plot->data())
    )) ;
}) ;

?>