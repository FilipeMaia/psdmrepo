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
header( 'Content-type: application/json' );
header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
    
function report_error ($msg) {
    print json_encode (
        array (
            'status' => 'error',
            'message' => $msg
        )
    );
    exit;
}
function report_result ($result=array()) {
    print json_encode (
        array_merge(
            array (
                'status' => 'success'
            ),
            $result
        )
    );
    exit;
}

/* Parse input parameters
 */
$exper_id   = null ;
$instr_name = null ;
$exper_name = null ;

if (isset( $_GET['exper_id'])) {

    $exper_id = intval(trim($_GET['exper_id'])) ;
    if (!$exper_id)
        report_error ('invalid value of the <exper_id> parameter') ;

} elseif (isset($_GET['instr_name']) && isset($_GET['exper_name'])) {

    $instr_name = strtoupper(trim($_GET['instr_name'])) ;
    $exper_name =            trim($_GET['exper_name']) ;

} else {
    report_error ('no experiment specification found amoung parameters');
}

try {
    LogBook::instance()->begin() ;

    $experiment = (is_null($exper_id) ?
        LogBook::instance()->find_experiment ($instr_name, $exper_name) :
        LogBook::instance()->find_experiment_by_id ($exper_id)) or report_error ('no such experiment') ;

    $last_run = $experiment->find_last_run() ;
    report_result (
        array (
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
        )
    ) ;
    LogBook::instance()->commit() ;

} catch( Exception $e ) { report_error( $e.'<pre>'.print_r( $e->getTrace(), true ).'</pre>' ); }

?>
