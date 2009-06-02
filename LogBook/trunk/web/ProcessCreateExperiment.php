<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new experiment
 * in the database.
 */
if(isset($_POST['name']))
    $name = $_POST['name'];
else
    die( "no valid experiment name" );

if(isset($_POST['begin_time'])) {
    $begin_time = LogBookTime::parse($_POST['begin_time']);
    if(is_null($begin_time))
        die("begin time has invalid format");
} else
    die( "no begin time for experiment" );

if(isset($_POST['end_time'])) {
    $end_time = $_POST['end_time'];
    if($end_time=='')
        $end_time=null;
    else {
        $end_time = LogBookTime::parse($_POST['end_time']);
        if(is_null($end_time))
            die("end time has invalid format");
    }
} else
    die( "no end time for experiment" );

/* Proceed with the operation
 */
try {

    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->create_experiment(
        $name, $begin_time, $end_time );

    $logbook->commit();

    /* Redirect to another page to see all experiments
     */
    header( 'Location: ListExperiments.php' ) ;

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>
