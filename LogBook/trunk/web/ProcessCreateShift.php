<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new shift
 * in the database.
 */

print_r($_POST);

$host     = "localhost";
$user     = "gapon";
$password = "";
$database = "logbook";

if(isset($_POST['leader']))
    $leader = $_POST['leader'];
else
    die( "no valid shift leader account" );

if(isset($_POST['experiment_name']))
    $experiment_name = $_POST['experiment_name'];
else
    die( "no valid experiment name" );

if(isset($_POST['begin_time'])) {
    $begin_time = LogBookTime::parse($_POST['begin_time']);
    if(is_null($begin_time))
        die("begin time has invalid format");
} else
    die( "no begin time for run" );

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
    die( "no end time for run" );

$logbook = new LogBook( $host, $user, $password, $database );
$experiment = $logbook->find_experiment_by_name( $experiment_name )
    or die("failed to find the experiment" );
$run = $experiment->create_shift( $leader, $begin_time, $end_time )
    or die("failed to create the shift" );
?>
