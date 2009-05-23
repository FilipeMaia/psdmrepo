<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new run
 * in the database.
 */

print_r($_POST);

$host     = "localhost";
$user     = "gapon";
$password = "";
$database = "logbook";

if(isset($_POST['param']))
    $param = $_POST['param'];
else
    die( "no valid parameter name" );

if(isset($_POST['experiment_name']))
    $experiment_name = $_POST['experiment_name'];
else
    die( "no valid experiment name" );

if(isset($_POST['num'])) {
    if( 1 != sscanf( $_POST['num'], "%ud", $num ))
        $num = -1;
 } else {
     die( "no run number" );
 }
 print("num='".$num."'\n");

if(isset($_POST['source']))
    $source = $_POST['source'];
else
    die( "no valid source" );

if(isset($_POST['value']))
    $value = $_POST['value'];
else
    die( "no parameter value" );

$update_allowed =isset($_POST['update_allowed']);


$logbook = new LogBook( $host, $user, $password, $database );
$experiment = $logbook->find_experiment_by_name( $experiment_name )
    or die("failed to find the experiment" );

/* Find the run if the positive number is given. If not - then look
 * for the latest run.
 */
if( $num < 0) {
    $run = $experiment->find_last_run()
        or die("failed to find the last run of the experiment" );
} else {
    $run = $experiment->find_run_by_num( $num )
        or die("failed to find the specified run of the experiment" );
}
$run->set_param_value( $param, $value, $source, LogBookTime::now(), $update_allowed )
    or die("failed to set a value of the run parameter" );
?>
