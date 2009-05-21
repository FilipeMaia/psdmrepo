<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new run
 * parameter in the database.
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

if(isset($_POST['type']))
    $type = $_POST['type'];
else
    die( "no valid parameter type" );

if(isset($_POST['descr']))
    $descr = $_POST['descr'];
else
    die( "no valid parameter description" );

$logbook = new LogBook( $host, $user, $password, $database );
$experiment = $logbook->find_experiment_by_name( $experiment_name )
    or die("failed to find the experiment" );
$run = $experiment->create_run_param($param, $type, $descr)
    or die("failed to create the run parameter" );
?>
