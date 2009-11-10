<?php

require_once('LogBook/LogBook.inc.php');

/*
 * This script will process a request for closing a run
 * in the database.
 */
if( !LogBookAuth::isAuthenticated()) return;

if( isset( $_POST['num'])) {
    if( 1 != sscanf( trim( $_POST['num'] ), "%ud", $num ))
        $num = null;
 } else {
     die( "no run number" );
 }

if( isset( $_POST['experiment_name'] )) {
    $experiment_name = trim( $_POST['experiment_name'] );
    if( $experiment_name == '' )
        die( "experiment name can't be empty" );
} else
    die( "no valid experiment name" );

if( isset( $_POST['end_time'] )) {
    $end_time = LusiTime::parse( trim( $_POST['end_time'] ));
    if( is_null( $end_time ))
        die( "end time has invalid format" );
} else
    die( "no end time for run" );

/* Proceed to the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die("no such experiment" );

    if( is_null( $num )) $run = $experiment->find_last_run();
    else                 $run = $experiment->find_run_by_num( $num );
    $run->close( $end_time );
?>
<!--
The page for reporting the information about all runs of the experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Newely created run</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <!------------------------------>
        <h1>All runs of an experiment</h1>
        <h2><?php echo $experiment->name(); ?></h2>
        <?php
        LogBookTestTable::Run()->show( $experiment->runs());
        ?>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>