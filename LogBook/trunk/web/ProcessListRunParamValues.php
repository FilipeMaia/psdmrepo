<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for displaying values of run
 * parameters of an experiment in one of the fgollowing scopes:
 * - all runs of that experiment
 * - the specified (just one) run of the experiment
 */
if(isset( $_POST['num'])) {
    if( 1 != sscanf( trim( $_POST['num'] ), "%ud", $num )) {

        /* No specific run given - then assume all runs
         */
        $num = null;
    }
 } else {
     die( "no run number" );
 }

if( isset( $_POST['experiment_name']))
    $experiment_name = trim( $_POST['experiment_name'] );
else
    die( "no valid experiment name" );

/* Proceed to the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die("no such experiment" );

    if( is_null( $num ))
        $run = null;
    else
        $run = $experiment->find_run_by_num( $num )
            or die( "no such run" );
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
        <h1>All runs of experiment: <?php echo $experiment->name(); ?></h1>
        <?php
        if( is_null( $run )) {
            $runs = $experiment->runs();
            foreach( $runs as $run ) {
                LogBookTestTable::RunVal()->show(
                    $run->values(),
                    '<h2>Run: '.$run->num().'</h2>' );
            }
        } else {
            LogBookTestTable::RunVal()->show(
                $run->values(),
                '<h2>Run: '.$run->num().'</h2>' );
        }
        ?>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>