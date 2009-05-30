<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new run
 * in the database.
 */
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

if(isset($_POST['source']))
    $source = $_POST['source'];
else
    die( "no valid source" );

if(isset($_POST['value']))
    $value = $_POST['value'];
else
    die( "no parameter value" );

$update_allowed =isset($_POST['update_allowed']);

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die( "no such experiment" );

    /* Find the run if the positive number is given. If not - then look
     * for the latest run.
     */
    if( $num < 0) {
        $run = $experiment->find_last_run()
            or die( "no last run in experiment" );
    } else {
        $run = $experiment->find_run_by_num( $num )
            or die( "no such run of the experiment" );
    }
    $run->set_param_value (
        $param, $value, $source, LogBookTime::now(), $update_allowed );
?>
<!--
The page for reporting the information about all summary parameters of the run.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Run parameters</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <!------------------------------>
        <h1>Values of run parameters</h1>
        <h2><?php echo $experiment->name(); ?></h2>
        <h3><?php echo $run->num(); ?></h3>
        <?php
        LogBookTestTable::RunVal()->show( $run->values());
        ?>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>