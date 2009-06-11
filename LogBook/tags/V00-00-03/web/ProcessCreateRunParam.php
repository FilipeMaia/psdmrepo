<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new run
 * parameter in the database.
 */
if( isset( $_POST['param'] )) {
    $param = trim( $_POST['param'] );
    if( $param == '' )
        die( "run parameter name can't be empty" );
} else
    die( "no valid parameter name" );

if( isset( $_POST['experiment_name'] )) {
    $experiment_name = $_POST['experiment_name'];
    if( $experiment_name == '' )
        die( "experiment name can't be empty" );
} else
    die( "no valid experiment name" );

if( isset( $_POST['type'] )) {
    $type = trim( $_POST['type'] );
    if( $type == '' )
        die( "parameter type can't be empty" );
} else
    die( "no valid parameter type" );

if( isset( $_POST['descr']))
    $descr = trim( $_POST['descr'] );
else
    die( "no valid parameter description" );

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die("no such experiment" );

    $param = $experiment->create_run_param( $param, $type, $descr );
?>
<!--
The page for reporting the information about all summary run parameters
of the experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Run parameters of the experiment</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <!------------------------------>
        <h1>Summary run parameters of the experiment</h1>
        <h2><?php echo $experiment->name(); ?></h2>
        <?php
        LogBookTestTable::RunParam()->show( $experiment->run_params());
        ?>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>