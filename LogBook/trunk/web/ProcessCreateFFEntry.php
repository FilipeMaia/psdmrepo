<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for creating a new free-form
 * entry in the database.
 */
$parent_entry_id = null;
if( isset($_POST['parent_entry_id'])) {
    $str = trim( $_POST['parent_entry_id'] );
    if( $str != '' && ( 1 != sscanf( $str, "%ud", $parent_entry_id )))
        die( "not a number at parent entry identifier" );
} else
    die( "no parent entry id" );

if( isset($_POST['experiment_name']))
    $experiment_name = $_POST['experiment_name'];
else
    die( "no valid experiment name" );

if( isset($_POST['relevance_time'])) {
    $relevance_time = LogBookTime::parse($_POST['relevance_time']);
    if( is_null( $relevance_time ))
        die("relevance time has invalid format");
} else
    die( "no relevance time for shift" );

if( isset($_POST['author']))
    $author = $_POST['author'];
else
    die( "no valid author account" );

if( isset($_POST['content_type']))
    $content_type = $_POST['content_type'];
else
    die( "no valid content type" );

if( isset($_POST['content']))
    $content = $_POST['content'];
else
    die( "no valid content" );

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    $experiment = $logbook->find_experiment_by_name( $experiment_name )
        or die("no such experiment" );

    if( is_null($parent_entry_id))
        $entry = $experiment->create_entry(
            $relevance_time, $author, $content_type, $content );
    else {
        $parent_entry = $experiment->find_entry_by_id( $parent_entry_id )
            or die( "no such entry" );
        $entry = $parent_entry->create_child(
            $author, $content_type, $content );
    }
?>
<!--
The page for reporting the information about all shifts of the experiment.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Newely created shift</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <!------------------------------>
        <h1>Free-form entry which just has been created</h1>
        <h2><?php echo $experiment->name(); ?></h2>
        <?php
        LogBookTestTable::Entry( "table_4" )->show( array( $entry ));
        ?>
    </body>
</html>
<?php

    $logbook->commit();

} catch( LogBookException $e ) {
    print $e->toHtml();
}
?>