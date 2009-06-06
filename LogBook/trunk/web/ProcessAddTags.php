<?php

require_once('LogBook.inc.php');

/*
 * This script will process a request for adding tags to
 * an existingfree-form entry.
 */
$entry_id = null;
if( isset( $_POST['entry_id'] )) {
    $str = trim( $_POST['entry_id'] );
    if( $str != '' && ( 1 != sscanf( $str, "%ud", $entry_id )))
        die( "not a number at an entry identifier" );
} else
    die( "no valid entry id" );

if( isset( $_POST['experiment_name'] )) {
    $experiment_name = trim( $_POST['experiment_name'] );
    if( $experiment_name == '' )
        die( "experiment name can't be empty" );
} else
    die( "no valid experiment name" );

// Read tags submitted with the entry
//
define( MAX_TAGS, 3 );          // max number of tags to attach

$tags = array();
for( $i = 0; $i < MAX_TAGS; $i++ ) {

    $tag_name_key  = 'tag_name'.$i;
    if( isset( $_POST[$tag_name_key] )) {

        $tag = trim( $_POST[$tag_name_key] );
        if( $tag != '' ) {

            $tag_value_key = 'tag_value'.$i;
            if( !isset( $_POST[$tag_value_key] ))
                die( "No valid value for tag {$tag_name_key}" );

            $value = trim( $_POST[$tag_value_key] );

            array_push(
                $tags,
                array(
                    'tag' => $tag,
                    'value' => $value ));
        }
    }
}

/* Proceed with the operation
 */
try {
    $logbook = new LogBook();
    $logbook->begin();

    if( is_null( $entry_id )) {
        $experiment = $logbook->find_experiment_by_name( $experiment_name )
            or die("no such experiment" );

        $entry = $experiment->find_last_entry()
            or die( "the experiment has no single entry yet - create the one first" );
    } else {
        $entry = $logbook->find_entry_by_id( $entry_id )
            or die( "no such entry" );
    }

    // Add tags
    //
    foreach( $tags as $t ) {
        $tag = $entry->add_tag( $t['tag'], $t['value'] );
    }
?>
<!--
The page for reporting the modified entry.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Modified Entry Report</title>
    </head>
    <link rel="stylesheet" type="text/css" href="LogBookTest.css" />
    <body>
        <h1>Modified Entry Report</h1>
        <h2><?php echo $entry->parent()->name(); ?></h2>
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