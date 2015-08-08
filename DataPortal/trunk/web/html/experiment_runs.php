<?php

# Needed to supress complains in the server's log files
date_default_timezone_set('America/Los_Angeles') ;

/*
 * This script is used as a web service to get the information about
 * an experiment from LogBook.
 *
 * TODO: Move the service to LogBook?
 */
require_once( 'logbook/logbook.inc.php' );

use LogBook\LogBook;
use LogBook\LogBookException;

if     ( isset( $_GET[ 'id'   ] )) { $exper_id   = (int) trim( $_GET[ 'id'   ] ); }
else if( isset( $_GET[ 'name' ] )) { $exper_name =       trim( $_GET[ 'name' ] ); }
else                               { die( 'no experiment identity parameter found in the requst' ); }

try {
    LogBook::instance()->begin();

    $experiment = isset( $exper_id ) ? LogBook::instance()->find_experiment_by_id         ( $exper_id   )
                                     : LogBook::instance()->find_experiment_by_unique_name( $exper_name );
    if( is_null( $experiment )) {
            die( 'no such experiment found for '.( isset( $exper_id ) ? "id={$exper_id}" : "name={$exper_name}" ));
    }
    foreach( $experiment->runs() as $run ) {
        printf("<br>r%04d: %s\n", $run->num(), $run->begin_time()->toStringDHM());
    }

    LogBook::instance()->commit();

} catch( LogBookException   $e ) { print $e->toHtml();exit; }

?>
