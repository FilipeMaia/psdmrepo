<?php

/**
 * This is an example of hwo to use the JSON Web services framework
 * in the functional way.
 */

require_once 'dataportal/dataportal.inc.php' ;

\DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {

    $id = $SVC->required_int ('id') ;

    $e = $SVC->regdb()->find_experiment_by_id ($id) ;
    if (is_null($e)) $SVC->abort ('no such experiment') ;

    $SVC->finish (array (
        'id'   => $e->id() ,
        'name' => $e->name()
    )) ;
}) ;

?>
