<?php

/*
 * Test the connection
 * 
 * The service won't do anything, just return some meaningless text
 * in a JSON object just to ensure a caller can get through the Web authntication.
 */
require_once 'dataportal/dataportal.inc.php' ;

DataPortal\ServiceJSON::run_handler ('GET', function ($SVC) {
    return array('boilerplate' => '...') ;
}) ;

?>
