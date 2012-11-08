<?php

require_once( 'dataportal/dataportal.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use DataPortal\DataPortal;

use RegDB\RegDB;
use RegDB\RegDBException;

$name_or_id = $_GET[ 'name_or_id' ];
if( isset( $name_or_id )) {
    $name_or_id = trim( $name_or_id );
    if( $name_or_id == '' ) {
        print_result( null );
        exit;
    }
}

function table_header() {
    return DataPortal::table_begin_html(
        array(
            array( 'name' => 'Experiment',  'width' => 105 ),
            array( 'name' => 'Id',          'width' =>  32 ),
            array( 'name' => 'Status',      'width' =>  85 ),
            array( 'name' => 'Begin',       'width' =>  90 ),
            array( 'name' => 'End',         'width' =>  90 ),
            array( 'name' => 'Contact',     'width' => 160 ),
            array( 'name' => 'Description', 'width' => 300 )
        )
    );
}
function table_row( $e ) {
    $name = '<a href="index.php?exper_id='.$e->id().'" class="link">'.$e->name().'</a>';
    $contact = preg_replace(
        '/(.*)[( ](.+@.+)[) ](.*)/',
        '<a class="link" href="javascript:show_email('."'$1','$2'".')" title="click to see e-mail address">$1</a>$3',
        $e->contact_info());
    return DataPortal::table_row_html(
        array(
            $name,
            $e->id(),
            DataPortal::decorated_experiment_status( $e ),
            $e->begin_time()->toStringDay(),
            $e->end_time()->toStringDay(),
            $contact,
            $e->description()
        )
    );
}

function print_result( $experiment ) {

    header( 'Content-type: text/plain' );
    header( "Cache-Control: no-cache, must-revalidate" ); // HTTP/1.1
    header( "Expires: Sat, 26 Jul 1997 05:00:00 GMT" );   // Date in the past
	
    echo table_header();
    if( !is_null( $experiment )) echo table_row( $experiment );
    echo DataPortal::table_end_html();
}

try {
    RegDB::instance()->begin();

    $id = (int)$name_or_id;
    print_result(
        $id > 0 ?
        RegDB::instance()->find_experiment_by_id( $id ) :
        RegDB::instance()->find_experiment_by_unique_name( $name_or_id )
    );
} catch (RegDBException $e) { print $e->toHtml(); }

?>
