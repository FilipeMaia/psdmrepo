<?php

require_once( 'authdb/authdb.inc.php' );
require_once( 'dataportal/dataportal.inc.php' );
require_once( 'logbook/logbook.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );
require_once( 'regdb/regdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

use DataPortal\DataPortal;

use LogBook\LogBook;
use LogBook\LogBookException;

use LusiTime\LusiTimeException;

use RegDB\RegDB;
use RegDB\RegDBException;


/* Override the default target page if teh one provide by a caller
 */
$on_select = 'index.php' ;
if (isset($_GET['on_select'])) $on_select = trim($_GET['on_select']) ;

/* Descending sort experiments by their identifiers
 */
function sort_experiments_by_id_desc( $experiments ) {
    usort( $experiments, function($a, $b) {    return $b->id() - $a->id();    });
    return $experiments;
}

/* Global variables used by the rest of the script
 */
$instrument_names = array();
$location_names   = array();

try {

    LogBook::instance()->begin();

    RegDB::instance()->begin();

    foreach( LogBook::instance()->regdb()->instruments() as $instrument )
        if( $instrument->is_location()) array_push( $location_names,   $instrument->name());
        else                            array_push( $instrument_names, $instrument->name());

    sort( $instrument_names );
?>




<!------------------- Document Begins Here ------------------------->
<?php
    require_once('dataportal/dataportal.inc.php');
    DataPortal::begin( "Experiment Selector" );
?>



<!------------------- Page-specific Styles ------------------------->
<style type="text/css"> 

  #experiment-search-result {
    margin-top:20px;
  }

</style>
<!----------------------------------------------------------------->






<?php
    DataPortal::scripts( "page_specific_init" );
?>


<!------------------ Page-specific JavaScript ---------------------->
<script type="text/javascript">

function page_specific_init() {
    $('#experiments').tabs();
    $('#experiments-by-year').tabs();

    $('#experiment-search-button').button().click( search_experiment );
    $('#experiment-search-input').keydown(
        function( event ) {
            // Only process RETURN key
            if( event.which == 13 ) search_experiment();
        }
       );
}

function search_experiment() {
    $( '#experiment-search-result' ).html( 'Searching...' );
    $.get(
           '../portal/ws/SearchExperiment.php',
           { name_or_id: $('#experiment-search input').val() },
           function( data ) {
            $( '#experiment-search-result' ).html( data );
        }
    );
}

function show_email( user, addr ) {
    $('#popupdialogs').html( '<p>'+addr+'</p>' );
    $('#popupdialogs').dialog({
        modal:  true,
        title:  'e-mail: '+user
    });
}

</script>
<!----------------------------------------------------------------->


<?php
    DataPortal::body( "Web Portal: Select Experiment" );
?>




<!------------------ Page-specific Document Body ------------------->
<?php

    function table_header ($is_facility=false) {
        $row = array (
            array ('name' => $is_facility ? 'Logbook' : 'Experiment',  'width' => 105),
            array ('name' => 'Id', 'width' => 32));
        if ($is_facility)
            array_push (
                $row,
                array ('name' => 'Num.Messages', 'width' => 90),
                array ('name' => 'Last Message', 'width' => 90));
        else
            array_push (
                $row,
                array ('name' => 'First Run', 'width' => 90),
                array ('name' => 'Last Run',  'width' => 90));
        array_push (
            $row,
            array ('name' => 'Contact',     'width' => 160),
            array ('name' => 'Description', 'width' => 300));
        return DataPortal::table_begin_html($row);
    }

    function table_row ($e, $is_facility=false) {
        global $on_select ;
        $row = array (
            '<a href="'.$on_select.'?exper_id='.$e->id().'" class="link">'.$e->name().'</a>',
            $e->id()
        );
        if ($is_facility) {
            $last_entry = $e->find_last_entry();
            array_push (
                $row,
                $e->num_entries(),
                is_null($last_entry) ? '' : $last_entry->insert_time()->toStringDay()
            );
        } else {
            $first_run = $e->find_first_run();
            $last_run  = $e->find_last_run();
            array_push (
                $row,
                is_null($first_run) ? '' : $first_run->begin_time()->toStringDay(),
                is_null( $last_run) ? '' :  $last_run->begin_time()->toStringDay()
            );
        }
        array_push (
            $row,
            DataPortal::decorated_experiment_contact_info( $e ),
            $e->description()
        );
        return DataPortal::table_row_html($row);            
    }

    $experiment_tabs = array();

    /* Experiments my account is a member of.
     */
    $html = table_header();
    $my_account = LogBook::instance()->regdb()->find_user_account( AuthDB::instance()->authName());
    foreach( sort_experiments_by_id_desc( LogBook::instance()->experiments()) as $e ) {
        if( $e->is_facility()) continue;
        if( $e->leader_account() == $my_account['uid']) {
            $html .= table_row( $e );
        } else {
            $experiment_group = $e->POSIX_gid();
            foreach( $my_account['groups'] as $group )
                if( $experiment_group == $group ) {
                    $html .= table_row( $e );
                    break;
                }
        }
    }
    $html .= DataPortal::table_end_html();
    array_push(
        $experiment_tabs,
        array(
            'name' => 'My Experiments',
            'id'   => 'experiment-my',
            'html' => $html,
            'class' => 'tab-inline-content'
        )
    );

    /* "Active" experiments. These are the most recent experimentch which have been
     * activated with the 'Experiment Switch'.
     */
    $html = table_header();
    foreach( RegDB::instance()->instruments() as $instrument ) {
        if( $instrument->is_location()) continue;

        $num_stations = $instrument->find_param_by_name( 'num_stations' );
        if( is_null($num_stations))
            throw new RegDBException (
                __METHOD__,
                "the instrument is not properly configured in the database, instrument: {$instrument->name()}" );

        for( $station=0; $station < intval($num_stations->value()); $station++ ) {
            $last_experiment_switch = RegDB::instance()->last_experiment_switch( $instrument->name(), $station);
                if( !is_null( $last_experiment_switch ))
                    $html .= table_row( LogBook::instance()->find_experiment_by_id( $last_experiment_switch['exper_id'] ));
        }
    }    
    $html .= DataPortal::table_end_html();
    array_push(
        $experiment_tabs,
        array(
            'name' => 'Active',
            'id'   => 'experiment-active',
            'html' => $html,
            'class' => 'tab-inline-content'
        )
    );
    
    /* All experiments in one tab. The tab has a subtab for each year of operation.
     */
    $experiment_by_year = array();
       foreach( sort_experiments_by_id_desc( LogBook::instance()->experiments()) as $e ) {
           if( $e->is_facility()) continue;
           $first_run = $e->find_first_run();
           $year = is_null($first_run) ? 0 : $first_run->begin_time()->year();
           if( !array_key_exists( $year,$experiment_by_year ))
               $experiment_by_year[$year] =
                   array(
                       'id' => 'experiments-by-year-'.$year,
                       'html' => table_header());
           $experiment_by_year[$year]['html'] .= table_row( $e );
       }
    $experiment_tabs_by_year = array();
       foreach( array_keys( $experiment_by_year ) as $year ) {
           array_push(
               $experiment_tabs_by_year,
               array(
                   'name' => $year == 0 ? 'No runs taken yet' : $year,
                   'id'   => 'experiments-by-year-'.$year,
                   'html' => $experiment_by_year[$year]['html'].DataPortal::table_end_html(),
                   'class' => 'tab-inline-content'
               )
           );
       }
       array_push(
        $experiment_tabs,
        array(
            'name' => 'All Experiments',
            'id'   => 'experiments-by-year',
            'html' => DataPortal::tabs_html( "experiments-by-year-contents", $experiment_tabs_by_year ),
            'class' => 'tab-inline-content'
        )
    );

    /* One tab per instrument.
     */
    foreach( $instrument_names as $i ) {
        $experiment_by_year = array();
        foreach( sort_experiments_by_id_desc( LogBook::instance()->experiments_for_instrument( $i )) as $e ) {
            if( $e->is_facility()) continue;
            $first_run = $e->find_first_run();
            $year = is_null($first_run) ? 0 : $first_run->begin_time()->year();
            if( !array_key_exists( $year,$experiment_by_year ))
                $experiment_by_year[$year] =
                    array(
                        'id' => 'experiments-by-year-'.$year,
                        'html' => table_header());
            $experiment_by_year[$year]['html'] .= table_row( $e );
        }
        $html = '';
        foreach( array_keys( $experiment_by_year ) as $year ) {
            $html .=
                '<p class="section1">'.($year == 0 ? 'No runs taken yet' : $year).'</p>'.
                $experiment_by_year[$year]['html'].DataPortal::table_end_html();
        }
        array_push(
            $experiment_tabs,
            array(
                'name' => $i,
                'id'   => 'experiments-'.$i,
                'html' => $html,
                'class' => 'tab-inline-content'
            )
        );
    }

    /* One tab per location.
     */
    foreach( $location_names as $location ) {

        $html = table_header(true);
        foreach( sort_experiments_by_id_desc( LogBook::instance()->experiments_for_instrument( $location )) as $e )
            if( $e->is_facility())
                $html .= table_row( $e, $e->is_facility());

        $tab_title = $location;
        switch($location) {
            case 'NEH': $tab_title = 'Facilities'; break;
            case 'EXT': $tab_title = 'External'; break;
        }
        $html .= DataPortal::table_end_html();
        array_push(
            $experiment_tabs,
            array(
                'name' => $tab_title,
                'id'   => 'experiments-'.$location,
                'html' => $html,
                'class' => 'tab-inline-content'
            )
        );
    }

    /* Search experiments by partial names or numeric identifiers
     */
    $html = <<<HERE
<div id="experiment-search" stype="display:none;">
  <div style="float:left;"><input id="experiment-search-input" type="text" title="enter experiment name or its numeric identifier"/></div>
  <div style="float:left; margin-left:10px; font-size:80%;"><button id="experiment-search-button">Search</button></div>
  <div style="clear:both;"></div>
  <div id="experiment-search-result"></div>
</div>

HERE;

    array_push(
        $experiment_tabs,
        array(
            'name' => 'Search',
            'id'   => 'experiment-search',
            'html' => $html,
            'class' => 'tab-inline-content'
        )
    );
    
    /* Print the whole tab and its contents (including sub-tabs).
     */
    DataPortal::tabs( "experiments", $experiment_tabs );
?>
<!----------------------------------------------------------------->






<?php
    DataPortal::end();
?>
<!--------------------- Document End Here -------------------------->

<?php

    LogBook::instance()->commit();

} catch( AuthDBException   $e ) { print $e->toHtml(); exit; }
  catch( LogBookException  $e ) { print $e->toHtml(); exit; }
  catch( LusiTimeException $e ) { print $e->toHtml(); exit; }
  catch( RegDBException    $e ) { print $e->toHtml(); exit; }
  catch( Exception         $e ) { print $e; exit; }

?>     