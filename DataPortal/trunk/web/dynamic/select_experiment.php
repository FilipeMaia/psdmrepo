<?php

$instruments = array();

try {

    require_once('RegDB/RegDB.inc.php');

	$regdb = new RegDB();
	$regdb->begin();

	foreach( $regdb->instruments() as $i ) {
		if( $i->is_location()) continue;
		array_push( $instruments, $i->name());
	}
	sort( $instruments );

} catch( AuthDBException $e ) {
	print $e->toHtml();
	exit;
} catch( RegDBException   $e ) {
	print $e->toHtml();
	exit;
}
?>




<!------------------- Document Begins Here ------------------------->
<?php
    require_once('DataPortal/DataPortal.inc.php');
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

    $('#experiment-search button').button();
    $('#experiment-search button').click( search_experiment );
    $('#experiment-search input').keydown(
    	function( event ) {
        	// Only process RETURN key
        	if( event.which == 13 ) search_experiment();
    	}
   	);
}

function search_experiment() {
	$( '#experiment-search-result' ).html( 'Searching...' );
	$.get(
	   	'SearchExperiment.php',
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
    DataPortal::body( "Data Portal: Select Experiment" );
?>




<!------------------ Page-specific Document Body ------------------->
<?php
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
    	return DataPortal::table_row_html(
    		array(
    			$name,
    			$e->id(),
    			DataPortal::decorated_experiment_status( $e ),
    			$e->begin_time()->toStringDay(),
    			$e->end_time()->toStringDay(),
    			DataPortal::decorated_experiment_contact_info( $e ),
    			$e->description()
   			)
    	);
	}

	$h = <<<HERE
<div id="experiment-search" stype="display:none;">
  <div>
    <input type="text" title="enter experiment name or its numeric identifier"/>
    <button>Search</button>
  </div>
  <div id="experiment-search-result"></div>
</div>

HERE;

	$experiment_tabs = array();
   	array_push(
		$experiment_tabs,
    	array(
    		'name' => 'Search',
    		'id'   => 'experiment-search',
    		'html' => $h
    	)
    );
	
    /* All experiments in one tab. The tab has a subtab for each year of operation.
     */
    $experiment_tabs_by_year = array();
   	$html = '';
   	$year = null;
   	foreach( $regdb->experiments() as $e ) {
   		if( $e->is_facility()) continue;
   		if( is_null( $year ) || ( $year != $e->begin_time()->year())) {
   			if( !is_null( $year )) {
   				$html .= DataPortal::table_end_html();
   				array_push(
   					$experiment_tabs_by_year,
   					array('name' => $year,
   						  'id'   => 'experiments-by-year-'.$year,
			   			  'html' => $html
   					)
   				);
   				$html = '';
   			}
   			$year = $e->begin_time()->year();
   			$html .= table_header();
   		}
   		$html .= table_row( $e );
   	}
   	if(( $html != '' ) && !is_null( $year )) {
   		$html .= DataPortal::table_end_html();
   		array_push(
   			$experiment_tabs_by_year,
   			array(
   				'name' => $year,
   				'id'   => 'experiments-by-year-'.$year,
   				'html' => $html
   			)
		);
   	}
   	array_push(
		$experiment_tabs,
    	array(
    		'name' => 'all experiments',
    		'id'   => 'experiments-by-year',
    		'html' => DataPortal::tabs_html( "experiments-by-year-contents", $experiment_tabs_by_year  )
    	)
    );

	/* One tab per instrument.
	 */
    foreach( $instruments as $i ) {
    	$html = '';
    	$year = null;
    	foreach( $regdb->experiments_for_instrument( $i ) as $e ) {
    		if( is_null( $year ) || ( $year != $e->begin_time()->year())) {
    			if( !is_null( $year )) {
    				$html .= DataPortal::table_end_html();
    			}
    			$year = $e->begin_time()->year();
    			$html .= '<p class="section1">'.$year.'</p>'.table_header();
     		}
    		$html .= table_row( $e );
    	}
    	$html .= DataPortal::table_end_html();
    	array_push(
    		$experiment_tabs,
    		array(
    			'name' => $i,
    			'id'   => 'experiments-'.$i,
    			'html' => $html
    		)
    	);
    }

   	/* Print the whole tab and its contents (including sub-tabs).
   	 */
	DataPortal::tabs( "experiments", $experiment_tabs );
?>
<!----------------------------------------------------------------->






<?php
    DataPortal::end();
?>
<!--------------------- Document End Here -------------------------->
