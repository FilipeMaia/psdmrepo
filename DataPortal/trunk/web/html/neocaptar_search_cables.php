<?php

require_once( 'authdb/authdb.inc.php' );

use AuthDB\AuthDB;
use AuthDB\AuthDBException;

$is_submitted = isset($_GET['is_submitted']);

class TableView {
	
	private $total = 0;
	private $is_submitted = false;

	public function __construct($is_submitted) {
		$this->is_submitted = $is_submitted;
		print <<<HERE
<table><tbody>

HERE;
	}
	public function __destruct() {
		print <<<HERE
</tbody></table>

HERE;
	}
	public function add_row( $jobnum, $cablenum, $system, $function, $type, $origin, $destination, $length, $routing ) {

		if( $this->total++ % 20 == 0 ) {

			if( $this->total > 1 ) echo <<<HERE
      <tr><td></td></tr>
      <tr><td></td></tr>
      <tr><td></td></tr>
HERE;
			echo <<<HERE
      <tr>
HERE;
			if( !$this->is_submitted ) echo <<<HERE
        <td nowrap="nowrap" class="table_hdr table_hdr_tight" >
          <select name="status">
            <option>status</option>
            <option>Planned</option>
            <option>Registered</option>
            <option>Labeled</option>
            <option>Fabrication</option>
            <option>Ready</option>
            <option>Installed</option>
            <option>Commissioned</option>
            <option>Damaged</option>
            <option>Retired</option>
          </select>
        </td>
        <td nowrap="nowrap" class="table_hdr">TOOLS</td>
HERE;
			echo <<<HERE
        <td nowrap="nowrap" class="table_hdr">job #</td>
        <td nowrap="nowrap" class="table_hdr">cable #</td>
		<td nowrap="nowrap" class="table_hdr">system</td>
        <td nowrap="nowrap" class="table_hdr">function</td>
        <td nowrap="nowrap" class="table_hdr">cable type</td>
        <td nowrap="nowrap" class="table_hdr">length</td>
        <td nowrap="nowrap" class="table_hdr">routing</td>
        <td nowrap="nowrap" ></td>
        <td nowrap="nowrap" class="table_hdr">ORIGIN / DESTINATION</td>
        <td nowrap="nowrap" class="table_hdr">loc</td>
        <td nowrap="nowrap" class="table_hdr">rack</td>
        <td nowrap="nowrap" class="table_hdr">ele</td>
        <td nowrap="nowrap" class="table_hdr">side</td>
        <td nowrap="nowrap" class="table_hdr">slot</td>
        <td nowrap="nowrap" class="table_hdr">conn #</td>
        <td nowrap="nowrap" class="table_hdr">pinlist</td>
        <td nowrap="nowrap" class="table_hdr">station</td>
        <td nowrap="nowrap" class="table_hdr">contype</td>
        <td nowrap="nowrap" class="table_hdr">instr</td>
       </tr>

HERE;

		}

		$origin_name     = $origin['name'];
		$origin_loc      = $origin['loc'];
		$origin_rack     = $origin['rack'];
		$origin_ele      = $origin['ele'];
		$origin_side     = $origin['side'];
		$origin_slot     = $origin['slot'];
		$origin_connum   = $origin['connum'];
		$origin_pinlist  = $origin['pinlist'];
		$origin_station  = $origin['station'];
		$origin_conntype = $origin['conntype'];
		$origin_instr    = $origin['instr'];

		echo <<<HERE
      <tr><td></td></tr>
      <tr><td></td></tr>
      <tr><td></td></tr>
      <tr class="table_row">
HERE;
		if( $this->is_submitted ) echo <<<HERE
        <td nowrap="nowrap" class="table_cell table_cell_left_closed table_cell_top table_cell_bottom ">{$jobnum}</td>
HERE;
		else echo <<<HERE
        <td nowrap="nowrap" class="table_cell table_cell_left_closed table_cell_top table_cell_bottom "><b>Planned</b></td>
        <td nowrap="nowrap" class="table_cell table_cell_bottom table_cell_right ">
          <button class="proj-cable-edit"   title="edit"  ><b>E</b></button>
          <button class="proj-cable-save"   title="save changes to the database"         >save</button>
          <button class="proj-cable-cancel" title="cancel editing and ignore any changes">cancel</button>
        </td>
        <td nowrap="nowrap" class="table_cell table_cell_left_closed table_cell_top table_cell_bottom ">{$jobnum}</td>
HERE;
		echo <<<HERE
        <td nowrap="nowrap" class="table_cell table_cell_top table_cell_bottom ">{$cablenum}</td>
		<td nowrap="nowrap" class="table_cell table_cell_top table_cell_bottom ">{$system}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top table_cell_bottom ">{$function}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top table_cell_bottom ">{$type}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top table_cell_bottom ">{$length}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top table_cell_bottom ">{$routing}</td>
        <td nowrap="nowrap" class="table_cell                table_cell_bottom "><span style="font-size:120%; font-style:bold;">&diams;&rarr;</span></td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_name}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_loc}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_rack}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_ele}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_side}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_slot}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_connum}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_pinlist}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_station}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_contype}</td>
        <td nowrap="nowrap" class="table_cell table_cell_top ">{$origin_instr}</td>
      </tr>

HERE;
		$destination_name     = $destination['name'];
		$destination_loc      = $destination['loc'];
		$destination_rack     = $destination['rack'];
		$destination_ele      = $destination['ele'];
		$destination_side     = $destination['side'];
		$destination_slot     = $destination['slot'];
		$destination_connum   = $destination['connum'];
		$destination_pinlist  = $destination['pinlist'];
		$destination_station  = $destination['station'];
		$destination_conntype = $destination['conntype'];
		$destination_instr    = $destination['instr'];

		echo <<<HERE
      <tr class="table_row">
HERE;
		if( $this->is_submitted ) echo <<<HERE
        <td nowrap="nowrap" class="table_cell table_cell_left_closed "></td>
HERE;
		else echo <<<HERE
        <td nowrap="nowrap" class="table_cell table_cell_left_closed "></td>
        <td nowrap="nowrap" class="table_cell table_cell_bottom table_cell_right ">
          <button class="proj-cable-clone"  title="clone" ><b>C</b></button>
          <button class="proj-cable-delete" title="delete"><b>D</b></button>
          <button class="proj-cable-history" title="history" ><b>H</b></button>
          <button class="proj-cable-label"   title="label"   ><b>L</b></button>
          <button class="proj-cable-submit"  title="submit"  ><b>S</b></button>
        </td>
        <td nowrap="nowrap" class="table_cell table_cell_left_closed "></td>
HERE;
		echo <<<HERE
		<td nowrap="nowrap" class="table_cell "></td>
        <td nowrap="nowrap" class="table_cell "></td>
		<td nowrap="nowrap" class="table_cell "></td>
        <td nowrap="nowrap" class="table_cell "></td>
        <td nowrap="nowrap" class="table_cell "></td>
        <td nowrap="nowrap" class="table_cell "></td>
        <td nowrap="nowrap" class="table_cell table_cell_top table_cell_bottom "><span style="font-size:120%; font-style:bold;">&larr;&diams;</span></td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_name}</td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_loc}</td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_rack}</td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_ele}</td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_side}</td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_slot}</td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_connum}</td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_pinlist}</td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_station}</td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_contype}</td>
        <td nowrap="nowrap" class="table_cell table_cell_highlight ">{$destination_instr}</td>
      </tr>

HERE;
	}
}


try {
	$authdb = AuthDB::instance();
	$authdb->begin();

	$table = new TableView( $is_submitted );

	for( $i = 0; $i < 78; $i++ ) {

		$jobnum   = $is_submitted ? sprintf( "GAP%03d", $i % 12 )  : '';
		$cablenum = $is_submitted ? sprintf( "LN%05d", 1200 + $i ) : '';
		$system   = 'CXI-DG2-PIP-01';
		$function = 'R52 To Cxi-Dg2 Pip-01 Hv Feed';
		$type     = $i % 3 ? 'CNT195FR' : 'CAT6TLN';
		$length   = 12 * ( $i % 14 );
		$routing  = 'HV:TDFEHF02';

		$origin = array();
		$origin['name'    ] = 'CXI-DG2-PIP-01-X';
		$origin['loc'     ] = 'CXI';
		$origin['rack'    ] = 'DG2';
		$origin['ele'     ] = '';
		$origin['side'    ] = '';
		$origin['slot'    ] = 'PIP-01';
		$origin['connum'  ] = 'X';
		$origin['pinlist' ] = '';
		$origin['station' ]  = '';
		$origin['conntype'] = 'STARCEL';
		$origin['instr'   ] = 5;

		$destination = array();
		$destination['name'    ] = 'B999-5605-PCI-J501';
		$destination['loc'     ] = 'B999';
		$destination['rack'    ] = '56';
		$destination['ele'     ] = '05';
		$destination['side'    ] = 'B';
		$destination['slot'    ] = 'PCI';
		$destination['connum'  ] = 'J501';
		$destination['pinlist' ] = '';
		$destination['station' ]  = '';
		$destination['conntype'] = 'SHV-10K';
		$destination['instr'   ] = 5;
		
		$table->add_row (
			$jobnum,
			$cablenum,
			$system,
			$function,
			$type,
			$origin,
			$destination,
			$length,
			$routing );
	}
	unset( $table );

	$regdb->commit();
	$logbook->commit();
	
} catch( AuthDBException     $e ) { print $e->toHtml(); }
  catch( LusiTimeException   $e ) { print $e->toHtml(); }
  
?>
