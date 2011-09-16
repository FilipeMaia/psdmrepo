/*
 * ======================================================
 *  Application: Data Files
 *  DOM namespace of classes and identifiers: datafiles-
 *  JavaScript global names begin with: datafiles
 * ======================================================
 */

function datafiles_create() {

	/* Add this anchor to access this object's variables from within
	 * for anonymous functions. Just using this.varname won't work
	 * due to a well known bug in JavaScript. */

	var that = this;

	/* ---------------------------------------
	 *  This application specific environment
	 * ---------------------------------------
	 *
	 * NOTE: These variables should be initialized externally.
	 */
	this.exp_id  = null;

	/* The context for v-menu items
	 */
	var context2_default = {
		'summary' : '',
		'files'   : ''
	};
	this.name = 'datafiles';
	this.full_name = 'File Manager';
	this.context1 = 'summary';
	this.context2 = '';
	this.select_default = function() { this.select(this.context1, this.context2); };
	this.select = function(ctx1, ctx2) {
		this.init();
		this.context1 = ctx1;
		this.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};

	/* --------------
	 *  Summary page
	 * --------------
	 */
	this.summary_update = function() {
		$('#datafiles-summary-info').html('Updating...');
		var params = {exper_id: that.exp_id};
		var jqXHR = $.get('../portal/DataFilesSummary.php',params,function(data) {
			var result = eval(data);
			if(result.Status != 'success') {
				$('#datafiles-summary-info').html(result.Message);
				return;
			}
			$('#datafiles-summary-info').html('[ Last update on: <b>'+result.updated+'</b> ]');
			$('#datafiles-summary-runs'         ).html(result.summary.runs);
			$('#datafiles-summary-firstrun'     ).html(result.summary.runs ? result.summary.min_run : 'n/a');
			$('#datafiles-summary-lastrun'      ).html(result.summary.runs ? result.summary.max_run : 'n/a');
			$('#datafiles-summary-xtc-size'     ).html(result.summary.xtc.size); 
			$('#datafiles-summary-xtc-files'    ).html(result.summary.xtc.files);
			$('#datafiles-summary-xtc-archived' ).html(result.summary.xtc.archived_html);
			$('#datafiles-summary-xtc-disk'     ).html(result.summary.xtc.disk_html);
			$('#datafiles-summary-hdf5-size'    ).html(result.summary.hdf5.size);
			$('#datafiles-summary-hdf5-files'   ).html(result.summary.hdf5.files);
			$('#datafiles-summary-hdf5-archived').html(result.summary.hdf5.archived_html);
			$('#datafiles-summary-hdf5-disk'    ).html(result.summary.hdf5.disk_html);
		},
		'JSON').error(function () {
			alert('failed because of: '+jqXHR.statusText);
		});

	};
	this.summary_init = function() {
		$('#datafiles-summary-refresh').button().click(function() { that.summary_update(); });
		this.summary_update();
	};

	/* ----------------------------
	 *  Files sorted by files page
	 * ----------------------------
	 */
	this.files_last_request = null;
	this.files_last_request_files = null;
	this.files_reverse_order = true;

	function file2html(f,run_url,display,extra_class) {
		var hightlight_class = f.type != 'XTC' ? 'datafiles-files-highlight' : '';
		var html =
'  <tr>'+
'    <td class="table_cell table_cell_left '+extra_class+'">'+run_url+'</td>'+
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.name+'</td>'+
			(display.type?
'	 <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.type+'</td>':'')+
			(display.size?
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.size+'</td>':'')+
			(display.created?
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.created+'</td>':'')+
			(display.checksum?
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.checksum+'</td>':'')+
			(display.archived?
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.archived+'</td>':'')+
			(display.local?
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class+'">'+f.local+'</td>':'')+
'  </tr>';
		return html;
	}
	this.files_display = function() {

		var display = new Object();
		display.type          = $('#datafiles-files-wa' ).find('input[name="type"]').attr('checked');
		display.size          = $('#datafiles-files-wa' ).find('input[name="size"]').attr('checked');
		display.created       = $('#datafiles-files-wa' ).find('input[name="created"]').attr('checked');
		display.archived      = $('#datafiles-files-wa' ).find('input[name="archived"]').attr('checked');
		display.local         = $('#datafiles-files-wa' ).find('input[name="local"]').attr('checked');
	    display.checksum      = $('#datafiles-files-wa' ).find('input[name="checksum"]').attr('checked');

		var num_runs = 0;
		var num_files = 0;

		var html =
'<table><tbody>'+
'  <tr>'+
'    <td class="table_hdr">Run</td>'+
'    <td class="table_hdr">File</td>'+
          	(display.type?
'    <td class="table_hdr">Type</td>':'')+
			(display.size?
'    <td class="table_hdr">Size</td>':'')+
            (display.created?
'    <td class="table_hdr">Created</td>':'')+
			(display.checksum?
'    <td class="table_hdr">Checksum</td>':'')+
            (display.archived?
'    <td class="table_hdr">Archived</td>':'')+
            (display.local?
'    <td class="table_hdr">On disk</td>':'')+
'  </tr>';

		if( $('#datafiles-files-wa' ).find('select[name="sort"]').val() == 'run' ) {
			var result = this.files_last_request;
			for(var i in result.runs) {
				++num_runs;
				var run = result.runs[i];
				var first = true;
				for(var j in run.files) {
					++num_files;
					var extra_class = (j != run.files.length - 1) ? 'table_cell_bottom' : '';
					var f = run.files[j];
					html += file2html( f, first ? run.url : '', display, extra_class );
					first = false;
				}
			}
		} else {
			var result = this.files_last_request;
			for(var i in result) ++num_runs;

			var files = this.files_last_request_files;
			for(var j in files) {
				++num_files;
				var f = files[j];
				html += file2html( f, f.run_url, display, '' );
			}
		}
		html +=
'</tbody></table>';

		$('#datafiles-files-list').html(html);
		$('#datafiles-files-info').html(
			'<span style="font-weight:bold;">'+num_files+'</span> files in <span style="font-weight:bold;">'+num_runs+'</span> runs, <b>'+result.total_size_gb+'</b> GB in total');
		$('#datafiles-files-updated').html(
			'[ Last update on: <b>'+result.updated+'</b> ]');
	}
	this.files_update = function() {
		$('#datafiles-files-updated').html('Updating...');
		var params   = {exper_id: that.exp_id, json: ''};
		var runs     = $('#datafiles-files-ctrl').find('input[name="runs"]'     ).val(); if(runs     != '')    params.runs     = runs;
		var types    = $('#datafiles-files-ctrl').find('select[name="types"]'   ).val(); if(types    != 'any') params.types    = types;
		var created  = $('#datafiles-files-ctrl').find('select[name="created"]' ).val(); if(created  != 'any') params.created  = created;
		var checksum = $('#datafiles-files-ctrl').find('select[name="checksum"]').val(); if(checksum != 'any') params.checksum = checksum == 'is known' ? 1 : 0;
		var archived = $('#datafiles-files-ctrl').find('select[name="archived"]').val(); if(archived != 'any') params.archived = archived == 'yes' ? 1 : 0;
		var local    = $('#datafiles-files-ctrl').find('select[name="local"]'   ).val(); if(local    != 'any') params.local    = local    == 'yes' ? 1 : 0;
		var jqXHR = $.get('../portal/SearchFiles.php',params,function(data) {
			var result = eval(data);
			if(result.Status != 'success') {
				$('#datafiles-files-updated').html(result.Message);
				return;
			}
			that.files_last_request = result;
			that.files_last_request_files = [];
			var k = 0;
			for(var i in result.runs) {
				var run = result.runs[i];
				for(var j in run.files) {
					var f = run.files[j];
					f.run_url = run.url;
					that.files_last_request_files[k++] = f;
				}
			}
			that.files_sort();
			if( that.files_reverse_order ) {
				that.files_last_request_files.reverse();
				that.files_last_request.runs.reverse();
			}
			that.files_display();
		},
		'JSON').error(function () {
			alert('failed because of: '+jqXHR.statusText);
		});

	};
	this.files_sort = function() {
		function compare_elements_by_run     (a, b) { return   a.runnum          - b.runnum; }
		function compare_elements_by_name    (a, b) { return ( a.name < b.name ? -1 : (a.name > b.name ? 1 : 0 )); }
		function compare_elements_by_type    (a, b) { return ( a.type < b.type ? -1 : (a.type > b.type ? 1 : compare_elements_by_run(a,b))); }
		function compare_elements_by_size    (a, b) { return   a.size_bytes      - b.size_bytes; }
		function compare_elements_by_created (a, b) { return   a.created_seconds - b.created_seconds; }
		function compare_elements_by_archived(a, b) { return ( a.archived < b.archived ? -1 : (a.archived > b.archived ? 1 : compare_elements_by_run(a,b))); }
		function compare_elements_by_disk    (a, b) { return ( a.local    < b.local    ? -1 : (a.local    > b.local    ? 1 : compare_elements_by_run(a,b))); }
		var sort_function = null;
		switch( $('#datafiles-files-wa' ).find('select[name="sort"]').val()) {
		case 'run':      sort_function = compare_elements_by_run;      break;
		case 'name':     sort_function = compare_elements_by_name;     break;
		case 'type':     sort_function = compare_elements_by_type;     break;
		case 'size':     sort_function = compare_elements_by_size;     break;
		case 'created':  sort_function = compare_elements_by_created;  break;
		case 'archived': sort_function = compare_elements_by_archived; break;
		case 'disk':     sort_function = compare_elements_by_disk;     break;
		}
		this.files_last_request_files.sort( sort_function );
	};
	this.files_init = function() {
		$('#datafiles-files-refresh').button().click(function() { that.files_update(); });
		$('#datafiles-files-ctrl').find('input').keyup(function(e) { if(e.keyCode == 13) that.files_update(); });
		$('#datafiles-files-ctrl').find('select').change(function() { that.files_update(); });
		$('#datafiles-files-reverse').button().click(function() {
			that.files_reverse_order = !that.files_reverse_order;
			that.files_last_request_files.reverse();
			that.files_last_request.runs.reverse();
			that.files_display();
		});
		$('#datafiles-files-wa' ).find('input[name="type"]').attr('checked','checked');
		$('#datafiles-files-wa' ).find('input[name="size"]').attr('checked','checked');
		$('#datafiles-files-wa' ).find('input[name="created"]').attr('checked','checked');
		$('#datafiles-files-wa' ).find('input[name="archived"]').attr('checked','checked');
		$('#datafiles-files-wa' ).find('input[name="local"]').attr('checked','checked');
		$('#datafiles-files-wa' ).find('input').change(function(){
			that.files_display();
		});
		$('#datafiles-files-wa' ).find('select[name="sort"]').change(function(){
			that.files_sort();
			if( that.files_reverse_order ) {
				that.files_last_request_files.reverse();
				that.files_last_request.runs.reverse();
			}
			that.files_display();
		});
		this.files_update();
	};

	/* ----------------------------------
	 *  Application initialization point
	 * ----------------------------------
	 *
	 * RETURN: true if the real initialization took place. Return false
	 *         otherwise.
	 */
	this.is_initialized = false;
	this.init = function() {
		if(that.is_initialized) return false;
		this.is_initialized = true;
		this.summary_init();
		this.files_init();
		return true;
	};
}

var datafiles = new datafiles_create();
