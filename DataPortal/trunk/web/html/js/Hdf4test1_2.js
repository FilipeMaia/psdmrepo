/*
 * ======================================================
 *  Application: HDF5 Translation
 *  DOM namespace of classes and identifiers: hdf-
 *  JavaScript global names begin with: hdf
 * ======================================================
 */

function hdf_create() {

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
		'manage' : '',
		'history' : '',
		'translators' : ''
	};
	this.name = 'hdf';
	this.full_name = 'HDF5 Translation';
	this.context1 = 'manage';
	this.context2 = '';
	this.select_default = function() { this.select(this.context1, this.context2); };
	this.select = function(ctx1, ctx2) {
		this.init();
		this.context1 = ctx1;
		this.context2 = ctx2 == null ? context2_default[ctx1] : ctx2;
	};

	/* --------------------------------------------
	 *  Menu item: Translation management requests 
	 * --------------------------------------------
	 */
	this.manage_last_request = null;

	this.translate_run = function(runnum) {
		$('#hdf-r-comment-'+runnum).html('<span style="color:red;">Processing...</span>');
		var params = {exper_id: that.exp_id, runnum: runnum};
		var jqXHR  = $.get('../portal/NewRequest.php',params,function(data) {
			var result = eval(data);
			if(result.Status == 'success') {
				$('#hdf-r-comment-'+runnum).html('<span style="color:green;">Translation request was queued</span>');
			} else {
				report_error('<span style="color:black;">Translation request was rejected because of: </span>'+result.Message);
			}
		},
		'JSON').error(function () {
			report_error('translation request for run '+runnum+' failed because of: '+jqXHR.statusText);
		});
	};
	this.translate_all = function() {
		var num_runs = 0;
		var result = this.manage_last_request;
		for(var i in result.requests) {
			var request = result.requests[i];
			var state = request.state;
			if(state.ready4translation) ++num_runs;
		}
		$('#popupdialogs').html(
			'<span class="ui-icon ui-icon-alert" style="float:left;"></span> '+
	    	'You are about to request HDF5 translaton of <b>'+num_runs+'</b> runs. '+
	    	'This may take a while. Are you sure you want to proceed with this operation?'
		 );
		$('#popupdialogs').dialog({
			resizable: false,
			modal: true,
			buttons: {
				"Yes": function() {
					$( this ).dialog('close');
					$('#hdf-manage-list').find('.translate').each(function() {
						$(this).button('disable');
						that.translate_run($(this).val());
					});
				},
				Cancel: function() {
					$(this).dialog('close');
				}
			},
			title: 'Confirmn HDF5 Translation Request'
		});
	};
	this.stop_translation = function(id) {
		var runnum = 0;
		var result = this.manage_last_request;
		for(var i in result.requests) {
			var state = result.requests[i].state;
			if( state.id == id ) { runnum = state.run_number; break; }
		}
		if(!runnum) {
			report_error('internal error: no run number found for request id: '+id);
		}
		$('#hdf-r-comment-'+runnum).html('<span style="color:red;">Processing...</span>');
		var params = {id: id};
		var jqXHR  = $.get('../portal/DeleteRequest.php',params,function(data) {
			var result = eval(data);
			if(result.Status == 'success') {
				$('#hdf-r-comment-'+runnum).html('<span style="color:green;">Translation was stopped</span>');
			} else {
				report_error('<span style="color:black;">Unable to stop translation because of: </span>'+result.Message);
			}
		},
		'JSON').error(function () {
			report_error('unable to perform the operation for run '+runnum+'  because of: '+jqXHR.statusText);
		});
	};
	this.stop_all_translation = function() {
		var num_runs = 0;
		var result = this.manage_last_request;
		for(var i in result.requests) {
			var request = result.requests[i];
			var state = request.state;
			if(state.status == 'QUEUED') ++num_runs;
		}
		$('#popupdialogs').html(
			'<span class="ui-icon ui-icon-alert" style="float:left;"></span> '+
	    	'You are about to withdraw HDF5 translaton requests for <b>'+num_runs+'</b> sitting in the translation queue. '+
	    	'This may take a while. Are you sure you want to proceed with this operation?'
		 );
		$('#popupdialogs').dialog({
			resizable: false,
			modal: true,
			buttons: {
				"Yes": function() {
					$( this ).dialog('close');
					$('#hdf-manage-list').find('.stop').each(function() {
						$(this).button('disable');
						that.stop_translation($(this).val());
					});
				},
				Cancel: function() {
					$(this).dialog('close');
				}
			},
			title: 'Confirmn HDF5 Translation Request Withdrawal'
		});
	};
	this.escalate_priority = function(id) {
		var runnum = 0;
		var result = this.manage_last_request;
		for(var i in result.requests) {
			var state = result.requests[i].state;
			if( state.id == id ) { runnum = state.run_number; break; }
		}
		if(!runnum) {
			report_error('internal error: no run number found for request id: '+id);
		}
		$('#hdf-r-comment-'+runnum).html('<span style="color:red;">Processing...</span>');
		var params = {exper_id: that.exp_id, id: id};
		var jqXHR  = $.get('../portal/EscalateRequestPriority.php',params,function(data) {
			var result = eval(data);
			if(result.Status == 'success') {
				$('#hdf-r-comment-'+runnum).html('');
				$('#hdf-r-priority-'+runnum).html(result.Priority);
			} else {
				report_error('<span style="color:black;">Unable to escalate the priority because of: </span>'+result.Message);
			}
		},
		'JSON').error(function () {
			report_error('unable to perform the operation for run '+runnum+'  because of: '+jqXHR.statusText);
		});
	};
	this.manage_display = function() {
		var result = this.manage_last_request;
		var html =
'<button class="reverse not4print" style="margin-right:20px;">Show in Reverse Order</button>'+
'<button class="translate_all not4print" value="" title="request translation for all selected runs which have not been translated">Translate selected runs</button>'+
'<button class="stop_all not4print" style="margin-left:5px;" value="" title="remove translation requests from the translation queue for selected runs">Stop translation of selected runs</button>'+
'<br><br>'+
'<table><tbody>'+
'  <tr>'+
'    <td class="table_hdr">Run</td>'+
'    <td class="table_hdr">End of Run</td>'+
'    <td class="table_hdr">File</td>'+
'    <td class="table_hdr">Size</td>'+
'    <td class="table_hdr">Status</td>'+
'    <td class="table_hdr">Changed</td>'+
'    <td class="table_hdr">Log</td>'+
'    <td class="table_hdr">Priority</td>'+
'    <td class="table_hdr">Actions</td>'+
'    <td class="table_hdr">Comments</td>'+
'  </tr>'+
'  <tr>'+
'    <td></td>'+
'    <td></td>'+
'    <td></td>'+
'    <td></td>'+
'    <td></td>'+
'    <td></td>'+
'    <td></td>'+
'    <td></td>'+
'    <td></td>'+
'    <td></td>'+
'  </tr>';

		var summary = {'FINISHED': 0, 'FAILED': 0, 'TRANSLATING': 0, 'QUEUED': 0, 'NOT-TRANSLATED': 0};
		for(var i in result.requests) {
			var request = result.requests[i];
			var state = request.state;
			summary[state.status]++;
			var files_xtc = { type: 'xtc', files: request.xtc };
			var files_hdf = { type: 'hdf5', files: request.hdf5 };
			var num_files = files_xtc.length + files_hdf.length;
			var extra_class = ( num_files > 0 ? 'table_cell_bottom' : '' )+' hdf-manage-run';
			var run_url =
				'<a class="link" href="/apps/logbook?action=select_run_by_id&id='+state.run_id+
				'" target="_blank" title="click to see a LogBook record for this run">'+state.run_number+'</a>';
			var log_url = state.log_available ? '<a class="link" href="translate/'+state.id+'/'+state.id+'.log" target="_blank" title="click to see the log file for the last translation attempt">log</a>' : '';
			html +=
'  <tr>'+
'    <td class="table_cell table_cell_left '+extra_class+'">'+run_url+'</td>'+
'    <td class="table_cell '+extra_class+'">'+state.end_of_run+'</td>'+
'	 <td class="table_cell '+extra_class+'"></td>'+
'    <td class="table_cell '+extra_class+'"></td>'+
'    <td class="table_cell '+extra_class+'">'+state.status+'</td>'+
'    <td class="table_cell '+extra_class+'">'+state.changed+'</td>'+
'    <td class="table_cell '+extra_class+'">'+log_url+'</td>'+
'    <td class="table_cell '+extra_class+'" id="hdf-r-priority-'+state.run_number+'">'+state.priority+'</td>'+
'    <td class="table_cell '+extra_class+'">'+state.actions+'</td>'+
'    <td class="table_cell table_cell_right '+extra_class+'" id="hdf-r-comment-'+state.run_number+'">'+state.comments+'</td>'+
'  </tr>';

			var file_downcounter = num_files;
			var collections = [ files_hdf, files_xtc ];
			for(var j in collections ) {

				var type = collections[j].type;
				var extra_class_if_hdf = type == 'hdf5' ? 'hdf-manage-xtc-files' : '';

				var files = collections[j].files;
				for(var k in files) {

					var file = files[k];
					var extra_class = --file_downcounter > 0 ? 'table_cell_bottom' : '';
					html +=
'  <tr>'+
'    <td class="table_cell table_cell_left '+extra_class+'"></td>'+
'    <td class="table_cell '+extra_class+'"></td>'+
'    <td class="table_cell '+extra_class+' '+extra_class_if_hdf+'">'+file.name+'</td>'+
'	 <td class="table_cell '+extra_class+' '+extra_class_if_hdf+'">'+file.size+'</td>'+
'    <td class="table_cell '+extra_class+' '+extra_class_if_hdf+'"></td>'+
'    <td class="table_cell '+extra_class+' '+extra_class_if_hdf+'"></td>'+
'    <td class="table_cell '+extra_class+' '+extra_class_if_hdf+'"></td>'+
'    <td class="table_cell '+extra_class+' '+extra_class_if_hdf+'"></td>'+
'    <td class="table_cell '+extra_class+' '+extra_class_if_hdf+'"></td>'+
'    <td class="table_cell table_cell_right '+extra_class+' '+extra_class_if_hdf+'"></td>'+
'  </tr>';
				}
			}
		}
		html +=
'</tbody></table>';

		$('#hdf-manage-list').html(html);
		$('#hdf-manage-list').find('.reverse').button().click(function() {
			that.manage_last_request.requests.reverse();
			that.manage_display();
		});
		$('#hdf-manage-list').find('.translate').button().click(function() {
			$(this).button('disable');
			that.translate_run($(this).val());
		});
		$('#hdf-manage-list').find('.translate_all').button().click(function() {
			$('#hdf-manage-list').find('.translate_all').button('disable');
			that.translate_all();
		});
		$('#hdf-manage-list').find('.stop').button().click(function() {
			$(this).button('disable');
			that.stop_translation($(this).val());
		});
		$('#hdf-manage-list').find('.stop_all').button().click(function() {
			$('#hdf-manage-list').find('.stop_all').button('disable');
			that.stop_all_translation();
		});
		$('#hdf-manage-list').find('.escalate').button().click(function() {
			that.escalate_priority($(this).val());
		});
		var html = '';
		for(var status in summary) {
			var counter = summary[status];
			if(counter) {
				if(html != '') html += ', ';
				html += status+': <b>'+counter+'</b>';
			}
		}
		html = '<b>'+result.requests.length+'</b> runs [ '+html+' ]';
		$('#hdf-manage-info').html(html);
		$('#hdf-manage-updated').html(
			'[ Last update on: <b>'+result.updated+'</b> ]');
	}
	this.manage_update = function() {
		$('#hdf-manage-updated').html('Updating...');
		var params = {exper_id: that.exp_id, show_files: '', json: ''};
		var runs   = $('#hdf-manage-ctrl').find('input[name="runs"]'  ).val(); if(runs  != '') params.runs  = runs;
		var status = $('#hdf-manage-ctrl').find('select[name="status"]').val(); if(status != 'any') params.status = status;
		var jqXHR  = $.get('../portal/SearchRequests.php',params,function(data) {
			var result = eval(data);
			if(result.Status != 'success') {
				$('#hdf-manage-updated').html(result.Message);
				return;
			}
			that.manage_last_request = result;
			that.manage_display();
		},
		'JSON').error(function () {
			alert('failed because of: '+jqXHR.statusText);
		});

	};
	this.manage_init = function() {
		$('#hdf-manage-refresh').button().click(function() { that.manage_update(); });
		$('#hdf-manage-ctrl').find('input').keyup(function(e) { if(e.keyCode == 13) that.manage_update(); });
		$('#hdf-manage-ctrl').find('select').change(function() { that.manage_update(); });
		this.manage_update();
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
		this.manage_init();
		return true;
	};
	/* --------------
	 *  Report error
	 * --------------
	 */
	function report_error(msg) {
		$('#popupdialogs').html(
			'<span class="ui-icon ui-icon-alert" style="float:left;"></span> '+msg
		 );
		$('#popupdialogs').dialog({
			resizable: false,
			modal: true,
			buttons: {
				Cancel: function() {
					$(this).dialog('close');
				}
			},
			title: 'Error Message'
		});
	};

}

var hdf = new hdf_create();
