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

    /* The last request data is shared beetween all pages of this
     * application. Hence if anyting request an update all pages will
     * get updated as well.
     */
    this.files_last_request = null;

	/* --------------
	 *  Summary page
	 * --------------
	 */
	this.summary_display = function() {
		$('#datafiles-summary-info').html('Updating...');

    $('#datafiles-summary-info').html('[ Last update on: <b>'+this.files_last_request.updated+'</b> ]');
    $('#datafiles-summary-runs'         ).html(this.files_last_request.summary.runs);
    $('#datafiles-summary-firstrun'     ).html(this.files_last_request.summary.runs ? this.files_last_request.summary.min_run : 'n/a');
    $('#datafiles-summary-lastrun'      ).html(this.files_last_request.summary.runs ? this.files_last_request.summary.max_run : 'n/a');
    $('#datafiles-summary-xtc-size'     ).html(this.files_last_request.summary.xtc.size); 
    $('#datafiles-summary-xtc-files'    ).html(this.files_last_request.summary.xtc.files);
    $('#datafiles-summary-xtc-archived' ).html(this.files_last_request.summary.xtc.archived_html);
    $('#datafiles-summary-xtc-disk'     ).html(this.files_last_request.summary.xtc.disk_html);
    $('#datafiles-summary-hdf5-size'    ).html(this.files_last_request.summary.hdf5.size);
    $('#datafiles-summary-hdf5-files'   ).html(this.files_last_request.summary.hdf5.files);
    $('#datafiles-summary-hdf5-archived').html(this.files_last_request.summary.hdf5.archived_html);
    $('#datafiles-summary-hdf5-disk'    ).html(this.files_last_request.summary.hdf5.disk_html);

	};
	this.summary_init = function() {
		$('#datafiles-summary-refresh').button().click(function() { that.files_update(); });
		this.files_update();
	};

	/* -----------------------------
	 *  Files groupped by runs page
	 * -----------------------------
	 */
	this.files_reverse_order = true;

    function migration_status2html(f) {
        var html = f.start_migration_delay_sec !== undefined ?
            f.start_migration_delay_sec :
            '';
        return html;
    }
    this.file_size_format = 'auto-format-file-size';
    this.file_size = function(f) {
        switch(this.file_size_format) {
            case 'auto-format-file-size': return f.size_auto;
            case 'Bytes'                : return f.size;
            case 'KBytes'               : return f.size_kb;
            case 'MBytes'               : return f.size_mb;
            case 'GBytes'               : return f.size_gb;
            default                     :
        }
    };

    this.page_size_default = 10;
    this.page_size = this.page_size_default;
    this.page_idx  = 0;
    this.page_min_ridx = 0;
    this.page_max_ridx = this.page_size - 1;
    this.repaginate = function () {
        this.page_idx = 0;
        this.page_min_ridx = this.page_idx * this.page_size;
        this.page_max_ridx = Math.min( this.files_last_request.runs.length, this.page_min_ridx + this.page_size ) - 1;
        var html = '';
        for( var pidx=0; pidx < Math.ceil( this.files_last_request.runs.length / this.page_size ); ++pidx ) {

            var min_run_idx = pidx * this.page_size;
            var max_run_idx = Math.min( this.files_last_request.runs.length, (pidx + 1) * this.page_size ) - 1;

            var min_run = this.files_last_request.runs[ min_run_idx ].runnum;
            var max_run = this.files_last_request.runs[ max_run_idx ].runnum;

            var title = 'runs from '+min_run+' through '+max_run;

            var totals  = {
                'RAW_DATA'   : { size_gb: 0, files: 0, archived: 0, local: 0 },
                'MEDIUM_TERM': { size_gb: 0, files: 0, archived: 0, local: 0 }
            };
            var overstay = false;
            for( var ridx = min_run_idx; ridx <= max_run_idx; ++ridx ) {
                var run = this.files_last_request.runs[ ridx ];
                var files = run.files;
                for( var j in files ) {
                    var f = files[j];
                    if( f.size_bytes ) totals[f.storage].size_gb += parseInt(f.size_gb);
                    totals[f.storage]['files'] += 1;
                    if( f.archived_flag ) totals[f.storage].archived += 1;
                    if( f.local_flag ) totals[f.storage].local += 1;
                }
                if( this.files_last_request.overstay.runs.indexOf(run.runnum) > -1 ) overstay = true;
            }
            var extra_class_archived = totals['RAW_DATA'].archived == totals['RAW_DATA'].files ? '' : 'df-r-highlighted';
            var extra_class_local    = totals['RAW_DATA'].local    == totals['RAW_DATA'].files ? '' : 'df-r-highlighted';
            html +=
'  <div class="df-r-hdr" id="df-r-hdr-'+pidx+'" onclick="datafiles.on_page_select('+pidx+');" title="'+title+'">'+
'    <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e df-r-tgl" id="df-r-tgl-'+pidx+'"></span></div>'+
'    <div style="float:left;" class="df-r-min-run">'+min_run+'</div>'+
'    <div style="float:left;" class="df-r-separator">'+( min_run == max_run ? '&nbsp;' : '&dash;' )+'</div>'+
'    <div style="float:left;" class="df-r-max-run">'+( min_run == max_run ? '&nbsp;' : max_run )+'</div>'+
'    <div style="float:left;" class="df-r-overstay">'+(overstay ? '<span class="ui-icon ui-icon-alert"></span>' : '<span>&nbsp;<span>')+'</div>'+
'    <div style="float:left;" class="df-r-size">'+totals['RAW_DATA'].size_gb+'</div>'+
'    <div style="float:left;" class="df-r-files">'+totals['RAW_DATA'].files+'</div>'+
'    <div style="float:left;" class="df-r-archived '+extra_class_archived+'">'+totals['RAW_DATA'].archived+'</div>'+
'    <div style="float:left;" class="df-r-local '+extra_class_local+'">'+totals['RAW_DATA'].local+'</div>'+
'    <div style="float:left;" class="df-r-size1">'+totals['MEDIUM_TERM'].size_gb+'</div>'+
'    <div style="float:left;" class="df-r-files1">'+totals['MEDIUM_TERM'].files+'</div>'+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="df-r-con df-r-hdn" id="df-r-con-'+pidx+'"></div>';
        }
        $('#datafiles-files-pages').html(
'<div style="float:left; margin-left:35px; width:100px;"><span class="datafiles-table-hdr">Run(s)</span></div>'+
'<div style="float:left;                   width: 85px;"><span class="datafiles-table-hdr">RAW_DATA:</span></div>'+
'<div style="float:left;                   width: 80px;"><span class="datafiles-table-hdr-plain">Size [GB]</span></div>'+
'<div style="float:left;                   width: 40px;"><span class="datafiles-table-hdr-plain">Files</span></div>'+
'<div style="float:left;                   width: 70px;"><span class="datafiles-table-hdr-plain">Archived</span></div>'+
'<div style="float:left;                   width: 70px;"><span class="datafiles-table-hdr-plain">Local</span></div>'+
'<div style="float:left;                   width:110px;"><span class="datafiles-table-hdr">MEDIUM_TERM:</span></div>'+
'<div style="float:left;                   width: 80px;"><span class="datafiles-table-hdr-plain">Size [GB]</span></div>'+
'<div style="float:left;                   width: 40px;"><span class="datafiles-table-hdr-plain">Files</span></div>'+
'<div style="clear:both;"></div>');
        $('#datafiles-files-list').html(html);

        this.on_page_select(this.page_idx);
    };
    this.on_page_select = function(page_idx) {
        this.page_idx = page_idx;
        this.page_min_ridx = this.page_idx * this.page_size;
        this.page_max_ridx = Math.min( this.files_last_request.runs.length, this.page_min_ridx + this.page_size ) - 1;
		var toggler='#df-r-tgl-'+this.page_idx;
		var container='#df-r-con-'+this.page_idx;
		if( $(container).hasClass('df-r-hdn')) {
			$(toggler).removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s');
			$(container).removeClass('df-r-hdn').addClass('df-r-vis');
			$('#df-r-con-'+this.page_idx).html('Loading...');
            this.files_display();
		} else {
			$(toggler).removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e');
			$(container).removeClass('df-r-vis').addClass('df-r-hdn');
			$(container).html('');
		}
    };

    this.file2html = function(f, run_url, first_of_a_kind, display, extra_class, page_idx, ridx) {
		var hightlight_class = f.type != 'XTC' ? 'datafiles-files-highlight' : '';
		var html =
'  <tr>'+
'    <td class="table_cell table_cell_left '+extra_class+'">'+run_url+'</td>'+
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.name+'</td>'+
			(display.type?
'	 <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.type+'</td>':'')+
			(display.size?
'    <td class="table_cell '+hightlight_class+' '+extra_class+'" style="text-align:right">'+this.file_size(f)+'</td>':'')+
			(display.created?
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.created+'</td>':'')+
			(display.checksum?
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.checksum+'</td>':'')+
			(display.archived?
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.archived+'</td>':'')+
			(display.local?
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.local+'</td>':'');
		if(display.storage) {
            html +=
'	 <td class="table_cell '+hightlight_class+' '+extra_class+'">PCDS</td>'+
'	 <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.storage+'</td>'+
'	 <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.allowed_stay+'</td>'+
'	 <td class="table_cell '+hightlight_class+' '+extra_class+'">';
            if(first_of_a_kind) {
                if(f.local_flag) {
                    html +=
'      <button class="delete_from_disk" style="font-size:7px;" onclick="datafiles.delete_from_disk('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+page_idx+','+ridx+')" title="delete all '+f.type+' files of run '+f.runnum+' from the '+f.storage+' disk storage">DELETE</button>';
                }
                if(f.storage == 'RAW_DATA') {
                    if(f.local_flag) {
                        html +=
'      <button class="move_to_medium_term" style="font-size:7px;" onclick="datafiles.move_to_medium_term('+f.runnum+',\''+f.type+'\')" title="move all '+f.type+' files of run '+f.runnum+' to MEDIUM_TERM disk storage">MOVE</button>';
                    }
                    if(f.archived_flag && !f.local_flag && !f.restore_flag) {
                        html +=
'      <button class="restore_from_archive" style="font-size:7px;" onclick="datafiles.restore_from_archive('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+page_idx+','+ridx+')" title="restore all '+f.type+' files of run '+f.runnum+' from tape archive to the MEDIUM_TERM disk storage">RESTORE</button>';
                    }
                }
            }
            html +=
'    </td>';
        }
        html +=
			(display.migration?
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class+'">'+migration_status2html(f)+'</td>':'')+
'  </tr>';
		return html;
	};
	this.files_display = function() {

		var display = new Object();
		display.type      = $('#datafiles-files-wa' ).find('input[name="type"]'     ).attr('checked');
		display.size      = $('#datafiles-files-wa' ).find('input[name="size"]'     ).attr('checked');
		display.created   = $('#datafiles-files-wa' ).find('input[name="created"]'  ).attr('checked');
	    display.checksum  = $('#datafiles-files-wa' ).find('input[name="checksum"]' ).attr('checked');
		display.archived  = $('#datafiles-files-wa' ).find('input[name="archived"]' ).attr('checked');
		display.local     = $('#datafiles-files-wa' ).find('input[name="local"]'    ).attr('checked');
		display.storage   = $('#datafiles-files-wa' ).find('input[name="storage"]'  ).attr('checked');
	    display.migration = $('#datafiles-files-wa' ).find('input[name="migration"]').attr('checked');

		var num_runs  = 0;
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
'    <td class="table_hdr">On Disk</td>':'')+
            (display.storage?
'    <td class="table_hdr">Site</td>'+
'    <td class="table_hdr">Storage Class</td>'+
'    <td class="table_hdr">Time left</td>'+
'    <td class="table_hdr">Operations</td>':'')+
            (display.migration?
'    <td class="table_hdr">DAQ-to-Disk Migration delay [s]</td>':'')+
'  </tr>';

        for( var i = this.page_min_ridx; i <= this.page_max_ridx; ++i) {
            ++num_runs;
            var run = this.files_last_request.runs[i];
            var first = true;
            var first_of_a_kind = {};
            for(var j in run.files) {
                ++num_files;
                var extra_class = (j != run.files.length - 1) ? 'table_cell_bottom' : '';
                var f = run.files[j];
                if( first_of_a_kind[f.storage]         === undefined ) first_of_a_kind[f.storage]         = {};
                if( first_of_a_kind[f.storage][f.type] === undefined ) first_of_a_kind[f.storage][f.type] = true;
                html += this.file2html( f, first ? run.url : '', first_of_a_kind[f.storage][f.type], display, extra_class, this.page_idx, i );
                first = false;
                first_of_a_kind[f.storage][f.type] = false;
            }
        }
		html +=
'</tbody></table>';

        var page = $('#df-r-con-'+this.page_idx);
        page.html(html);
        page.find('.move_to_medium_term').button();
        page.find('.restore_from_archive').button();
        page.find('.delete_from_disk').button();
	}
    this.stats_display = function() {
        var stats = { runs: 0, files: 0, size_gb: 0, storage: {}};
        var runs = this.files_last_request.runs;
        for(var i in runs) {
            var run = runs[i];
            for(var j in run.files) {
                var f = run.files[j];
                if( stats.storage[f.storage] === undefined ) stats.storage[f.storage] = {};
                if( stats.storage[f.storage][f.type] === undefined ) stats.storage[f.storage][f.type] = {files: 0, size_gb: 0};
                stats.storage[f.storage][f.type].files += 1;
                if( f.size_bytes != 0 ) {
                    var size_gb = parseInt(f.size_gb);
                    stats.storage[f.storage][f.type].size_gb += size_gb;
                    stats.size_gb += size_gb;
                }
                stats.files += 1;
            }
            stats.runs += 1;
        }
        var overstay = this.files_last_request.overstay;
        $('#datafiles-files-info').html(
'<div style="float:left; padding:3px;">'+
'  <span style="font-weight:bold;">'+stats.files+'</span> files in <span style="font-weight:bold;">'+stats.runs+'</span> runs, <b>'+stats.size_gb+'</b> GB in total'+
'</div>'+
            ( overstay.total_runs ?
'<div style="float:left; margin-left:20px; padding:2px; border: solid 1px;">'+
'  <span style="float:left;"></span>'+
'  <span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
'  <span style="float:left;"> <b>'+overstay.total_files+'</b> files in <b>'+overstay.total_runs+'</b> runs, <b>'+overstay.total_size_gb+'</b> GB overstay in <b>RAW_DATA</b></span>'+
'  <span style="float:left;"></span>'+
'</div>' :
'' )
        );
		$('#datafiles-files-updated').html(
'[ Last update on: <b>'+this.files_last_request.updated+'</b> ]'
        );
    };
	this.files_update = function() {
		$('#datafiles-summary-info').html('Updating...');
		$('#datafiles-files-updated').html('Updating...');
		var params   = {exper_id: this.exp_id};
		var runs     = $('#datafiles-files-ctrl').find('input[name="runs"]'     ).val(); if(runs     != '')    params.runs     = runs;
		var types    = $('#datafiles-files-ctrl').find('select[name="types"]'   ).val(); if(types    != 'any') params.types    = types;
		var created  = $('#datafiles-files-ctrl').find('select[name="created"]' ).val(); if(created  != 'any') params.created  = created;
		var checksum = $('#datafiles-files-ctrl').find('select[name="checksum"]').val(); if(checksum != 'any') params.checksum = checksum == 'is known' ? 1 : 0;
		var archived = $('#datafiles-files-ctrl').find('select[name="archived"]').val(); if(archived != 'any') params.archived = archived == 'yes' ? 1 : 0;
		var local    = $('#datafiles-files-ctrl').find('select[name="local"]'   ).val(); if(local    != 'any') params.local    = local    == 'yes' ? 1 : 0;
		var storage  = $('#datafiles-files-ctrl').find('select[name="storage"]' ).val(); if(storage  != 'any') params.storage  = storage;
		var jqXHR = $.get('../portal/SearchFiles.php',params,function(data) {
			var result = eval(data);
			if(result.Status != 'success') {
				report_error(result.Message);
				return;
			}
			that.files_last_request = result;
			for(var i in result.runs) {
				var run = result.runs[i];
				for(var j in run.files) {
					var f = run.files[j];
                    f.run_url = run.url;
                }
            }
			if( that.files_reverse_order ) {
				that.files_last_request.runs.reverse();
			}
            that.repaginate();
			that.files_display();
			that.stats_display();
        	that.summary_display();
		},
		'JSON').error(function () {
			report_error('failed because of: '+jqXHR.statusText);
		});
	};
	this.files_init = function() {
		$('#datafiles-files-refresh').button().click(function() { that.files_update(); });
		$('#datafiles-files-ctrl').find('input').keyup(function(e) { if(e.keyCode == 13) that.files_update(); });
		$('#datafiles-files-ctrl').find('select').change(function() { that.files_update(); });
		$('#datafiles-files-reverse').button().click(function() {
			that.files_reverse_order = !that.files_reverse_order;
			that.files_last_request.runs.reverse();
            that.repaginate();
			that.files_display();
		});
		$('#datafiles-files-wa' ).find('input[name="storage"]' ).attr('checked','checked');
		$('#datafiles-files-wa' ).find('input[name="type"]'    ).attr('checked','checked');
		$('#datafiles-files-wa' ).find('input[name="size"]'    ).attr('checked','checked');
		$('#datafiles-files-wa' ).find('input[name="created"]' ).attr('checked','checked');
		$('#datafiles-files-wa' ).find('input[name="archived"]').attr('checked','checked');
		$('#datafiles-files-wa' ).find('input[name="local"]'   ).attr('checked','checked');
		$('#datafiles-files-wa' ).find('input').change(function(){
			that.files_display();
		});
		$('#datafiles-files-wa' ).find('select[name="format"]').change(function(){
            that.file_size_format = $(this).val();
			that.files_display();
		});
        $('#datafiles-files-wa').find('select[name="page_size"]').change(function(){
            if( $(this).val() == 'auto-page-size' ) that.page_size = that.page_size_default;
            else                                    that.page_size = parseInt($(this).val());
            that.repaginate();
			that.files_display();
		});
		this.files_update();
	};
    this.move_to_medium_term = function(runnum,type) {
        ask_yes_no(
            'Confirm File Migration',
            'Are you sure you want to move all <b>'+type+'</b> files of run <b>'+runnum+'</b> to the <b>MEDIUM_TERM</b> disk storage?<br><br>'+
            'Note this operation will succeed only if your experiment has sufficient quota to accomodate new files. '+
            'Once moved the files will be able to stay in the MEDIUM_TERM storage as long as it\'s permited by <b>LCLS Data Retention Policies</b>.',
            function() {
                report_error('The back-end support for this operation is yet to be implemented');
            },
            null
        );
    };
    this.restore_from_archive = function(runnum, type, storage, page_idx, ridx) {
        ask_yes_no(
            'Confirm File Recovery from Tape Archive',
            'Are you sure you want to restore all <b>'+type+'</b> files of run <b>'+runnum+'</b> from Tape Archive to the <b>MEDIUM_TERM</b> disk storage?<br><br>'+
            'Note this operation will succeed only if your experiment has sufficient quota to accomodate new files. '+
            'Once restored the files will be able to stay in the MEDIUM_TERM storage as long as it\'s permited by <b>LCLS Data Retention Policies</b>.',
            function() {
                that.restore_from_archive_impl(
                    runnum,
                    type,
                    storage,
                    function() {
                        // Update entries for all relevant files from the transient data structure
                        //
                        var run = that.files_last_request.runs[ridx];
                        for( var i in run.files ) {
                            var f = run.files[i];
                            if((f.runnum == runnum) && (f.type == type) && (f.storage == storage)) {
                                f.local = '<span style="color:black;">Restoring from archive...</span>';
                                f.restore_flag = 1;
                                f.restore_requested_time = '';
                                f.restore_requested_uid = auth_remote_user;
                            }
                        }
                        // Redisplay the corresponding page
                        //
                        that.on_page_select(page_idx);
                        that.on_page_select(page_idx);
                    }
                );
            },
            null
        );
    };
	this.restore_from_archive_impl = function(runnum, type, storage, on_success) {

		var params  = {exper_id: this.exp_id, runnum: runnum, type: type, storage: storage };

		var jqXHR = $.get(
            '../portal/RestoreFiles.php',
            params,
            function(data) {
                var result = eval(data);
                if(result.status != 'success') {
                    report_error(result.message);
                    return;
                }
                if( on_success ) on_success();
            },
            'JSON'
        ).error(function () {
            report_error('failed because of: '+jqXHR.statusText); });
	};
    this.delete_from_disk = function(runnum, type, storage, page_idx, ridx) {
        var warning = '';
        if(storage == 'MEDIUM_TERM')
            warning =
            'The files you\'re about delete are not archived. '+
            'Proceeding with this operation may result in irreversable loss of informaton. '+
            'Also note that this operation will be reported to the PI of the experiment. ';
        else if(storage == 'RAW_DATA')
            warning =
            'Remember that these files can be later restored from Tape Archive';
        ask_yes_no(
            'Confirm File Deletion',
            'Are you sure you want to delete all <b>'+type+'</b> files of run <b>'+runnum+'</b> from the <b>'+storage+'</b> disk storage?<br><br>'+
            warning,
            function() {
                that.delete_from_disk_impl(
                    runnum,
                    type,
                    storage,
                    function() {
                        // Update entries for all relevant files from the transient data structure
                        //
                        var run = that.files_last_request.runs[ridx];
                        for( var i in run.files ) {
                            var f = run.files[i];
                            if((f.runnum == runnum) && (f.type == type) && (f.storage == storage)) {
                                f.local_flag = 0;
                                f.local = '<span style="color:red;">No</span>';
                                f.allowed_stay_sec = 0;
                                f.allowed_stay = '';
                            }
                        }
                        // Redisplay the corresponding page
                        //
                        that.on_page_select(page_idx);
                        that.on_page_select(page_idx);
                    }
                );
            },
            null
        );
    };
	this.delete_from_disk_impl = function(runnum, type, storage, on_success) {

		var params  = {exper_id: this.exp_id, runnum: runnum, type: type, storage: storage };

		var jqXHR = $.get(
            '../portal/DeleteFiles.php',
            params,
            function(data) {
                var result = eval(data);
                if(result.status != 'success') {
                    report_error(result.message);
                    return;
                }
                if( on_success ) on_success();
            },
            'JSON'
        ).error(function () {
            report_error('failed because of: '+jqXHR.statusText); });
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
