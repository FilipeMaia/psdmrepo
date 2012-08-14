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

    this.files_display_header = function() {

        var min_run = this.files_last_request.runs.length ? this.files_last_request.runs[ 0 ].runnum : 0;
        var max_run = this.files_last_request.runs.length ? this.files_last_request.runs[ this.files_last_request.runs.length -1 ].runnum : 0;
        if( max_run < min_run ) {
            var swap = max_run;
            max_run = min_run;
            min_run = swap;
        }
        var totals  = {
            'TOTAL'      : { files: 0, size_gb: 0 },
            'TAPE'       : { files: 0, size_gb: 0 },
            'MEDIUM-TERM': { files: 0, size_gb: 0 },
            'SHORT-TERM' : { files: 0, size_gb: 0 }
        };
        for( var ridx = 0; ridx < this.files_last_request.runs.length; ++ridx ) {
            var run = this.files_last_request.runs[ ridx ];
            var files = run.files;
            for( var j in files ) {
                var f = files[j];
                var size_gb = f.archived_flag || f.local_flag ? parseInt(f.size_gb) : 0;
                totals['TOTAL'  ].files   += 1;
                totals['TOTAL'  ].size_gb += size_gb
                if( f.archived_flag ) {
                    totals['TAPE'].files   += 1;
                    totals['TAPE'].size_gb += size_gb;
                }
                totals[f.storage].files   += 1;
                totals[f.storage].size_gb += size_gb
            }
        }
        $('#datafiles-files-pages #header').html(
'<div style="float:left; margin-left:35px; width:120px;">'+
'  <div><span class="datafiles-table-hdr">&nbsp;</span></div>'+
'  <div style="padding-bottom:6px;"><span class="datafiles-table-hdr">R U N (s)</span></div>'+
'  <div><span class="datafiles-table-hdr">&nbsp;</span></div>'+
'</div>'+
'<div style="float:left; width:160px;">'+
'  <div ><span class="datafiles-table-hdr">T O T A L &nbsp; D A T A</span></div>'+
'  <div style="padding-bottom:6px; padding-left:25px;"><span class="datafiles-table-hdr">any storage</span></div>'+
'  <div style="float:left; width:80px;"><span class="datafiles-table-hdr-plain">[# files]</span></div>'+
'  <div style="float:left; width:60px;"><span class="datafiles-table-hdr-plain">[GB]</span></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div style="float:left; width:160px;">'+
'  <div                            ><span class="datafiles-table-hdr">L O N G - T E R M</span></div>'+
'  <div style="padding-bottom:6px; padding-left:15px;"><span class="datafiles-table-hdr">10 years, tape</span></div>'+
'  <div style="float:left; width:80px;"><span class="datafiles-table-hdr-plain">[# files]</span></div>'+
'  <div style="float:left; width:60px;"><span class="datafiles-table-hdr-plain">[GB]</span></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div style="float:left; width:160px;">'+
'  <div                            ><span class="datafiles-table-hdr">S H O R T - T E R M</span></div>'+
'  <div style="padding-bottom:6px; padding-left:20px;"><span class="datafiles-table-hdr">'+this.files_last_request.policies['SHORT-TERM'].retention_months+' months, disk</span></div>'+
'  <div style="float:left; width:80px;"><span class="datafiles-table-hdr-plain">[# files]</span></div>'+
'  <div style="float:left; width:60px;"><span class="datafiles-table-hdr-plain">[GB]</span></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div style="float:left; width:160px;">'+
'  <div                            ><span class="datafiles-table-hdr">M E D I U M - T E R M</span></div>'+
'  <div style="padding-bottom:6px; padding-left:25px;"><span class="datafiles-table-hdr">'+this.files_last_request.policies['MEDIUM-TERM'].retention_months+' months, disk</span></div>'+
'  <div style="float:left; width:80px;"><span class="datafiles-table-hdr-plain">[# files]</span></div>'+
'  <div style="float:left; width:60px;"><span class="datafiles-table-hdr-plain">[GB]</span></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div style="clear:both;"></div>'
        );
        $('#datafiles-files-pages #summary').html(
'<div style="float:left; margin-left:18px; width:140px;">'+
'  <div style="float:left;" class="df-r-min-run">'  +min_run+'</div>'+
'  <div style="float:left;" class="df-r-separator">'+( min_run == max_run ? '&nbsp;' : '-' )+'</div>'+
'  <div style="float:left;" class="df-r-max-run">'  +( min_run == max_run ? '&nbsp;' : max_run )+'</div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div style="float:left; width:160px;">'+
'  <div style="float:left;" class="df-r-total-files"><b>'+totals['TOTAL'].files  +'</b></div>'+
'  <div style="float:left;" class="df-r-total-size" ><b>'+totals['TOTAL'].size_gb+'</b></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div style="float:left; width:160px;">'+
'  <div style="float:left;" class="df-r-total-files"><b>'+totals['TAPE'].files  +'</b></div>'+
'  <div style="float:left;" class="df-r-total-size" ><b>'+totals['TAPE'].size_gb+'</b></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div style="float:left; width:160px;">'+
'  <div style="float:left;" class="df-r-total-files"><b>'+totals['SHORT-TERM'].files  +'</b></div>'+
'  <div style="float:left;" class="df-r-total-size" ><b>'+totals['SHORT-TERM'].size_gb+'</b></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div style="float:left; width:160px;">'+
'  <div style="float:left;" class="df-r-total-files"><b>'+totals['MEDIUM-TERM'].files  +'</b></div>'+
'  <div style="float:left;" class="df-r-total-size" ><b>'+totals['MEDIUM-TERM'].size_gb+'</b></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+
'<div style="clear:both;"></div>'
        );
    };
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
            if( max_run < min_run ) {
                var swap = max_run;
                max_run = min_run;
                min_run = swap;
            }
            var title = 'runs from '+min_run+' through '+max_run;

            html +=
'  <div class="df-r-hdr" id="df-r-hdr-'+pidx+'" onclick="datafiles.on_page_select('+pidx+');" title="'+title+'">'+
'    <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e df-r-tgl" id="df-r-tgl-'+pidx+'"></span></div>'+
'    <div style="float:left;" class="df-r-min-run">'        +min_run+'</div>'+
'    <div style="float:left;" class="df-r-separator">'      +( min_run == max_run ? '&nbsp;' : '-' )+'</div>'+
'    <div style="float:left;" class="df-r-max-run">'        +( min_run == max_run ? '&nbsp;' : max_run )+'</div>'+
'    <div style="float:left;" class="df-r-total-files"></div>'+
'    <div style="float:left;" class="df-r-total-size"></div>'+
'    <div style="float:left;" class="df-r-tape-files"></div>'+
'    <div style="float:left;" class="df-r-tape-size"></div>'+
'    <div style="float:left;" class="df-r-raw-files"></div>'+
'    <div style="float:left;" class="df-r-raw-size"></div>'+
'    <div style="float:left;" class="df-r-raw-overstay"></div>'+
'    <div style="float:left;" class="df-r-medium-files"></div>'+
'    <div style="float:left;" class="df-r-medium-size"></div>'+
'    <div style="float:left;" class="df-r-medium-overstay"></div>'+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="df-r-con df-r-hdn" id="df-r-con-'+pidx+'"></div>';
        }
        $('#datafiles-files-list').html(html);

        for( var pidx=0; pidx < Math.ceil( this.files_last_request.runs.length / this.page_size ); ++pidx )
            this.update_page_header(pidx);

        this.on_page_select(this.page_idx);
    };
    this.update_page_header = function(pidx) {

        var min_run_idx = pidx * this.page_size;
        var max_run_idx = Math.min( this.files_last_request.runs.length, (pidx + 1) * this.page_size ) - 1;

        var totals  = {
            'TOTAL'      : { files: 0, size_gb: 0 },
            'TAPE'       : { files: 0, size_gb: 0 },
            'MEDIUM-TERM': { files: 0, size_gb: 0 },
            'SHORT-TERM' : { files: 0, size_gb: 0 }
        };
        var overstay_raw    = false;
        var overstay_medium = false;
        for( var ridx = min_run_idx; ridx <= max_run_idx; ++ridx ) {
            var run = this.files_last_request.runs[ ridx ];
            var files = run.files;
            for( var j in files ) {
                var f = files[j];
                var size_gb = f.archived_flag || f.local_flag ? parseInt(f.size_gb) : 0;
                totals['TOTAL'  ].files   += 1;
                totals['TOTAL'  ].size_gb += size_gb
                if( f.archived_flag ) {
                    totals['TAPE'].files   += 1;
                    totals['TAPE'].size_gb += size_gb;
                }
                if( f.local_flag ) {
                    totals[f.storage].files   += 1;
                    totals[f.storage].size_gb += size_gb
                }
            }
            if(( this.files_last_request.overstay['SHORT-TERM' ] != undefined ) && ( this.files_last_request.overstay['SHORT-TERM' ]['runs'][run.runnum] != undefined )) overstay_raw    = true;
            if(( this.files_last_request.overstay['MEDIUM-TERM'] != undefined ) && ( this.files_last_request.overstay['MEDIUM-TERM']['runs'][run.runnum] != undefined )) overstay_medium = true;
        }
        var elem = $('#df-r-hdr-'+pidx);
        elem.find('.df-r-total-files'    ).html(totals['TOTAL'].files);
        elem.find('.df-r-total-size'     ).html(totals['TOTAL'].size_gb);
        elem.find('.df-r-tape-files'     ).html(totals['TAPE' ].files);
        elem.find('.df-r-tape-size'      ).html(totals['TAPE' ].size_gb);
        elem.find('.df-r-raw-files'      ).html(totals['SHORT-TERM'].files);
        elem.find('.df-r-raw-size'       ).html(totals['SHORT-TERM'].size_gb);
        elem.find('.df-r-raw-overstay'   ).html(overstay_raw ? '<span class="ui-icon ui-icon-alert"></span>' : '<span>&nbsp;<span>');
        elem.find('.df-r-medium-files'   ).html(totals['MEDIUM-TERM'].files);
        elem.find('.df-r-medium-size'    ).html(totals['MEDIUM-TERM'].size_gb);
        elem.find('.df-r-medium-overstay').html(overstay_medium ? '<span class="ui-icon ui-icon-alert"></span>' : '<span>&nbsp;<span>');

        $('#datafiles-files-table-ctrl #quota-usage span').html('Quota usage: '+
            this.files_last_request.policies['MEDIUM-TERM'].quota_used_gb+' GB (out of '+this.files_last_request.policies['MEDIUM-TERM'].quota_gb+')');
    }
    this.on_page_select = function(pidx) {
        this.page_idx = pidx;
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

    this.file2html = function(f, run_url, first_of_a_kind, display, extra_class1, extra_class2, pidx, ridx) {
		var hightlight_class = ''; // f.type != 'XTC' ? 'datafiles-files-highlight' : '';
		var html =
'  <tr>'+
'    <td class="table_cell table_cell_left '+extra_class1+'">'+run_url+'</td>'+
'    <td class="table_cell '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.name+'</td>'+
			(display.type?
'	 <td class="table_cell '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.type+'</td>':'')+
			(display.size?
'    <td class="table_cell '+hightlight_class+' '+extra_class1+' '+extra_class2+'" style="text-align:right">&nbsp;'+this.file_size(f)+'</td>':'')+
			(display.created?
'    <td class="table_cell '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.created+'</td>':'')+
			(display.checksum?
'    <td class="table_cell '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.checksum+'</td>':'');
        html +=
'    <td class="table_cell '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.archived+'</td>';
		if(display.storage) {
            switch(f.storage) {
                case 'SHORT-TERM':
                    html +=
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.local+'</td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.allowed_stay['SHORT-TERM'].expiration+'</td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.allowed_stay['SHORT-TERM'].allowed_stay+'</td>'+
'    <td class="table_cell                  '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;';
                    if(first_of_a_kind) {
                        if(f.local_flag) {
                            html +=
'      <button class="delete_from_raw" style="font-size:7px;" onclick="datafiles.delete_from_disk('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+pidx+','+ridx+')" title="delete all '+f.type+' files of run '+f.runnum+' from the '+f.storage+' disk storage">DELETE</button>'+
'      <button class="move_to_medium_term" style="font-size:7px;" onclick="datafiles.move_to_medium_term('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+pidx+','+ridx+')" title="save all '+f.type+' files of run '+f.runnum+' to MEDIUM-TERM disk storage">SAVE TO MEDIUM</button>';
                        }
                        if(f.archived_flag && !f.local_flag && !f.restore_flag) {
                            html +=
'      <button class="restore_from_archive" style="font-size:7px;" onclick="datafiles.restore_from_archive('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+pidx+','+ridx+')" title="restore all '+f.type+' files of run '+f.runnum+' from tape archive to the '+f.storage+' disk storage">RESTORE FROM TAPE</button>';
                        }
                    }
                    html +=
'    </td>'+
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;</td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;</td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;</td>'+
'	 <td class="table_cell                  '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;</td>';
                    break;
                case 'MEDIUM-TERM':
                    html +=
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;</td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;</td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;</td>'+
'	 <td class="table_cell                  '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;</td>'+
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.local+'</td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.allowed_stay['MEDIUM-TERM'].expiration+'</td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.allowed_stay['MEDIUM-TERM'].allowed_stay+'</td>'+
'    <td class="table_cell                  '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;';
                    if(first_of_a_kind) {
                        if(f.local_flag) {
                            html +=
'      <button class="delete_from_medium" style="font-size:7px;" onclick="datafiles.delete_from_disk('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+pidx+','+ridx+')" title="remove all '+f.type+' files of run '+f.runnum+' from the '+f.storage+' disk storage and move them back to the SHORT-TERM storage">REMOVE</button>';
                        }
                        if(f.archived_flag && !f.local_flag && !f.restore_flag) {
                            html +=
'      <button class="restore_from_archive" style="font-size:7px;" onclick="datafiles.restore_from_archive('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+pidx+','+ridx+')" title="restore all '+f.type+' files of run '+f.runnum+' from tape archive to the '+f.storage+' disk storage">RESTORE FROM TAPE</button>';
                        }
                    }
                    html +=
'    </td>';
                    break;
            }
        } else {
            html +=
'    <td class="table_cell '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+f.local+'</td>';
        }
        html +=
			(display.migration?
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class1+' '+extra_class2+'">&nbsp;'+migration_status2html(f)+'</td>':'')+
'  </tr>';
		return html;
	};
	this.files_display = function() {

		var display = new Object();
		display.type      = $('#datafiles-files-wa' ).find('input[name="type"]'     ).attr('checked');
		display.size      = $('#datafiles-files-wa' ).find('input[name="size"]'     ).attr('checked');
		display.created   = $('#datafiles-files-wa' ).find('input[name="created"]'  ).attr('checked');
	    display.checksum  = $('#datafiles-files-wa' ).find('input[name="checksum"]' ).attr('checked');
		display.storage   = $('#datafiles-files-wa' ).find('input[name="storage"]'  ).attr('checked');
	    display.migration = $('#datafiles-files-wa' ).find('input[name="migration"]').attr('checked');

        var rowspan = display.storage ? 2 : 1;

		var num_runs  = 0;
		var num_files = 0;

		var html =
'<table><tbody>'+
'  <tr>'+
'    <td class="table_hdr" rowspan='+rowspan+' align="right"  >Run</td>'+
'    <td class="table_hdr" rowspan='+rowspan+' align="center" >File</td>'+
          	(display.type?
'    <td class="table_hdr" rowspan='+rowspan+' align="center" >Type</td>':'')+
			(display.size?
'    <td class="table_hdr" rowspan='+rowspan+' align="right" >Size</td>':'')+
            (display.created?
'    <td class="table_hdr" rowspan='+rowspan+' align="center" >Created</td>':'')+
			(display.checksum?
'    <td class="table_hdr" rowspan='+rowspan+' align="center" >Checksum</td>':'');
        html +=
'    <td class="table_hdr" rowspan='+rowspan+' align="center" >On tape</td>';
        if(display.storage)
            html +=
'    <td class="table_hdr " colspan=4 align="center" >SHORT-TERM</td>'+
'    <td class="table_hdr " colspan=4 align="center" >MEDIUM-TERM</td>';
        else
            html +=
'    <td class="table_hdr">On Disk</td>';
(display.migration?
'    <td class="table_hdr" rowspan='+rowspan+' align="center" >DAQ-to-Disk Migration delay [s]</td>':'')+
'  </tr>';
        if(display.storage)
            html +=
'  <tr>'+
'    <td class="table_hdr " align="center" >on disk</td>'+
'    <td class="table_hdr " align="center" >expiration</td>'+
'    <td class="table_hdr " align="center" >allowed stay</td>'+
'    <td class="table_hdr " align="center" >actions</td>'+
'    <td class="table_hdr " align="center" >on disk</td>'+
'    <td class="table_hdr " align="center" >expiration</td>'+
'    <td class="table_hdr " align="center" >allowed stay</td>'+
'    <td class="table_hdr " align="center" >actions</td>'+
'  </tr>';

        for( var i = this.page_min_ridx; i <= this.page_max_ridx; ++i) {
            ++num_runs;
            var run = this.files_last_request.runs[i];
            var first = true;
            var previous_type = '';
            var first_of_a_kind = {};
            for(var j in run.files) {
                ++num_files;
                var f = run.files[j];
                var extra_class1 = (j != run.files.length - 1) ? 'table_cell_bottom' : '';
                var extra_class2 = (previous_type != '') && (previous_type != f.type) ? 'table_cell_top' : '';
                previous_type = f.type;
                if( first_of_a_kind[f.storage]         === undefined ) first_of_a_kind[f.storage]         = {};
                if( first_of_a_kind[f.storage][f.type] === undefined ) first_of_a_kind[f.storage][f.type] = true;
                html += this.file2html( f, first ? run.url : '', first_of_a_kind[f.storage][f.type], display, extra_class1, extra_class2, this.page_idx, i );
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
        page.find('.delete_from_raw').button().button('disable');
        page.find('.delete_from_medium').button();
        $('#datafiles-files-table-ctrl #quota-usage').css('display', display.storage ? 'block' : 'none' );
	};
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
        var html = '';
        for( var storage in overstay ) {
            if(html == '')
                html +=
'<div>';
            else
                html +=
'  <span style="float:left;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>';
            html +=
'  <span style="float:left;"></span>'+
'  <span class="ui-icon ui-icon-alert" style="float:left;"></span>'+
'  <span style="float:left;"> <b>'+overstay[storage].total_files+'</b> files in <b>'+overstay[storage].total_runs+'</b> runs, <b>'+parseInt(overstay[storage].total_size_gb)+'</b> GB overstay in <b>'+storage+'</b> storage</span>';
        }
        if(html != '')
            html +=
'</div>';
        $('#datafiles-files-info').html(html);
		$('#datafiles-files-updated').html(
'[ Last update on: <b>'+this.files_last_request.updated+'</b> ]'
        );
    };
	this.files_update = function() {
		$('#datafiles-summary-info').html('Updating...');
		$('#datafiles-files-updated').html('Updating...');
		var params   = {exper_id: this.exp_id};
		var runs     = $('#datafiles-files-ctrl').find('input[name="runs"]'     ).val(); if(runs     != '') params.runs     = runs;
		var types    = $('#datafiles-files-ctrl').find('select[name="types"]'   ).val(); if(types    != '') params.types    = types;
		var created  = $('#datafiles-files-ctrl').find('select[name="created"]' ).val(); if(created  != '') params.created  = created;
		var checksum = $('#datafiles-files-ctrl').find('select[name="checksum"]').val(); if(checksum != '') params.checksum = checksum == 'is known' ? 1 : 0;
		var archived = $('#datafiles-files-ctrl').find('select[name="archived"]').val(); if(archived != '') params.archived = archived == 'yes' ? 1 : 0;
		var local    = $('#datafiles-files-ctrl').find('select[name="local"]'   ).val(); if(local    != '') params.local    = local    == 'no'  ? 0 : 1;
		switch($('#datafiles-files-ctrl').find('select[name="local"]').val()) {
        case '90 DAYS': params.storage = 'SHORT-TERM';  break;
        case '2 YEARS': params.storage = 'MEDIUM-TERM'; break;
        }
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
            that.files_display_header();
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
		$('#datafiles-files-reset').button().click(function() {
    		$('#datafiles-files-ctrl').find('input').val('');
            $('#datafiles-files-ctrl').find('select').val('');
            that.files_update();
        });
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

    this.confirm_move = true;
    this.move_to_medium_term = function(runnum, type, storage, pidx, ridx) {
        if( !this.confirm_move ) {
            this.move_to_medium_term_impl(runnum, type, storage, pidx, ridx );
            return;
        }
        ask_yes_no(
            'Confirm File Migration',
            'Are you sure you want to save all <b>'+type+'</b> files of run <b>'+runnum+'</b> to the <b>MEDIUM-TERM</b> disk storage?<br><br>'+
            'Note this operation will succeed only if your experiment has sufficient quota to accomodate new files. '+
            'Once saved the files will be able to stay in the MEDIUM-TERM storage as long as it\'s permited by <b>LCLS Data Retention Policies</b>.<br><br>'+
            '<span class="ui-icon ui-icon-info" style="float:left; margin-right:4px;"></span><input type="checkbox" id="datafiles-confirm-move" /> check to prevent this dialog for the rest of the current session',
            function() {
                that.confirm_move = $('#datafiles-confirm-move').attr('checked') ? false : true;
                that.move_to_medium_term_impl(runnum, type, storage, pidx, ridx );
            },
            null
        );
    };
	this.move_to_medium_term_impl = function(runnum, type, storage, pidx, ridx) {

		var params  = {exper_id: this.exp_id, runnum: runnum, type: type, storage: storage };

		var jqXHR = $.get(
            '../portal/MoveFiles.php',
            params,
            function(data) {
                var result = eval(data);
                if(result.status != 'success') {
                    report_error(result.message);
                    return;
                }
                that.files_last_request.policies['MEDIUM-TERM'].quota_used_gb = result.medium_quota_used_gb;

                // Update entries for all relevant files from the transient data structure
                //
                var run = that.files_last_request.runs[ridx];
                if( run.runnum != runnum ) report_error('internal error in Data.js:datafiles_create.move_to_medium_term_impl()');
                for( var i in run.files ) {
                    var f = run.files[i];
                    if((f.type == type) && (f.storage == storage)) {
                        f.storage = 'MEDIUM-TERM';
                    }
                }
                // Redisplay the corresponding page
                //
                that.update_page_header(pidx);
                that.on_page_select(pidx);
                that.on_page_select(pidx);
            },
            'JSON'
        ).error(function () {
            report_error('failed because of: '+jqXHR.statusText); });
	};

    this.restore_from_archive = function(runnum, type, storage, pidx, ridx) {
        ask_yes_no(
            'Confirm File Recovery from Tape Archive',
            'Are you sure you want to restore all <b>'+type+'</b> files of run <b>'+runnum+'</b> from Tape Archive to the <b>'+storage+'</b> disk storage?<br><br>'+
            'Note this operation will succeed only if your experiment has sufficient quota to accomodate new files. '+
            'Once restored the files will be able to stay in the MEDIUM-TERM storage as long as it\'s permited by <b>LCLS Data Retention Policies</b>.',
            function() {
                that.restore_from_archive_impl(
                    runnum,
                    type,
                    storage,
                    function(result) {

                        that.files_last_request.policies['MEDIUM-TERM'].quota_used_gb = result.medium_quota_used_gb;

                        // Update entries for all relevant files from the transient data structure
                        //
                        var run = that.files_last_request.runs[ridx];
                        for( var i in run.files ) {
                            var f = run.files[i];
                            if((f.runnum == runnum) && (f.type == type) && (f.storage == storage)) {
                                f.local = '<span style="color:black;">Restoring from tape...</span>';
                                f.restore_flag = 1;
                                f.restore_requested_time = '';
                                f.restore_requested_uid = auth_remote_user;
                            }
                        }
                        // Redisplay the corresponding page
                        //
                        that.update_page_header(pidx);
                        that.on_page_select(pidx);
                        that.on_page_select(pidx);
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
                if( on_success ) on_success(result);
            },
            'JSON'
        ).error(function () {
            report_error('failed because of: '+jqXHR.statusText); });
	};
    this.delete_from_disk = function(runnum, type, storage, pidx, ridx) {
        var warning = '';
        if(storage == 'MEDIUM-TERM')
            warning =
            'This operation will move all files of the selected run back to the SHORT-TERM term storage from where they would be eventually deleted. '+
            'So be advised that proceeding with this operation may result in irreversable loss of informaton. '+
            'Also note that this operation may be reported to the PI of the experiment. ';
        else if(storage == 'SHORT-TERM')
            warning =
            'Remember that these files can be later restored from tope';
        ask_yes_no(
            'Confirm File Removal',
            'Are you sure you want to remove all <b>'+type+'</b> files of run <b>'+runnum+'</b> from the <b>'+storage+'</b> disk storage?<br><br>'+
            warning,
            function() {
                that.delete_from_disk_impl(
                    runnum,
                    type,
                    storage,
                    function(result) {

                        that.files_last_request.policies['MEDIUM-TERM'].quota_used_gb = result.medium_quota_used_gb;

                        // Update entries for all relevant files from the transient data structure
                        //
                        var run = that.files_last_request.runs[ridx];
                        for( var i in run.files ) {
                            var f = run.files[i];
                            if((f.runnum == runnum) && (f.type == type) && (f.storage == storage)) {
                                f.storage = 'SHORT-TERM';
                            }
                        }
                        // Redisplay the corresponding page
                        //
                        that.update_page_header(pidx);
                        that.on_page_select(pidx);
                        that.on_page_select(pidx);
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
                if( on_success ) on_success(result);
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
