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

    function storage_name(storage) {
        var storage_name = storage;
        switch(storage) {
            case 'RAW_DATA':    storage_name = '90 DAYS'; break;
            case 'MEDIUM_TERM': storage_name = '2 YEARS'; break;
        }
        return storage_name;
    }

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

        var totals  = {
            'TOTAL'      : { files: 0, size_gb: 0 },
            'TAPE'       : { files: 0, size_gb: 0 },
            'MEDIUM_TERM': { files: 0, size_gb: 0 },
            'RAW_DATA'   : { files: 0, size_gb: 0 }
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
'  <div style="padding-bottom:6px;"><span class="datafiles-table-hdr">R U N (s)</span></div>'+
'  <div><span class="datafiles-table-hdr">&nbsp;</span></div>'+
'</div>'+

'<div style="float:left; width:160px;">'+
'  <div style="padding-bottom:6px;"><span class="datafiles-table-hdr">T O T A L &nbsp; D A T A</span></div>'+
'  <div style="float:left; width:80px;"><span class="datafiles-table-hdr-plain">[# files]</span></div>'+
'  <div style="float:left; width:60px;"><span class="datafiles-table-hdr-plain">[GB]</span></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+

'<div style="float:left; width:160px;">'+
'  <div style="padding-bottom:6px;"><span class="datafiles-table-hdr">1 0 &nbsp; Y E A R S &nbsp; T A P E</span></div>'+
'  <div style="float:left; width:80px;"><span class="datafiles-table-hdr-plain">[# files]</span></div>'+
'  <div style="float:left; width:60px;"><span class="datafiles-table-hdr-plain">[GB]</span></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+

'<div style="float:left; width:160px;">'+
'  <div style="padding-bottom:6px;"><span class="datafiles-table-hdr">9 0 &nbsp; D A Y S &nbsp; D I S K</span></div>'+
'  <div style="float:left; width:80px;"><span class="datafiles-table-hdr-plain">[# files]</span></div>'+
'  <div style="float:left; width:60px;"><span class="datafiles-table-hdr-plain">[GB]</span></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+

'<div style="float:left; width:160px;">'+
'  <div style="padding-bottom:6px;"><span class="datafiles-table-hdr">2 &nbsp; Y E A R S &nbsp; D I S K</span></div>'+
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
'  <div style="float:left;" class="df-r-total-files"><b>'+totals['RAW_DATA'].files  +'</b></div>'+
'  <div style="float:left;" class="df-r-total-size" ><b>'+totals['RAW_DATA'].size_gb+'</b></div>'+
'  <div style="clear:both;"></div>'+
'</div>'+

'<div style="float:left; width:160px;">'+
'  <div style="float:left;" class="df-r-total-files"><b>'+totals['MEDIUM_TERM'].files  +'</b></div>'+
'  <div style="float:left;" class="df-r-total-size" ><b>'+totals['MEDIUM_TERM'].size_gb+'</b></div>'+
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

            var title = 'runs from '+min_run+' through '+max_run;

            var totals  = {
                'TOTAL'      : { files: 0, size_gb: 0 },
                'TAPE'       : { files: 0, size_gb: 0 },
                'MEDIUM_TERM': { files: 0, size_gb: 0 },
                'RAW_DATA'   : { files: 0, size_gb: 0 }
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
                if(( this.files_last_request.overstay['RAW_DATA'   ] != undefined ) && ( this.files_last_request.overstay['RAW_DATA'   ]['runs'][run.runnum] != undefined )) overstay_raw    = true;
                if(( this.files_last_request.overstay['MEDIUM_TERM'] != undefined ) && ( this.files_last_request.overstay['MEDIUM_TERM']['runs'][run.runnum] != undefined )) overstay_medium = true;
            }
            html +=
'  <div class="df-r-hdr" id="df-r-hdr-'+pidx+'" onclick="datafiles.on_page_select('+pidx+');" title="'+title+'">'+
'    <div style="float:left;"><span class="toggler ui-icon ui-icon-triangle-1-e df-r-tgl" id="df-r-tgl-'+pidx+'"></span></div>'+
'    <div style="float:left;" class="df-r-min-run">'        +min_run+'</div>'+
'    <div style="float:left;" class="df-r-separator">'      +( min_run == max_run ? '&nbsp;' : '-' )+'</div>'+
'    <div style="float:left;" class="df-r-max-run">'        +( min_run == max_run ? '&nbsp;' : max_run )+'</div>'+
'    <div style="float:left;" class="df-r-total-files">'    +totals['TOTAL'].files        +'</div>'+
'    <div style="float:left;" class="df-r-total-size">'     +totals['TOTAL'].size_gb      +'</div>'+
'    <div style="float:left;" class="df-r-tape-files">'     +totals['TAPE' ].files        +'</div>'+
'    <div style="float:left;" class="df-r-tape-size">'      +totals['TAPE' ].size_gb      +'</div>'+
'    <div style="float:left;" class="df-r-raw-files">'      +totals['RAW_DATA'].files     +'</div>'+
'    <div style="float:left;" class="df-r-raw-size">'       +totals['RAW_DATA'].size_gb   +'</div>'+
'    <div style="float:left;" class="df-r-raw-overstay">'   +(overstay_raw ? '<span class="ui-icon ui-icon-alert"></span>' : '<span>&nbsp;<span>')+'</div>'+
'    <div style="float:left;" class="df-r-medium-files">'   +totals['MEDIUM_TERM'].files  +'</div>'+
'    <div style="float:left;" class="df-r-medium-size">'    +totals['MEDIUM_TERM'].size_gb+'</div>'+
'    <div style="float:left;" class="df-r-medium-overstay">'+(overstay_medium ? '<span class="ui-icon ui-icon-alert"></span>' : '<span>&nbsp;<span>')+'</div>'+
'    <div style="clear:both;"></div>'+
'  </div>'+
'  <div class="df-r-con df-r-hdn" id="df-r-con-'+pidx+'"></div>';
        }
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
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.archived+'</td>':'');
		if(display.storage) {
            switch(f.storage) {
                case 'RAW_DATA':
                    html +=
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class+'">'+f.local+'</td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class+'">'+f.allowed_stay+'</td>'+
'    <td class="table_cell                  '+hightlight_class+' '+extra_class+'">';
                    if(first_of_a_kind) {
                        if(f.local_flag) {
                            html +=
'      <button class="delete_from_raw" style="font-size:7px;" onclick="datafiles.delete_from_disk('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+page_idx+','+ridx+')" title="delete all '+f.type+' files of run '+f.runnum+' from the '+storage_name(f.storage)+' disk storage">DELETE</button>'+
'      <button class="move_to_medium_term" style="font-size:7px;" onclick="datafiles.move_to_medium_term('+f.runnum+',\''+f.type+'\')" title="move all '+f.type+' files of run '+f.runnum+' to '+storage_name('MEDIUM_TERM')+' disk storage">MOVE</button>';
                        }
                        if(f.archived_flag && !f.local_flag && !f.restore_flag) {
                            html +=
'      <button class="restore_from_archive" style="font-size:7px;" onclick="datafiles.restore_from_archive('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+page_idx+','+ridx+')" title="restore all '+f.type+' files of run '+f.runnum+' from tape archive to the '+storage_name(f.storage)+' disk storage">RESTORE</button>';
                        }
                    }
                    html +=
'    </td>'+
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class+'"></td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class+'"></td>'+
'	 <td class="table_cell                  '+hightlight_class+' '+extra_class+'"></td>';
                    break;
                case 'MEDIUM_TERM':
                    html +=
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class+'"></td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class+'"></td>'+
'	 <td class="table_cell                  '+hightlight_class+' '+extra_class+'"></td>'+
'    <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class+'">'+f.local+'</td>'+
'	 <td class="table_cell table_cell_right '+hightlight_class+' '+extra_class+'">'+f.allowed_stay+'</td>'+
'    <td class="table_cell                  '+hightlight_class+' '+extra_class+'">';
                    if(first_of_a_kind) {
                        if(f.local_flag) {
                            html +=
'      <button class="delete_from_medium" style="font-size:7px;" onclick="datafiles.delete_from_disk('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+page_idx+','+ridx+')" title="delete all '+f.type+' files of run '+f.runnum+' from the '+storage_name(f.storage)+' disk storage">DELETE</button>';
                        }
                        if(f.archived_flag && !f.local_flag && !f.restore_flag) {
                            html +=
'      <button class="restore_from_archive" style="font-size:7px;" onclick="datafiles.restore_from_archive('+f.runnum+',\''+f.type+'\',\''+f.storage+'\','+page_idx+','+ridx+')" title="restore all '+f.type+' files of run '+f.runnum+' from tape archive to the '+storage_name(f.storage)+' disk storage">RESTORE</button>';
                        }
                    }
                    html +=
'    </td>';
                    break;
            }
        } else {
            html +=
'    <td class="table_cell '+hightlight_class+' '+extra_class+'">'+f.local+'</td>';
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
'    <td class="table_hdr">On tape</td>':'');
        if(display.storage)
            html +=
'    <td class="table_hdr ">90 Days Disk</td>'+
'    <td class="table_hdr ">expire</td>'+
'    <td class="table_hdr ">actions</td>'+
'    <td class="table_hdr ">2 Years Disk</td>'+
'    <td class="table_hdr ">expire</td>'+
'    <td class="table_hdr ">actions</td>';
        else
            html +=
'    <td class="table_hdr">On Disk</td>';
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
        page.find('.delete_from_raw').button().button('disable');
        page.find('.delete_from_medium').button();
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
'  <span style="float:left;"> <b>'+overstay[storage].total_files+'</b> files in <b>'+overstay[storage].total_runs+'</b> runs, <b>'+parseInt(overstay[storage].total_size_gb)+'</b> GB overstay in <b>'+storage_name(storage)+'</b> storage</span>';
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
        case '90 DAYS': params.storage = 'RAW_DATA';    break;
        case '2 YEARS': params.storage = 'MEDIUM_TERM'; break;
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
    this.move_to_medium_term = function(runnum,type) {
        ask_yes_no(
            'Confirm File Migration',
            'Are you sure you want to move all <b>'+type+'</b> files of run <b>'+runnum+'</b> to the <b>'+storage_name('MEDIUM_TERM')+'</b> disk storage?<br><br>'+
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
            'Are you sure you want to restore all <b>'+type+'</b> files of run <b>'+runnum+'</b> from Tape Archive to the <b>'+storage_name(storage)+'</b> disk storage?<br><br>'+
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
                                f.local = '<span style="color:black;">Restoring from tape...</span>';
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
            'The files you\'re about to delete may not be saved to tape. '+
            'Proceeding with this operation may result in irreversable loss of informaton. '+
            'Also note that this operation will be reported to the PI of the experiment. ';
        else if(storage == 'RAW_DATA')
            warning =
            'Remember that these files can be later restored from tope';
        ask_yes_no(
            'Confirm File Deletion',
            'Are you sure you want to delete all <b>'+type+'</b> files of run <b>'+runnum+'</b> from the <b>'+storage_name(storage)+'</b> disk storage?<br><br>'+
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
