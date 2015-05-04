define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/StackOfRows', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, StackOfRows, FwkApplication, Fwk) {

    cssloader.load('../portal/css/Filemanager_Files.css') ;

    /**
     * This class implements the body of the stack of runs
     *
     * @param {Filemanager_Files} parent
     * @param {Number} min_run_idx
     * @param {Number} max_run_idx
     * @returns {Filemanager_Body}
     */
    function Filemanager_Body (parent, pidx, min_run_idx, max_run_idx, options) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        StackOfRows.StackRowBody.call(this) ;

        // ------------------------
        // Parameters of the object
        // ------------------------

        this.parent = parent ;
        this.experiment = parent.experiment ;
        this.access_list = parent.access_list ;

        this._files_last_request = parent._files_last_request ;
        this._pidx = pidx ;
        this._min_run_idx = min_run_idx ;
        this._max_run_idx = max_run_idx ;
        this._options = options ;

        // ----------------------------
        // Static variables & functions
        // ----------------------------

        // ------------------------------------------------
        // Override event handler defined in thw base class
        // ------------------------------------------------

        this._is_rendered = false ;

        this._table_cont = null ;

        this.render = function () {

            if (this._is_rendered) return ;
            this._is_rendered = true ;

            this.container.html('<div class="table-cont"></div>') ;
            this._table_cont = this.container.children('.table-cont') ;

            this._files_display() ;
        } ;

        /**
         * Re-display the body with the specified options.
         *
         * @param {Array} options
         * @returns {undefined}
         */
        this.set_options = function (options) {
            this._options = options ;
            this._files_display() ;
        } ;

        this._files_display = function () {

            if (!this._is_rendered) return ;

            var html =
'<table class="files" border="0" cellspacing="0">' +
'  <thead>' ;

            if (this._options.storage) html +=
'    <tr align="center" >' +
'      <td rowspan="2" align="right" > Run         </td>' +
'      <td rowspan="2" align="left"  > File        </td>' +
'      <td rowspan="2"               > Type        </td>' +
'      <td rowspan="2" align="right" > Size        </td>' +
'      <td rowspan="2"               > Created     </td>' + (this._options.checksum ?
'      <td rowspan="2" align="left"  > Checksum    </td>':'') +
'      <td colspan="4"               > SHORT-TERM  </td>' +
'      <td colspan="4"               > MEDIUM-TERM </td>' +
'      <td rowspan="2"               > On Tape     </td>' +
'    </tr>' +
'    <tr align="center" >' +
'      <td > on disk      </td>' +
'      <td > expiration   </td>' +
'      <td > allowed stay </td>' +
'      <td > actions      </td>' +
'      <td > on disk      </td>' +
'      <td > expiration   </td>' +
'      <td > allowed stay </td>' +
'      <td > actions      </td>' +
'    </tr>' ;
            else html +=
'    <tr align="center" >' +
'      <td align="right" > Run      </td>' +
'      <td align="left"  > File     </td>' +
'      <td               > Type     </td>' +
'      <td align="right" > Size     </td>' +
'      <td               > Created  </td>' + (this._options.checksum ?
'      <td align="left"  > Checksum </td>':'') +
'      <td               > On Disk  </td>' +
'      <td               > On Tape  </td>' +
'    </tr>' ;

            html +=
'  </thead>' +
'  <tbody>' ;

            var num_runs  = 0 ;
            var num_files = 0 ;

            for (var i = this._min_run_idx; i <= this._max_run_idx; ++i) {
                ++num_runs ;
                var run = this._files_last_request.runs[i] ;
                var first = true ;
                var first_of_a_kind = {} ;
                for (var j in run.files) {
                    ++num_files ;
                    var f = run.files[j] ;
                    if (first_of_a_kind[f.storage]         === undefined) first_of_a_kind[f.storage]         = {} ;
                    if (first_of_a_kind[f.storage][f.type] === undefined) first_of_a_kind[f.storage][f.type] = true ;
                    html += this.file2html(f, first ? run.url : '', first_of_a_kind[f.storage][f.type], i) ;
                    first = false ;
                    first_of_a_kind[f.storage][f.type] = false ;
                }
            }
            html +=
'  </tbody>' +
'</table>' ;

            this._table_cont.html(html) ;

            this._table_cont.find('.control-button').button().click(function () {
                var tr = $(this).closest('tr') ;
                var runnum    = parseInt(tr.attr('runnum')) ,
                    type      =          tr.attr('type') ,
                    storage   =          tr.attr('storage') ,
                    ridx      = parseInt(tr.attr('ridx')) ;
                switch (this.name) {
                    case 'move_to_medium_term' :
                    case 'move_to_short_term' :
                        _that.move_files(runnum, type, storage, ridx) ;
                        break ;
                    case 'delete_from_medium_term'  :
                    case 'delete_from_short_term'  :
                        _that.delete_from_disk(runnum, type, storage, ridx) ;
                        break ;
                    case 'restore_from_archive' :
                        _that.restore_from_archive(runnum, type, storage, ridx) ;
                        break ;
                }
            }) ;
        };

        this.file2html = function (f, run_url, first_of_a_kind, ridx) {

            var extra = 'class="' + f.type +
                (run_url         ? ' run-header'     : '') +
                (first_of_a_kind ? ' first-in-class' : '') +
                '"' ;

            var html =
'  <tr '+extra+' align="center" ridx='+ridx+'" runnum="'+f.runnum+'" type="'+f.type+'" storage="' + f.storage + '" >' +
'    <td align="right" >'       + run_url           + '</td>' +
'    <td align="left"  >'       + f.name            + '</td>' +
'    <td               >'       + f.type            + '</td>' +
'    <td align="right" >&nbsp;' + this.file_size(f) + '</td>' +
'    <td               >&nbsp;' + f.created         + '</td>' + (this._options.checksum ?
'    <td align="left"  >&nbsp;' + f.checksum        + '</td>' : '') ;

            if (this._options.storage) {

                switch (f.storage) {

                    case 'SHORT-TERM' :

                        html +=
'    <td >&nbsp;' + f.local                                   + '</td>' +
'    <td >&nbsp;' + f.allowed_stay['SHORT-TERM'].expiration   + '</td>' +
'    <td >&nbsp;' + f.allowed_stay['SHORT-TERM'].allowed_stay + '</td>' +
'    <td > ' ;
                        if (first_of_a_kind) {
                            if (f.local_flag) {
                                var title = 'save all ' + f.type + ' files of run ' + f.runnum +
                                            ' to MEDIUM-TERM disk storage' ;
                                html +=
'      <button class="control-button" name="move_to_medium_term" title="' + title + '" > MOVE TO MEDIUM </button>' ;
                            }
                            if (f.local_flag && f.archived_flag && this.access_list.datafiles.is_data_administrator) {
                                var title = 'delete all ' + f.type + ' files of run ' + f.runnum +
                                            ' from the ' + f.storage + ' disk storage' ;
                                html +=
'      <button class="control-button" name="delete_from_short_term" title="' + title + '" > DELETE </button>' ;
                            }
                            if (f.archived_flag && !f.local_flag && !f.restore_flag) {
                                var title = 'restore all ' + f.type + ' files of run ' + f.runnum +
                                            ' from tape archive to the ' + f.storage + ' disk storage' ;
                                html +=
'      <button class="control-button" name="restore_from_archive" title="' + title + '" > RESTORE FROM TAPE </button>' ;
                            }
                        }
                        html +=
'    </td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' ;
                        break ;

                    case 'MEDIUM-TERM' :

                        html +=
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;</td>' +
'    <td >&nbsp;' + f.local                                    + '</td>' +
'    <td >&nbsp;' + f.allowed_stay['MEDIUM-TERM'].expiration   + '</td>' +
'    <td >&nbsp;' + f.allowed_stay['MEDIUM-TERM'].allowed_stay + '</td>' +
'    <td > ' ;
                        if (first_of_a_kind) {
                            if (f.local_flag) {
                                var title = 'remove all ' + f.type + ' files of run ' + f.runnum + ' from the ' + f.storage +
                                            ' disk storage and move them back to the SHORT-TERM storage' ;
                                html +=
'      <button class="control-button" name="move_to_short_term" title="' + title + '" > MOVE TO SHORT </button>' ;
                            }
                            if (f.local_flag && f.archived_flag && this.access_list.datafiles.is_data_administrator) {
                                var title = 'delete all ' + f.type + ' files of run ' + f.runnum + ' from the '
                                            + f.storage + ' disk storage' ;
                                html +=
'      <button class="control-button" name="delete_from_medium_term" title="' + title + '" > DELETE </button>' ;
                            }
                            if (f.archived_flag && !f.local_flag && !f.restore_flag) {
                                var title = 'restore all ' + f.type + ' files of run ' + f.runnum +
                                            ' from tape archive to the ' + f.storage + ' disk storage' ;
                                html +=
'      <button class="control-button" name="restore_from_archive" title="' + title + '" > RESTORE FROM TAPE </button>' ;
                            }
                        }
                        html +=
'    </td>';
                        break ;
                }
            } else {
                html +=
'    <td >&nbsp;' + f.local + '</td>' ;
            }
            html +=
'    <td >&nbsp;' + f.archived + '</td>' +
'  </tr>' ;
            return html ;
        } ;

        this.file_size = function (f) {
            switch (this._options.format) {
                case 'auto-format-file-size' : return f.size_auto ;
                case 'Bytes'                 : return f.size ;
                case 'KBytes'                : return f.size_kb ;
                case 'MBytes'                : return f.size_mb ;
                case 'GBytes'                : return f.size_gb ;
                default                      : break ;
            }
        } ;

        this.move_files = function (runnum, type, storage, ridx) {

            if (!this.parent.confirm_move) {
                this.move_files_impl(runnum, type, storage, ridx) ;
                return ;
            }

            var warning = '' ;

            switch (storage) {
                case 'SHORT-TERM' :
                    warning =
'Are you sure you want to save all <b>'+type+'</b> files of run <b>'+runnum+'</b> to the <b>MEDIUM-TERM</b> disk storage?<br><br>' +
'Note this operation will succeed only if your experiment has sufficient quota to accomodate new files. ' +
'Once saved the files will be able to stay in the MEDIUM-TERM storage as long as it\'s permited by <b>LCLS Data Retention Policies</b>.<br><br>' ;
                    break ;
                case 'MEDIUM-TERM' :
                    warning =
'Are you sure you want to move all <b>'+type+'</b> files of run <b>'+runnum+'</b> back to the <b>SHORT-TERM</b> storage?<br><br>' +
'Keep in mind that data retention period is typically much shorted for files stored on the <b>SHORT-TERM</b> storage and when expired the files may be automatically deleted from disk. ' +
'So be advised that proceeding with this operation may result in loss of informaton. ' +
'This operation may be reported to the PI of the experiment.<br><br>' ;
                    break ;
            }
            Fwk.ask_yes_no (
                'Confirm File Move' ,
                warning +
'<span class="ui-icon ui-icon-info" style="float:left; margin-right:4px;"></span><input type="checkbox" id="datafiles-confirm-move" /> check to prevent this dialog for the rest of the current session' ,
                function () {
                    _that.parent.confirm_move = $('#datafiles-confirm-move').attr('checked') ? false : true ;
                    _that.move_files_impl(runnum, type, storage, ridx) ;
                }
            ) ;
        } ;

        this.move_files_impl = function (runnum, type, storage, ridx) {
            var name = '' ;
            switch (storage) {
                case 'SHORT-TERM' : name = 'move_to_medium_term' ; break ;
                case 'MEDIUM-TERM': name = 'move_to_short_term'  ;  break ;
            }
            var button = this._table_cont.find('tr[ridx="'+'"]').find('button[name="'+name+'"]').button() ;
            button.button('disable') ;

            Fwk.web_service_GET (
                '../portal/ws/filemgr_files_move.php' ,
                {   exper_id: this.experiment.id ,
                    runnum  : runnum ,
                    type    : type ,
                    storage : storage
                } ,
                function (data) {

                    _that._files_last_request.policies['MEDIUM-TERM'].quota_used_gb = data.medium_quota_used_gb ;

                    // Update entries for all relevant files from the transient data structure
                    //
                    var run = _that._files_last_request.runs[ridx] ;

                    for (var i in run.files) {
                        var f = run.files[i] ;
                        if ((f.type === type) && (f.storage === storage)) {
                            switch( storage ) {
                                case 'SHORT-TERM'  : f.storage = 'MEDIUM-TERM' ; break ;
                                case 'MEDIUM-TERM' : f.storage = 'SHORT-TERM'  ; break ;
                            }
                        }
                    }
                    _that._files_display() ;
                    _that.parent.update_page_header(_that._pidx) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    button.button('enable') ;
                }
            ) ;
        } ;

        this.restore_from_archive = function(runnum, type, storage, ridx) {

            if (!this.parent.confirm_restore) {
                this.restore_from_archive_impl(runnum, type, storage, ridx) ;
                return ;
            }
            Fwk.ask_yes_no (
'Confirm File Recovery from Tape Archive' ,
'Are you sure you want to restore all <b>'+type+'</b> files of run <b>'+runnum+'</b> from Tape Archive to the <b>'+storage+'</b> disk storage?<br><br>' +
'Note this operation will succeed only if your experiment has sufficient quota to accomodate new files. ' +
'Once restored the files will be able to stay in the MEDIUM-TERM storage as long as it\'s permited by <b>LCLS Data Retention Policies</b>.<br><br>' +
'<span class="ui-icon ui-icon-info" style="float:left; margin-right:4px;"></span><input type="checkbox" id="datafiles-confirm-restore" /> check to prevent this dialog for the rest of the current session' ,
                function () {
                    _that.parent.confirm_restore = $('#datafiles-confirm-restore').attr('checked') ? false : true ;
                    _that.restore_from_archive_impl(runnum, type, storage, ridx) ;
                }
            ) ;
        } ;
        this.restore_from_archive_impl = function (runnum, type, storage, ridx) {

            var button = this._table_cont.find('tr[ridx="'+'"]').find('button[name="restore_from_archive"]').button() ;
            button.button('disable') ;

            Fwk.web_service_GET (
                '../portal/ws/filemgr_files_restore.php',
                {   exper_id : this.experiment.id ,
                    runnum   : runnum ,
                    type     : type ,
                    storage  : storage
                } ,
                function (data) {

                    _that._files_last_request.policies['MEDIUM-TERM'].quota_used_gb = data.medium_quota_used_gb;

                    // Update entries for all relevant files from the transient data structure
                    //
                    var run = _that._files_last_request.runs[ridx] ;
                    for (var i in run.files) {
                        var f = run.files[i] ;
                        if ((f.runnum === runnum) && (f.type === type) && (f.storage === storage)) {
                            f.local = '<span style="color:black;">Restoring from tape...</span>' ;
                            f.restore_flag = 1 ;
                            f.restore_requested_time = '' ;
                            f.restore_requested_uid = _that.access_list.user.uid ;
                        }
                    }
                    _that._files_display() ;
                    _that.parent.update_page_header(_that._pidx) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    button.button('enable') ;
                }
            ) ;
        } ;

        this.delete_from_disk = function (runnum, type, storage, ridx) {

            if( !this.parent.confirm_delete ) {
                this.delete_from_disk_impl(runnum, type, storage, ridx) ;
                return ;
            }

            var warning = '' ;

            switch (storage) {
                case 'SHORT-TERM' : 
                case 'MEDIUM-TERM':
                    warning =
'Are you sure you want to delete all <b>'+type+'</b> files of run <b>'+runnum+'</b> from disk?<br><br>' +
'So be advised that proceeding with this operation may result in irreversable loss of informaton. ' +
'This operation may be reported to the PI of the experiment.<br><br>' ;
                    break ;

                default :
                    Fwk.report_error('datafiles.delete_from_disk_impl() implementation error') ;
                    return ;
            }
            Fwk.ask_yes_no (
                'Confirm File Deletion' ,
                warning +
'<span class="ui-icon ui-icon-info" style="float:left; margin-right:4px;"></span><input type="checkbox" id="datafiles-confirm-delete" /> check to prevent this dialog for the rest of the current session' ,
                function () {
                    parent.confirm_delete = $('#datafiles-confirm-delete').attr('checked') ? false : true ;
                    _that.delete_from_disk_impl(runnum, type, storage, ridx) ;
                }
            ) ;
        } ;
        this.delete_from_disk_impl = function (runnum, type, storage, ridx) {

            var name = '' ;
            switch (storage) {
                case 'SHORT-TERM' : name = 'delete_from_short_term'  ; break ;
                case 'MEDIUM-TERM': name = 'delete_from_medium_term' ; break ;
            }
            var button = this._table_cont.find('tr[ridx="'+'"]').find('button[name="'+name+'"]').button() ;
            button.button('disable') ;

            Fwk.web_service_GET (
                '../portal/ws/filemgr_files_delete.php',
                {   exper_id : this.experiment.id ,
                    runnum   : runnum ,
                    type     : type ,
                    storage  : storage
                } ,
                function (data) {

                    _that._files_last_request.policies['MEDIUM-TERM'].quota_used_gb = data.medium_quota_used_gb ;

                    // Remove entries for all relevant files from the transient data structure
                    //
                    var run = _that._files_last_request.runs[ridx] ;
                    for (var i in run.files) {
                        var f = run.files[i] ;
                        if ((f.runnum === runnum) && (f.type === type) && (f.storage === storage)) {
                            f.local = '<span style="color:red;">No</span>' ;
                            f.local_flag = 0 ;
                            f.allowed_stay[f.storage].seconds = '' ;
                            f.allowed_stay[f.storage].expiration = '' ;
                            f.allowed_stay[f.storage].allowed_stay = '' ;
                        }
                    }
                    _that._files_display() ;
                    _that.parent.update_page_header(_that._pidx) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    button.button('enable') ;
                }
            ) ;
        } ;
    }
    Class.define_class (Filemanager_Body, StackOfRows.StackRowBody, {}, {}) ;

    /**
     * The fake class representing the body of the summary row.
     * 
     * @param {Filemanager_Files} parent
     * @returns {Filemanager_SummaryBody}
     */
    function Filemanager_SummaryBody (parent) {

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        StackOfRows.StackRowBody.call(this) ;

        // ------------------------
        // Parameters of the object
        // ------------------------

        this.parent = parent ;
        this.experiment = parent.experiment ;
        this.access_list = parent.access_list ;

        // ----------------------------
        // Static variables & functions
        // ----------------------------

        // ------------------------------------------------
        // Override event handler defined in thw base class
        // ------------------------------------------------

        this._is_rendered = false ;

        this.render = function () {

            if (this._is_rendered) return ;
            this._is_rendered = true ;

            this.container.html('') ;
        } ;
    }
    Class.define_class (Filemanager_SummaryBody, StackOfRows.StackRowBody, {}, {}) ;

    /**
     * The application for displaying the detailed info about the data files of the experiment
     *
     * @returns {Filemanager_Files}
     */
    function Filemanager_Files (experiment, access_list) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this.on_activate = function() {
            this.on_update() ;
        } ;

        this.on_deactivate = function() {
            this._init() ;
        } ;

        this._update_interval_sec = 300 ;
        this._prev_update_sec = null ;

        this.on_update = function () {
            if (this.active) {
                this._init() ;
                var now_sec = Fwk.now().sec ;
                if (!this._prev_update_sec || (now_sec - this._prev_update_sec) > this._update_interval_sec) {
                    this._prev_update_sec = now_sec ;
                    this._load() ;
                }
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.experiment  = experiment ;
        this.access_list = access_list ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._wa = null ;
        this._updated = null ;
        this._runs = null ;
        this._types = null ;
        this._checksum = null ;
        this._archived = null ;
        this._local = null ;
        this._info = null ;
        this._quota = null ;
        this._body_ctrl = null ;
        this._viewer = null ;

        this._reverse_order = false ;

        this._files_last_request = null ;

        this._table = null ;
        this._table_rows_cache = [] ;   // cache the rows to communicate with them later
                                        // when/if changes in the display parameters occure

        this._table_total_row_cache = null ;    // cache the summary row for the same reason

        this.confirm_move    = true ;
        this.confirm_restore = true ;
        this.confirm_delete  = true ;

        this._init = function () {
            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this.container.html('<div id="datafiles-xtchdf5"></div>') ;
            this._wa = this.container.find('div#datafiles-xtchdf5') ;

            if (!this.access_list.datafiles.read) {
                this.wa.html(this.access_list.no_page_access_html) ;
                return ;
            }
            var html =
'<div id="ctrl">' +
'  <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'  <div style="clear:both;"></div>' +
'  <div class="group" title=' +
'"Put a run number or a range of runs to activate the filter. \n' +
'Range can be specified like this: \n' +
'  10 \n' +
'  10-20 \n' +
'  10- \n' +
'  -20 \n' +
'Note that the range cane be open on the either end. \n' +
'Press RETURN to activate search." >' +
'    <span class="label">Search runs:</span>' +
'    <input class="update-trigger" type="text" name="runs" value="" />' +
'  </div>' +
'  <div class="group" title="Select non-blank option to activate the filter">' +
'    <span class="label">Types:</span>' +
'    <select class="update-trigger" name="types">' +
'      <option></option>' +
'      <option>XTC</option>' +
'      <option>HDF5</option>' +
'    </select>' +
'  </div>' +
'  <div class="group" title="Select non-blank option to activate the filter">' +
'    <span class="label">Checksum:</span>' +
'    <select class="update-trigger" name="checksum">' +
'      <option></option>' +
'      <option>none</option>' +
'      <option>is known</option>' +
'    </select>' +
'  </div>' +
'  <div class="group" title="Select non-blank option to activate the filter">' +
'    <span class="label">On tape:</span>' +
'    <select class="update-trigger" name="archived">' +
'      <option></option>' +
'      <option>yes</option>' +
'      <option>no</option>' +
'    </select>' +
'  </div>' +
'  <div class="group" title="Select non-blank option to activate the filter">' +
'    <span class="label">On disk:</span>' +
'    <select class="update-trigger" name="local">' +
'      <option></option>' +
'      <option>SHORT-TERM</option>' +
'      <option>MEDIUM-TERM</option>' +
'      <option>no</option>' +
'    </select>' +
'  </div>' +
'  <div class="buttons" style="float:left;" >' +
'    <button class="control-button" name="reset"   title="reset the form">RESET FORM</button>' +
'    <button class="control-button" name="update" title="click to update the list of files"><img src="../webfwk/img/Update.png" /></button>' +
'  </div>' +
'  <div style="clear:both;"></div>' +
'</div>' +
'<div id="body" >' +
'  <div class="info" id="info"    style="float:left;">&nbsp;</div>' +
'  <div style="clear:both;"></div>' +
'  <div id="ctrl" >' +
'    <div>' +
'      <select name="page-size">' +
'        <option>auto-page-size</option>' +
'        <option>1</option>' +
'        <option>5</option>' +
'        <option>10</option>' +
'        <option>50</option>' +
'      </select>' +
'      <select class="display-trigger" name="format">' +
'        <option>auto-format-file-size</option>' +
'        <option>Bytes</option>' +
'        <option>KBytes</option>' +
'        <option>MBytes</option>' +
'        <option>GBytes</option>' +
'      </select>' +
'      <input  class="display-trigger" type="checkbox" name="storage"  checked="checked" /><span class="label">Storage details</span>' +
'      <input  class="display-trigger" type="checkbox" name="checksum"                   /><span class="label">Checksum</span>' +
'      <button class="control-button" name="reverse">Show in Reverse Order</button>' +
'    </div>' +
'    <div id="quota-usage" >Loading Quota Info...</div>' +
'  </div>' +
'  <div id="viewer"></div>' +
'</div>' ;
            this._wa.html(html) ;

            var ctrl = this._wa.find('#ctrl') ;

            this._updated  = ctrl.find('#updated') ;
            this._runs     = ctrl.find('input[name="runs"]') ;
            this._types    = ctrl.find('select[name="types"]') ;
            this._checksum = ctrl.find('select[name="checksum"]') ;
            this._archived = ctrl.find('select[name="archived"]') ;
            this._local    = ctrl.find('select[name="local"]') ;

            ctrl.find('.update-trigger').change(function () { _that._load() ; }) ;
            ctrl.find('.control-button').button().click(function () {
                var op = this.name ;
                switch (op) {
                    case 'reset'  : _that._reset() ; break ;
                    case 'update' : _that._load() ; break ;
                }
            }) ;

            var body = this._wa.find('#body') ;

            this._info    = body.find('#info') ;

            this._body_ctrl = body.find('#ctrl') ;
            this._body_ctrl.find('.display-trigger').change(function () {
                var options = _that._get_display_options() ;
                for (var i in _that._table_rows_cache)
                    _that._table_rows_cache[i].body.set_options(options) ;
            }) ;
            this._body_ctrl.find('select[name="page-size"]').change(function () { _that._display() ; }) ;
            this._body_ctrl.find('button[name="reverse"]').button().click(function () {
                _that._reverse_order = !_that._reverse_order ;
                if (_that._files_last_request.runs) _that._files_last_request.runs.reverse() ;
                _that._display() ;
            }) ;

            this._quota  = body.find('#quota-usage') ;
            this._viewer = body.find('#viewer') ;

            this._load() ;
        } ;
        this._reset = function () {
            this._runs.val('') ;
            this._types.val('') ;
            this._checksum.val('') ;
            this._archived.val('') ;
            this._local.val('') ;
            this._load() ;
        } ;
        this._load = function () {

            var params   = {exper_id: this.experiment.id} ;
            var runs     = this._runs.val() ;     if (runs     !== '') params.runs     = runs ;
            var types    = this._types.val() ;    if (types    !== '') params.types    = types ;
            var checksum = this._checksum.val() ; if (checksum !== '') params.checksum = checksum === 'is known' ? 1 : 0 ;
            var archived = this._archived.val() ; if (archived !== '') params.archived = archived === 'yes'      ? 1 : 0 ;
            var local    = this._local.val() ;    if (local    !== '') params.local    = local    === 'no'       ? 0 : 1 ;
            switch (local) {
                case 'SHORT-TERM'  : params.storage = 'SHORT-TERM' ;  break ;
                case 'MEDIUM-TERM' : params.storage = 'MEDIUM-TERM' ; break ;
            }

            this._updated.html('Updating...') ;

            Fwk.web_service_GET (
                '../portal/ws/filemgr_files_search.php' ,
                params ,
                function (data) {

                    _that._files_last_request = data ;

                    // Employing the inverted logic because the runs are reported in
                    // the ascending order. And by default we're supposed to show runs in
                    // the reversed order.

                    if (!_that._reverse_order) _that._files_last_request.runs.reverse() ;

                    _that._updated.html('Updated: <b>'+data.updated+'</b>') ;
                    _that._display() ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ; 
                }
            ) ;
        } ;

        this._table_hdr = [
            {id: 'runs_begin',      title: 'RUNS',        width:  40} ,
            {id: 'runs_end',        title: '&nbsp;',      width:  55} ,
            {id: '|'} ,
            {id: 'total',           title: '&Sum;',       width:  10} ,
            {id: 'total_files',     title: 'files',       width:  40, align: "right"} ,
            {id: 'total_gb',        title: 'GB',          width:  60, align: "right"} ,
            {id: '_',                                     width:  10} ,
            {id: '|'} ,
            {id: 'short',           title: 'SHORT-TERM',  width: 100} ,
            {id: 'short_overstay',  title: '&nbsp',       width:  20, align: "right"} ,
            {id: 'short_files',     title: 'files',       width:  40, align: "right"} ,
            {id: 'short_gb',        title: 'GB',          width:  60, align: "right"} ,
            {id: '_',                                     width:  10} ,
            {id: '|'} ,
            {id: 'medium',          title: 'MEDIUM-TERM', width: 100} ,
            {id: 'medium_overstay', title: '&nbsp',       width:  20, align: "right"} ,
            {id: 'medium_files',    title: 'files',       width:  40, align: "right"} ,
            {id: 'medium_gb',       title: 'GB',          width:  60, align: "right"} ,
            {id: '_',                                     width:  10} ,
            {id: '|'} ,
            {id: 'long',            title: 'LONG-TERM',   width: 100} ,
            {id: 'long_files',      title: 'files',       width:  40, align: "right"} ,
            {id: 'long_gb',         title: 'GB',          width:  60, align: "right"} ,
            {id: '_',                                     width:  10}
        ] ;

        this._display = function () {

            var options = this._get_display_options() ;

            this._table = new StackOfRows.StackOfRows (
                this._table_hdr ,
                [] ,
                {   theme: 'stack-theme-mustard' ,
                    allow_replicated_headers: true
                }
            ) ;
            this._table.display(this._viewer) ;

            this._table_rows_cache = [] ;

            for (var pidx=0; pidx < Math.ceil(this._files_last_request.runs.length / options.page_size); ++pidx) {
                var row = this._create_row(pidx, options) ;
                this._table.append(row) ;
                this._table_rows_cache[pidx] = row ;
            }

            var row = this._create_summary_row() ;
            this._table.insert_front(row) ;
            this._table_total_row_cache = row ;

            this._update_quota_info() ;
        } ;

        this._create_row = function (pidx, options) {

            var min_run_idx = pidx * options.page_size ;
            var max_run_idx = Math.min(this._files_last_request.runs.length, (pidx + 1) * options.page_size) - 1 ;

            var min_run = this._files_last_request.runs[min_run_idx].runnum ;
            var max_run = this._files_last_request.runs[max_run_idx].runnum ;
            if (max_run < min_run) {
                var swap = max_run ;
                max_run = min_run ;
                min_run = swap ;
            }

            var totals  = {
                'TOTAL'       : {files: 0, size_gb: 0} ,
                'TAPE'        : {files: 0, size_gb: 0} ,
                'MEDIUM-TERM' : {files: 0, size_gb: 0} ,
                'SHORT-TERM'  : {files: 0, size_gb: 0}
            } ;
            var overstay_short  = false ;
            var overstay_medium = false ;
            for (var ridx = min_run_idx; ridx <= max_run_idx; ++ridx) {
                var run = this._files_last_request.runs[ridx] ;
                var files = run.files ;
                for(var j in files) {
                    var f = files[j] ;
                    var size_gb = f.archived_flag || f.local_flag ? parseInt(f.size_gb) : 0 ;
                    totals['TOTAL'].files   += 1 ;
                    totals['TOTAL'].size_gb += size_gb ;
                    if (f.archived_flag) {
                        totals['TAPE'].files   += 1 ;
                        totals['TAPE'].size_gb += size_gb ;
                    }
                    if (f.local_flag) {
                        totals[f.storage].files   += 1 ;
                        totals[f.storage].size_gb += size_gb ;
                    }
                }
                if ((this._files_last_request.overstay['SHORT-TERM' ] !== undefined) && (this._files_last_request.overstay['SHORT-TERM' ]['runs'][run.runnum] !== undefined)) overstay_short  = true ;
                if ((this._files_last_request.overstay['MEDIUM-TERM'] !== undefined) && (this._files_last_request.overstay['MEDIUM-TERM']['runs'][run.runnum] !== undefined)) overstay_medium = true ;
            }

            var row = {
                title: {
                    runs_begin:      '<div style="text-align: right; font-weight: bold;">' + min_run + '</div>' ,
                    runs_end:        max_run === min_run ? '&nbsp;' : '<div style="font-weight: bold;">&nbsp;-&nbsp;' + max_run + '</div>' ,
                    total:           '&nbsp;' ,
                    total_files:     totals['TOTAL'].files ,
                    total_gb:        totals['TOTAL'].size_gb ,
                    short:           '&nbsp;' ,
                    short_overstay:  overstay_short ? '<span class="ui-icon ui-icon-alert"></span>' : '&nbsp;' ,
                    short_files:     totals['SHORT-TERM'].files ,
                    short_gb:        totals['SHORT-TERM'].size_gb ,
                    medium:          '&nbsp;' ,
                    medium_overstay: overstay_medium ? '<span class="ui-icon ui-icon-alert"></span>' : '&nbsp;' ,
                    medium_files:    totals['MEDIUM-TERM'].files ,
                    medium_gb:       totals['MEDIUM-TERM'].size_gb ,
                    long:            '&nbsp;' ,
                    long_files:      totals['TAPE' ].files ,
                    long_gb:         totals['TAPE' ].size_gb
                } ,
                body: new Filemanager_Body (
                    this ,
                    pidx ,
                    min_run_idx ,
                    max_run_idx ,
                    options
                )
            } ;
            return row ;
        } ;

        this._create_summary_row = function () {

            var row = {
                title: {
                    runs_begin:      '&nbsp;' ,
                    runs_end:        '&nbsp;' ,
                    total:           '&nbsp;' ,
                    total_files:     '&nbsp;' ,
                    total_gb:        '&nbsp;' ,
                    short:           '<b>'+this._files_last_request.policies['SHORT-TERM'].retention_months+'mo @disk</b>' ,
                    short_overstay:  '&nbsp;' ,
                    short_files:     '&nbsp;' ,
                    short_gb:        '&nbsp;' ,
                    medium:          '<b>'+this._files_last_request.policies['MEDIUM-TERM'].retention_months+'mo @disk</b>' ,
                    medium_overstay: '&nbsp;' ,
                    medium_files:    '&nbsp;' ,
                    medium_gb:       '&nbsp;' ,
                    long:            '<b>10yrs @tape</b>' ,
                    long_files:      '&nbsp;' ,
                    long_gb:         '&nbsp;'
                } ,
                body: new Filemanager_SummaryBody(this) ,
                color_theme: 'stack-theme-default' ,
                block_expand: true
            } ;

            if (this._files_last_request.runs.length) {

                var totals  = {
                    'TOTAL'       : {files: 0, size_gb: 0} ,
                    'TAPE'        : {files: 0, size_gb: 0} ,
                    'MEDIUM-TERM' : {files: 0, size_gb: 0} ,
                    'SHORT-TERM'  : {files: 0, size_gb: 0}
                } ;
                var overstay_short  = false ;
                var overstay_medium = false ;

                var min_run = this._files_last_request.runs[0].runnum ;
                var max_run = this._files_last_request.runs[this._files_last_request.runs.length-1].runnum ;
                if (max_run < min_run) {
                    var swap = max_run ;
                    max_run  = min_run ;
                    min_run  = swap ;
                }

                for (var ridx in this._files_last_request.runs) {
                    var run = this._files_last_request.runs[ridx] ;
                    var files = run.files ;
                    for(var j in files) {
                        var f = files[j] ;
                        var size_gb = f.archived_flag || f.local_flag ? parseInt(f.size_gb) : 0 ;
                        totals['TOTAL'].files   += 1 ;
                        totals['TOTAL'].size_gb += size_gb ;
                        if (f.archived_flag) {
                            totals['TAPE'].files   += 1 ;
                            totals['TAPE'].size_gb += size_gb ;
                        }
                        if (f.local_flag) {
                            totals[f.storage].files   += 1 ;
                            totals[f.storage].size_gb += size_gb ;
                        }
                    }
                    if ((this._files_last_request.overstay['SHORT-TERM' ] !== undefined) && (this._files_last_request.overstay['SHORT-TERM' ]['runs'][run.runnum] !== undefined)) overstay_short  = true ;
                    if ((this._files_last_request.overstay['MEDIUM-TERM'] !== undefined) && (this._files_last_request.overstay['MEDIUM-TERM']['runs'][run.runnum] !== undefined)) overstay_medium = true ;
                }

                row.title.runs_begin      = '<div style="text-align: right; font-weight: bold;">' + min_run + '</div>' ;
                row.title.runs_end        =  max_run === min_run ? '&nbsp;' : '<div style="font-weight: bold;">&nbsp;-&nbsp;' + max_run + '</div>' ;
                row.title.total_files     =  totals['TOTAL'].files ;
                row.title.total_gb        =  totals['TOTAL'].size_gb ;
                row.title.short_overstay  = overstay_short ? '<span class="ui-icon ui-icon-alert"></span>' : '&nbsp;' ;
                row.title.short_files     =  totals['SHORT-TERM'].files ;
                row.title.short_gb        = totals['SHORT-TERM'].size_gb ;
                row.title.medium_overstay = overstay_medium ? '<span class="ui-icon ui-icon-alert"></span>' : '&nbsp;' ;
                row.title.medium_files    = totals['MEDIUM-TERM'].files ;
                row.title.medium_gb       = totals['MEDIUM-TERM'].size_gb ;
                row.title.long_files      = totals['TAPE' ].files ;
                row.title.long_gb         = totals['TAPE' ].size_gb ;
            }
            return row ;
        } ;


        this._update_quota_info = function () {

            var options = this._get_display_options() ;

            this._quota.css('display', options.storage ? 'block' : 'none') ;
            this._quota.html (
'MEDIUM-TERM Quota Usage: ' + this._files_last_request.policies['MEDIUM-TERM'].quota_used_gb +
' / ' + this._files_last_request.policies['MEDIUM-TERM'].quota_gb + ' GB'
            ) ;

            var overstay = this._files_last_request.overstay ;
            var html = '' ;
            for (var storage in overstay) {
                if (html === '')
                    html +=
'<div>' ;
                else
                    html +=
'  <span style="float:left;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>' ;
                html +=
'  <span style="float:left;"></span>' +
'  <span class="ui-icon ui-icon-alert" style="float:left;"></span>' +
'  <span style="float:left;"> <b>'+overstay[storage].total_files+'</b> files in <b>'+overstay[storage].total_runs+'</b> runs, <b>'+parseInt(overstay[storage].total_size_gb)+'</b> GB overstay in <b>'+storage+'</b> storage</span>' ;
            }
            if (html != '')
                html +=
'</div>' ;
            this._info.html(html) ;
        } ;

        this.update_page_header = function (pidx) {

            var old_row = this._table_rows_cache[pidx] ;
            var new_row = this._create_row(pidx, this._get_display_options()) ;
            this._table.update_row(old_row.body.row_id, new_row) ;
            this._table_rows_cache[pidx] = new_row ;

            var summary_row = this._create_summary_row() ;
            this._table.update_row(this._table_total_row_cache.body.row_id, summary_row) ;
            this._table_total_row_cache = summary_row ;

            this._update_quota_info() ;
        } ;

        this._get_display_options = function () {
            var page_size_str = this._body_ctrl.find('select[name="page-size"]').val() ;
            var options = {
                page_size : page_size_str === 'auto-page-size' ? 10 : parseInt(page_size_str) ,
                checksum  : this._body_ctrl.find('input[name="checksum"]') .attr('checked') ? 1 : 0 ,
                storage   : this._body_ctrl.find('input[name="storage"]')  .attr('checked') ? 1 : 0 ,
                format    : this._body_ctrl.find('select[name="format"]').val()
            } ;
            return options ;
        } ;
    }
    Class.define_class (Filemanager_Files, FwkApplication, {}, {}) ;

    return Filemanager_Files ;
}) ;
