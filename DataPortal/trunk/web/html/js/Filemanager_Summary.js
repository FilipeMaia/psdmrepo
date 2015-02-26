define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../portal/css/Filemanager_Summary.css') ;

    /**
     * The application for displaying the integral info about the data files of the experiment
     *
     * @returns {Filemanager_Summary}
     */
    function Filemanager_Summary (experiment, access_list) {

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

        this._prev_update_sec = null ;

        this.on_update = function () {
            if (this.active) {
                this._init() ;
                var now_sec = Fwk.now().sec ;
                if (!this._prev_update_sec || (now_sec - this._prev_update_sec) > 20) {
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

        this._init = function () {
            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this.container.html('<div id="datafiles-summary"></div>') ;
            this._wa = this.container.find('div#datafiles-summary') ;

            if (!this.access_list.datafiles.read) {
                this.wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            var html =
'<div id="ctrl" > ' +
  '<div class="info" id="updated" style="float:right;" ></div> ' +
  '<div style="clear:both;" ></div> ' +
'</div> ' +
'<div id="table" style="float:left;" > ' +
  '<table><tbody> ' +
    '<tr><td class="table_cell table_cell_left"># of runs</td> ' +
        '<td class="table_cell table_cell_right" id="runs">Loading...</td></tr> ' +
    '<tr><td class="table_cell table_cell_left">First run #</td> ' +
        '<td class="table_cell table_cell_right" id="firstrun">Loading...</td></tr> ' +
    '<tr><td class="table_cell table_cell_left">Last run #</td> ' +
        '<td class="table_cell table_cell_right" id="lastrun">Loading...</td></tr> ' +
    '<tr><td class="table_cell table_cell_left" valign="center">XTC</td> ' +
        '<td class="table_cell table_cell_right"> ' +
          '<table cellspacing=0 cellpadding=0><tbody> ' +
            '<tr><td class="table_cell table_cell_left">Size [GB]</td> ' +
                '<td class="table_cell table_cell_right" id="xtc-size">Loading...</td></tr> ' +
            '<tr><td class="table_cell table_cell_left"># of files</td> ' +
                '<td class="table_cell table_cell_right" id="xtc-files">Loading...</td></tr> ' +
            '<tr><td class="table_cell table_cell_left">Archived to HPSS</td> ' +
                '<td class="table_cell table_cell_right" id="xtc-archived">Loading...</td></tr> ' +
            '<tr><td class="table_cell table_cell_left  table_cell_bottom">Available on disk</td> ' +
                '<td class="table_cell table_cell_right table_cell_bottom" id="xtc-disk">Loading...</td></tr> ' +
            '</tbody></table> ' +
        '</td></tr> ' +
    '<tr><td class="table_cell table_cell_left table_cell_bottom" valign="center">HDF5</td> ' +
        '<td class="table_cell table_cell_right table_cell_bottom"> ' +
          '<table cellspacing=0 cellpadding=0><tbody> ' +
            '<tr><td class="table_cell table_cell_left">Size [GB]</td> ' +
                '<td class="table_cell table_cell_right" id="hdf5-size">Loading...</td></tr> ' +
            '<tr><td class="table_cell table_cell_left"># of files</td> ' +
                '<td class="table_cell table_cell_right" id="hdf5-files">Loading...</td></tr> ' +
            '<tr><td class="table_cell table_cell_left">Archived to HPSS</td> ' +
                '<td class="table_cell table_cell_right" id="hdf5-archived">Loading...</td></tr> ' +
            '<tr><td class="table_cell table_cell_left  table_cell_bottom">Available on disk</td> ' +
                '<td class="table_cell table_cell_right table_cell_bottom" id="hdf5-disk">Loading...</td></tr> ' +
            '</tbody></table> ' +
        '</td></tr> ' +
  '</tbody></table> ' +
'</div> ' +
'<div id="buttons" style="float:left;" > ' +
  '<button class="control-button" name="update" title="click to update the summary information"><img src="../webfwk/img/Update.png" /></button> ' +
'</div> ' +
'<div style="clear:both;" ></div> ' ;
            this._wa.html(html) ;
            this._wa.find('button[name="update"]').button().click(function () { _that._load() ; }) ;
            this._updated = this._wa.find('#updated') ;
            this._load() ;
        } ;

        this._load = function () {

            this._updated.html('Updating...') ;

            Fwk.web_service_GET (
                '../portal/ws/filemgr_files_search.php' ,
                {exper_id: this.experiment.id} ,
                function (data) {

                    _that._updated.html('Updated: <b>'+data.updated+'</b>') ;

                    _that._wa.find('#runs'         ).html(data.summary.runs) ;
                    _that._wa.find('#firstrun'     ).html(data.summary.runs ? data.summary.min_run : 'n/a') ;
                    _that._wa.find('#lastrun'      ).html(data.summary.runs ? data.summary.max_run : 'n/a') ;
                    _that._wa.find('#xtc-size'     ).html(data.summary.xtc.size) ; 
                    _that._wa.find('#xtc-files'    ).html(data.summary.xtc.files) ;
                    _that._wa.find('#xtc-archived' ).html(data.summary.xtc.archived_html) ;
                    _that._wa.find('#xtc-disk'     ).html(data.summary.xtc.disk_html) ;
                    _that._wa.find('#hdf5-size'    ).html(data.summary.hdf5.size) ;
                    _that._wa.find('#hdf5-files'   ).html(data.summary.hdf5.files) ;
                    _that._wa.find('#hdf5-archived').html(data.summary.hdf5.archived_html) ;
                    _that._wa.find('#hdf5-disk'    ).html(data.summary.hdf5.disk_html) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ; 
                }
            ) ;
        } ;
    }
    Class.define_class (Filemanager_Summary, FwkApplication, {}, {}) ;

    return Filemanager_Summary ;
}) ;

