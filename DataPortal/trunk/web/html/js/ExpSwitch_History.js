define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk) {

    cssloader.load('../portal/css/ExpSwitch_History.css') ;

    /**
     * @brief The application for displaying a history experiment activations
     *
     * @returns {ExpSwitch_History}
     */
    function ExpSwitch_History (instrument, access_list) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        FwkApplication.call(this) ;

        // ------------------------------------------------
        // Override event handler defined in the base class
        // ------------------------------------------------

        this.on_activate = function() {
            this._init() ;
        } ;

        this.on_deactivate = function() {
            this._init() ;
        } ;

        this.on_update = function () {
            if (this.active) {
                this._init() ;
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this.instrument  = instrument ;
        this.access_list = access_list ;

        // --------------------
        // Own data and methods
        // --------------------

        this._wa = null ;    // work area container

        this._is_initialized = false ;

        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this._ctrl().children('button[name="refresh"]').button().click(function () {
                _that._load() ;
            }) ;
            this._load() ;
        } ;
        this._wa = function (html) {
            if (this._wa_elem) {
                if (html !== undefined) {
                    this._wa_elem.html(html) ;
                }
            } else {
                this.container.html('<div id="expswitch-history"></div>') ;
                this._wa_elem = this.container.find('div#expswitch-history') ;
                if (html === undefined) {
                    html =
'<div id="ctrl"> ' +
  '<button class="control-button" ' +
          'name="refresh" ' +
          'title="refresh the list of experiment activations" >REFRESH</button> ' +
'</div> ' +
'<div id="body"> ' +
  '<div class="info" id="info"    style="float:left;">&nbsp;</div> ' +
  '<div class="info" id="updated" style="float:right;">&nbsp;</div> ' +
  '<div style="clear:both;"></div> ' +
  '<div id="table">Loading...</div> ' +
'</div> ' ;
                }
                this._wa_elem.html(html) ;
            }
            return this._wa_elem ;
        } ;

        this._ctrl = function () {
            if (!this._ctrl_elem) {
                this._ctrl_elem = this._wa().children('#ctrl') ;
            }
            return this._ctrl_elem ;
        } ;
        this._body = function () {
            if (!this._body_elem) {
                this._body_elem = this._wa().children('#body') ;
            }
            return this._body_elem ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._body().children('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._table = function () {
            if (!this._table_obj) {
                this._table_obj = new Table (
                    this._body().children('#table') ,
                    [   {name: 'Station',      align: 'right', type: Table.Types.Number} ,
                        {name: 'Experiment',   align: 'right'} ,
                        {name: 'Switch Time',  align: 'right', default_sort_forward: false} ,
                        {name: 'Requested By', align: 'right'}] ,

                    [] ,

                    {   default_sort_column: 2 ,
                        default_sort_forward: false}
                ) ;
                this._table_obj.display() ;
            }
            return this._table_obj ;
        } ;

        this._load = function () {

            this._set_updated('Updating...') ;

            Fwk.web_service_GET (
                '../portal/ws/expswitch_history.php' ,
                {instr_name: this.instrument} ,
                function (data) {
                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                    _that._display(data) ;
                }   
            ) ;
        } ;
        this._display = function (data) {
            var rows = _.map(data.history, function (row) {
                return [
                    row.station ,
                    '<a href="index.php?exper_id='+row.exper_id+'" target="_blank" class="link" title="go to the Web Portal of the experiment">'+row.exper_name+'</a>' ,
                    '<b>'+row.switch_time.ymd+'</b>&nbsp;&nbsp;'+row.switch_time.hms ,
                    row.requestor_gecos] ;
            }) ;
            this._table().load(rows) ;
        } ;
    }
    Class.define_class (ExpSwitch_History, FwkApplication, {}, {}) ;

    return ExpSwitch_History ;
}) ;