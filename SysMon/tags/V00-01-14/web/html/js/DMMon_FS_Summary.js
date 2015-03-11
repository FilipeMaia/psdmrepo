define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../sysmon/css/DMMon_FS_Summary.css') ;

    /**
     * The application for displaying the summary on file system usage stats
     *
     * @returns {DMMon_FS_Summary}
     */
    function DMMon_FS_Summary (app_config) {

        var _that = this ;

        var TB = 1024 * 1024 * 1024;

        // ----------------------------------------
        // Always call the base class's constructor
        // ----------------------------------------

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

        // Automatically refresh the page at specified interval only

        this._update_ival_sec = 10 ;
        this._prev_update_sec = 0 ;

        this.on_update = function () {
            if (this.active) {
                var now_sec = Fwk.now().sec ;
                if (now_sec - this._prev_update_sec > this._update_ival_sec) {
                    this._prev_update_sec = now_sec ;
                    this._init() ;
                    this._load() ;
                }
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this._app_config = app_config ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._filesystems = [] ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="dmmon-fs-summary" class="dmmon-fs-summary" >' +

  '<div id="controls" > ' +
    '<div style="float:right;" > ' +
      '<button name="update" class="control-button" title="update from the database" >UPDATE</button> ' +
    '</div> ' +
    '<div style="clear:both;" ></div> ' +
  '</div> ' +

  '<div id="update_info" > ' +
    '<div class="info" id="info"    style="float:left;"  >&nbsp;</div> ' +
    '<div class="info" id="updated" style="float:right;" >&nbsp;</div> ' +
    '<div style="clear:both;"></div> ' +
  '</div> ' +

  '<div id="plot" class="presentation" ></div> ' +

'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#dmmon-fs-summary') ;
            }
            return this._wa_elem ;
        } ;
        this._set_info = function (html) {
            if (!this._info_elem) this._info_elem = this._wa().children('#update_info').children('#info') ;
            this._info_elem.html(html) ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().children('#update_info').children('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._plot = function () {

            // Always destroy an existing chart to prevent memory leaks
            // when attaching the chart to teh same container

            if (this._plot_obj) {
                this._plot_obj.destroy() ;
            }
            var chartdef = {
                chart: {
                    renderTo:  this._wa().children('#plot').get(0) ,
                    animation: Highcharts.svg ,                     // don't animate in old IE
                    type:      'column' ,
                    margin:    [50, 20, 70, 40] ,
                } ,
                title: {
                    text: '<b>Disk Space Utilization [%]</b>'
                } ,
                xAxis: {
                    categories: _.map(this._filesystems, function(fs) { return fs.name ; })
                } ,
                yAxis: {
                    min: 0
                } ,
                legend: {
                    enabled: true
                } ,
                exporting: {
                    enabled: true
                } ,
                tooltip: {
                    pointFormat: '<span style="color:{series.color}">{series.name}</span>: <b>{point.y} TB</b> ({point.percentage:.0f}%)<br/>' ,
                    shared: true
                } ,
                plotOptions: {
                    column: {
                        stacking: 'percent'
                    }
                } ,
                series: [{
                    name: 'Available' ,
                    data: _.map(this._filesystems, function(fs) { return Math.floor(fs.available/TB) ; }) ,
                    color: 'lightgrey'
                } , {
                    name: 'Used' ,
                    data: _.map(this._filesystems, function(fs) { return Math.floor(fs.used/TB) ; }) ,
                    color: 'rgba(124, 181, 236, 0.7)'
                }]
            } ;
            this._plot_obj = new Highcharts.Chart(chartdef) ;
            return this._plot_obj ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this._wa().find('button.control-button').button().click(function () {
                switch (this.name) {
                    case 'update' :
                        _that._load() ;
                        break ;
                }
            }) ;

            this._load() ;
        } ;
        this._load = function () {
            this._action (
                'Loading...' ,
                '../sysmon/ws/dmmon_fs_summary.php' ,
                {}
            ) ;
        } ;
        this._action = function (name, url, params) {

            this._set_updated(name) ;

            Fwk.web_service_GET (url, params, function (data) {

                _that._filesystems = data.filesystems ;
                _that._display() ;

                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
        this._display = function () {

            var plot = this._plot() ;

            plot.reflow() ;     // in case if the container geometry has changed
            plot.redraw() ;     // batch update for efficiency
        } ;
    }
    Class.define_class (DMMon_FS_Summary, FwkApplication, {}, {}) ;
    
    return DMMon_FS_Summary ;
}) ;


