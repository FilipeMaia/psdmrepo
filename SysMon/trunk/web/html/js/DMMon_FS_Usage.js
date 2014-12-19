define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../sysmon/css/DMMon_FS_Usage.css') ;

    /**
     * The application for displaying the file system usage stats
     *
     * @returns {DMMon_FS_Usage}
     */
    function DMMon_FS_Usage (app_config) {

        var _that = this ;

        var TB = 1024 * 1024 * 1024;

        var _LOAD_INTERVAL = [
            {name: 'day',   sec:     24*3600} ,
            {name: 'week',  sec:   7*24*3600} ,
            {name: 'month', sec:  30*24*3600} ,
            {name: 'year',  sec: 365*24*3600} ,
            {name: 'all',   sec:           0}
        ] ;
        function _DEFAULT_INTERVAL () {
            return _LOAD_INTERVAL[0] ;
        }

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

        // Automatically refresh the page at specified interval only

        this._update_ival_sec = 10 ;
        this._prev_update_sec = 0 ;

        this.on_update = function () {
            if (this.active) {
                var now_sec = Fwk.now().sec ;
                if (now_sec - this._prev_update_sec > this._update_ival_sec) {
                    this._prev_update_sec = now_sec ;
                    this._init() ;
                    this._load_all() ;
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
        this._id2fs = function (fs_id) {
            for (var i in this._filesystems) {
                var fs = this._filesystems[i] ;
                if (fs.id == fs_id) return fs ;
            }
            console.log('DMMon_FS_Usage._id2fs: ERROR - no file system loaded for id: '+fs_id) ;
            return null ;
        }

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="dmmon-fs-usage" class="dmmon-fs-usage" >' +

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

  '<div id="tabs" ></div> ' +

'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#dmmon-fs-usage') ;
            }
            return this._wa_elem ;
        } ;
        this._tabs = function () {

            if (!this._tabs_obj) {
                var html =
'<ul>' +        _.reduce(this._filesystems, function (html, fs) { return html +=
  '<li><a href="#'+fs.id+'" >'+fs.name+'</a></li> ' ; }, '') +
'</ul> ' +      _.reduce(this._filesystems, function (html, fs) { return html +=
'<div id="'+fs.id+'" > ' +
  '<div class="tab-cont" > ' +
    '<div id="view-'+fs.id+'" class="control-group" > ' +
      '<input type="radio" id="view-'+fs.id+'-table" name="view-'+fs.id+'" checked="checked" ><label for="view-'+fs.id+'-table" title="view as a table" ><img src="../sysmon/img/enumeration1.png" /></label> ' +
      '<input type="radio" id="view-'+fs.id+'-plot"  name="view-'+fs.id+'"                   ><label for="view-'+fs.id+'-plot"  title="view as a plot"  ><img src="../sysmon/img/scatter1.png" /></label> ' +
    '</div> ' +
    '<div class="control-group" style="padding-top:10px;" > ' +
      '<span class="control-group-title" >Display last:</span> ' +
      '<select name="interval" > ' + _.reduce(_LOAD_INTERVAL, function (html, interval) { return html +=
        '<option value="'+interval.sec+'" >'+interval.name+'</option> ' ; }) +
      '</select> ' +
    '</div> ' +
    '<div class="control-group-end" ></div> ' +
    '<div id="table" class="presentation" ></div> ' +
    '<div id="plot"  class="presentation" ></div> ' +
  '</div> ' +
'</div> ' ; }, '') ;
                this._tabs_obj = this._wa().find('#tabs').html(html) ;
                this._tabs_obj.tabs() ;
            }
            return this._tabs_obj ;
        } ;
        this._view_selector = function (fs_id, name) {
            if (!this._view_selector_elem) {
                this._view_buttonset_elem = {} ;
                this._view_selector_elem  = {} ;
            }
            if (!this._view_selector_elem[fs_id]) {
                this._view_buttonset_elem[fs_id] = this._tabs().children('#'+fs_id).find('div#view-'+fs_id).buttonset() ;
                this._view_selector_elem [fs_id] = {} ;
                var view_names = ['table', 'plot'] ;
                for (var i in view_names) {
                    var view_name = view_names[i] ;
                    var elem = this._view_buttonset_elem[fs_id].children('#view-'+fs_id+'-'+view_name) ;
                    // Add custom properties which can be used later to identify a context
                    // in which the objet was set up. This should work better than using
                    // HTML element's attributes.
                    jQuery.data(elem[0], "fs_id", fs_id)
                    jQuery.data(elem[0], "name",  view_name)
                    elem.click (function () { _that._display(jQuery.data(this, "fs_id")) ; }) ;
                    this._view_selector_elem[fs_id][view_name] = elem ;
                }
            }
            return this._view_selector_elem[fs_id][name] ;
        } ;
        this._interval_selector = function (fs_id) {
            if (!this._interval_selector_elem) {
                this._interval_selector_elem = {} ;
            }
            if (!this._interval_selector_elem[fs_id]) {
                var elem = this._tabs().children('#'+fs_id).find('div.control-group').children('select[name="interval"]') ;
                // Add custom data which can be used later to identify a context
                // in which the objet was set up. This should work better than using
                // HTML element's attributes.
                jQuery.data(elem[0], "fs_id", fs_id)
                elem.change(function () { _that._load(jQuery.data(this, "fs_id")) ; }) ;
                this._interval_selector_elem[fs_id] = elem ;
            }
            return this._interval_selector_elem[fs_id] ;
        } ;
        this._set_info = function (html) {
            if (!this._info_elem) this._info_elem = this._wa().children('#update_info').children('#info') ;
            this._info_elem.html(html) ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().children('#update_info').children('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._view = function (fs_id, name) {
            if (!this._view_elem)      this._view_elem      = {} ;
            if (!this._view_elem[fs_id]) this._view_elem[fs_id] = {} ;
            if (!this._view_elem[fs_id][name]) {
                this._view_elem[fs_id][name] = this._tabs().children('div#'+fs_id).find('div#'+name) ;
            }
            return this._view_elem[fs_id][name] ;
        } ;
        this._table = function (fs_id) {
            if (!this._table_obj) this._table_obj = {} ;
            if (!this._table_obj[fs_id]) {
                var rows = [] ;
                var hdr = [
                    {   name: 'time'} ,
                    {   name: 'used [TB]',      type: SimpleTable.Types.Number, sorted: false, align: "right"} ,
                    {   name: 'available [TB]', type: SimpleTable.Types.Number, sorted: false, align: "right"} ,
                    {   name: 'available [%]',  type: SimpleTable.Types.Number, sorted: false, align: "right"}
                ] ;
                var rows = [] ;
                this._table_obj[fs_id] = new SimpleTable.constructor (
                    this._view(fs_id, 'table') ,
                    hdr ,
                    rows ,
                    {   default_sort_forward: false
                    }
                ) ;
                this._table_obj[fs_id].display() ;
            }
            return this._table_obj[fs_id] ;
        } ;
        this._plot = function (fs_id) {

            if (!this._plot_obj) {
                this._plot_obj = {} ;

                // Set global options only once when initializing the data structures

                Highcharts.setOptions({
                    global: {
                        useUTC: false
                    }
                }) ;

            }
            if (!this._plot_obj[fs_id]) {

                var fs  = this._id2fs(fs_id) ;

                var chartdef = {
                    chart: {
                        renderTo:  this._view(fs_id, 'plot').get(0) ,
                        animation: Highcharts.svg ,                     // don't animate in old IE
                        type:      'area' ,
                        margin:    [50, 20, 70, 40] ,
                    } ,
                    title: {
                        text: '<b>Used [%]</b>'
                    } ,
                    xAxis: {
                        type: 'datetime' ,
                        gridLineWidth:  1 /*,
                        minPadding:     0.2 ,
                        maxPadding:     0.2 ,
                        maxZoom:       60*/
                    } ,
                    yAxis: {
                        title: {
                            text: ''
                        } ,
                        minPadding:  0.1 ,
                        maxPadding:  0.1 ,
                        min:         0 ,
                        max:         100 ,
                        plotLines: [{
                            value: 0 ,
                            width: 1 ,
                            color: '#808080'
                        }]
                    } ,
                    legend: {
                        enabled: true
                    } ,
                    exporting: {
                        enabled: true
                    } ,
                    plotOptions: {
                        series: {
                            lineWidth: 2 ,
                            point: {
                            }
                        } ,
                        area: {
                            marker: {
                                enabled: false
                            }
                        }
                                /* ,
                        column: {
                            dataLabels: {
                                enabled: true
                            } ,
                        }*/
                    } ,
                    series: [
                        {   name: fs.name ,
                            data: []
                        }
                    ]
                } ;
                this._plot_obj[fs_id] = new Highcharts.Chart(chartdef) ;
            }
            return this._plot_obj[fs_id] ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this._wa().find('button.control-button').button().click(function () {
                switch (this.name) {
                    case 'update' :
                        _that._load_all() ;
                        break ;
                }
            }) ;

            this._load_all() ;
        } ;
        this._load_all = function () {

            if (_.isEmpty(this._filesystems)) {

                // Load them all using the default interval. Usually, we're going
                // to do it when initializing the UI for the first time when no data
                // has been loaded yet.

                this._action (
                    'Loading...' ,
                    '../sysmon/ws/dmmon_fs_get.php' ,
                    {   interval_sec: _DEFAULT_INTERVAL().sec
                    }
                ) ;

            } else {

                // Load each file system individually taking into consideration
                // its customized interval.
            
                for(var i in this._filesystems) {
                    var fs = this._filesystems[i] ;
                    this._load(fs.id) ;
                }
            }
        } ;
        this._load = function (fs_id) {
            this._action (
                'Loading...' ,
                '../sysmon/ws/dmmon_fs_get.php' ,
                {   fs_id:        fs_id ,
                    interval_sec: this._interval_selector(fs_id).val()  // retreive the interval from the UI
                }
            ) ;
        } ;
        this._action = function (name, url, params) {
            
            // TODO: The logic flow of this function is a bit convoluted.
            //       Consider redesigning it to deal with a dictionary of file systems,
            //       where the database ID of a file system is a key of that file system.
            //       Dictionaries are much easier to deal with than arrays.
            //       Another option would be to separate file system descriptions
            //       from the statistics:
            //
            //         fsdef: [{group: <group>, name: <name>}, ... {}]
            //         stat:  {<fs_id_1>: [], ... <fs_id_N>: []}
            //
            //       As an additional (though, it's questiobale if that's too important)
            //       benefit The later would allow separate loading of the UI configuration
            //       from the actual updates.

            this._set_updated(name) ;

            Fwk.web_service_GET (url, params, function (data) {

                if (_.isUndefined(params.fs_id)) {

                    _that._filesystems = data.filesystems ;
                    _that._display_all() ;

                } else {
                    
                    // Display just the specified one

                    for (var i in _that._filesystems) {
                        if (_that._filesystems[i].id === params.fs_id) {
                            _that._filesystems[i] = data.filesystems[0] ;
                            _that._display(params.fs_id) ;
                            break ;
                        }
                    }
                }
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
        this._display_all = function () {
            for (var i in this._filesystems) {
                this._display(this._filesystems[i].id) ;
            }
        } ;
        this._display = function (fs_id) {
            var fs = this._id2fs(fs_id) ;
            if (fs)
                if (this._view_selector(fs.id, 'table').attr('checked')) {
                    this._view(fs.id, 'table').removeClass('view-hidden') .addClass('view-visible') ;
                    this._view(fs.id, 'plot') .removeClass('view-visible').addClass('view-hidden') ;
                    this._display_table(fs) ;
                } else if (this._view_selector(fs.id, 'plot').attr('checked')) {
                    this._view(fs.id, 'table').removeClass('view-visible').addClass('view-hidden') ;
                    this._view(fs.id, 'plot') .removeClass('view-hidden') .addClass('view-visible') ;
                    this._display_plot(fs) ;
                }
        } ;
        this._display_table = function (fs) {
            var rows = [] ;
            for (var i in fs.stats) {
                var s = fs.stats[i] ;
                rows.push([
                    s.insert_time.day+'&nbsp;&nbsp;&nbsp;&nbsp;<span style="font-weight:normal;" >'+s.insert_time.hms+'</span>' ,
                    Math.floor(s.used / TB) ,
                    Math.floor(s.available / TB) ,
                    Math.floor((s.available / (s.used + s.available)) * 100.)
                ]) ;
            }
            this._table(fs.id).load(rows) ;
        } ;
        this._display_plot = function (fs) {

            var plot   = this._plot(fs.id) ;
            var series = plot.series[0] ;

            var data = [] ;
            for (var i in fs.stats) {
                var s = fs.stats[i] ;
                var t_msec = s.insert_time.sec * 1000 ;
                data.push ([
                    t_msec ,
                    Math.floor((s.used / (s.used + s.available)) * 100.)
                ]) ;
            }
            series.setData(data, false) ;   // do the bulk redraw later
            plot.reflow() ;     // in case if the container geometry has changed
            plot.redraw() ;     // batch update for efficiency
        } ;
    }
    Class.define_class (DMMon_FS_Usage, FwkApplication, {}, {}) ;
    
    return DMMon_FS_Usage ;
}) ;


