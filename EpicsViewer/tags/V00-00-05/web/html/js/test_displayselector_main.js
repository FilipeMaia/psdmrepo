require.config ({
    baseUrl: '..' ,

    waitSeconds : 15,
    urlArgs     : "bust="+new Date().getTime() ,

    paths: {
        'jquery'      : '/jquery/js/jquery-1.8.2' ,
        'jquery-ui'   : '/jquery/js/jquery-ui-1.9.1.custom.min' ,
        'underscore'  : '/underscore/underscore-min' ,
        'webfwk'      : 'webfwk/js' ,
        'EpicsViewer' : 'EpicsViewer/js'
    } ,
    shim : {
        'jquery' : {
            exports : '$'
        } ,
        'jquery-ui' : {
            exports : '$' ,
            deps : ['jquery']
        } ,
        'jquery.resize' :  {
            deps : ['jquery']
        } ,
        'jquery.mousewheel' :  {
            deps : ['jquery']
        } ,
        'underscore' : {
            exports  : '_'
        }
    }
}) ;

require ([
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/Widget' , 'EpicsViewer/DisplaySelector' ,

    // Make sure the core libraries are preloaded so that the applications
    // won't borther with loading them individually

    'jquery', 'jquery-ui', 'underscore'] ,

function (cssloader, Class, Widget, DisplaySelector) {

    cssloader.load('/jquery/css/custom-theme-1.9.1/jquery-ui.custom.css') ;

    function DummyPlot (parent, message) {
        Widget.Widget.call(this) ;
        this._parent = parent ;
        this._message = message ;
        this.render = function () {
            if (this._isRendered) return ;
            this._isRendered = true ;
            this.container.html(this._message) ;
        } ;
    }
    Class.define_class(DummyPlot, Widget.Widget, {}, {}) ;

    $(function () {
        var ds = new DisplaySelector($('#display'), [
            {   id:     "timeseries" ,
                name:   "T<sub>series</sub>" ,
                descr:  "Time series plots for PVs and functions" ,
                widget: new DummyPlot(this, 'TimeSeries')} ,
            
            {   id:     "waveform" ,
                name:   "W<sub>form</sub>" ,
                desc:   "Waveform plots for PVs and functions" ,
                widget: new DummyPlot(this, 'Waveform')} ,

            {   id:     "correlation" ,
                name:   "C<sub>plot</sub>" ,
                descr:  "Correlation plots for select PVs and functions" ,
                widget: new DummyPlot(this, 'Correlation plot')} ,
            
            {   id:     "histogram" ,
                name:   "H-gram" ,
                descr:  "Histograms for all relevant PVs and functions" ,
                widget: new DummyPlot(this, 'Histograms')} ,

            {   id:     "info" ,
                name:   "Info" ,
                descr:  "Detailed information on PVs, functions and plots" ,
                widget: new DummyPlot(this, 'Info')}
        ]) ;
        ds.activate('waveform') ;
    }) ;
}) ;


