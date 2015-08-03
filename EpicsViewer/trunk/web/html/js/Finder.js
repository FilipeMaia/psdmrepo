/**
 * PV Finder UI
 */
define ([
    'CSSLoader' ,
    'WebService'
] ,

function (
    CSSLoader ,
    WebService) {

    CSSLoader.load('css/Finder.css') ;

    var _KEY_ENTER = 13 ,
        _KEY_ESC = 27 ;

    /**
     * The PV Finder UI helps a user to find PVs then calls the speified
     * call back function when a selection is made.
     *
     * @param {Object} cont
     * @param {Object} config
     * @returns {_Finder}
     */
    function _Finder (cont, config) {

        var _that = this ;

        // The callback to report user selected PV
        this._on_select = config.on_select ;

        // Initialize UI
        //
        // TODO: This may change if the class will become
        //       a standard Widget.

        cont.addClass('finder') ;
        cont.html (
'<div id="logo" ><img src="img/View.png" /></div> ' +
'<div id="input" ' +
  'data="' +
    'Enter GLOB pattern and press ENTER to find PVs. \n' +
    'Press ESC to clear the input and close the search window." > ' +
'  <input type="text" /> ' +
'</div> ' +
'<div id="results" ></div> ' +
'<button id="new_funct" data="Define a new function for generating a synthetic time series \n' +
'The series will get a unique name and it can be plotted along side PVs." >ADD FUNCTION</button> ' +
'<button id="new_cplot" data="Define a correlation plot" >C-PLOT CONFIG</button> ' +
'<div style="clear:both;" ></div> '
        ) ;
        this._input   = cont.find('#input > input') ;
        this._results = cont.children('#results') ;
        this._new_funct = cont.find('#new_funct').button() ;
        this._new_cplot = cont.find('#new_cplot').button() ;

        this._input.keyup(function (e) {
            switch (e.keyCode) {
                case _KEY_ENTER:
                    var pattern = $(this).val() ;
                    if (pattern === '') {
                        _that._results.removeClass('visible') ;
                        return ;
                    }
                    _that._load_pvs(pattern) ;
                    break ;

                case _KEY_ESC:
                    _that._results.removeClass('visible') ;
                    $(this).val('') ;
                    break ;
            }
        }) ;
        this._new_funct.click(function () {
            alert('This feature has not been implemented') ;
        }) ;
        this._new_cplot.click(function () {
            alert('This feature has not been implemented') ;
        }) ;

        
        // Make sure the window wher ewe report our findings stays within
        // the limits of the visible viewport.
        this._resize = function () {
            this._results.css (
                'max-height' ,
                (window.innerHeight - this._results.offset().top - 40)+'px'
            ) ;
        } ;
        this._resize() ;
        $(window).resize(function () { _that._resize() ; }) ;

        // Processing user selection

        this._pvs = [] ;

        this._load_pvs = function (pattern) {
            this._results.removeClass('visible') ;
            var params = {
                pv: pattern ,
                limit: 4000
            } ;
            WebService.GET (
                window.global_options.retrieval_url_base + "/bpl/getMatchingPVs" ,
                params ,
                function (data) {
                    _that._pvs = data ;
                    _that._display_pvs() ;
                }
            ) ;
        } ;
        this._display_pvs = function () {
            if (!this._pvs.length) return ;

            this._results.addClass('visible') ;
            this._results.html (
                _.reduce(this._pvs, function (html, pvname) { return html +=
'<div class="pvname">'+pvname+'</div>'; }, '' ) +
'<div class="pvname_endoflist"></div>'
            ) ;
            this._results.children('.pvname').click(function () {
                var pvname = $(this).text() ;
                _that._on_select(pvname) ;
                _that._results.removeClass('visible') ;
            }) ;
        } ;
    }
    return _Finder ;
}) ;