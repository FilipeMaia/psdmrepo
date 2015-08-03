define ([
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/Widget' ,
    'underscore'] ,

function (
    CSSLoader, Class, Widget) {

    CSSLoader.load('../EpicsViewer/css/DisplaySelector.css') ;

    /**
     * @brief display area manager/selector
     *
     * DESCRIPTION:
     * 
     * This widget provides a layout and a selector for user-defined displays:
     * 
     *   var ds = new DisplaySelector ($('#display'), [
     *       {  id:     'timeseries' ,
     *          title:  'T<sub>series</sub>' ,
     *          descr:  'This is the TimeSeries plotter. \nUse it at your discretion.' ,
     *          widget: new TimeSeriesDisplay(this)} ,
     *
     *       {  id:     'waveform' ,
     *          title:  'T<sub>series</sub>' ,
     *          descr:  '...' ,
     *          widget: new WafeFormDisplay(this)}
     *   ]) ;
     *
     *   ds.activate('waveform').load(...) ;
     *   ds.get('waveform').load(...) ;
     * 
     * IMPORTANT: the widget objects passed with the configuration
     * must be subclasses of the Widget class. That's important because
     * the selector class will also asked them to be rendered after
     * the selector renders itself.
     *
     * @param object cont
     * @param object displays
     * @returns {DisplaySelector}
     */
    function DisplaySelector (cont, displays) {

        var _that = this ;

        // -------------------------------------------
        //   Always call the c-tor of the base class
        // -------------------------------------------

        Widget.Widget.call(this) ;

        // ------------------------------
        //   Data members of the object
        // ------------------------------

        this._ids = [] ;                // display identifiers in the original order
        this._id2display = {} ;         // display descriptors

        this._is_rendered = false ;     // rendering is done only once

        // -----------------------
        //   Parse configuration
        // -----------------------

        var displays2parse = displays || [] ;
        Widget.ASSERT (
            _.isArray(displays2parse) && displays2parse.length) ;

        _.each(displays2parse, function (disp) {

            Widget.ASSERT(_.isObject(disp)) ;

            var id     = Widget.PROP_STRING(disp, 'id') ;
            var widget = Widget.PROP_OBJECT(disp, 'widget') ;
            Widget.ASSERT(widget instanceof Widget.Widget) ;

            _that._ids.push(id) ;
            _that._id2display[id] = {
                name:   Widget.PROP_STRING(disp, 'name',  id) , // use the identifier if no name is provided
                descr:  Widget.PROP_STRING(disp, 'descr', '') , // empty descripton by default
                widget: widget
            } ;
        }) ;

        /**
         * @brief Implement the widget rendering protocol as required by
         *        the base class Widget.
         *
         * @returns {undefined}
         */
        this.render = function () {

            if (this._is_rendered) return ;
            this._is_rendered = true ;

            // Render the widget
            this.container
                .addClass('disp-sel')
                .html(
'<div id="selector" > ' + _.reduce(this._ids, function (html, id) { var d = _that._id2display[id] ; return html +=
  '<div class="disp-sel-item" id="'+id+'" data="'+d.descr+'" >'+d.name+'</div> ' ; }, '') +
  '<div class="disp-sel-item-last" ></div> ' +
'</div> ' +
'<div id="area" > ' + _.reduce(this._ids, function (html, id) { var d = _that._id2display[id] ; return html +=
  '<div class="disp-sel-area" id="'+id+'" ></div> ' ; }, '') +
'</div> ' /*+
'<div id="end" ></div> '*/) ;


            var selector = this.container.children('#selector') ,
                area     = this.container.children('#area') ,
                last     = this.container.children('#selector').children('.disp-sel-item-last');
            area.css('width',  (this.container.innerWidth() - selector.innerWidth()) + 'px') ;
            console.log('window.innerHeight:'+window.innerHeight) ;
            last.css('height', (window.innerHeight - last.offset().top - 52) + 'px') ;
            $(window).resize(function () {
                area.css('width', (_that.container.innerWidth() - selector.innerWidth()) + 'px') ;
                console.log('window.innerHeight:'+window.innerHeight) ;
                last.css('height', (window.innerHeight - last.offset().top - 52) + 'px') ;
//                _that._display() ;  // redisplay is needed to prevent plots
//                                    // from being scaled.
            }) ;



            // propagate rendering to the displays
            _.each(this._ids, function (id) {
                _that._id2display[id].widget.display (
                    $(_that.container.children('#area').children('.disp-sel-area#'+id))
                ) ;
            }) ;

            // Activate the first item in the list
            this.activate(this._ids[0]) ;

            // Process user selection
            this.container.children('#selector').children('.disp-sel-item').click(function () {
                _that.activate($(this).attr('id')) ;
            }) ;

        } ;

        /**
         * Return an widget object for the specified identifier
         *
         * @param {string} id
         * @returns {Widget}
         */
        this.get = function (id) {
            return this._getDisplay(id).widget ;
        } ;

        /**
         * Activate an widget area for the specified identifier and
         * return the widget object.
         *
         * @param {string} id
         * @returns {Widget}
         */
        this.activate = function (id) {
            var widget = this.get(id) ;
            // selectors
            this.container.children('#selector').children('.disp-sel-item')    .removeClass('disp-sel-item-active') ;
            this.container.children('#selector').children('.disp-sel-item#'+id).addClass   ('disp-sel-item-active') ;
            // areas
            this.container.children('#area').children('.disp-sel-area')    .removeClass('disp-sel-area-active') ;
            this.container.children('#area').children('.disp-sel-area#'+id).addClass   ('disp-sel-area-active') ;
            // tell the widget that it's now active
            widget.on_activate() ;
            return widget ;
        } ;

        this._getDisplay = function (id) {
            Widget.ASSERT(this._is_rendered) ;
            Widget.ASSERT(_.has(this._id2display, id)) ;
            return this._id2display[id] ;
        } ;
        this._activeId = function () {
            Widget.ASSERT(this._is_rendered) ;
            var elem = this.container.children('#selector').children('.disp-sel-item-active') ;
            Widget.ASSERT(elem.length) ;
            return elem[0].attr('id') ;
        } ;

        // Trigger rendering with the provided container        
        this.display(cont) ;
    }
    Class.define_class(DisplaySelector, Widget.Widget, {}, {}) ;

    return DisplaySelector ;
}) ;