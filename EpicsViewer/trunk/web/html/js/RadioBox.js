define ([
    'CSSLoader' ,
    'Class' ,
    'Widget' ,
    'underscore'] ,

function (
    cssloader ,
    Class ,
    Widget) {

    cssloader.load('css/RadioBox.css') ;

/**
 * The radio-box widget encapsulates buttons
 *
 * USAGE:
 * 
 *   TO BE DONE LATER...
 *
 * @param array buttons
 * @param function onchange
 * @param object options
 * @returns {RadioBox}
 */
function RadioBox (buttons, onchange, options) {

    var _that = this ;

    // Always call the c-tor of the base class

    Widget.Widget.call(this) ;

    // Verify parameters of the object

    Widget.ASSERT(_.isArray(buttons) && buttons.length) ;
    this._buttons = _.map(buttons, function (button) {
        Widget.ASSERT (
            _.isObject(button) &&
            _.has(button, 'name') && _.isString(button.name) && button.name !== '' &&
            _.has(button, 'text') && _.isString(button.text) && button.text !== '') ;

        function get_prop (button, prop) {
            if (_.has(button, prop)) {
                var val = button[prop] ;
                Widget.ASSERT(_.isString(val)) ;
                return val ;
            }
            return '' ;
        }
        var result = {
            name:  button.name ,
            text:  _.escape(button.text) ,
            title: _.escape(get_prop(button, 'title')) ,
            class: get_prop(button, 'class') ,
            style: get_prop(button, 'style')
        } ;
        return result ;
    }) ;

    Widget.ASSERT(_.isFunction(onchange)) ;
    this._onchange = onchange ;

    this._options = {
        activate: this._buttons[0].name ,   // the button to be activa by default
        pointer:  false                     // the optional pointer to the active button
    } ;
    if (options) {
        Widget.ASSERT(_.isObject(options)) ;
        if (_.has(options, 'activate')) {
            var button2activate = _.find(this._buttons, function (button) {
                return button.name === options.activate ;
            }) ;
            Widget.ASSERT(button2activate) ;
            this._options.activate = button2activate.name ;
        }
        this._options.pointer = _.has(options, 'pointer') && options.pointer ;
    }

    // Rendering is done only once

    this._is_rendered = false ;
    this._button_elements = null ;

    /**
     * Implement the widget rendering protocol
     *
     * @returns {undefined}
     */
    this.render = function () {

        if (this._is_rendered) return ;
        this._is_rendered = true ;

        // Render the widget
        var html =
'<div class="radio-box" >' ;
        for (var i in this._buttons) {
            var button = this._buttons[i] ;
            html +=
  '<div style="float:left;" >' +
    '<button name="'+button.name+'"' +
           ' class="radio-box-button '+button.class+'"' +
           ' style="'+button.style+'"' +
           ' data="'+button.title+'" >'+button.text+'</button>' +
    '<div class="radio-box-hint" >&nbsp;</div>' +
  '</div>' ;
        }
        html +=
  '<div style="clear:both;" >' +
'</div>' ;
        this.container.html(html) ;

        // Initialize the dynamic state of the widget as per its
        // parameters.
        this._button_elements = this.container.find('.radio-box-button').
            button().
            button('enable') ;
        this.container.find('.radio-box-button[name="'+this._options.activate+'"]').
            button().
            button('disable').
            addClass('radio-box-active').
            next().
            addClass('radio-box-hint-active') ;

        // Initialize event handlers on user actions
        this._button_elements.click(function () {       
            _that.activate(this.name) ;
            _that._onchange(this.name) ;
        }) ;

    } ;
    
    /**
     * Return the name of the presently active button.
     * 
     * @returns {string}
     */
    this.active = function () {
        Widget.ASSERT(this._is_rendered) ;
        return this.container.find('.radio-box-button.radio-box-active').attr('name') ;
    } ;
    
    /**
     * Activate the specified button.
     * 
     * NOTES:
     * - throw an exception if not a valid button
     *
     * @param {string} name
     * @returns {undefined}
     */
    this.activate = function (name) {
        this._ASSERT_IS_VALID(name) ;
        this.container.find('.radio-box-button[name="'+this.active()+'"]').
            button('enable').
            removeClass('radio-box-active').
            next().
            removeClass('radio-box-hint-active') ;
        this.container.find('.radio-box-button[name="'+name+'"]').
            button('disable').
            addClass('radio-box-active').
            next().
            addClass('radio-box-hint-active') ;
    } ;
    
    /**
     * Disable all but the presently active buttons. For the later do noting.
     *
     * @param {boolean} yes
     * @returns {undefined}
     */
    this.disableAll = function (yes) {
        Widget.ASSERT(this._is_rendered) ;
        if (yes) {
            this._button_elements.button('disable') ;
        } else {
            this._button_elements.button('enable') ;
            this.activate(this.active()) ;  // the active button should stay disabled
        }
    } ;
    
    /**
     * Disable the specified button unless it's presently active. In the later
     * case do nothing.
     * 
     * @param {type} name
     * @param {boolean} yes
     * @returns {undefined}
     */
    this.disable = function (name, yes) {
        this._ASSERT_IS_VALID(name) ;
        var button = this.container.find('.radio-box-button[name="'+name+'"]') ;
        if (yes) {
            button.button('disable') ;
        } else {
            // the active button should stay disabled, so we can't do it here
            if (name !== this.active()) {
                button.button('enable') ;
            }
        }
    } ;
    
    /**
     * Assert that the specified button name is a valid one.
     *
     * @param {string} name
     * @returns {undefined}
     */
    this._ASSERT_IS_VALID = function (name) {
        Widget.ASSERT(this._is_rendered) ;
        Widget.ASSERT(_.isString(name)) ;
        var button = _.find(this._buttons, function (button) {
            return button.name === name ;
        }) ;
        Widget.ASSERT(button) ;
    } ;
}
Class.define_class(RadioBox, Widget.Widget, {}, {}) ;

    return RadioBox ;
}) ;