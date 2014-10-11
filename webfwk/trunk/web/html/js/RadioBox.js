define ([
    'webfwk/CSSLoader', 'webfwk/Class','webfwk/Widget' ,

    'underscore'] ,

function (
    cssloader, Class, Widget) {

    cssloader.load('../webfwk/css/RadioBox.css') ;

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

    function _ASSERT (expression) {
        if (!expression) throw new Widget.WidgetError('RadioBox::'+arguments.callee.caller.name) ;
    }

    // Verify parameters of the object

    _ASSERT(_.isArray(buttons) && buttons.length) ;
    this._buttons = _.map(buttons, function (button) {
        _ASSERT (
            _.isObject(button) &&
            _.has(button, 'name') && _.isString(button.name) && button.name !== '' &&
            _.has(button, 'text') && _.isString(button.text) && button.text !== '') ;

        function get_prop (button, prop) {
            if (_.has(button, prop)) {
                var val = button[prop] ;
                _ASSERT(_.isString(val)) ;
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

    _ASSERT(_.isFunction(onchange)) ;
    this._onchange = onchange ;

    this._options = {
        activate: this._buttons[0].name ,   // the button to be activa by default
        pointer:  false                     // the optional pointer to the active button
    } ;
    if (options) {
        _ASSERT(_.isObject(options)) ;
        if (_.has(options, 'activate')) {
            var button2activate = _.find(this._buttons, function (button) {
                return button.name === options.activate ;
            }) ;
            _ASSERT(button2activate) ;
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

        var html =
'<div class="radio-box" >' ;
        for (var i in this._buttons) {
            var button = this._buttons[i] ;
            html +=
  '<div style="float:left;" >' +
    '<button name="'+button.name+'"' +
           ' class="radio-box-button '+button.class+'"' +
           ' style="'+button.style+'"' +
           ' title="'+button.title+'" >'+button.text+'</button>' +
    '<div class="radio-box-hint" >&nbsp;</div>' +
  '</div>' ;
        }
        html +=
  '<div style="clear:both;" >' +
'</div>' ;
        this.container.html(html) ;

        this._button_elements = this.container.find('.radio-box-button').
            button().
            button('enable') ;

        this._button_elements.
            next().
            removeClass('radio-box-hint-active') ;

        this.container.find('.radio-box-button[name="'+this._options.activate+'"]').
            button().
            button('disable').
            addClass('radio-box-active').
            next().
            addClass('radio-box-hint-active') ;

        this._button_elements.click(function () {

            _that._button_elements.
                button('enable').
                removeClass('radio-box-active').
                next().
                removeClass('radio-box-hint-active') ; 

            $(this).
                button('disable').
                addClass('radio-box-active').
                next().
                addClass('radio-box-hint-active') ;
        
            _that._onchange(this.name) ;
        }) ;
    } ;
    
    this.active = function () {
        _ASSERT(this._is_rendered) ;
        return this.container.find('.radio-box-button.radio-box-active"]').attr('name') ;
    } ;
    
    /**
     * Activate the specified button.
     * 
     * NOTES:
     * - throw an exception if not a valid button
     *
     * @param string name
     * @returns {undefined}
     */
    this.activate = function (name) {
        _ASSERT(this._is_rendered) ;
        _ASSERT(_.isString(name)) ;
        var button2activate = _.find(this._buttons, function (button) {
            return button.name === name ;
        }) ;
        _ASSERT(button2activate) ;
        this._button_elements.
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
}
Class.define_class(RadioBox, Widget.Widget, {}, {}) ;

    return RadioBox ;
}) ;