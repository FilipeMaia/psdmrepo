/**
 * The 2-column tabular widget representing properties and their values
 *
 * DEPENDENCIES:
 *      underscrore.js
 *      jquery
 *      jquery ui
 *      Widget.js
 *
 * STYLING:
 *      proplist.css
 *
 * USAGE:
 *
 * @param array properties
 * @returns {PropList}
 */
function PropList (propdefs) {

    var _that = this ;

    // Always call the c-tor of the base class

    Widget.call(this) ;

    function _ASSERT (expression) {
        if (!expression) throw new WidgetError('PropList::'+arguments.callee.caller.name) ;
    }
    function get_prop (propdef, prop, default_val, validator) {
        if (_.has(propdef, prop)) {
            var val = propdef[prop] ;
            if (validator) _ASSERT(validator(val)) ;
            return val ;
        }
        if (default_val === undefined) _ASSERT() ;
        return default_val ;
    }
    function get_prop_string (propdef, prop, default_val) {
        return get_prop(propdef, prop, default_val, _.isString) ;
    }

    // Digest property definitions

    _ASSERT(_.isArray(propdefs) && propdefs.length) ;

    this._propnames = [] ;      // -- names of properties given in the original order
    this._propdefs = {} ;       // -- property definitions

    _.each(propdefs, function (propdef) {
        _ASSERT (
            _.isObject(propdef) &&
            _.has(propdef, 'name') && _.isString(propdef.name) && propdef.name !== '' &&
            _.has(propdef, 'text') && _.isString(propdef.text) && propdef.text !== '') ;

        var name = get_prop_string(propdef, 'name') ;
        _ASSERT(name !== '') ;

        _that._propnames.push(name) ;
        _that._propdefs[name]  = {
            text:  _.escape(get_prop_string(propdef, 'text', name)) ,   // -- use the name if no text is provide
            value: get_prop_string(propdef, 'value', '') ,
            type:  get_prop_string(propdef, 'type',  'text') ,
            class: get_prop_string(propdef, 'class', '') ,
            style: get_prop_string(propdef, 'style', '')
        } ;
    }) ;

    // Rendering is done only once

    this._is_rendered = false ;

    /**
     * Implement the widget rendering protocol
     *
     * @returns {undefined}
     */
    this.render = function () {

        if (this._is_rendered) return ;
        this._is_rendered = true ;

        var html =
'<div class="prop-list" >' +
'  <table>' +
'    <tbody>' +
        _.reduce(this._propnames , function (html, name) {
            var propdef = _that._propdefs[name] ;
            html +=
'      <tr name="'+name+'">' +
'        <td class="prop-list-name" >'+propdef.text+'</td>' +
'        <td class="prop-list-value" '+propdef.class+'" style="'+propdef.style+'"></td>' +
'      </tr>' ;
            return html ;
        }, '') +
'    </tbody>' +
'  </table>' +
'</div>' ;
        this.container.html(html) ;

        // Set initial values (if any)

        _.each(this._propnames, function (name) {
            var propdef = _that._propdefs[name] ;
            _that.set_value(name, propdef.value) ;
        }) ;
    } ;

    /**
     * @brief Set the value of the specified property
     * 
     * The function will also find and cache a JQuery object corresponding to the HTML
     * element where the value is stored.
     *
     * @param {String} name
     * @param {String|Number} value
     * @returns {undefined}
     */
    this.set_value = function (name, value) {
        var propdef = this._propdefs[name] ;
        _ASSERT(propdef) ;
        propdef.value = value ;
        if (this._is_rendered) {
            var value_elem = propdef.value_elem ;
            if (!value_elem) {
                value_elem = this.container.find('tr[name="'+name+'"]').children('td.prop-list-value') ;
                propdef.value_elem = value_elem ;
            }
            switch (propdef.type) {
                case 'html' : value_elem.html(value) ; break ;
                default     : value_elem.text(value) ; break ;
            }
        }
    } ;

    /**
     * Load values of select properties.
     * 
     * NOTES:
     * - throw an exception if not a valid object
     *
     * @param {String|Array|Object} values
     * @returns {undefined}
     */
    this.load = function (values) {
        if (_.isString(values)) {                       // - Set the same value to all
            _.each(this._propnames, function (name) {
                _that.set_value(name, values) ;
            }) ;

        } else if (_.isArray(values)) {                 // - Set values from an input array using
            _.each(values, function (value, i) {        // at most the total number of properties.
                var name = _that._propnames[i] ;
                _ASSERT(name !== undefined) ;
                _that.set_value(name, value) ;
            }) ;
        } else if (_.isObject(values)) {                // - Set values for propertis whose names are
            _.each(values, function (value, name) {     // matching the ones found in the input object.
                _that.set_value(name, value) ;
            }) ;
        } else {
            _ASSERT(false) ;
        }
    } ;
}
define_class(PropList, Widget, {}, {}) ;