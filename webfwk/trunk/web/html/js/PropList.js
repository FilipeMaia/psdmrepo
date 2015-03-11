define ([
    //'underscore' ,
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/Widget'] ,

function (
    //_ ,
    cssloader, Class, Widget) {

    cssloader.load('../webfwk/css/PropList.css') ;

    /**
     * The 2-column tabular widget representing properties and their values
     *
     * USAGE:
     * 
     *   TO BE COMPLETED...
     *
     * @param array properties
     * @returns {PropList}
     */
    function PropList (propdefs) {

        var _that = this ;

        // Always call the c-tor of the base class

        Widget.Widget.call(this) ;

        function _ASSERT (expression) {
            if (!expression) throw new Widget.WidgetError('PropList::'+arguments.callee.caller.name) ;
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
        function get_prop_bool (propdef, prop, default_val) {
            return get_prop(propdef, prop, default_val, _.isBoolean) ;
        }
        function get_prop_int (propdef, prop, default_val) {
            return get_prop(propdef, prop, default_val, _.isNumber) ;
        }
        function get_prop_enum (propdef, prop, allowed_values) {
            var val = get_prop_string (propdef, prop, allowed_values[0]) ;
            _ASSERT(_.find(allowed_values, function (v) {return v === val ; }) !== undefined) ;
            return val ;
        }

        // Digest property definitions

        _ASSERT(_.isArray(propdefs) && propdefs.length) ;

        this._propnames = [] ;      // -- names of properties given in the original order
        this._propdefs = {} ;       // -- property definitions

        this._groups = 0 ;          // -- group number generator

        _.each(propdefs, function (propdef) {
            _ASSERT (
                _.isObject(propdef) &&
                ((_.has(propdef, 'group') && _.isString(propdef.group) && propdef.group !== '') ||
                 (_.has(propdef, 'name')  && _.isString(propdef.name)  && propdef.name  !== '' &&
                  _.has(propdef, 'text')  && _.isString(propdef.text)  && propdef.text  !== '')
                )
            ) ;

            // Make sure either the group or name is provided

            var group = get_prop_string(propdef, 'group', null) ;
            var name  = get_prop_string(propdef, 'name',  null) ;

            if (group) {
                
                // Generate a unique name which should not interfere with
                // the names of user-supplied properties.
                // 
                // TODO: evaluate user-supplied property names to prevent using
                //       reserved names.
                
                name = '__group:'+_that._groups ;
                _that._groups++ ;

                _that._propnames.push(name) ;
                _that._propdefs[name]  = {
                    is_group: true ,
                    value:    group ,
                    class:    get_prop_string(propdef, 'class', '') ,
                    style:    get_prop_string(propdef, 'style', '') ,
                    title:    get_prop_string(propdef, 'title', null)
                } ;
            } else {
                _that._propnames.push(name) ;
                _that._propdefs[name]  = {
                    is_group:  false ,
                    text:      _.escape(get_prop_string(propdef, 'text', name)) ,   // -- use the name if no text is provide
                    value:     get_prop_string(propdef, 'value', '') ,
                    type:      get_prop_string(propdef, 'type',  'text') ,
                    class:     get_prop_string(propdef, 'class', '') ,
                    style:     get_prop_string(propdef, 'style', '') ,
                    edit_mode: get_prop_bool  (propdef, 'edit_mode', false) ,
                    title:     get_prop_string(propdef, 'title', null) ,
                    editing:   false
                } ;
                if (_that._propdefs[name].edit_mode) {
                    _that._propdefs[name].edit_size = get_prop_int (propdef, 'edit_size', 4) ;
                    _that._propdefs[name].editor    = get_prop_enum(propdef, 'editor', ['text', 'checkbox']) ;
                }
            }
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
                var data_opt = _.isNull(propdef.title) ? '' : 'data="'+propdef.title+'"' ;
                if (propdef.is_group) {
                    html +=
'      <tr name="'+name+'" '+data_opt+' >' +
'        <td class="prop-list-value prop-list-group '+propdef.class+'" style="'+propdef.style+'" colspan="2" ></td>' +
'      </tr>' ;
                } else {
                    html +=
'      <tr name="'+name+'" '+data_opt+' >' +
'        <td class="prop-list-name" >'+propdef.text+'</td>' +
'        <td class="prop-list-value  '+propdef.class+'" style="'+propdef.style+'" ></td>' +
'      </tr>' ;
                }
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
         * Return the current value of the property.
         * 
         * Note, that if the property value is being edited then the current state
         * of the properting value would be returned.
         * @param {type} name
         * @returns {_L5.PropList._propdefs.value}
         */
        this.get_value = function (name) {

            var propdef = this._propdefs[name] ;

            _ASSERT(propdef) ;

            if (!this._is_rendered || !(propdef.edit_mode && propdef.editing)) return propdef.value ;

            switch (propdef.editor) {
                case 'text':     return propdef.value_elem.children('input').val() ;
                case 'checkbox': return propdef.value_elem.children('input').attr('checked') ? 1 : 0 ;
            }
            _ASSERT() ;
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
                if (propdef.edit_mode && propdef.editing) {
                    switch (propdef.editor) {
                        case 'text':
                            value_elem.children('input').val(value) ;
                            break ;
                        case 'checkbox':
                            if (value) value_elem.children('input').attr      ('checked', 'checked') ;
                            else       value_elem.children('input').removeAttr('checked') ;
                            break ;
                        default:
                            _ASSERT() ;
                    }
                } else {
                    switch (propdef.type) {
                        case 'html' :
                            value_elem.html('&nbsp;'+value) ;
                            break ;
                        default :
                            if (propdef.edit_mode && propdef.editor === 'checkbox') {
                                if (value) value_elem.html('<div style="width:8px; height:8px; background-color:red;">&nbsp;</div>') ;
                                else       value_elem.html('&nbsp;') ;
                            } else {
                                if (value === '') value_elem.html('&nbsp;') ;
                                else              value_elem.text(value) ;
                            }
                            break ;
                    }
                }
            }
        } ;

        /**
         * Turn the property value's cell into the editing mode (if permitted)
         *
         * @param {String} name
         * @returns {undefined}
         */
        this.edit_value = function (name) {

            var propdef = this._propdefs[name] ;

            _ASSERT(propdef) ;
            _ASSERT(propdef.edit_mode) ;

            if (this._is_rendered) {

                if (propdef.editing) return ;

                var value_elem = propdef.value_elem ;
                if (!value_elem) {
                    value_elem = this.container.find('tr[name="'+name+'"]').children('td.prop-list-value') ;
                    propdef.value_elem = value_elem ;
                }
                value_elem.addClass('prop-list-value-editing') ;
                switch (propdef.editor) {
                    case 'text':
                        value_elem.html('<input type="text" size="'+propdef.edit_size+'" />') ;
                        value_elem.children('input').val(propdef.value) ;
                        break ;
                    case 'checkbox':
                        value_elem.html('<input type="checkbox" />') ;
                        if (propdef.value) value_elem.children('input').attr('checked', 'checked') ;
                        break ;
                    default:
                        _ASSERT() ;
                }
                propdef.editing = true ;
            }
        } ;

        /**
         * Extract the presently edited value of the property from its cell and turn
         * the cell into the viewing mode.
         *
         * @param {String} name
         * @returns {undefined}
         */
        this.view_value = function (name) {

            var propdef = this._propdefs[name] ;

            _ASSERT(propdef) ;
            _ASSERT(propdef.edit_mode) ;

            if (this._is_rendered) {

                if (!propdef.editing) return ;

                var value_elem = propdef.value_elem ;
                if (!value_elem) {
                    value_elem = this.container.find('tr[name="'+name+'"]').children('td.prop-list-value') ;
                    propdef.value_elem = value_elem ;
                }
                value_elem.removeClass('prop-list-value-editing') ;
                switch (propdef.editor) {
                    case 'text':
                        propdef.value = value_elem.children('input').val() ;
                        break ;
                    case 'checkbox':
                        propdef.value = value_elem.children('input').attr('checked') ? 1 : 0 ;
                        break ;
                    default:
                        _ASSERT() ;
                }
                switch (propdef.type) {
                    case 'html' :
                        value_elem.html('&nbsp;'+propdef.value) ;
                        break ;
                    default :
                        if (propdef.editor === 'checkbox') {
                            if (propdef.value) value_elem.html('<div style="width:8px; height:8px; background-color:red;">&nbsp;</div>') ;
                            else               value_elem.html('&nbsp;') ;
                        } else {
                            if (propdef.value === '') value_elem.html('&nbsp;') ;
                            else                      value_elem.text(propdef.value) ;
                        }
                        break ;
                }
                propdef.editing = false ;
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
    Class.define_class(PropList, Widget.Widget, {}, {}) ;

    return PropList ;
}) ;