define ([
    'webfwk/Class', 'webfwk/Widget' ,
    'underscore'] ,

function (
    Class, Widget) {

    /**
     * @brief The simpe input widget based on the HTML "input" element
     *
     * DESCRIPTION:
     * 
     * This widget is a wrapper around HTML input element. Here is how it
     * should be used:
     * 
     *   var selector = new SelectOption ($('select[name="instrument"]'), {
     *       disabled: true ,
     *       options: [
     *           {value: '',    default: true} ,
     *           {value: 'AMO', text: 'The AMO Instrument'} ,
     *           {value: 'SXR', text: 'The SXR Instrument'}
     *       ] ,
     *       on_change: function (val) { alert(val) ; } ,
     *       config_handler: Fwk.config_handler('e-Log:Live','instrument')
     *   }) ;
     *   selector.enable() ;
     *   selector.set_value('AMO') ;
     *   alert(input.value());
     * 
     * @param object cont
     * @param object config
     * @returns {SelectOption}
     */
    function SelectOption (cont, config) {

        var _that = this ;

        // -------------------------------------------
        //   Always call the c-tor of the base class
        // -------------------------------------------

        Widget.Widget.call(this) ;

        // ------------------------------
        //   Data members of the object
        // ------------------------------

        this._config = {} ;

        this._is_rendered = false ;     // rendering is done only once

        // -----------------------
        //   Parse configuration
        // -----------------------

        var config2parse = config || {} ;
        Widget.ASSERT(_.isObject(config2parse)) ;

        this._config.disabled       = Widget.PROP_FUNCTION(config2parse, 'disabled ',      false) ;
        this._config.on_change      = Widget.PROP_FUNCTION(config2parse, 'on_change',      null) ;
        this._config.config_handler = Widget.PROP_OBJECT  (config2parse, 'config_handler', null) ;
        this._config.options        = [] ;
        this._config.default_value  = null ;

        Widget.ASSERT (
            _.has(config, 'options') &&
            _.isArray(config.options) && config.options.length) ;

        _.each(config.options, function (opt) {

            Widget.ASSERT(_.isObject(opt)) ;

            var value = Widget.PROP_STRING(opt, 'value') ;

            _that._config.options.push({
                value: value ,
                text:  Widget.PROP_STRING(opt, 'text', value)  // use the value if no text is provided
            }) ;
            
            // Make sure the default value is initialized either with
            // an exlicitly requested one, or with the first one found
            // during the iteration.
            if (Widget.PROP_BOOL(opt, 'default', false) || _.isNull(_that._config.default_value))
                _that._config.default_value  = value ;
        }) ;


        this.value = function () {
            Widget.ASSERT(this._is_rendered) ;
            return this.container.val() ;
        } ;

        this.set_value = function (val) {
            Widget.ASSERT(this._is_rendered) ;
            
            // State change requests made by users through this public
            // method should alays be reported to the persistent backend.
            this._set_value(val, true) ;
        } ;

        this._set_value = function (val, update_persistent_state) {

            // Validate the parameter's value
            Widget.ASSERT(!_.isUndefined (
                _.find(this._config.options, function (opt) {
                    return opt.value === val ;
                })
            )) ;

            // Update transient state
            this.container.val(val) ;

            // Update persistent backend if requested and if the one
            // is available.
            if (this._config.config_handler && update_persistent_state)
                this._config.config_handler.save(val) ;
        } ;

        this.is_disabled = function () {
            Widget.ASSERT(this._is_rendered) ;
            return this.container.attr('disabled') ? 1 : 0 ;
        } ;
        this.disable = function () {
            Widget.ASSERT(this._is_rendered) ;
            this.container.attr('disabled', 'disabled') ;
        } ;

        this.enable = function () {
            Widget.ASSERT(this._is_rendered) ;
            this.container.removeAttr('disabled') ;
        } ;

        /**
         * @brief Implement the widget rendering protocol as required by
         *        the base class Widget.
         *
         * @returns {undefined}
         */
        this.render = function () {

            if (this._is_rendered) return ;
            this._is_rendered = true ;

            // Render options
            this.container.html(_.reduce (
                this._config.options ,
                function (html, opt) {
                    return html += '<option value="'+opt.value+'" >'+opt.text+'</option> ' ;
                } ,
                ''
            )) ;

            // Set default value. And no update to the persistent backend
            // is made since the default value is usually a random choice
            // of a client application designer.
            this._set_value(this._config.default_value, false) ;

            // Process user input
            this.container.change(function () {

                var val = _that.container.val() ;
                
                // Notify a subscriber (if any)
                if (_that._config.on_change)
                    _that._config.on_change(val) ;

                // Update the persistent backend
                if (_that._config.config_handler)
                    _that._config.config_handler.save(val) ;
            }) ;

            // Load last used value (if any) from the persistent backend
            if (this._config.config_handler)
                this._config.config_handler.load(function (val) {
                    // No update to the persistent backend. It just makes no sense
                    // to do so from here sine we have just got the latest value
                    // from there.
                    _that._set_value(val, false) ;

                    // Notify a subscriber (if any)
                    if (_that._config.on_change)
                        _that._config.on_change(val) ;
                }) ;
            
            // Set desired state if requested
            if (this._config.disabled)
                this.disable() ;
        } ;

        // Trigger rendering with the provided container        
        this.display(cont) ;
    }
    Class.define_class(SelectOption, Widget.Widget, {}, {}) ;

    return SelectOption ;
}) ;