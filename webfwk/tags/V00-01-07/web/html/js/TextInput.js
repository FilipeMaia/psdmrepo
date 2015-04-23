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
     *   var input = new TextInput ($('input[name="timeout"]'), {
     *       disabled: true ,
     *       default_value: '600' ,
     *       on_validate: function (str) { var val = parseInt(str) ; return val < 600 : 600 : val ; } ,
     *       on_change: function (val) { alert(val) ; } ,
     *       config_handler: Fwk.config_handler('e-Log:Live','timeout')
     *   }) ;
     *   input.enable() ;
     *   input.set_value(123) ;
     *   alert(input.value());
     * 
     * @param object cont
     * @param object config
     * @returns {TextInput}
     */
    function TextInput (cont, config) {

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

        // ----------------------------------
        //   Parse configuration parameters
        // ----------------------------------

        var config2parse = config || {} ;
        Widget.ASSERT(_.isObject(config2parse)) ;

        this._config.disabled       = Widget.PROP_FUNCTION(config2parse, 'disabled ',      false) ;
        this._config.default_value  = Widget.PROP_STRING  (config2parse, 'default_value',  '') ;
        this._config.on_validate    = Widget.PROP_FUNCTION(config2parse, 'on_validate',    null) ;
        this._config.on_change      = Widget.PROP_FUNCTION(config2parse, 'on_change',      null) ;
        this._config.config_handler = Widget.PROP_OBJECT  (config2parse, 'config_handler', null) ;

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

            // Validate and correct (if needed) the nput value
            var validated = this._config.on_validate ? this._config.on_validate(val) : val ;

            // Update transient state
            this.container.val(validated) ;

            // Update persistent state if requested and if the one
            // is available.
            if (this._config.config_handler && update_persistent_state)
                this._config.config_handler.save(val) ;
        }

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

            // Set default value. And no update to the persistent backend
            // is made since the default value is usually a random choice
            // of a client application designer.
            this._set_value(this._config.default_value, false) ;

            // Process user input
            this.container.change(function () {

                // Validate and correct (if needed) the input
                var val = _that.container.val() ;
                if (_that._config.on_validate) {
                    val = _that._config.on_validate(val) ;
                    _that.container.val(val) ;
                }
                
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
    Class.define_class(TextInput, Widget.Widget, {}, {}) ;

    return TextInput ;
}) ;