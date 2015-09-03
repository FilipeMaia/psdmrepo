define ([
    'webfwk/Class', 'webfwk/Widget' ,
    'underscore'] ,

function (
    Class, Widget) {

    /**
     * @brief The simpe input widget based on the HTML "input" checkbox element
     *
     * DESCRIPTION:
     * 
     * This widget is a wrapper around HTML input element. Here is how it
     * should be used:
     * 
     *   var checkbox = new Checkbox ($('input[name="yes_or_no"]'), {
     *       disabled: true ,
     *       default_value: '1' ,
     *       on_change: function (val) { alert(val) ; } ,
     *       config_handler: Fwk.config_handler('e-Log:Live','yes_or_no')
     *   }) ;
     *   checkbox.enable() ;
     *   checkbox.set_value(0) ;
     *   alert(checkbox.value());
     * 
     * @param object cont
     * @param object config
     * @returns {Checkbox}
     */
    function Checkbox (cont, config) {

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
        this._config.default_value  = Widget.PROP_NUMBER  (config2parse, 'default_value',  0) ;
        this._config.on_change      = Widget.PROP_FUNCTION(config2parse, 'on_change',      null) ;
        this._config.config_handler = Widget.PROP_OBJECT  (config2parse, 'config_handler', null) ;

        this.value = function () {
            Widget.ASSERT(this._is_rendered) ;
            return this.container.attr('checked') ? 1 : 0 ;
        } ;

        this.set_value = function (val) {
            Widget.ASSERT(this._is_rendered) ;

            // State change requests made by users through this public
            // method should alays be reported to the persistent backend.
            this._set_value(val, true) ;
        } ;
 
        this._set_value = function (val, update_persistent_state) {

            // Update transient state
            if (val) this.container.attr      ('checked', 'checked') ;
            else     this.container.removeAttr('checked') ;

            // Update persistent state if requested and if the one
            // is available.
            if (this._config.config_handler && update_persistent_state)
                this._config.config_handler.save(val ? 1 : 0) ;
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

                var val = _that.container.attr('checked') ? 1 : 0 ;
                
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
    Class.define_class(Checkbox, Widget.Widget, {}, {}) ;

    return Checkbox ;
}) ;