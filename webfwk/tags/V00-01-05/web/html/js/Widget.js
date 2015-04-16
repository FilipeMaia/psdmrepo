define ([
    'webfwk/Class'] ,

function (Class) {

    /**
     * The exception class used by Widgets
     * 
     * @param String message
     * @returns {WidgetError}
     */
    function WidgetError (message) {
        this.message = message ;
    }
    Class.define_class (WidgetError, Error, {}, {}) ;

    /**
     * The base class for widgets
     *
     * @returns {undefined}
     */
    function Widget () {
    }

    Class.define_class (Widget, null, {}, {

    /**
     * Display the widget ta the specified location
     * 
     * @param String or JQuery Object - a container where to render the widget
     */
    display : function (container) {
        switch (typeof container) { 
            case 'string' : this.container = $('#'+container) ; break ;
            case 'object' : this.container = $(container) ; break ;
            default :
                throw new WidgetError('Widget: the container parameter is mission or it has wrong type') ;
        }
        this.render() ;
    } ,

    /**
     * Render the widget. This method MUST be implemented by a subclasse
     */
    render : function() {
        throw new WidgetError('Widget: no rendering provided by the derived class') ;
    }

    }) ;

    return {
        WidgetError : WidgetError ,
        Widget : Widget
    } ;
}) ;