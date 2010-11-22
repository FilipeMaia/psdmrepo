/* 
 * Application menu bar generator.
 */
var oMenuBar = null;

function menubar_create( element, menubar_data ) {

    if( null != oMenuBar ) return;

    var i;

    /* Initialize the markup for the menu.
     */
    var markup = '<div class="bd"><ul class="first-of-type">';
    for( i = 0; i < menubar_data.length; i++ ) {
        var data = menubar_data[i];
        if( i == 0 ) markup += '<li class="yuimenubaritem first-of-type">';
        else         markup += '<li class="yuimenubaritem">';
        markup += '<a class="yuimenubaritemlabel" href="'+data['href']+'"';
        var title_style = data['title_style'];
        if( title_style != null ) markup += ' style="'+title_style+'"';
        markup += '>'+data['title']+'</a></li>';
    }
    markup += '</ul></div>';

    document.getElementById( element ).innerHTML = markup;

    /* Generate the dynamic menu.
     */
    var aSubmenuData = new Array();
    for( i = 0; i < menubar_data.length; i++ ) {
        var m = menubar_data[i];
        if( m.id == null ) aSubmenuData.push( {} );
        else               aSubmenuData.push( { id: m.id, itemdata: m.itemdata } );
    }

    var ua = YAHOO.env.ua,
        oAnim;  // Animation instance

    oMenuBar = new YAHOO.widget.MenuBar (
        element,
        {   //autosubmenudisplay: true,
            hidedelay: 750,
            lazyload: true
        }
    );
    function onSubmenuBeforeShow(p_sType, p_sArgs) {

        var oBody,
            oElement,
            oShadow,
            oUL;

        if (this.parent) {

            oElement = this.element;

            oShadow = oElement.lastChild;
            oShadow.style.height = "0px";

            if (oAnim && oAnim.isAnimated()) {
                oAnim.stop();
                oAnim = null;
            }
            oBody = this.body;

            //  Check if the menu is a submenu of a submenu.
            if (this.parent &&
                !(this.parent instanceof YAHOO.widget.MenuBarItem)) {

                if (ua.gecko || ua.opera) {
                    oBody.style.width = oBody.clientWidth + "px";
                }
                if (ua.ie == 7) {
                    oElement.style.width = oElement.clientWidth + "px";
                }
            }
            oBody.style.overflow = "hidden";

            oUL = oBody.getElementsByTagName("ul")[0];
            oUL.style.marginTop = ("-" + oUL.offsetHeight + "px");
        }
    }
    function onTween(p_sType, p_aArgs, p_oShadow) {

        if (this.cfg.getProperty("iframe")) {
            this.syncIframe();
        }
        if (p_oShadow) {
            p_oShadow.style.height = this.element.offsetHeight + "px";
        }
    }
    function onAnimationComplete(p_sType, p_aArgs, p_oShadow) {

        var oBody = this.body,
            oUL = oBody.getElementsByTagName("ul")[0];

        if (p_oShadow) {
            p_oShadow.style.height = this.element.offsetHeight + "px";
        }
        oUL.style.marginTop = "";
        oBody.style.overflow = "";

        //  Check if the menu is a submenu of a submenu.

        if (this.parent &&
            !(this.parent instanceof YAHOO.widget.MenuBarItem)) {

            // Clear widths set by the "beforeshow" event handler

            if (ua.gecko || ua.opera) {
                oBody.style.width = "";
            }
            if (ua.ie == 7) {
                this.element.style.width = "";
            }
        }
    }
    function onSubmenuShow(p_sType, p_sArgs) {

        var oElement,
            oShadow,
            oUL;

        if (this.parent) {

            oElement = this.element;
            oShadow = oElement.lastChild;
            oUL = this.body.getElementsByTagName("ul")[0];

            oAnim = new YAHOO.util.Anim(oUL,
                { marginTop: { to: 0 } },
                .5, YAHOO.util.Easing.easeOut);

            oAnim.onStart.subscribe(function () {
                oShadow.style.height = "100%";
            });
            oAnim.animate();

            if (YAHOO.env.ua.ie) {
                oShadow.style.height = oElement.offsetHeight + "px";
                oAnim.onTween.subscribe(onTween, oShadow, this);
            }
            oAnim.onComplete.subscribe(onAnimationComplete, oShadow, this);
        }
    }
    oMenuBar.subscribe("beforeRender", function () {

        var nSubmenus = aSubmenuData.length,
            i;

        if (this.getRoot() == this) {
            for (i = 0; i < nSubmenus; i++) {
                this.getItem(i).cfg.setProperty("submenu",  aSubmenuData[i]);
                this.getItem(i).cfg.setProperty("disabled", aSubmenuData[i].disabled);
            }
        }
    });
    oMenuBar.subscribe("beforeShow", onSubmenuBeforeShow);
    oMenuBar.subscribe("show", onSubmenuShow);
    oMenuBar.render();
    for( i = 0; i < menubar_data.length; i++ ) {
        m = menubar_data[i];
        oMenuBar.getItem(i).cfg.setProperty( "disabled", m.disabled );
    }
}

function menubar_disable( idx ) {
    if( oMenuBar != null )
        oMenuBar.getItem(idx).cfg.setProperty("disabled", true );
}
function menubar_enable( idx ) {
    if( oMenuBar != null )
        oMenuBar.getItem(idx).cfg.setProperty("disabled", false );
}
