/* 
 * Constructors for dialogs.
 */

function post_message( id, title, text ) {

    document.getElementById( id ).innerHTML =
        '<div class="hd">'+title+'</div>'+
        '<div class="bd">'+
        '  <center><p>'+text+'</p></center>'+
        '</div>';

    var handleOk = function() { this.hide(); };

    var dialog1 =
        new YAHOO.widget.Dialog (
            id,
			{   width : "480px",
                fixedcenter : true,
				visible : true,
                close: true,
                modal:true,
				constraintoviewport : true,
				buttons : [
                    { text:"Ok", handler: handleOk, isDefault:true }
                ]
			}
        );
    dialog1.render();
}

function post_warning( id, text ) {
    post_message (
        id,
        '<span style="color:red; font-size:16px;">Warning</span>',
        text );
}

function post_info( id, text ) {
    post_message (
        id,
        '<span style="color:green; font-size:16px;">Info</span>',
        text );
}

function ask_yesno( id, title, text, onYes, onNo ) {

    document.getElementById( id ).innerHTML =
        '<div class="hd">'+title+'</div>'+
        '<div class="bd">'+
        '  <center><p>'+text+'</p></center>'+
        '</div>';

    var handleYes = function() {
        this.hide();
        onYes();
    };
    var handleNo = function() {
        this.hide();
        onNo();
    };
    var dialog1 = new YAHOO.widget.Dialog (
        id,
        {   width : "480px",
            fixedcenter : true,
            visible : true,
            close: true,
            modal:true,
            constraintoviewport : true,
            buttons : [
                { text:"Yes", handler: handleYes },
                { text:"No",  handler: handleNo, isDefault:true }
            ]
        }
    );
    dialog1.render();
}

function ask_yesno_confirmation( id, text, onYes, onNo ) {
    ask_yesno (
        id,
        '<span style="color:red; font-size:16px;">Confirmation Request</span>',
        text,
        onYes, onNo
    );
}
