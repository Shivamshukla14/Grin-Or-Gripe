document.addEventListener("DOMContentLoaded", function() {
    let CurrentURL = window.location.pathname
    $('#workspace-result').hide();  //hides the targeted element
    if(CurrentURL.split("/")[1].includes("app")){
        if((CurrentURL.split("/")[2]) || (CurrentURL.split("/")[1].includes("app4"))){
            $('#workspace-result').show();
        }
        $('html, body').animate({
            scrollTop: $("#workspace").offset().top
        }, 100);
    } else {
        $('#workspace').hide()  //hides the targeted element
        $('#Workspace-nav').hide()  //hides the targeted element
    }
  });

function myFunction() {
    document.getElementById("tweet").value = "";
};
function ShowResult(){
    $('#workspace-result').show();  //hides the targeted element
}


$(document).ready(function() {
    $("#messageAreaDocuBot").on("submit", function(event) {
        $("div.spanner").addClass("show");
        const date = new Date();
        const hour = date.getHours();
        const minute = date.getMinutes();
        const str_time = hour+":"+minute;
        var rawText = $("#text").val();

        var userHtml = '<div class="d-flex justify-content-end mb-4"><div class="msg_cotainer_send">' + rawText + '<span class="msg_time_send">'+ str_time + '</span></div><div class="img_cont_msg"><img src="https://i.ibb.co/d5b84Xw/Untitled-design.png" class="rounded-circle user_img_msg"></div></div>';
        
        $("#text").val("");
        $("#messageFormeight").append(userHtml);

        $.ajax({
            data: {
                msg: rawText,	
            },
            type: "POST",
            url: "/app4/chat",
        }).done(function(data) {
            $("div.spanner").removeClass("show");
            var botHtml = '<div class="d-flex justify-content-start mb-4"><div class="img_cont_msg"><img src="../static/images/logo_head.png" class="rounded-circle user_img_msg"></div><div class="msg_cotainer">' + data + '<span class="msg_time">' + str_time + '</span></div></div>';
            $("#messageFormeight").append($.parseHTML(botHtml));
        });
        event.preventDefault();
    });
});

// Developer Section

$(document).ready(function() {

	/* About me slider */
	$('.about-me-slider').slick({
		slidesToShow: 1,
		prevArrow: '<span class="span-arrow slick-prev"><</span>',
		nextArrow: '<span class="span-arrow slick-next">></span>'
	});
	
});
