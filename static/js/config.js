$('#myForm').submit( function(event){
	event.preventDefault();
	$.ajax({
		url: '/config',
		type: 'POST',
		data: $('#myForm').serialize(),
		success: function(response){
			console.log('config sent! response = ' + response);
		},
		error: function(error){
			console.log('encountered an error = ' + error);
		}
	});
})