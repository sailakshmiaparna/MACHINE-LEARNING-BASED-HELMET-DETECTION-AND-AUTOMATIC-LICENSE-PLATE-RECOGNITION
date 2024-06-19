function processVideo() {
    const videoInput = document.getElementById('videoInput');
    const videoPlayer = document.getElementById('videoPlayer');

    const file = videoInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const videoUrl = event.target.result;
            videoPlayer.src = videoUrl;

            // Send video file to backend for processing
            // Implement AJAX request or fetch API to send the file data to your backend
        };
        reader.readAsDataURL(file);
    } else {
        alert('Please select a video file.');
    }
}
