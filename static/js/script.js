const dropArea = document.getElementById("drop_area")
const inputFile = document.getElementById("input-file")
const form = document.getElementById("form_drop_area")

document.getElementById('input-file').addEventListener('change', function(e) {
    if (e.target.files[0]) {
        uploadDocument();
    }
  });

function uploadDocument() {
    document.getElementById("input_button").click();
    let fileLink = URL.createObjectURL(inputFile.files[0]);
}

dropArea.addEventListener("dragover", function(e){
    e.preventDefault();
});
dropArea.addEventListener("drop", function(e){
    e.preventDefault();
    inputFile.files = e.dataTransfer.files;
    uploadDocument();
});