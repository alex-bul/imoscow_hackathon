let DOWNLOAD_LINK = null;
let DOWNLOAD_FILE = null;

document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault();
    let videoInput = document.getElementById('videoInput');
    let video = videoInput.files[0];

    let formData = new FormData();
    formData.append('video', video);

    document.getElementById('loader').classList.remove('d-none');

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(function (data) {
        let resultTable = document.getElementById('resultTable');
        let tbody = resultTable.getElementsByTagName('tbody')[0];
        let filename = data.filename;
        console.log(data)
        if(data.objects.length){
            for(let item of data.objects){
                console.log(item);
                let newRow = tbody.insertRow();
                let objectNameCell = newRow.insertCell();
                let countCell = newRow.insertCell();
                let timeCell = newRow.insertCell();

                objectNameCell.innerText = item.object_name;
                countCell.innerText = item.count;
                timeCell.innerText = item.time;
            }
            document.getElementById('loader').classList.add('d-none');
            document.getElementById('no_violation').classList.add('d-none');
            document.getElementById('resultTable').classList.remove('d-none');
            document.getElementById('download_finish').classList.remove('d-none');

            DOWNLOAD_FILE = filename
            DOWNLOAD_LINK = "/download?filename=" + filename
        }
        else {
            document.getElementById('no_violation').classList.remove('d-none');
        }

    })
    .catch(function (error) {
        console.error('Error:', error);
        document.getElementById('loader').classList.add('d-none');
    });

    videoInput.value = '';
});

function downloadFile() {
    var link = document.createElement('a');
    link.href = DOWNLOAD_LINK;
    link.download = DOWNLOAD_FILE;
    link.click();
}
