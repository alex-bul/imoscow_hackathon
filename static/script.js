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
        let filename = data.filename;
        let resultBlock = document.getElementById('result_block');
        resultBlock.innerHTML = '';
        console.log(data)
        document.getElementById('loader').classList.add('d-none');
        if(data.objects.length){
            for(let item of data.objects){
                let cardHTML = `
                    <div class="mt-3 col-10">
                        <div class="card mb-3 d-flex flex-row" style="height: 400px">
                            <div class="card-body">
                                <h4 class="card-title">${item.object_name}</h4>
                                <p class="card-text">Время: ${item.time} сек.</p>
                                <p class="card-text">Количество: ${item.count}</p>
                                <p class="card-text">Имя кадра: ${item.frame_name}</p>
                                <a href="${item.frame_url}"
                                download class="btn btn-primary">Скачать фото</a>
                            </div>
                            <img src="${item.frame_url}" class="img-fluid rounded m-2"
                                 style="height: auto; object-fit: cover; display: block">
                        </div>
                    </div>
                  `;

                resultBlock.innerHTML += cardHTML;
            }
            document.getElementById('no_violation').classList.add('d-none');
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
