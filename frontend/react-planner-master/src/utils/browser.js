

export function browserDownload(json) {
  let fileOutputLink = document.createElement('a');

  let filename = 'output' + Date.now() + '.json';
  filename = window.prompt('Insert output filename', filename);
  if (!filename) return;

  let output = JSON.stringify(json);
  let data = new Blob([output], {type: 'text/plain'});
  let url = window.URL.createObjectURL(data);
  fileOutputLink.setAttribute('download', filename);
  fileOutputLink.href = url;
  fileOutputLink.style.display = 'none';
  document.body.appendChild(fileOutputLink);
  fileOutputLink.click();
  document.body.removeChild(fileOutputLink);
}

export function browserUpload() {
  return new Promise(function (resolve, reject) {

    let fileInput = document.createElement('input');
    fileInput.type = 'file';

    fileInput.addEventListener('change', function (event) {
      let file = event.target.files[0];
      let reader = new FileReader();
      reader.addEventListener('load', (fileEvent) => {
        let loadedData = fileEvent.target.result;
        resolve(loadedData);
      });
      reader.readAsText(file);
    });

    fileInput.click();
  });
}

export function browserImageUpload() {
  return new Promise(function (resolve, reject) {

    let fileInput = document.createElement('input');
    fileInput.type = 'file';

    fileInput.addEventListener('change', function (event) {
      let file = event.target.files[0];
      let reader = new FileReader();
      reader.addEventListener('load', (fileEvent) => {
        let loadedData = fileEvent.target.result;
        resolve(loadedData);
      });
      reader.readAsDataURL(file);
    });

    fileInput.click();
  });
}

export function sendSDPostRequest(image){
    const url = '/api/sdapi/v1/img2img';
    const senddata = {
        prompt: "<lora:floorplan:1>,a floorplan of an exhibition hall, sd_test",
        init_images:[image],
        Steps:20,
        Seed:-1,
        sampler_name:"Euler a",
        alwayson_scripts: {
          controlnet: {
            args: [
                {
                    module: "all",
                }
            ]
          }
        }
      };

    // 将数据转换为JSON字符串
    const body = JSON.stringify(senddata);

    // 设置请求的头部
    const headers = {
      'Content-Type': 'application/json'
    };
    return new Promise(function (resolve, reject) {
      // 发送POST请求
      fetch(url, {
        method: 'POST',
        headers: headers,
        body: body,
      })
      .then(response => response.text())
      .then(json => {
        // 处理返回的数据
        resolve(json)
        console.log(json)
      })
      .catch((error) => {
        // 处理错误
        console.error('Error:', error);
      });
    });
  }

export function sendRecognizationPostRequest(image){
  const data = {
    image:image
  }
  const url = "/flask2/imagechange"
  const body = JSON.stringify(data);
  // 设置请求的头部
    const headers = {
      'Content-Type': 'application/json'
    };
  fetch(url, {
        method: 'POST',
        headers: headers,
        body: body,
      })
      .then(response => response.text())
      .then(json => {
        // 处理返回的数据
        resolve(json)
        console.log(json)
      })
      .catch((error) => {
        // 处理错误
        console.error('Error:', error);
      });
}
