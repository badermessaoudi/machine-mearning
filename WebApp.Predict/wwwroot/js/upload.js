//const serviceUrl = 'https://localhost:44380/api/ImageClassification/classifyImage';

//const serviceUrl = 'http://localhost:5000/api/ImageClassification/classifyImage';

const serviceUrl = 'api/ImageClassification/classifyImage';
const form = document.querySelector('form');

form.addEventListener('submit', e => {
    e.preventDefault();

    const files = document.querySelector('[type=file]').files;
    const formData = new FormData();

    formData.append('imageFile', files[0]);


    fetch(serviceUrl, {
        method: 'POST',
        body: formData
    }).then((resp) => resp.json())
      .then(function (response) {
          console.info('Response', response);
          console.log('Response', response);

          var output = document.getElementById('pokemon-image');
          output.src = URL.createObjectURL(files[0]);
          output.onload = function () {
              URL.revokeObjectURL(output.src) // free memory
          }
          document.getElementById('image-id').innerHTML = response.imageId;
          document.getElementById('name-screen').innerHTML = response.predictedLabel;
          document.getElementById('probability').innerHTML = (response.probability * 100).toFixed(3) + "%";
          document.getElementById('execution-time').innerHTML = response.predictionExecutionTime + " mlSecs";

          return response;
        });


});