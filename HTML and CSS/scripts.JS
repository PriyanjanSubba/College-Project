

  function previewImages() {
    const previewContainer = document.getElementById("image-preview");
    previewContainer.innerHTML = '';

    const files = document.getElementById("photo-upload").files;
    for (let i = 0; i < files.length; i++) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(files[i]);
      img.style.width = "100px";
      img.style.marginRight = "10px";
      img.style.borderRadius = "8px";
      previewContainer.appendChild(img);
    }
  }

  function uploadPhotos() {
    const progressBar = document.getElementById("progress-bar");
    const progressStatus = document.getElementById("progress-status");
    const progressContainer = document.getElementById("progress-container");

    progressContainer.style.display = "block";
    
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      progressBar.value = progress;
      progressStatus.innerHTML = `${progress}%`;

      if (progress >= 100) {
        clearInterval(interval);
        alert("Upload completed!");
      }
    }, 500);
  }

