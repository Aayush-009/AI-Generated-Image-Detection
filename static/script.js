const imageInput = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");

const previewContainer = document.getElementById("previewContainer");
const preview = document.getElementById("preview");
const fileInfo = document.getElementById("fileInfo");

const detectBtn = document.getElementById("detectBtn");
const resetBtn = document.getElementById("resetBtn");
const anotherBtn = document.getElementById("anotherBtn");

const emptyResult = document.getElementById("emptyResult");
const loading = document.getElementById("loading");
const result = document.getElementById("result");

const resultBadge = document.getElementById("resultBadge");
const predictionText = document.getElementById("predictionText");
const resultDescription = document.getElementById("resultDescription");

const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");

let selectedFile = null;


// ========================================
// CHOOSE IMAGE
// ========================================

imageInput.addEventListener("change", function () {

    if (this.files.length > 0) {
        handleFile(this.files[0]);
    }

});


// ========================================
// DRAG OVER
// ========================================

dropZone.addEventListener("dragover", function (event) {

    event.preventDefault();

    dropZone.classList.add("dragover");

});


// ========================================
// DRAG LEAVE
// ========================================

dropZone.addEventListener("dragleave", function () {

    dropZone.classList.remove("dragover");

});


// ========================================
// DROP IMAGE
// ========================================

dropZone.addEventListener("drop", function (event) {

    event.preventDefault();

    dropZone.classList.remove("dragover");

    if (event.dataTransfer.files.length > 0) {
        handleFile(event.dataTransfer.files[0]);
    }

});


// ========================================
// HANDLE FILE
// ========================================

function handleFile(file) {

    const allowedTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/heic",
        "image/heif"
    ];

    const fileExtension =
        file.name.toLowerCase().split(".").pop();

    const allowedExtensions = [
        "jpg",
        "jpeg",
        "png",
        "heic",
        "heif"
    ];

    if (
        !allowedTypes.includes(file.type) &&
        !allowedExtensions.includes(fileExtension)
    ) {

        alert("Please select a JPG, JPEG, PNG, HEIC, or HEIF image.");

        return;
    }


    // 10 MB limit

    if (file.size > 10 * 1024 * 1024) {

        alert("Image must be smaller than 10 MB.");

        return;
    }


    selectedFile = file;


    // Image preview

    preview.src = URL.createObjectURL(file);

    previewContainer.style.display = "block";


    // File information

    const sizeMB =
        (file.size / (1024 * 1024)).toFixed(2);

    fileInfo.textContent =
        file.name + " • " + sizeMB + " MB";


    // Show ready state

    emptyResult.style.display = "flex";

    loading.style.display = "none";

    result.style.display = "none";

}


// ========================================
// ANALYZE IMAGE
// ========================================

detectBtn.addEventListener(
    "click",
    async function () {

        if (!selectedFile) {

            alert("Please choose an image first.");

            return;
        }


        const formData = new FormData();

        formData.append("image", selectedFile);


        // Show loading

        emptyResult.style.display = "none";

        result.style.display = "none";

        loading.style.display = "flex";

        detectBtn.disabled = true;


        try {

            const response = await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


            // Get server response as text

            const text = await response.text();


            let data;


            // Safely convert response to JSON

            try {

                data = JSON.parse(text);

            }

            catch {

                throw new Error(
                    "Server did not return a valid response. Please try again."
                );

            }


            // Check server status

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Prediction failed."
                );

            }


            // Hide loading

            loading.style.display = "none";

            detectBtn.disabled = false;


            // Show result

            showResult(
                data.prediction,
                data.confidence
            );

        }


        catch (error) {

            loading.style.display = "none";

            detectBtn.disabled = false;

            emptyResult.style.display = "flex";


            alert(
                "Error: " + error.message
            );


            console.error(error);

        }

    }
);


// ========================================
// SHOW RESULT
// ========================================

function showResult(
    prediction,
    confidence
) {

    result.style.display = "block";


    confidenceText.textContent =
        confidence + "%";


    confidenceBar.style.width =
        confidence + "%";


    const predictionUpper =
        prediction.toUpperCase();


    if (predictionUpper === "FAKE") {

        resultBadge.textContent =
            "AI GENERATED";


        predictionText.textContent =
            "AI-Generated Image";


        resultDescription.textContent =
            "The model detected patterns associated with an AI-generated image.";

    }

    else {

        resultBadge.textContent =
            "REAL IMAGE";


        predictionText.textContent =
            "Real Image";


        resultDescription.textContent =
            "The model detected patterns associated with a real image.";

    }

}


// ========================================
// RESET
// ========================================

function resetDetector() {

    selectedFile = null;


    imageInput.value = "";


    preview.src = "";


    previewContainer.style.display =
        "none";


    fileInfo.textContent =
        "";


    result.style.display =
        "none";


    loading.style.display =
        "none";


    emptyResult.style.display =
        "flex";


    confidenceBar.style.width =
        "0%";


    confidenceText.textContent =
        "0%";


    detectBtn.disabled =
        false;

}


// ========================================
// RESET BUTTON
// ========================================

resetBtn.addEventListener(
    "click",
    resetDetector
);


// ========================================
// ANALYZE ANOTHER IMAGE
// ========================================

anotherBtn.addEventListener(
    "click",
    resetDetector
);
