let selectedDisease = "";
let selectedInputs = [];

// Disease features with descriptions
const diseaseFeatures = {
  mastitis: [
    { 
      name: "temperature", 
      label: "Temperature (°C)", 
      type: "number", 
      min: 37.5, 
      max: 41.0, 
      step: 0.1 
    },
    { 
      name: "hardness", 
      label: "Udder Hardness", 
      type: "select",
      options: [
        {value: 0, text: "0: Very soft, normal udder"},
        {value: 1, text: "1: Slight firmness, early warning"},
        {value: 2, text: "2: Mild firmness, possible minor infection"},
        {value: 3, text: "3: Moderate firmness, notable inflammation"},
        {value: 4, text: "4: Severe firmness, advanced infection"},
        {value: 5, text: "5: Extremely hard, critical condition"}
      ]
    },
    { 
      name: "milk_color", 
      label: "Milk Color", 
      type: "select",
      options: [
        {value: 0, text: "0: Normal milk color, no infection"},
        {value: 1, text: "1: Slight off-white, little concern"},
        {value: 2, text: "2: Cloudy, early stages of infection"},
        {value: 3, text: "3: Some discoloration, infection present"},
        {value: 4, text: "4: Signs of blood or pus, late stage"},
        {value: 5, text: "5: Severely abnormal, critical condition"}
      ]
    },
    { 
      name: "pain", 
      label: "Pain Level", 
      type: "select",
      options: [
        {value: 0, text: "0: No pain, healthy"},
        {value: 1, text: "1: Minor pain, slight discomfort"},
        {value: 2, text: "2: Light discomfort, early infection"},
        {value: 3, text: "3: Moderate pain, needs attention"},
        {value: 4, text: "4: Severe pain, chronic"},
        {value: 5, text: "5: Extreme pain, emergency care"}
      ]
    },
    { 
      name: "milk_yield", 
      label: "Milk Yield Reduction", 
      type: "select",
      options: [
        {value: 0, text: "0: Yields normal, udder healthy"},
        {value: 1, text: "1: Slightly reduced yield"},
        {value: 2, text: "2: Noticeably reduced yield"},
        {value: 3, text: "3: Significantly reduced yield"},
        {value: 4, text: "4: Severely reduced yield"},
        {value: 5, text: "5: Almost no milk production"}
      ]
    }
  ],
  foot: [
    { 
      name: "temperature", 
      label: "Temperature (°C)", 
      type: "number", 
      min: 37.5, 
      max: 41.0, 
      step: 0.1 
    },
    { 
      name: "milk_production", 
      label: "Milk Production", 
      type: "select",
      options: [
        {value: 0, text: "0: Normal production"},
        {value: 1, text: "1: Slightly reduced"},
        {value: 2, text: "2: Reduced"},
        {value: 3, text: "3: Significantly reduced"},
        {value: 4, text: "4: Severely reduced"},
        {value: 5, text: "5: Almost no production"}
      ]
    },
    { 
      name: "lethargy", 
      label: "Lethargy", 
      type: "select",
      options: [
        {value: 0, text: "0: Active"},
        {value: 1, text: "1: Sluggish or no activity"}
      ]
    },
    { 
      name: "difficulty_walking", 
      label: "Difficulty Walking", 
      type: "select",
      options: [
        {value: 0, text: "0: Walking normal"},
        {value: 1, text: "1: Limping or can't walk"}
      ]
    },
    { 
      name: "foot_blister", 
      label: "Foot Blister", 
      type: "select",
      options: [
        {value: 0, text: "0: No blister"},
        {value: 1, text: "1: Very small blister"},
        {value: 2, text: "2: Small, red blister"},
        {value: 3, text: "3: Medium blister, slight discomfort"},
        {value: 4, text: "4: Large/multiple blisters"},
        {value: 5, text: "5: Severe blistering, ruptured/bleeding"}
      ]
    },
    { 
      name: "foot_swelling", 
      label: "Foot Swelling", 
      type: "select",
      options: [
        {value: 0, text: "0: No swelling"},
        {value: 1, text: "1: Slight puffiness"},
        {value: 2, text: "2: Slight swelling"},
        {value: 3, text: "3: Moderate swelling"},
        {value: 4, text: "4: Severe swelling, limping"},
        {value: 5, text: "5: Very swollen, can't walk"}
      ]
    },
    { 
      name: "hoof_detachment", 
      label: "Hoof Detachment", 
      type: "select",
      options: [
        {value: 0, text: "0: Hoof normal"},
        {value: 1, text: "1: Hairline cracks"},
        {value: 2, text: "2: Slight loosening"},
        {value: 3, text: "3: Slightly partially detached"},
        {value: 4, text: "4: Majorly detached"},
        {value: 5, text: "5: Entirely detached"}
      ]
    }
  ],
  mouth: [
    { 
      name: "temperature", 
      label: "Temperature (°C)", 
      type: "number", 
      min: 37.5, 
      max: 41.0, 
      step: 0.1 
    },
    { 
      name: "milk_production", 
      label: "Milk Production", 
      type: "select",
      options: [
        {value: 0, text: "0: Normal production"},
        {value: 1, text: "1: Slightly reduced"},
        {value: 2, text: "2: Reduced"},
        {value: 3, text: "3: Significantly reduced"},
        {value: 4, text: "4: Severely reduced"},
        {value: 5, text: "5: Almost no production"}
      ]
    },
    { 
      name: "lethargy", 
      label: "Lethargy", 
      type: "select",
      options: [
        {value: 0, text: "0: Active"},
        {value: 1, text: "1: Sluggish or no activity"}
      ]
    },
    { 
      name: "mouth_ulcers", 
      label: "Mouth Ulcers", 
      type: "select",
      options: [
        {value: 0, text: "0: None"},
        {value: 1, text: "1: One small ulcer"},
        {value: 2, text: "2: Few small ulcers"},
        {value: 3, text: "3: Moderate-size ulcers"},
        {value: 4, text: "4: Large or several ulcers"},
        {value: 5, text: "5: Severe ulcers preventing eating"}
      ]
    },
    { 
      name: "mouth_blister", 
      label: "Mouth Blisters", 
      type: "select",
      options: [
        {value: 0, text: "0: None"},
        {value: 1, text: "1: Tiny blisters"},
        {value: 2, text: "2: Small fluid blisters"},
        {value: 3, text: "3: Medium blisters"},
        {value: 4, text: "4: Large or multiple blisters"},
        {value: 5, text: "5: Severe blistering (ruptured)"}
      ]
    },
    { 
      name: "salivation", 
      label: "Salivation", 
      type: "select",
      options: [
        {value: 0, text: "0: Normal"},
        {value: 1, text: "1: Excessive drooling"}
      ]
    },
    { 
      name: "nasal_discharge", 
      label: "Nasal Discharge", 
      type: "select",
      options: [
        {value: 0, text: "0: No discharge"},
        {value: 1, text: "1: Runny/mucous discharge"}
      ]
    }
  ]
};

// Navigation functions
function goToStep2() {
  selectedDisease = document.getElementById("disease").value;
  if (!selectedDisease) {
    alert("Please select a disease.");
    return;
  }

  document.getElementById("step1").classList.add("hidden");
  document.getElementById("step2").classList.remove("hidden");

  if (selectedDisease === "mastitis") {
    document.getElementById("mastitis-options").classList.remove("hidden");
    document.getElementById("fmd-options").classList.add("hidden");
  } else if (selectedDisease === "fmd") {
    document.getElementById("fmd-options").classList.remove("hidden");
    document.getElementById("mastitis-options").classList.add("hidden");
  }
}

function goBackToStep1() {
  document.getElementById("step2").classList.add("hidden");
  document.getElementById("step1").classList.remove("hidden");
}

function goToStep3() {
  // Get selected inputs
  selectedInputs = [];
  
  if (selectedDisease === "mastitis") {
    document.querySelectorAll('input[name="mastitisInput"]:checked').forEach(checkbox => {
      selectedInputs.push(checkbox.value);
    });
  } else if (selectedDisease === "fmd") {
    document.querySelectorAll('input[name="fmdTextInput"]:checked').forEach(checkbox => {
      selectedInputs.push(`text_${checkbox.value}`);
    });
    document.querySelectorAll('input[name="fmdImageInput"]:checked').forEach(checkbox => {
      selectedInputs.push(`image_${checkbox.value}`);
    });
  }

  if (selectedInputs.length === 0) {
    alert("Please select at least one input type.");
    return;
  }

  document.getElementById("step2").classList.add("hidden");
  document.getElementById("step3").classList.remove("hidden");
  renderInputFields();
}

function goBackToStep2() {
  document.getElementById("step3").classList.add("hidden");
  document.getElementById("step2").classList.remove("hidden");
}

function renderInputFields() {
  const textContainer = document.getElementById("textFields");
  const imgContainer = document.getElementById("imageFields");
  textContainer.innerHTML = "";
  imgContainer.innerHTML = "";

  const createInputField = (feature) => {
    const div = document.createElement("div");
    const label = document.createElement("label");
    label.textContent = feature.label;
    div.appendChild(label);

    if (feature.type === "number") {
      const input = document.createElement("input");
      input.type = "number";
      input.name = feature.name;
      input.min = feature.min;
      input.max = feature.max;
      input.step = feature.step || 1;
      input.required = true;
      div.appendChild(input);
    } else if (feature.type === "select") {
      const select = document.createElement("select");
  select.name = feature.name;
  select.required = true;

  // Add the placeholder option
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "--Select--";
  placeholder.disabled = true;
  placeholder.selected = true;
  select.appendChild(placeholder);

  // Add actual feature options
  feature.options.forEach(option => {
    const opt = document.createElement("option");
    opt.value = option.value;
    opt.textContent = option.text;
    select.appendChild(opt);
      });
      
      div.appendChild(select);
    }
    
    return div;
  };

  const createTextInputSection = (label, features, prefix) => {
    const section = document.createElement("div");
    section.className = "input-section";
    section.innerHTML = `<h3>${label}</h3>`;
    
    features.forEach(feature => {
      const field = createInputField(feature);
      if (prefix) {
        field.querySelector("input, select").name = `${prefix}_${feature.name}`;
      }
      section.appendChild(field);
    });
    
    return section;
  };

  const createFileInput = (label, name) => {
    const div = document.createElement("div");
    div.className = "input-section";
    div.innerHTML = `
      <h3>${label}</h3>
      <input type="file" name="${name}" accept="image/*">
    `;
    return div;
  };

  if (selectedDisease === "mastitis") {
    if (selectedInputs.includes("text")) {
      textContainer.appendChild(
        createTextInputSection("Mastitis Symptoms", diseaseFeatures.mastitis, "mastitis")
      );
    }
    if (selectedInputs.includes("image")) {
      imgContainer.appendChild(createFileInput("Udder Image", "udderImage"));
    }
  } else if (selectedDisease === "fmd") {
    selectedInputs.forEach(input => {
      const [type, part] = input.split("_");
      if (type === "text") {
        textContainer.appendChild(
          createTextInputSection(
            `${part.charAt(0).toUpperCase() + part.slice(1)} Symptoms`, 
            diseaseFeatures[part], 
            `${part}_text`
          )
        );
      } else if (type === "image") {
        imgContainer.appendChild(
          createFileInput(
            `${part.charAt(0).toUpperCase() + part.slice(1)} Image`, 
            `${part}Image`
          )
        );
      }
    });
  }
}

function displayResults(data) {
  const resultContainer = document.getElementById("resultDisplay");
  resultContainer.innerHTML = "";
  
  if (selectedDisease === "mastitis") {
    const textResult = data.text_result;
    const imageResult = data.image_result;
    let combinedResult = null;
    
    // Display text results if available
    if (textResult) {
      const isInfected = textResult === 'Mastitis Detected';
      const confidence = (data.text_confidence * 100).toFixed(1);
      
      const textDiv = document.createElement("div");
      textDiv.className = "result-item";
      textDiv.innerHTML = `
        <div class="result-title">Text Analysis:</div>
        <div class="result-value ${isInfected ? 'danger' : 'success'}">
          ${textResult} (${confidence}% confidence)
        </div>
      `;
      resultContainer.appendChild(textDiv);
    }
    
    // Display image results if available
    if (imageResult) {
      const isInfected = imageResult === 'Infected';
      const confidence = (data.image_confidence * 100).toFixed(1);
      
      const imgDiv = document.createElement("div");
      imgDiv.className = "result-item";
      imgDiv.innerHTML = `
        <div class="result-title">Image Analysis:</div>
        <div class="result-value ${isInfected ? 'danger' : 'success'}">
          ${imageResult} (${confidence}% confidence)
        </div>
      `;
      resultContainer.appendChild(imgDiv);
    }
    
    // Calculate and display combined result if both inputs were provided
    if (textResult && imageResult) {
      const textWeight = data.text_confidence;
      const imageWeight = data.image_confidence;
      const combinedProb = (textWeight + imageWeight) / 2;
      const isInfected = combinedProb > 0.5;
      const confidence = (combinedProb * 100).toFixed(1);
      
      combinedResult = isInfected ? 'Mastitis Detected' : 'No Mastitis';
      
      const combinedDiv = document.createElement("div");
      combinedDiv.className = "final-result";
      combinedDiv.innerHTML = `
        <div class="result-title">Final Diagnosis (Combined):</div>
        <div class="result-value ${isInfected ? 'danger' : 'success'}">
          ${combinedResult} (${confidence}% confidence)
        </div>
        <p>Combined analysis of both text and image inputs</p>
      `;
      resultContainer.appendChild(combinedDiv);
    }
    
    // Display errors if any
    if (data.text_error) {
      const errorDiv = document.createElement("div");
      errorDiv.className = "error-message";
      errorDiv.textContent = `Text Analysis Error: ${data.text_error}`;
      resultContainer.appendChild(errorDiv);
    }
    
    if (data.image_error) {
      const errorDiv = document.createElement("div");
      errorDiv.className = "error-message";
      errorDiv.textContent = `Image Analysis Error: ${data.image_error}`;
      resultContainer.appendChild(errorDiv);
    }
  } 
  else if (selectedDisease === "fmd") {
    // Process FMD results
    const results = [];
    
    // Foot text results
    if (data.foot_text_result) {
      const isInfected = data.foot_text_result === 'Infected';
      const confidence = (data.foot_text_confidence * 100).toFixed(1);
      
      results.push({
        type: 'foot_text',
        title: 'Foot Symptoms Analysis',
        result: data.foot_text_result,
        confidence: confidence,
        isInfected: isInfected
      });
    }
    
    // Mouth text results
    if (data.mouth_text_result) {
      const isInfected = data.mouth_text_result === 'Infected';
      const confidence = (data.mouth_text_confidence * 100).toFixed(1);
      
      results.push({
        type: 'mouth_text',
        title: 'Mouth Symptoms Analysis',
        result: data.mouth_text_result,
        confidence: confidence,
        isInfected: isInfected
      });
    }
    
    // Foot image results
    if (data.foot_image_result) {
      const isInfected = data.foot_image_result === 'Infected';
      const confidence = (data.foot_image_confidence * 100).toFixed(1);
      
      results.push({
        type: 'foot_image',
        title: 'Foot Image Analysis',
        result: data.foot_image_result,
        confidence: confidence,
        isInfected: isInfected
      });
    }
    
    // Mouth image results
    if (data.mouth_image_result) {
      const isInfected = data.mouth_image_result === 'Infected';
      const confidence = (data.mouth_image_confidence * 100).toFixed(1);
      
      results.push({
        type: 'mouth_image',
        title: 'Mouth Image Analysis',
        result: data.mouth_image_result,
        confidence: confidence,
        isInfected: isInfected
      });
    }
    
    // Display all individual results
    results.forEach(item => {
      const div = document.createElement("div");
      div.className = "result-item";
      div.innerHTML = `
        <div class="result-title">${item.title}:</div>
        <div class="result-value ${item.isInfected ? 'danger' : 'success'}">
          ${item.result} (${item.confidence}% confidence)
        </div>
      `;
      resultContainer.appendChild(div);
    });
    
    // Display combined result if available
    if (data.combined_result) {
      const isInfected = data.combined_result === 'Infected';
      const confidence = (data.combined_confidence * 100).toFixed(1);
      
      const combinedDiv = document.createElement("div");
      combinedDiv.className = "final-result";
      combinedDiv.innerHTML = `
        <div class="result-title">Final FMD Diagnosis:</div>
        <div class="result-value ${isInfected ? 'danger' : 'success'}">
          ${data.combined_result} (${confidence}% confidence)
        </div>
        <p>Combined analysis of all provided inputs</p>
      `;
      resultContainer.appendChild(combinedDiv);
    }
  }
  
  // Show the results step
  document.getElementById("step3").classList.add("hidden");
  document.getElementById("step4").classList.remove("hidden");
}

function goBackToStep3() {
  document.getElementById("step4").classList.add("hidden");
  document.getElementById("step3").classList.remove("hidden");
}

function restartProcess() {
  document.getElementById("step4").classList.add("hidden");
  document.getElementById("step1").classList.remove("hidden");
  document.getElementById("inputForm").reset();
  selectedDisease = "";
  selectedInputs = [];
}

document.getElementById("inputForm").addEventListener("submit", function (e) {
  e.preventDefault();
  const form = e.target;
  const formData = new FormData(form);
  
  // Add selected input types to form data
  selectedInputs.forEach(input => {
    formData.append("inputTypes[]", input);
  });

  const url = selectedDisease === "mastitis"
    ? "http://localhost:5000/predict/mastitis"
    : "http://localhost:5000/predict/fmd";

  // Show loading state
  const predictBtn = form.querySelector('.predict-button');
  const originalText = predictBtn.textContent;
  predictBtn.textContent = "Processing...";
  predictBtn.disabled = true;

  fetch(url, {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => displayResults(data))
    .catch(err => {
      document.getElementById("resultDisplay").innerHTML = `
        <div class="error-message">Error: ${err.message}</div>
      `;
      document.getElementById("step3").classList.add("hidden");
      document.getElementById("step4").classList.remove("hidden");
    })
    .finally(() => {
      predictBtn.textContent = originalText;
      predictBtn.disabled = false;
    });
});