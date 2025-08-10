let selectedDisease = "";
let selectedInputs = [];
let currentResults = null;

// Navigation functions
function goToStep2() {
  selectedDisease = document.getElementById("disease").value;
  if (!selectedDisease) {
    alert(i18next.t('alerts.select_disease'));
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
    alert(i18next.t('alerts.select_input'));
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
    label.textContent = i18next.t(`labels.${feature.name}`) || feature.label;
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

      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = `--${i18next.t('labels.select')}--`;
      placeholder.disabled = true;
      placeholder.selected = true;
      select.appendChild(placeholder);

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
        createTextInputSection(
          i18next.t('sections.mastitis_symptoms'),
          window.translatedDiseaseFeatures.mastitis,
          "mastitis"
        )
      );
    }
    if (selectedInputs.includes("image")) {
      imgContainer.appendChild(
        createFileInput(
          i18next.t('labels.udder_image'),
          "udderImage"
        )
      );
    }
  } else if (selectedDisease === "fmd") {
    selectedInputs.forEach(input => {
      const [type, part] = input.split("_");
      if (type === "text") {
        textContainer.appendChild(
          createTextInputSection(
            i18next.t(`sections.${part}_symptoms`),
            window.translatedDiseaseFeatures[part],
            `${part}_text`
          )
        );
      } else if (type === "image") {
        imgContainer.appendChild(
          createFileInput(
            i18next.t(`labels.${part}_image`),
            `${part}Image`
          )
        );
      }
    });
  }
}

function safeGenerateSuggestionItems(translationKey) {
  try {
    const items = i18next.t(translationKey, { returnObjects: true });
    
    // Handle case where translation returns a string instead of array
    if (typeof items === 'string') {
      console.warn(`Expected array for ${translationKey}, got string. Splitting by newline.`);
      return items.split('\n').filter(item => item.trim() !== '');
    }
    
    if (!Array.isArray(items)) {
      console.error(`Translation for ${translationKey} is not an array:`, items);
      return [];
    }
    
    return items;
  } catch (error) {
    console.error(`Error processing ${translationKey}:`, error);
    return [];
  }
}

function generateMastitisSuggestions(confidence, isInfected) {
  let suggestions = `<div class='suggestion-section'><strong>${i18next.t('suggestions.mastitis_title')}</strong><br><ul>`;
  let items = [];

  if (isInfected) {
    if (confidence > 0.8) {
      items = safeGenerateSuggestionItems('suggestions.mastitis_high');
    } else if (confidence >= 0.5) {
      items = safeGenerateSuggestionItems('suggestions.mastitis_medium');
    } else {
      items = safeGenerateSuggestionItems('suggestions.mastitis_low');
    }
  } else {
    items = safeGenerateSuggestionItems('suggestions.mastitis_negative');
  }

  suggestions += items.map(item => `<li>${item}</li>`).join('');
  suggestions += "</ul></div>";
  return suggestions;
}

function generateFMDSuggestions(confidence, isInfected) {
  let suggestions = `<div class='suggestion-section'><strong>${i18next.t('suggestions.fmd_title')}</strong><br><ul>`;
  let items = [];

  if (isInfected) {
    if (confidence > 0.8) {
      items = safeGenerateSuggestionItems('suggestions.fmd_high');
    } else if (confidence >= 0.5) {
      items = safeGenerateSuggestionItems('suggestions.fmd_medium');
    } else {
      items = safeGenerateSuggestionItems('suggestions.fmd_low');
    }
  } else {
    items = safeGenerateSuggestionItems('suggestions.fmd_negative');
  }

  suggestions += items.map(item => `<li>${item}</li>`).join('');
  suggestions += "</ul></div>";
  return suggestions;
}

function safeGenerateSuggestions(translationKey) {
  try {
    const items = i18next.t(translationKey, { returnObjects: true });
    if (!Array.isArray(items)) {
      console.error(`Expected array for ${translationKey}, got:`, items);
      return '';
    }
    return items.map(item => `<li>${item}</li>`).join('');
  } catch (error) {
    console.error('Error generating suggestions:', error);
    return '';
  }
}
function displayResults(data) {
  currentResults = data;
  const showSuggestionInCombinedOnly = data.text_result && data.image_result;
  const resultContainer = document.getElementById("resultDisplay");
  resultContainer.innerHTML = "";
  
  // Status translation mapping
  // In the displayResults function, update the statusTranslations object:
// In the displayResults function, update the statusTranslations object:
const statusTranslations = {
  'Infected': i18next.t('results.infected'),
  'Not Infected': i18next.t('results.non_infected'),
  'Mastitis Detected': i18next.t('results.infected'),
  'No Mastitis': i18next.t('results.non_infected'),
  'Non-infected': i18next.t('results.non_infected'),
  'Healthy': i18next.t('results.non_infected') // Add this line
};
  
  const translateStatus = (status) => statusTranslations[status] || status;

  if (selectedDisease === "mastitis") {
    const textResult = data.text_result;
    const imageResult = data.image_result;
    
    if (textResult) {
      const translatedResult = translateStatus(textResult);
      const isInfected = textResult === 'Mastitis Detected';
      const confidence = (data.text_confidence * 100).toFixed(1);
      
      const textDiv = document.createElement("div");
      const suggestions = showSuggestionInCombinedOnly ? "" : generateMastitisSuggestions(data.text_confidence, isInfected);
      textDiv.className = "result-item";
      textDiv.innerHTML = `
        <div class="result-title">${i18next.t('results.text_analysis')}:</div>
        <div class="result-value ${isInfected ? 'danger' : 'success'}">
          ${translatedResult} (${confidence}% ${i18next.t('labels.confidence')})
        </div>
        ${suggestions}
      `;
      resultContainer.appendChild(textDiv);
    }
    
if (imageResult) {
  const translatedResult = translateStatus(imageResult);
  const isInfected = imageResult === 'Infected';
  const confidence = (data.image_confidence * 100).toFixed(1);
  
  const imgDiv = document.createElement("div");
  const suggestions = showSuggestionInCombinedOnly ? "" : generateMastitisSuggestions(data.image_confidence, isInfected);
  imgDiv.className = "result-item";
  imgDiv.innerHTML = `
    <div class="result-title">${i18next.t('results.image_analysis')}:</div>
    <div class="result-value ${isInfected ? 'danger' : 'success'}">
      ${translatedResult} (${confidence}% ${i18next.t('labels.confidence')})
    </div>
    ${suggestions}
  `;
  resultContainer.appendChild(imgDiv);
}    
    if (textResult && imageResult) {
  const combinedProb = (data.text_confidence + data.image_confidence) / 2;
  const isInfected = combinedProb > 0.5;
  const combinedResult = isInfected ? 'Mastitis Detected' : 'No Mastitis';
  const translatedCombinedResult = translateStatus(combinedResult);
  
  const combinedDiv = document.createElement("div");
  combinedDiv.className = "final-result";
  combinedDiv.innerHTML = `
    <div class="result-title">${i18next.t('results.final_diagnosis')}:</div>
    <div class="result-value ${isInfected ? 'danger' : 'success'}">
      ${translatedCombinedResult} (${(combinedProb * 100).toFixed(1)}% ${i18next.t('labels.confidence')})
    </div>
    <p>${i18next.t('results.combined_analysis')}</p>
    ${generateMastitisSuggestions(combinedProb, isInfected)}
  `;
  resultContainer.appendChild(combinedDiv);
}
    
    if (data.text_error) {
      const errorDiv = document.createElement("div");
      errorDiv.className = "error-message";
      errorDiv.textContent = `${i18next.t('errors.text')}: ${data.text_error}`;
      resultContainer.appendChild(errorDiv);
    }
    
    if (data.image_error) {
      const errorDiv = document.createElement("div");
      errorDiv.className = "error-message";
      errorDiv.textContent = `${i18next.t('errors.image')}: ${data.image_error}`;
      resultContainer.appendChild(errorDiv);
    }
  } 
  else if (selectedDisease === "fmd") {
    const results = [];
    let validFMDInputs = 0;
    
        ['foot_text', 'mouth_text', 'foot_image', 'mouth_image'].forEach(type => {
      const resultKey = `${type}_result`;
      if (data[resultKey]) {
        const [part] = type.split('_');
        results.push({
          type,
          title: i18next.t(`results.${part}_${type.includes('text') ? 'symptoms' : 'image'}_analysis`),
          result: data[resultKey],
          translatedResult: translateStatus(data[resultKey]), // Add this line
          confidence: data[`${type}_confidence`],
          isInfected: data[resultKey] === 'Infected'
        });
      }
    });
    
    results.forEach(item => {
      const div = document.createElement("div");
      const suggestions = data.combined_result ? null : generateFMDSuggestions(item.confidence, item.isInfected);
      div.className = "result-item";
      div.innerHTML = `
        <div class="result-title">${item.title}:</div>
        <div class="result-value ${item.isInfected ? 'danger' : 'success'}">
          ${item.translatedResult} (${(item.confidence * 100).toFixed(1)}% ${i18next.t('labels.confidence')})
        </div>
        ${suggestions || ''}
      `;
      resultContainer.appendChild(div);
    });
    
    if (data.combined_result) {
      const combinedDiv = document.createElement("div");
      combinedDiv.className = "final-result";
      const translatedCombinedResult = translateStatus(data.combined_result);
      combinedDiv.innerHTML = `
        <div class="result-title">${i18next.t('results.final_diagnosis')}:</div>
        <div class="result-value ${data.combined_result === 'Infected' ? 'danger' : 'success'}">
          ${translatedCombinedResult} (${(data.combined_confidence * 100).toFixed(1)}% ${i18next.t('labels.confidence')})
        </div>  
        <p>${i18next.t('results.combined_analysis')}</p>
        ${generateFMDSuggestions(data.combined_confidence, data.combined_result === 'Infected')}
      `;
      resultContainer.appendChild(combinedDiv);
    }
    

    
    ['foot_text', 'mouth_text', 'foot_image', 'mouth_image'].forEach(type => {
      const errorKey = `${type}_error`;
      if (data[errorKey]) {
        const errorDiv = document.createElement("div");
        errorDiv.className = "error-message";
        errorDiv.textContent = `${i18next.t(`errors.${type.replace('_', '-')}`)}: ${data[errorKey]}`;
        resultContainer.appendChild(errorDiv);
      }
    });
  }
  
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
// Add this at the top of your script.js
document.getElementById('languageSwitcher').addEventListener('change', function(e) {
  i18next.changeLanguage(e.target.value, (err, t) => {
    if (err) return console.error(err);
    updateContent();
    initializeDiseaseFeatures();
    
    // Force reload input fields if on step 3
    if (!document.getElementById('step3').classList.contains('hidden')) {
      renderInputFields();
    }
  });
});

// Add this to ensure the switcher shows the correct language
i18next.on('languageChanged', () => {
  document.getElementById('languageSwitcher').value = i18next.language;
  updateContent();
  initializeDiseaseFeatures();
});

document.getElementById("inputForm").addEventListener("submit", function (e) {
  e.preventDefault();
  const form = e.target;
  const formData = new FormData(form);
  
  selectedInputs.forEach(input => {
    formData.append("inputTypes[]", input);
  });

  const url = selectedDisease === "mastitis"
    ? "http://localhost:5000/predict/mastitis"
    : "http://localhost:5000/predict/fmd";

  const predictBtn = form.querySelector('.predict-button');
  const originalText = predictBtn.textContent;
  predictBtn.textContent = i18next.t('buttons.processing');
  predictBtn.disabled = true;

  fetch(url, {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => displayResults(data))
    // Replace your error handling in the fetch call with:
.catch(err => {
  const errorMessage = err.message || 'Unknown error occurred';
  document.getElementById("resultDisplay").innerHTML = `
    <div class="error-message">${i18next.t('errors.prediction', { error: errorMessage })}</div>
  `;
  document.getElementById("step3").classList.add("hidden");
  document.getElementById("step4").classList.remove("hidden");
  console.error("Prediction error:", err); // Log full error for debugging
})
    .finally(() => {
      predictBtn.textContent = originalText;
      predictBtn.disabled = false;
    });
});

function toggleChatbot() {
  const chatbot = document.getElementById('chatbot');
  chatbot.style.display = (chatbot.style.display === 'none' || chatbot.style.display === '') ? 'flex' : 'none';
  
  if (!localStorage.getItem('chatbotOpened') && chatbot.style.display === 'flex') {
    const messagesDiv = document.getElementById("chatbot-messages");
    messagesDiv.innerHTML += `<div class="bubble bot">🐄 ${i18next.t('chatbot.welcome')}</div>`;
    localStorage.setItem('chatbotOpened', 'true');
  }
}

async function sendChatbotMessage() {
  const input = document.getElementById("chatbot-input");
  const message = input.value.trim();
  if (!message) return;

  const messagesDiv = document.getElementById("chatbot-messages");
  messagesDiv.innerHTML += `<div class="bubble user">👨‍🌾 ${message}</div>`;

  const needsLocation = /vet|doctor|clinic|hospital/i.test(message);
  
  if (needsLocation && navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      async position => {
        try {
          const response = await fetch("http://localhost:5000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message: message,
              latitude: position.coords.latitude,
              longitude: position.coords.longitude
            })
          });

          const data = await response.json();
          const formattedMessage = data.response.replace(/(https?:\/\/[^\s]+)/g, 
            `<a href="$1" target="_blank">${i18next.t('chatbot.map_link')}</a>`);
          messagesDiv.innerHTML += `<div class="bubble bot">🐄 ${formattedMessage}</div>`;
        } catch (error) {
          messagesDiv.innerHTML += `<div class="bubble bot">⚠️ ${i18next.t('errors.chatbot')}</div>`;
        }
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
      },
      () => {
        messagesDiv.innerHTML += `<div class="bubble bot">⚠️ ${i18next.t('errors.location')}</div>`;
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
      }
    );
  } else {
    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
      });

      const data = await response.json();
      messagesDiv.innerHTML += `<div class="bubble bot">🐄 ${data.response}</div>`;
    } catch (error) {
      messagesDiv.innerHTML += `<div class="bubble bot">⚠️ ${i18next.t('errors.chatbot')}</div>`;
    }
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  input.value = "";
}

async function loadLogoBase64(path) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = path;
    img.crossOrigin = 'Anonymous';
    img.onload = function () {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = reject;
  });
}

async function generatePDF() {
  const downloadBtn = document.getElementById('downloadBtn');
  const originalText = downloadBtn.textContent;
  
  try {
      downloadBtn.textContent = await i18next.t('buttons.processing');
    downloadBtn.disabled = true;

    const doc = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4'
    });

    doc.setFont('helvetica', 'normal');
    doc.setTextColor(40, 40, 40);

    // Load and add logo to top right corner
    try {
      const logoData = await loadLogoBase64('./logo.png');
      const logoWidth = 25;
      const logoHeight = 20;
      const logoX = doc.internal.pageSize.width - logoWidth - 10;
      const logoY = 10;
      doc.addImage(logoData, 'PNG', logoX, logoY, logoWidth, logoHeight);
    } catch (e) {
      console.log('Logo not loaded, using text header only');
    }

    // Header section
    doc.setFontSize(18);
    doc.text('Bovine Health Report', 20, 20);
    
    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 27);

    const diseaseName = selectedDisease === 'mastitis' ? 'Mastitis' : 'Foot and Mouth Disease';
    doc.setFontSize(14);
    doc.text(`${diseaseName} Diagnosis Report`, 20, 35);

    doc.setDrawColor(200, 200, 200);
    doc.line(20, 40, 190, 40);

    let yPos = 50;

    // Input Data Section
    doc.setFontSize(12);
    doc.text('Input Data Provided:', 20, yPos);
    yPos += 8;

    // Process text inputs
    const processedInputs = new Set();
    if (selectedInputs.some(input => input.includes('text'))) {
      const textInputs = document.querySelectorAll('input[type="number"], select');
      textInputs.forEach(input => {
        if (yPos > 270) { 
          doc.addPage(); 
          yPos = 20; 
        }
        
        let label = cleanLabel(input.previousElementSibling?.textContent || input.name);
        label = label.replace(/^labels\./, '').replace(/_/g, ' ');
        
        if (!label || processedInputs.has(label)) return;
        
        processedInputs.add(label);
        const value = input.value;
        
        doc.setFontSize(10);
        doc.text(`${label}: ${value}`, 25, yPos);
        yPos += 6;
      });
    }

    // Process image inputs - Modified to handle multiple images
   // Process image inputs - Always add all images
const imageInputs = document.querySelectorAll('input[type="file"]');
for (const input of imageInputs) {
  if (input.files && input.files[0]) {
    if (yPos > 180) {
      doc.addPage();
      yPos = 20;
    }

    let label = cleanLabel(input.previousElementSibling?.textContent || input.name);
    label = label.replace(/^labels\./, '').replace(/_/g, ' ');

    // REMOVE processedInputs check for images
    doc.setFontSize(10);
    doc.text(`${label}:`, 25, yPos);
    yPos += 6;

    try {
      const img = new Image();
      const reader = new FileReader();

      await new Promise((resolve, reject) => {
        reader.onload = function(e) {
          img.src = e.target.result;
          img.onload = function() {
            const maxWidth = 120;
            const ratio = maxWidth / img.width;
            const height = img.height * ratio;

            doc.addImage(img, 'JPEG', 25, yPos, maxWidth, height);
            yPos += height + 8;
            resolve();
          };
          img.onerror = reject;
        };
        reader.readAsDataURL(input.files[0]);
      });
    } catch (error) {
      console.error('Error processing image:', error);
      doc.text(`[Image not loaded]`, 30, yPos);
      yPos += 12;
    }
  }
}


    // Results Section
    doc.setFontSize(12);
    if (yPos > 270) {
      doc.addPage();
      yPos = 20;
    }
    doc.text('Analysis Results:', 20, yPos);
    yPos += 8;

    if (currentResults) {
      if (selectedDisease === "mastitis") {
        if (currentResults.text_result) {
          yPos = addResultSection(doc, 'Text Analysis', currentResults.text_result, currentResults.text_confidence, yPos);
          yPos += 8;
        }
        if (currentResults.image_result) {
          yPos = addResultSection(doc, 'Image Analysis', currentResults.image_result, currentResults.image_confidence, yPos);
          yPos += 8;
        }
        if (currentResults.text_result && currentResults.image_result) {
          const combinedProb = (currentResults.text_confidence + currentResults.image_confidence) / 2;
          const combinedResult = combinedProb > 0.5 ? 'Mastitis Detected' : 'No Mastitis';
          
          doc.setFontSize(14);
          doc.setFont('helvetica', 'bold');
          doc.text('FINAL DIAGNOSIS:', 20, yPos);
          yPos += 8;
          
          const statusColor = combinedProb > 0.5 ? [200, 0, 0] : [0, 150, 0];
          doc.setFontSize(12);
          doc.setTextColor(...statusColor);
          doc.text(`${combinedResult} (${(combinedProb * 100).toFixed(1)}% confidence)`, 25, yPos);
          
          doc.setFont('helvetica', 'normal');
          doc.setTextColor(40, 40, 40);
          yPos += 12;
        }
      } else if (selectedDisease === "fmd") {
        ['foot_text', 'mouth_text', 'foot_image', 'mouth_image'].forEach(type => {
          if (currentResults[`${type}_result`]) {
            const [part] = type.split('_');
            const title = `${part.charAt(0).toUpperCase() + part.slice(1)} ${type.includes('text') ? 'Symptoms' : 'Image'}`;
            yPos = addResultSection(doc, title, currentResults[`${type}_result`], currentResults[`${type}_confidence`], yPos);
            yPos += 8;
          }
        });
        
        if (currentResults.combined_result) {
          doc.setFontSize(14);
          doc.setFont('helvetica', 'bold');
          doc.setTextColor(0, 0, 0);
          doc.text('FINAL DIAGNOSIS:', 20, yPos);
          yPos += 8;
          
          const statusColor = currentResults.combined_result === 'Infected' ? [200, 0, 0] : [0, 150, 0];
          doc.setFontSize(12);
          doc.setTextColor(...statusColor);
          doc.text(`${currentResults.combined_result} (${(currentResults.combined_confidence * 100).toFixed(1)}% confidence)`, 25, yPos);
          
          doc.setFont('helvetica', 'normal');
          doc.setTextColor(40, 40, 40);
          yPos += 12;
        }
      }
    }

    // Footer
    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);
    doc.text('© 2025 Bovine Health Assistant', 105, 290, { align: 'center' });

    // Save PDF
    const cleanDiseaseName = diseaseName.replace(/ /g, '_').toLowerCase();
    doc.save(`bovine_${cleanDiseaseName}_report_${new Date().toISOString().slice(0,10)}.pdf`);

  }  catch (error) {
    console.error('PDF Generation Error:', error);
    alert(await i18next.t('errors.pdf_generation'));
  } finally {
    // Restore the original button text
    downloadBtn.textContent = originalText;
    downloadBtn.disabled = false;
  }
}

// Modified addResultSection to handle larger fonts for final diagnosis
function addResultSection(doc, title, result, confidence, yPos) {
  const isInfected = result.includes('Infected') || result.includes('Detected');
  const statusColor = isInfected ? [200, 0, 0] : [0, 150, 0];
  
  const isFinalDiagnosis = title.includes('FINAL DIAGNOSIS');
  const titleFontSize = isFinalDiagnosis ? 14 : 10;
  const resultFontSize = isFinalDiagnosis ? 12 : 10;
  
  doc.setFontSize(titleFontSize);
  doc.setTextColor(40, 40, 40);
  doc.text(`${title}:`, 20, yPos);
  
  doc.setFontSize(resultFontSize);
  doc.setTextColor(...statusColor);
  const resultText = `${result} (${(confidence * 100).toFixed(1)}% confidence)`;
  doc.text(resultText, 20 + doc.getTextWidth(title + ': ') + 2, yPos);
  
  return yPos + (isFinalDiagnosis ? 8 : 6);
}

// Helper function to clean labels
function cleanLabel(text) {
  if (!text) return '';
  // Remove unwanted characters and normalize
  return text
    .replace(/[^a-zA-Z0-9\s:_.,%-]/g, '') 
    .replace(/^labels\./, '')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}



// Helper functions for English recommendations
function getMastitisRecommendations(results) {
  const isInfected = results.combined_result === 'Mastitis Detected' || 
                    results.text_result === 'Mastitis Detected' || 
                    results.image_result === 'Infected';
  const confidence = results.combined_confidence || results.text_confidence || results.image_confidence || 0;

  if (isInfected) {
    if (confidence > 0.8) {
      return [
        "Immediate veterinary attention required",
        "Isolate the affected cow from the herd",
        "Administer prescribed antibiotics",
        "Monitor milk for abnormalities",
        "Implement strict hygiene measures"
      ];
    } else if (confidence >= 0.5) {
      return [
        "Veterinary consultation recommended",
        "Monitor cow's temperature and behavior",
        "Check for udder swelling or pain",
        "Consider milk culture test",
        "Review milking equipment sanitation"
      ];
    } else {
      return [
        "Monitor for further symptoms",
        "Check milking procedures",
        "Consider retesting in 24 hours",
        "Ensure proper udder hygiene",
        "Watch for changes in milk appearance"
      ];
    }
  } else {
    return [
      "No immediate action needed",
      "Continue regular health monitoring",
      "Maintain good udder hygiene practices",
      "Ensure proper milking equipment maintenance",
      "Schedule regular veterinary check-ups"
    ];
  }
}

function getFMDRecommendations(results) {
  const isInfected = results.combined_result === 'Infected';
  const confidence = results.combined_confidence || 0;

  if (isInfected) {
    if (confidence > 0.8) {
      return [
        "Immediately notify local veterinary authorities",
        "Strict isolation of affected animals",
        "Disinfect all equipment and premises",
        "Stop all movement of animals",
        "Implement biosecurity measures"
      ];
    } else if (confidence >= 0.5) {
      return [
        "Contact veterinarian for confirmation",
        "Isolate potentially affected animals",
        "Disinfect equipment and footwear",
        "Monitor herd for additional cases",
        "Restrict farm visitors"
      ];
    } else {
      return [
        "Monitor animals closely",
        "Check for fever or loss of appetite",
        "Consider veterinary consultation",
        "Review vaccination records",
        "Watch for excessive salivation or lameness"
      ];
    }
  } else {
    return [
      "No signs of FMD detected",
      "Maintain good biosecurity practices",
      "Ensure animals are properly vaccinated",
      "Monitor herd health regularly",
      "Report any suspicious symptoms immediately"
    ];
  }
}

// Helper function to add result sections


// Helper function to add result sections


// Enhanced error display
function showPDFError(errorMessage) {
  const resultDisplay = document.getElementById("resultDisplay");
  if (!resultDisplay) return;
  
  const errorDiv = document.createElement("div");
  errorDiv.className = "error-message";
  errorDiv.innerHTML = `
    <strong>PDF Generation Error</strong>
    <p>${errorMessage}</p>
    <button class="retry-button" onclick="generatePDF()">Try Again</button>
  `;
  
  resultDisplay.appendChild(errorDiv);
}