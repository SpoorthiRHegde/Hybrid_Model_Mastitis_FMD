let selectedDisease = "";
let selectedInputs = [];
let currentResults = null;

function showAboutPage() {
  console.log("Showing about page");
  document.getElementById('app-container').classList.add('hidden');
  document.getElementById('about-container').classList.remove('hidden');
}
// Initialize when document is loaded
document.addEventListener('DOMContentLoaded', function() {
  initializeChatbot();
  
  // Make sure the main page is visible by default
  document.getElementById('app-container').classList.remove('hidden');
  document.getElementById('about-container').classList.add('hidden');
});
// Fix the showMainPage function
function showMainPage() {
  console.log("Showing main page");
  document.getElementById('about-container').classList.add('hidden');
  document.getElementById('app-container').classList.remove('hidden');
}

function openTab(evt, tabName) {
  document.querySelectorAll('.tab-content').forEach(tab => {
    tab.classList.remove('active');
  });
  
  document.querySelectorAll('.tab-link').forEach(tab => {
    tab.classList.remove('active');
  });
  
  document.getElementById(tabName).classList.add('active');
  
  evt.currentTarget.classList.add('active');
}

function goToStep2() {
  selectedDisease = document.getElementById("disease").value;
  if (!selectedDisease) {
    alert(i18next.t('alerts.select_disease'));
    return;
  }

  document.getElementById("step1").classList.add("hidden");
  document.getElementById("step2").classList.remove("hidden");

  document.getElementById("mastitis-options").classList.add("hidden");
  document.getElementById("fmd-options").classList.add("hidden");
  document.getElementById("both-options").classList.add("hidden");

  if (selectedDisease === "mastitis") {
    document.getElementById("mastitis-options").classList.remove("hidden");
  } else if (selectedDisease === "fmd") {
    document.getElementById("fmd-options").classList.remove("hidden");
  } else if (selectedDisease === "both") {
    document.getElementById("both-options").classList.remove("hidden");
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
  } else if (selectedDisease === "both") {
    document.querySelectorAll('input[name="bothMastitisInput"]:checked').forEach(checkbox => {
      selectedInputs.push(`mastitis_${checkbox.value}`);
    });
    document.querySelectorAll('input[name="bothFMDTextInput"]:checked').forEach(checkbox => {
      selectedInputs.push(`fmd_text_${checkbox.value}`);
    });
    document.querySelectorAll('input[name="bothFMDImageInput"]:checked').forEach(checkbox => {
      selectedInputs.push(`fmd_image_${checkbox.value}`);
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

  const createInputField = (feature, prefix) => {
    const div = document.createElement("div");
    const label = document.createElement("label");
    label.textContent = i18next.t(`labels.${feature.name}`) || feature.label;
    div.appendChild(label);

    if (feature.type === "number") {
      const input = document.createElement("input");
      input.type = "number";
      input.name = `${prefix}_${feature.name}`;
      input.min = feature.min;
      input.max = feature.max;
      input.step = feature.step || 1;
      input.required = true;
      div.appendChild(input);
    } else if (feature.type === "select") {
      const select = document.createElement("select");
      select.name = `${prefix}_${feature.name}`;
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
      const field = createInputField(feature, prefix);
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
  } else if (selectedDisease === "both") {
    selectedInputs.forEach(input => {
      const parts = input.split("_");
      
      if (parts[0] === "mastitis") {
        const type = parts[1];
        if (type === "text") {
          textContainer.appendChild(
            createTextInputSection(
              i18next.t('sections.mastitis_symptoms'),
              window.translatedDiseaseFeatures.mastitis,
              "mastitis"
            )
          );
        } else if (type === "image") {
          imgContainer.appendChild(
            createFileInput(
              i18next.t('labels.udder_image'),
              "udderImage"
            )
          );
        }
      } else if (parts[0] === "fmd") {
        const type = parts[1];
        const part = parts[2];
        
        if (type === "text") {
          textContainer.appendChild(
            createTextInputSection(
              i18next.t(`sections.${part}_symptoms`),
              window.translatedDiseaseFeatures[part],
              `fmd_${part}_text`
            )
          );
        } else if (type === "image") {
          imgContainer.appendChild(
            createFileInput(
              i18next.t(`labels.${part}_image`),
              `fmd_${part}Image`
            )
          );
        }
      }
    });
  }
}

function safeGenerateSuggestionItems(translationKey) {
  try {
    const items = i18next.t(translationKey, { returnObjects: true });
    
    if (typeof items === 'string') {
      return items.split('\n').filter(item => item.trim() !== '');
    }
    
    if (!Array.isArray(items)) {
      return [];
    }
    
    return items;
  } catch (error) {
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

function displayResults(data) {
  currentResults = data;
  const resultContainer = document.getElementById("resultDisplay");
  resultContainer.innerHTML = "";
  
  const statusTranslations = {
    'Infected': i18next.t('results.infected'),
    'Not Infected': i18next.t('results.non_infected'),
    'Mastitis Detected': i18next.t('results.infected'),
    'No Mastitis': i18next.t('results.non_infected'),
    'Non-infected': i18next.t('results.non_infected'),
    'Healthy': i18next.t('results.non_infected')
  };
  
  const translateStatus = (status) => statusTranslations[status] || status;

  const hasMastitisResults = data.mastitis_text_result || data.text_result || 
                            data.mastitis_image_result || data.image_result;
  
  if (hasMastitisResults) {
    const mastitisHeader = document.createElement("h3");
    mastitisHeader.textContent = i18next.t('mastitis') + " " + i18next.t('results.results');
    mastitisHeader.className = "disease-header";
    resultContainer.appendChild(mastitisHeader);
    
    const textResult = data.mastitis_text_result || data.text_result;
    const textConfidence = data.mastitis_text_confidence || data.text_confidence;
    
    const imageResult = data.mastitis_image_result || data.image_result;
    const imageConfidence = data.mastitis_image_confidence || data.image_confidence;
    
    const combinedResult = data.mastitis_combined_result || data.combined_result;
    const combinedConfidence = data.mastitis_combined_confidence || data.combined_confidence;
    
    const mastitisResultCount = [textResult, imageResult].filter(Boolean).length;
    
    if (mastitisResultCount > 1) {
      if (textResult) {
        const translatedResult = translateStatus(textResult);
        const isInfected = textResult.includes('Mastitis') || textResult.includes('Infected');
        
        const textDiv = document.createElement("div");
        textDiv.className = "result-item";
        textDiv.innerHTML = `
          <div class="result-title">${i18next.t('results.text_analysis')}:</div>
          <div class="result-value ${isInfected ? 'danger' : 'success'}">
            ${translatedResult} ${textConfidence ? `(${(textConfidence * 100).toFixed(1)}% ${i18next.t('labels.confidence')})` : ''}
          </div>
          ${generateMastitisSuggestions(textConfidence || 0, isInfected)}
        `;
        resultContainer.appendChild(textDiv);
      }
      
      if (imageResult) {
        const translatedResult = translateStatus(imageResult);
        const isInfected = imageResult.includes('Infected');
        
        const imgDiv = document.createElement("div");
        imgDiv.className = "result-item";
        imgDiv.innerHTML = `
          <div class="result-title">${i18next.t('results.image_analysis')}:</div>
          <div class="result-value ${isInfected ? 'danger' : 'success'}">
            ${translatedResult} ${imageConfidence ? `(${(imageConfidence * 100).toFixed(1)}% ${i18next.t('labels.confidence')})` : ''}
          </div>
          ${generateMastitisSuggestions(imageConfidence || 0, isInfected)}
        `;
        resultContainer.appendChild(imgDiv);
      }
    }
    
    if (combinedResult || mastitisResultCount === 1) {
      const finalResult = combinedResult || textResult || imageResult;
      const finalConfidence = combinedConfidence || textConfidence || imageConfidence;
      const isInfected = finalResult.includes('Mastitis') || finalResult.includes('Infected');
      const translatedFinalResult = translateStatus(finalResult);
      
      const finalDiv = document.createElement("div");
      finalDiv.className = "final-result";
      
      if (mastitisResultCount > 1) {
        finalDiv.innerHTML = `
          <div class="result-title">${i18next.t('results.final_diagnosis')} (Mastitis):</div>
          <div class="result-value ${isInfected ? 'danger' : 'success'}">
            ${translatedFinalResult} ${finalConfidence ? `(${(finalConfidence * 100).toFixed(1)}% ${i18next.t('labels.confidence')})` : ''}
          </div>
          <p>${i18next.t('results.combined_analysis')}</p>
          ${generateMastitisSuggestions(finalConfidence || 0, isInfected)}
        `;
      } else {
        finalDiv.innerHTML = `
          <div class="result-title">${i18next.t('results.final_diagnosis')} (Mastitis):</div>
          <div class="result-value ${isInfected ? 'danger' : 'success'}">
            ${translatedFinalResult} ${finalConfidence ? `(${(finalConfidence * 100).toFixed(1)}% ${i18next.t('labels.confidence')})` : ''}
          </div>
          ${generateMastitisSuggestions(finalConfidence || 0, isInfected)}
        `;
      }
      
      resultContainer.appendChild(finalDiv);
    }
    
    if (data.mastitis_text_error || data.text_error) {
      const errorDiv = document.createElement("div");
      errorDiv.className = "error-message";
      errorDiv.textContent = `${i18next.t('errors.text')}: ${data.mastitis_text_error || data.text_error}`;
      resultContainer.appendChild(errorDiv);
    }
    
    if (data.mastitis_image_error || data.image_error) {
      const errorDiv = document.createElement("div");
      errorDiv.className = "error-message";
      errorDiv.textContent = `${i18next.t('errors.image')}: ${data.mastitis_image_error || data.image_error}`;
      resultContainer.appendChild(errorDiv);
    }
  }
  
  let hasFMDResults = false;
  
  if (selectedDisease === "fmd" || selectedDisease === "both") {
    const fmdResultTypes = ['foot_text', 'mouth_text', 'foot_image', 'mouth_image', 
                           'fmd_foot_text', 'fmd_mouth_text', 'fmd_foot_image', 'fmd_mouth_image'];
    
    fmdResultTypes.forEach(type => {
      if (data[`${type}_result`]) hasFMDResults = true;
    });
    
    if (hasFMDResults || data.fmd_combined_result) {
      if (hasMastitisResults) {
        const separator = document.createElement("hr");
        separator.className = "result-separator";
        resultContainer.appendChild(separator);
      }
      
      const fmdHeader = document.createElement("h3");
      fmdHeader.textContent = i18next.t('fmd') + " " + i18next.t('results.results');
      fmdHeader.className = "disease-header";
      resultContainer.appendChild(fmdHeader);
      
      const results = [];
      fmdResultTypes.forEach(type => {
        const resultKey = `${type}_result`;
        if (data[resultKey]) {
          const parts = type.split('_');
          const part = parts.length > 2 ? parts[1] : parts[0];
          const analysisType = parts.length > 2 ? parts[2] : parts[1];
          
          results.push({
            type,
            title: i18next.t(`results.${part}_${analysisType}_analysis`),
            result: data[resultKey],
            translatedResult: translateStatus(data[resultKey]),
            confidence: data[`${type}_confidence`],
            isInfected: data[resultKey] === 'Infected'
          });
        }
      });
      
      const fmdResultCount = results.length;
      
      if (fmdResultCount > 1) {
        results.forEach(item => {
          const div = document.createElement("div");
          div.className = "result-item";
          div.innerHTML = `
            <div class="result-title">${item.title}:</div>
            <div class="result-value ${item.isInfected ? 'danger' : 'success'}">
              ${item.translatedResult} ${item.confidence ? `(${(item.confidence * 100).toFixed(1)}% ${i18next.t('labels.confidence')})` : ''}
            </div>
            ${generateFMDSuggestions(item.confidence || 0, item.isInfected)}
          `;
          resultContainer.appendChild(div);
        });
      }
      
      const fmdCombinedResult = data.fmd_combined_result;
      const fmdCombinedConfidence = data.fmd_combined_confidence;
      
      if (fmdCombinedResult || fmdResultCount > 0) {
        const finalResult = fmdCombinedResult || (results.length > 0 ? results[0].result : null);
        const finalConfidence = fmdCombinedConfidence || (results.length > 0 ? results[0].confidence : null);
        
        if (finalResult) {
          const isInfected = finalResult === 'Infected';
          const translatedFinalResult = translateStatus(finalResult);
          
          const finalDiv = document.createElement("div");
          finalDiv.className = "final-result";
          
          if (fmdResultCount > 1) {
            finalDiv.innerHTML = `
              <div class="result-title">${i18next.t('results.final_diagnosis')} (FMD):</div>
              <div class="result-value ${isInfected ? 'danger' : 'success'}">
                ${translatedFinalResult} ${finalConfidence ? `(${(finalConfidence * 100).toFixed(1)}% ${i18next.t('labels.confidence')})` : ''}
              </div>  
              <p>${i18next.t('results.combined_analysis')}</p>
              ${generateFMDSuggestions(finalConfidence || 0, isInfected)}
            `;
          } else {
            finalDiv.innerHTML = `
              <div class="result-title">${i18next.t('results.final_diagnosis')} (FMD):</div>
              <div class="result-value ${isInfected ? 'danger' : 'success'}">
                ${translatedFinalResult} ${finalConfidence ? `(${(finalConfidence * 100).toFixed(1)}% ${i18next.t('labels.confidence')})` : ''}
              </div>
              ${generateFMDSuggestions(finalConfidence || 0, isInfected)}
            `;
          }
          
          resultContainer.appendChild(finalDiv);
        }
      }
      
      fmdResultTypes.forEach(type => {
        const errorKey = `${type}_error`;
        if (data[errorKey]) {
          const errorDiv = document.createElement("div");
          errorDiv.className = "error-message";
          errorDiv.textContent = `${i18next.t(`errors.${type.replace(/_/g, '-')}`)}: ${data[errorKey]}`;
          resultContainer.appendChild(errorDiv);
        }
      });
    }
  }
  
  if (resultContainer.innerHTML === "") {
    resultContainer.innerHTML = `
      <div class="error-message">
        ${i18next.t('errors.no_results')}
      </div>
      <div>Debug info: ${JSON.stringify(data)}</div>
    `;
  }
  
  document.getElementById("step3").classList.add("hidden");
  document.getElementById("step4").classList.remove("hidden");
}

function goBackToStep3() {
  document.getElementById("step4").classList.add("hidden");
  document.getElementById("step3").classList.remove("hidden");
}
// Add this function to initialize the chatbot properly
function initializeChatbot() {
  const chatbot = document.getElementById('chatbot');
  chatbot.classList.add('hidden');
  chatbot.classList.remove('visible');
}
function restartProcess() {
  document.getElementById("step4").classList.add("hidden");
  document.getElementById("step1").classList.remove("hidden");
  document.getElementById("inputForm").reset();
  selectedDisease = "";
  selectedInputs = [];
}

document.getElementById('languageSwitcher').addEventListener('change', function(e) {
  i18next.changeLanguage(e.target.value, (err, t) => {
    if (err) return console.error(err);
    updateContent();
    initializeDiseaseFeatures();
    
    if (!document.getElementById('step3').classList.contains('hidden')) {
      renderInputFields();
    }
  });
});

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

  let url;
  if (selectedDisease === "mastitis") {
    url = "http://localhost:5000/predict/mastitis";
  } else if (selectedDisease === "fmd") {
    url = "http://localhost:5000/predict/fmd";
  } else if (selectedDisease === "both") {
    url = "http://localhost:5000/predict/both";
  }

  const predictBtn = form.querySelector('.predict-button');
  const originalText = predictBtn.textContent;
  predictBtn.textContent = i18next.t('buttons.processing');
  predictBtn.disabled = true;

  fetch(url, {
    method: "POST",
    body: formData
  })
    .then(res => {
      const contentType = res.headers.get("content-type");
      if (contentType && contentType.includes("application/json")) {
        return res.json();
      } else {
        return res.text().then(text => {
          throw new Error(`Server returned non-JSON response: ${text.substring(0, 100)}...`);
        });
      }
    })
    .then(data => displayResults(data))
    .catch(err => {
      const errorMessage = err.message || 'Unknown error occurred';
      document.getElementById("resultDisplay").innerHTML = `
        <div class="error-message">${i18next.t('errors.prediction', { error: errorMessage })}</div>
      `;
      document.getElementById("step3").classList.add("hidden");
      document.getElementById("step4").classList.remove("hidden");
      console.error("Prediction error:", err);
    })
    .finally(() => {
      predictBtn.textContent = originalText;
      predictBtn.disabled = false;
    });
});

function toggleChatbot() {
  const chatbot = document.getElementById('chatbot');
  if (chatbot.classList.contains('hidden')) {
    chatbot.classList.remove('hidden');
    chatbot.classList.add('visible');
    
    // Add welcome message if first time opening
    if (!localStorage.getItem('chatbotOpened')) {
      const messagesDiv = document.getElementById("chatbot-messages");
      messagesDiv.innerHTML += `<div class="bubble bot">🐄 ${i18next.t('chatbot.welcome')}</div>`;
      localStorage.setItem('chatbotOpened', 'true');
    }
  } else {
    chatbot.classList.remove('visible');
    chatbot.classList.add('hidden');
  }
}


async function sendChatbotMessage() {
  const input = document.getElementById("chatbot-input");
  const message = input.value.trim();
  if (!message) return;

  const messagesDiv = document.getElementById("chatbot-messages");
  messagesDiv.innerHTML += `<div class="bubble user">👨‍🌾 ${message}</div>`;
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

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

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const data = await response.json();
          const formattedMessage = data.response.replace(/(https?:\/\/[^\s]+)/g, 
            `<a href="$1" target="_blank">${i18next.t('chatbot.map_link')}</a>`);
          messagesDiv.innerHTML += `<div class="bubble bot">🐄 ${formattedMessage}</div>`;
        } catch (error) {
          console.error("Chatbot error:", error);
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

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      messagesDiv.innerHTML += `<div class="bubble bot">🐄 ${data.response}</div>`;
    } catch (error) {
      console.error("Chatbot error:", error);
      messagesDiv.innerHTML += `<div class="bubble bot">⚠️ ${i18next.t('errors.chatbot')}</div>`;
    }
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  input.value = "";
}

// Add event listener for Enter key in chatbot input
document.getElementById("chatbot-input").addEventListener("keypress", function(e) {
  if (e.key === "Enter") {
    e.preventDefault();
    sendChatbotMessage();
  }
});

async function sendChatbotMessage() {
  const input = document.getElementById("chatbot-input");
  const message = input.value.trim();
  if (!message) return;

  const messagesDiv = document.getElementById("chatbot-messages");
  messagesDiv.innerHTML += `<div class="bubble user">👨‍🌾 ${message}</div>`;
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

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

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          
          const data = await response.json();
          const formattedMessage = data.response.replace(/(https?:\/\/[^\s]+)/g, 
            `<a href="$1" target="_blank">${i18next.t('chatbot.map_link')}</a>`);
          messagesDiv.innerHTML += `<div class="bubble bot">🐄 ${formattedMessage}</div>`;
        } catch (error) {
          console.error("Chatbot error:", error);
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

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      messagesDiv.innerHTML += `<div class="bubble bot">🐄 ${data.response}</div>`;
    } catch (error) {
      console.error("Chatbot error:", error);
      messagesDiv.innerHTML += `<div class="bubble bot">⚠️ ${i18next.t('errors.chatbot')}</div>`;
    }
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }

  input.value = "";
}

// Add event listener for Enter key in chatbot input
document.getElementById("chatbot-input").addEventListener("keypress", function(e) {
  if (e.key === "Enter") {
    e.preventDefault();
    sendChatbotMessage();
  }
});

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

    doc.setFontSize(18);
    doc.text('Bovine Health Report', 20, 20);
    
    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 27);

    let diseaseName;
    if (selectedDisease === "mastitis") {
      diseaseName = 'Mastitis';
    } else if (selectedDisease === "fmd") {
      diseaseName = 'Foot and Mouth Disease';
    } else if (selectedDisease === "both") {
      diseaseName = 'Mastitis & Foot and Mouth Disease';
    }
    
    doc.setFontSize(14);
    doc.text(`${diseaseName} Diagnosis Report`, 20, 35);

    doc.setDrawColor(200, 200, 200);
    doc.line(20, 40, 190, 40);

    let yPos = 50;

    doc.setFontSize(12);
    doc.text('Input Data Provided:', 20, yPos);
    yPos += 8;

    const processedInputs = new Set();
    const textInputs = document.querySelectorAll('input[type="number"], select, input[type="text"]');
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

    const imageInputs = document.querySelectorAll('input[type="file"]');
    for (const input of imageInputs) {
      if (input.files && input.files[0]) {
        if (yPos > 180) {
          doc.addPage();
          yPos = 20;
        }

        let label = cleanLabel(input.previousElementSibling?.textContent || input.name);
        label = label.replace(/^labels\./, '').replace(/_/g, ' ');

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

    doc.setFontSize(12);
    if (yPos > 270) {
      doc.addPage();
      yPos = 20;
    }
    doc.text('Analysis Results:', 20, yPos);
    yPos += 8;

    if (currentResults) {
      if (selectedDisease === "mastitis" || selectedDisease === "both") {
        const mastitisTextResult = currentResults.mastitis_text_result || currentResults.text_result;
        const mastitisTextConfidence = currentResults.mastitis_text_confidence || currentResults.text_confidence;
        const mastitisImageResult = currentResults.mastitis_image_result || currentResults.image_result;
        const mastitisImageConfidence = currentResults.mastitis_image_confidence || currentResults.image_confidence;
        const mastitisCombinedResult = currentResults.mastitis_combined_result || currentResults.combined_result;
        const mastitisCombinedConfidence = currentResults.mastitis_combined_confidence || currentResults.combined_confidence;

        if (mastitisTextResult) {
          yPos = addResultSection(doc, 'Mastitis Text Analysis', mastitisTextResult, mastitisTextConfidence, yPos);
          yPos += 8;
        }
        if (mastitisImageResult) {
          yPos = addResultSection(doc, 'Mastitis Image Analysis', mastitisImageResult, mastitisImageConfidence, yPos);
          yPos += 8;
        }
        if (mastitisCombinedResult) {
          doc.setFontSize(14);
          doc.setFont('helvetica', 'bold');
          doc.text('MASTITIS FINAL DIAGNOSIS:', 20, yPos);
          yPos += 8;
          
          const statusColor = mastitisCombinedResult.includes('Infected') || mastitisCombinedResult.includes('Detected') ? [200, 0, 0] : [0, 150, 0];
          doc.setFontSize(12);
          doc.setTextColor(...statusColor);
          doc.text(`${mastitisCombinedResult} (${(mastitisCombinedConfidence * 100).toFixed(1)}% confidence)`, 25, yPos);
          
          doc.setFont('helvetica', 'normal');
          doc.setTextColor(40, 40, 40);
          yPos += 12;
        }
      }

      if (selectedDisease === "fmd" || selectedDisease === "both") {
        if (selectedDisease === "both" && yPos > 200) {
          doc.addPage();
          yPos = 20;
        }
        
        const fmdResultTypes = ['foot_text', 'mouth_text', 'foot_image', 'mouth_image', 
                               'fmd_foot_text', 'fmd_mouth_text', 'fmd_foot_image', 'fmd_mouth_image'];
        
        fmdResultTypes.forEach(type => {
          if (currentResults[`${type}_result`]) {
            const parts = type.split('_');
            const part = parts.length > 2 ? parts[1] : parts[0];
            const analysisType = parts.length > 2 ? parts[2] : parts[1];
            const title = `${part.charAt(0).toUpperCase() + part.slice(1)} ${analysisType.charAt(0).toUpperCase() + analysisType.slice(1)} Analysis`;
            
            yPos = addResultSection(doc, title, currentResults[`${type}_result`], currentResults[`${type}_confidence`], yPos);
            yPos += 8;
          }
        });
        
        const fmdCombinedResult = currentResults.fmd_combined_result || currentResults.combined_result;
        const fmdCombinedConfidence = currentResults.fmd_combined_confidence || currentResults.combined_confidence;
        
        if (fmdCombinedResult) {
          doc.setFontSize(14);
          doc.setFont('helvetica', 'bold');
          doc.text('FMD FINAL DIAGNOSIS:', 20, yPos);
          yPos += 8;
          
          const statusColor = fmdCombinedResult === 'Infected' ? [200, 0, 0] : [0, 150, 0];
          doc.setFontSize(12);
          doc.setTextColor(...statusColor);
          doc.text(`${fmdCombinedResult} (${(fmdCombinedConfidence * 100).toFixed(1)}% confidence)`, 25, yPos);
          
          doc.setFont('helvetica', 'normal');
          doc.setTextColor(40, 40, 40);
          yPos += 12;
        }
      }
    }

    doc.setFontSize(8);
    doc.setTextColor(150, 150, 150);

    const disclaimer = 
      "© 2025 Bovine Health Assistant. This tool is intended for informational and educational purposes only " +
      "and should not be considered a substitute for professional veterinary advice.";

    const wrappedDisclaimer = doc.splitTextToSize(disclaimer, 170);

    doc.text(wrappedDisclaimer, doc.internal.pageSize.width / 2, 285, { align: 'center' });

    const cleanDiseaseName = diseaseName.replace(/ /g, '_').toLowerCase();
    doc.save(`bovine_${cleanDiseaseName}_report_${new Date().toISOString().slice(0,10)}.pdf`);

  } catch (error) {
    console.error('PDF Generation Error:', error);
    alert(await i18next.t('errors.pdf_generation'));
  } finally {
    downloadBtn.textContent = originalText;
    downloadBtn.disabled = false;
  }
}

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

function cleanLabel(text) {
  if (!text) return '';
  return text
    .replace(/[^a-zA-Z0-9\s:_.,%-]/g, '') 
    .replace(/^labels\./, '')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}