i18next
  .use(i18nextHttpBackend)
  .use(i18nextBrowserLanguageDetector)
  .init({
    fallbackLng: 'en',
    debug: true,
    supportedLngs: ['en', 'hi', 'kn'], // Explicitly list supported languages
    detection: {
      order: ['querystring', 'cookie', 'navigator', 'htmlTag'],
      caches: ['cookie'],
      lookupQuerystring: 'lng',
      lookupCookie: 'i18next',
      htmlTag: document.documentElement
    },
    backend: {
      loadPath: '/i18n/{{lng}}.json'
    },
    interpolation: {
      escapeValue: false,
      skipOnVariables: false
    }
  }, function(err, t) {
    if (err) return console.error(err);
    updateContent();
    initializeDiseaseFeatures();
     console.log('Current language:', i18next.language);
  });

function updateContent() {
  // Update all elements with data-i18n attribute
  document.querySelectorAll('[data-i18n]').forEach(element => {
    const key = element.getAttribute('data-i18n');
    element.innerHTML = i18next.t(key);
  });

  // Update placeholders
  document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
    const key = element.getAttribute('data-i18n-placeholder');
    element.setAttribute('placeholder', i18next.t(key));
  });
}

function initializeDiseaseFeatures() {
  const diseaseFeatures = {
    mastitis: [
      { 
        name: "temperature", 
        label: i18next.t('disease_features.mastitis.temperature'), 
        type: "number", 
        min: 37.5, 
        max: 41.0, 
        step: 0.1 
      },
      { 
        name: "hardness", 
        label: i18next.t('disease_features.mastitis.hardness'), 
        type: "select",
        options: i18next.t('disease_features.mastitis.options.hardness', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "milk_color", 
        label: i18next.t('disease_features.mastitis.milk_color'), 
        type: "select",
        options: i18next.t('disease_features.mastitis.options.milk_color', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "pain", 
        label: i18next.t('disease_features.mastitis.pain'), 
        type: "select",
        options: i18next.t('disease_features.mastitis.options.pain', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "milk_yield", 
        label: i18next.t('disease_features.mastitis.milk_yield'), 
        type: "select",
        options: i18next.t('disease_features.mastitis.options.milk_yield', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      }
    ],
    foot: [
      { 
        name: "temperature", 
        label: i18next.t('disease_features.foot.temperature'), 
        type: "number", 
        min: 37.5, 
        max: 41.0, 
        step: 0.1 
      },
      { 
        name: "milk_production", 
        label: i18next.t('disease_features.foot.milk_production'), 
        type: "select",
        options: i18next.t('disease_features.foot.options.milk_production', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "lethargy", 
        label: i18next.t('disease_features.foot.lethargy'), 
        type: "select",
        options: i18next.t('disease_features.foot.options.lethargy', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "difficulty_walking", 
        label: i18next.t('disease_features.foot.difficulty_walking'), 
        type: "select",
        options: i18next.t('disease_features.foot.options.difficulty_walking', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "foot_blister", 
        label: i18next.t('disease_features.foot.foot_blister'), 
        type: "select",
        options: i18next.t('disease_features.foot.options.foot_blister', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "foot_swelling", 
        label: i18next.t('disease_features.foot.foot_swelling'), 
        type: "select",
        options: i18next.t('disease_features.foot.options.foot_swelling', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "hoof_detachment", 
        label: i18next.t('disease_features.foot.hoof_detachment'), 
        type: "select",
        options: i18next.t('disease_features.foot.options.hoof_detachment', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      }
    ],
    mouth: [
      { 
        name: "temperature", 
        label: i18next.t('disease_features.mouth.temperature'), 
        type: "number", 
        min: 37.5, 
        max: 41.0, 
        step: 0.1 
      },
      { 
        name: "milk_production", 
        label: i18next.t('disease_features.mouth.milk_production'), 
        type: "select",
        options: i18next.t('disease_features.mouth.options.milk_production', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "lethargy", 
        label: i18next.t('disease_features.mouth.lethargy'), 
        type: "select",
        options: i18next.t('disease_features.mouth.options.lethargy', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "mouth_ulcers", 
        label: i18next.t('disease_features.mouth.mouth_ulcers'), 
        type: "select",
        options: i18next.t('disease_features.mouth.options.mouth_ulcers', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "mouth_blister", 
        label: i18next.t('disease_features.mouth.mouth_blister'), 
        type: "select",
        options: i18next.t('disease_features.mouth.options.mouth_blister', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "salivation", 
        label: i18next.t('disease_features.mouth.salivation'), 
        type: "select",
        options: i18next.t('disease_features.mouth.options.salivation', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      },
      { 
        name: "nasal_discharge", 
        label: i18next.t('disease_features.mouth.nasal_discharge'), 
        type: "select",
        options: i18next.t('disease_features.mouth.options.nasal_discharge', { returnObjects: true }).map((text, value) => ({
          value,
          text
        }))
      }
    ]
  };
  
  window.translatedDiseaseFeatures = diseaseFeatures;
}

document.getElementById('languageSwitcher').addEventListener('change', function(e) {
  i18next.changeLanguage(e.target.value, (err, t) => {
    if (err) return console.error(err);
    updateContent();
    initializeDiseaseFeatures();
    
    // If we're on step 3, re-render the input fields with new translations
    if (!document.getElementById('step3').classList.contains('hidden')) {
      renderInputFields();
    }
  });
});

i18next.on('languageChanged', () => {
  updateContent();
  initializeDiseaseFeatures();
});