document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const steps = document.querySelectorAll('.step');
    const diseaseSelect = document.getElementById('disease');
    const inputTypeSelect = document.getElementById('input-type');
    const next1Btn = document.getElementById('next1');
    const next2Btn = document.getElementById('next2');
    const prev2Btn = document.getElementById('prev2');
    const prev3Btn = document.getElementById('prev3');
    const submitBtn = document.getElementById('submit-btn');
    const restartBtn = document.getElementById('restart-btn');
    const footTextCheck = document.getElementById('foot-text');
    const mouthTextCheck = document.getElementById('mouth-text');
    const footImageCheck = document.getElementById('foot-image');
    const mouthImageCheck = document.getElementById('mouth-image');
    const resultElement = document.getElementById('result');

    let currentDisease = '';

    // Initialize
    updateDiseaseSelection();
    validateFMDInputs();

    // Event Listeners
    diseaseSelect.addEventListener('change', updateDiseaseSelection);
    next1Btn.addEventListener('click', goToStep2);
    next2Btn.addEventListener('click', goToStep3);
    prev2Btn.addEventListener('click', goToStep1);
    prev3Btn.addEventListener('click', goToStep2);
    submitBtn.addEventListener('click', handleSubmit);
    restartBtn.addEventListener('click', restartProcess);
    [footTextCheck, mouthTextCheck, footImageCheck, mouthImageCheck].forEach(checkbox => {
        checkbox.addEventListener('change', validateFMDInputs);
    });

    function updateDiseaseSelection() {
        currentDisease = diseaseSelect.value;
        document.getElementById('step2-title').textContent = `Select Input Type for ${currentDisease === 'mastitis' ? 'Mastitis' : 'FMD'}`;
        
        document.getElementById('mastitis-input').classList.toggle('hidden', currentDisease !== 'mastitis');
        document.getElementById('fmd-input').classList.toggle('hidden', currentDisease !== 'fmd');
        
        validateFMDInputs();
    }

    function validateFMDInputs() {
        if (currentDisease === 'fmd') {
            next2Btn.disabled = !(footTextCheck.checked || mouthTextCheck.checked || 
                                 footImageCheck.checked || mouthImageCheck.checked);
        }
    }

    function goToStep1() { showStep(1); }
    
    function goToStep2() { 
        if (!diseaseSelect.value) {
            alert('Please select a disease');
            return;
        }
        showStep(2); 
    }
    
    function goToStep3() { 
        if (currentDisease === 'mastitis' && !inputTypeSelect.value) {
            alert('Please select an input type');
            return;
        }
        if (currentDisease === 'fmd' && !footTextCheck.checked && !mouthTextCheck.checked && 
            !footImageCheck.checked && !mouthImageCheck.checked) {
            alert('Please select at least one input option');
            return;
        }
        updateInputFields();
        showStep(3);
    }

    function showStep(stepNumber) {
        steps.forEach((step, index) => {
            step.classList.toggle('hidden', index !== stepNumber - 1);
            step.classList.toggle('active', index === stepNumber - 1);
        });
    }

    function updateInputFields() {
        document.querySelectorAll('.input-section').forEach(section => {
            section.classList.add('hidden');
        });
        
        if (currentDisease === 'mastitis') {
            const inputType = inputTypeSelect.value;
            if (inputType === 'text' || inputType === 'both') {
                document.getElementById('mastitis-text-fields').classList.remove('hidden');
            }
            if (inputType === 'image' || inputType === 'both') {
                document.getElementById('mastitis-image-field').classList.remove('hidden');
            }
        } else {
            if (footTextCheck.checked) document.getElementById('fmd-foot-text-fields').classList.remove('hidden');
            if (mouthTextCheck.checked) document.getElementById('fmd-mouth-text-fields').classList.remove('hidden');
            if (footImageCheck.checked) document.getElementById('fmd-foot-image-field').classList.remove('hidden');
            if (mouthImageCheck.checked) document.getElementById('fmd-mouth-image-field').classList.remove('hidden');
        }
    }

    async function handleSubmit(event) {
        event.preventDefault();
        
        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing...';
        showStep(4);
        resultElement.innerHTML = '<p class="processing">Processing your request...</p>';
        
        try {
            const formData = new FormData();
            formData.append('disease', currentDisease);
            
            if (currentDisease === 'mastitis') {
                const inputType = inputTypeSelect.value;
                formData.append('inputType', inputType);
                
                if (inputType === 'text' || inputType === 'both') {
                    ['temperature', 'hardness', 'pain', 'milk_yield', 'milk_color'].forEach(id => {
                        const value = document.getElementById(id).value;
                        formData.append(id, value !== '' ? value : '0');
                    });
                }
                
                if (inputType === 'image' || inputType === 'both') {
                    const fileInput = document.getElementById('mastitis-image');
                    if (fileInput.files.length > 0) {
                        formData.append('image', fileInput.files[0]);
                    }
                }
            } else {
                if (footTextCheck.checked) {
                    formData.append('foot_text', 'true');
                    ['ft_temp', 'ft_milk', 'ft_lethargy', 'ft_walk', 'ft_blister', 'ft_swelling', 'ft_hoof'].forEach(id => {
                        const value = document.getElementById(id).value;
                        formData.append(id, value !== '' ? value : '0');
                    });
                }
                
                if (mouthTextCheck.checked) {
                    formData.append('mouth_text', 'true');
                    ['mt_temp', 'mt_milk', 'mt_lethargy', 'mt_ulcers', 'mt_blister', 'mt_salivation', 'mt_discharge'].forEach(id => {
                        const value = document.getElementById(id).value;
                        formData.append(id, value !== '' ? value : '0');
                    });
                }
                
                if (footImageCheck.checked) {
                    const fileInput = document.getElementById('foot-image');
                    if (fileInput.files.length > 0) {
                        formData.append('foot_image', 'true');
                        formData.append('foot_image_file', fileInput.files[0]);
                    }
                }
                
                if (mouthImageCheck.checked) {
                    const fileInput = document.getElementById('mouth-image');
                    if (fileInput.files.length > 0) {
                        formData.append('mouth_image', 'true');
                        formData.append('mouth_image_file', fileInput.files[0]);
                    }
                }
            }
            
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}`);
            }
            
            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error:', error);
            displayError(error.message || 'Failed to process request');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Detect Disease';
        }
    }

    function displayResults(data) {
        resultElement.innerHTML = '';
        
        if (!data) {
            displayError('No response received from server');
            return;
        }

        if (data.status === 'success') {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result-content';
            
            const title = document.createElement('h3');
            title.textContent = data.result || 'Detection Results';
            resultDiv.appendChild(title);
            
            if (data.details) {
                const detailsDiv = document.createElement('div');
                detailsDiv.className = 'result-details';
                
                for (const [key, detail] of Object.entries(data.details)) {
                    if (!detail) continue;
                    
                    const section = document.createElement('div');
                    section.className = 'result-section';
                    
                    const sectionTitle = document.createElement('h4');
                    sectionTitle.textContent = key.replace('_', ' ').toUpperCase();
                    section.appendChild(sectionTitle);
                    
                    if (detail.error) {
                        const errorPara = document.createElement('p');
                        errorPara.className = 'error-text';
                        errorPara.textContent = `Error: ${detail.error}`;
                        section.appendChild(errorPara);
                    } else if (detail.result) {
                        const resultPara = document.createElement('p');
                        resultPara.className = 'result-text';
                        resultPara.textContent = `Result: ${detail.result}`;
                        section.appendChild(resultPara);
                        
                        if (detail.features) {
                            const featuresPara = document.createElement('p');
                            featuresPara.textContent = `Features: ${JSON.stringify(detail.features)}`;
                            section.appendChild(featuresPara);
                        }
                        
                        if (detail.filename) {
                            const filePara = document.createElement('p');
                            filePara.textContent = `File: ${detail.filename}`;
                            section.appendChild(filePara);
                        }
                    }
                    
                    detailsDiv.appendChild(section);
                }
                
                resultDiv.appendChild(detailsDiv);
            }
            
            resultElement.appendChild(resultDiv);
        } else {
            displayError(data.message || 'Unknown error occurred');
        }
    }

    function displayError(message) {
        resultElement.innerHTML = '';
        
        const errorDiv = document.createElement('div');
        
        const title = document.createElement('h3');
        title.className = 'error-title';
        title.textContent = 'Error';
        errorDiv.appendChild(title);
        
        const errorMsg = document.createElement('p');
        errorMsg.className = 'error-text';
        errorMsg.textContent = message;
        errorDiv.appendChild(errorMsg);
        
        resultElement.appendChild(errorDiv);
    }

    function restartProcess() {
        // Reset form
        diseaseSelect.value = '';
        inputTypeSelect.value = '';
        document.querySelectorAll('input[type="number"]').forEach(input => input.value = '');
        document.querySelectorAll('input[type="file"]').forEach(input => input.value = '');
        [footTextCheck, mouthTextCheck, footImageCheck, mouthImageCheck].forEach(cb => cb.checked = false);
        
        // Reset UI
        currentDisease = '';
        showStep(1);
    }
});