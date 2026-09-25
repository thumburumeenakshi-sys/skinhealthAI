document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const imageInput = document.getElementById('image-input');
    const cameraBtn = document.getElementById('camera-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const previewName = document.getElementById('preview-name');
    const previewMeta = document.getElementById('preview-meta');
    const validationTags = document.getElementById('validation-tags');
    const loadingBox = document.getElementById('loading-box');
    const loadingStepText = document.getElementById('loading-step-text');
    const resultsSection = document.getElementById('results-section');
    const cameraModal = document.getElementById('camera-modal');
    const closeModalBtn = document.getElementById('close-modal');
    const cameraStream = document.getElementById('camera-stream');
    const capturePhotoBtn = document.getElementById('capture-photo-btn');
    const historyContainer = document.getElementById('history-table-body');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const diseaseTabsContainer = document.getElementById('disease-tabs-container');
    const diseaseDetailContent = document.getElementById('disease-detail-content');

    let selectedFile = null;
    let cameraMediaStream = null;
    let globalClassesData = {};

    // Load initial model info and history
    fetchModelInfo();
    fetchHistory();

    // --- Drag and Drop Handlers ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    dropzone.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // --- Image Selection & Validation ---
    function handleFileSelect(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            alert('Unsupported file format! Please upload JPG, PNG, or WEBP images.');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert('File size exceeds 10 MB limit!');
            return;
        }

        selectedFile = file;
        previewName.textContent = file.name;
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        previewMeta.textContent = `Size: ${sizeMB} MB | Type: ${file.type.split('/')[1].toUpperCase()}`;

        // Local Image Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            
            const img = new Image();
            img.onload = () => {
                validationTags.innerHTML = '';
                const resTag = document.createElement('span');
                resTag.className = 'tag tag-success';
                resTag.textContent = `Resolution: ${img.width}x${img.height}px`;
                validationTags.appendChild(resTag);

                const sizeTag = document.createElement('span');
                sizeTag.className = 'tag tag-success';
                sizeTag.textContent = `Size: ${sizeMB} MB OK`;
                validationTags.appendChild(sizeTag);

                if (img.width < 100 || img.height < 100) {
                    const warnTag = document.createElement('span');
                    warnTag.className = 'tag tag-warning';
                    warnTag.textContent = 'Low Resolution Warning (<100px)';
                    validationTags.appendChild(warnTag);
                }
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);

        previewContainer.style.display = 'block';
        analyzeBtn.disabled = false;
        resultsSection.style.display = 'none';
        
        previewContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // --- Camera Modal Handlers ---
    cameraBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
            cameraMediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            cameraStream.srcObject = cameraMediaStream;
            cameraModal.style.display = 'flex';
        } catch (err) {
            alert('Unable to access camera: ' + err.message + '\nPlease check camera permissions or upload an image file.');
        }
    });

    closeModalBtn.addEventListener('click', closeCamera);
    
    function closeCamera() {
        if (cameraMediaStream) {
            cameraMediaStream.getTracks().forEach(track => track.stop());
        }
        cameraModal.style.display = 'none';
    }

    capturePhotoBtn.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = cameraStream.videoWidth || 640;
        canvas.height = cameraStream.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob((blob) => {
            const capturedFile = new File([blob], `captured_lesion_${Date.now()}.jpg`, { type: 'image/jpeg' });
            handleFileSelect(capturedFile);
            closeCamera();
        }, 'image/jpeg', 0.95);
    });

    // --- Analyze Button Click ---
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        loadingBox.style.display = 'block';
        resultsSection.style.display = 'none';

        const steps = [
            "Validating image resolution, file size, and Laplacian blur score...",
            "Applying OpenCV LAB color space CLAHE contrast enhancement...",
            "Running MobileNetV2 Deep Neural Network forward pass...",
            "Computing ranked Softmax class probabilities and medical guidance..."
        ];

        let stepIndex = 0;
        loadingStepText.textContent = steps[0];
        const stepInterval = setInterval(() => {
            stepIndex = (stepIndex + 1) % steps.length;
            loadingStepText.textContent = steps[stepIndex];
        }, 700);

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            clearInterval(stepInterval);
            loadingBox.style.display = 'none';
            analyzeBtn.disabled = false;

            const data = await response.json();
            if (data.success) {
                renderResults(data);
                fetchHistory();
            } else {
                alert('Prediction Error: ' + (data.errors ? data.errors.join('\n') : 'Unknown error'));
            }
        } catch (err) {
            clearInterval(stepInterval);
            loadingBox.style.display = 'none';
            analyzeBtn.disabled = false;
            alert('Server connection error: ' + err.message);
        }
    });

    // --- Render Prediction Results ---
    function renderResults(data) {
        const top = data.top_prediction;
        
        document.getElementById('res-top-name').textContent = top.name;
        document.getElementById('res-top-category').textContent = top.category;
        
        const riskBadge = document.getElementById('res-risk-badge');
        riskBadge.textContent = `Risk Level: ${top.risk_level}`;
        riskBadge.className = 'risk-badge ';
        if (top.risk_level.toLowerCase().includes('low')) {
            riskBadge.classList.add('risk-low');
        } else if (top.risk_level.toLowerCase().includes('moderate')) {
            riskBadge.classList.add('risk-moderate');
        } else {
            riskBadge.classList.add('risk-high');
        }

        const confValueEl = document.getElementById('res-confidence-val');
        confValueEl.textContent = `${top.confidence.toFixed(1)}%`;
        
        const confCircle = document.getElementById('res-confidence-circle');
        confCircle.style.background = `radial-gradient(circle, #ffffff 60%, transparent 61%), conic-gradient(var(--primary) ${top.confidence * 3.6}deg, #e2e8f0 0deg)`;

        const alertLowConf = document.getElementById('alert-low-confidence');
        if (data.is_low_confidence) {
            document.getElementById('low-conf-notice-text').textContent = data.low_confidence_notice;
            alertLowConf.style.display = 'flex';
        } else {
            alertLowConf.style.display = 'none';
        }

        const rankedContainer = document.getElementById('ranked-list-container');
        rankedContainer.innerHTML = '';

        data.ranked_conditions.forEach((item, index) => {
            const itemEl = document.createElement('div');
            itemEl.className = 'ranked-item';
            
            itemEl.innerHTML = `
                <div class="ranked-header">
                    <span><strong>${index + 1}. ${item.name}</strong> (${item.code})</span>
                    <span style="color: var(--primary); font-weight: 700;">${item.confidence.toFixed(1)}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: ${item.confidence}%;"></div>
                </div>
            `;
            rankedContainer.appendChild(itemEl);
        });

        document.getElementById('res-desc-text').textContent = top.description;
        document.getElementById('res-guidance-text').textContent = top.guidance;
        document.getElementById('res-derm-rec-text').textContent = top.dermatologist_recommendation;

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // --- Render Interactive Diseases Explanation Guide ---
    function renderDiseasesGuide(classesMap) {
        globalClassesData = classesMap;
        if (!diseaseTabsContainer || !diseaseDetailContent) return;

        diseaseTabsContainer.innerHTML = '';
        const codes = Object.keys(classesMap);

        codes.forEach((code, index) => {
            const info = classesMap[code];
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = `disease-tab-btn ${index === 0 ? 'active' : ''}`;
            btn.textContent = `${info.code} - ${info.name.split('/')[0].trim()}`;
            btn.onclick = () => selectDiseaseTab(code);
            diseaseTabsContainer.appendChild(btn);
        });

        if (codes.length > 0) {
            selectDiseaseTab(codes[0]);
        }
    }

    function selectDiseaseTab(code) {
        const info = globalClassesData[code];
        if (!info) return;

        const buttons = diseaseTabsContainer.querySelectorAll('.disease-tab-btn');
        buttons.forEach(btn => {
            if (btn.textContent.includes(code)) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        let riskClass = 'risk-low';
        if (info.risk_level.toLowerCase().includes('moderate')) riskClass = 'risk-moderate';
        if (info.risk_level.toLowerCase().includes('high') || info.risk_level.toLowerCase().includes('critical')) riskClass = 'risk-high';

        diseaseDetailContent.innerHTML = `
            <div class="disease-header">
                <div class="disease-title-area">
                    <h3>${info.name} <span class="disease-code-tag">${info.code}</span></h3>
                    <p style="color: var(--text-muted); font-size: 0.95rem; margin-top: 4px;"><strong>Category:</strong> ${info.category || 'Dermatological Condition'}</p>
                </div>
                <span class="risk-badge ${riskClass}">Risk Level: ${info.risk_level}</span>
            </div>

            <div style="background: #f8fafc; padding: 18px; border-radius: 10px; border: 1px solid var(--border-color); margin-bottom: 20px;">
                <h4 style="color: var(--primary); margin-bottom: 6px;">📖 Clinical Description & Pathophysiology</h4>
                <p style="color: var(--text-secondary); font-size: 0.95rem;">${info.description}</p>
            </div>

            <div class="disease-subgrid">
                <div class="sub-card">
                    <h4>💡 Precautionary Measures & Skincare</h4>
                    <p style="font-size: 0.92rem; color: var(--text-secondary);">${info.guidance}</p>
                </div>
                <div class="sub-card">
                    <h4>🩺 Dermatologist Recommendation</h4>
                    <p style="font-size: 0.92rem; color: var(--text-secondary);">${info.dermatologist_recommendation}</p>
                </div>
            </div>
        `;
    }

    // --- Fetch History ---
    async function fetchHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();
            
            if (data.success && historyContainer) {
                historyContainer.innerHTML = '';
                if (data.history.length === 0) {
                    historyContainer.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 24px;">No saved prediction history yet.</td></tr>`;
                    return;
                }

                data.history.forEach(rec => {
                    const row = document.createElement('tr');
                    const isLow = rec.is_low_confidence ? ' <span style="color: var(--accent-amber); font-weight:700;">(Low Conf)</span>' : '';
                    const dbSource = rec.source || 'SQLite';
                    row.innerHTML = `
                        <td><img src="${rec.image_path}" class="history-thumb" alt="lesion"></td>
                        <td><strong>${rec.predicted_name}</strong>${isLow}</td>
                        <td><span style="color: var(--primary); font-weight: 700;">${rec.confidence.toFixed(1)}%</span></td>
                        <td>${rec.timestamp}</td>
                        <td><span style="font-size: 0.8rem; background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-weight:600;">${dbSource}</span></td>
                        <td>
                            <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 0.8rem;" onclick="deleteRecord(${rec.id})">Delete</button>
                        </td>
                    `;
                    historyContainer.appendChild(row);
                });
            }
        } catch (err) {
            console.error('Error loading history:', err);
        }
    }

    window.deleteRecord = async (id) => {
        try {
            const res = await fetch('/api/history/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ record_id: id })
            });
            const data = await res.json();
            if (data.success) fetchHistory();
        } catch (e) {
            console.error(e);
        }
    };

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', async () => {
            if (confirm('Are you sure you want to clear all prediction history?')) {
                await fetch('/api/history/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ record_id: 'all' })
                });
                fetchHistory();
            }
        });
    }

    // --- Fetch Model Info & Render Diseases Guide ---
    async function fetchModelInfo() {
        try {
            const res = await fetch('/api/model-info');
            const data = await res.json();
            if (data.success) {
                if (data.classes) {
                    renderDiseasesGuide(data.classes);
                }
                if (data.evaluation_metrics && data.evaluation_metrics.accuracy !== undefined) {
                    const m = data.evaluation_metrics;
                    document.getElementById('metric-accuracy').textContent = `${(m.accuracy * 100).toFixed(1)}%`;
                    document.getElementById('metric-precision').textContent = `${(m.precision_macro * 100).toFixed(1)}%`;
                    document.getElementById('metric-recall').textContent = `${(m.recall_macro * 100).toFixed(1)}%`;
                    document.getElementById('metric-f1').textContent = `${(m.f1_score_macro * 100).toFixed(1)}%`;
                }
            }
        } catch (e) {
            console.log('Model info fetch error:', e);
        }
    }
});
