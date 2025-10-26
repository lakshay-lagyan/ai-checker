        // Configuration
        const API_URL = 'https://ai-text-detector-xris.onrender.com';
        
        // State
        let sessionHistory = [];
        let currentFile = null;
        let sessionStats = {
            totalDetections: 0,
            totalConfidence: 0
        };

        // DOM Elements
        const textInput = document.getElementById('textInput');
        const charCount = document.getElementById('charCount');
        const analyzeTextBtn = document.getElementById('analyzeTextBtn');
        const analyzeTextBtnText = document.getElementById('analyzeTextBtnText');
        const analyzeTextBtnSpinner = document.getElementById('analyzeTextBtnSpinner');
        
        const fileUploadArea = document.getElementById('fileUploadArea');
        const fileInput = document.getElementById('fileInput');
        const selectedFile = document.getElementById('selectedFile');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const removeFile = document.getElementById('removeFile');
        const analyzeFileBtn = document.getElementById('analyzeFileBtn');
        const analyzeFileBtnText = document.getElementById('analyzeFileBtnText');
        const analyzeFileBtnSpinner = document.getElementById('analyzeFileBtnSpinner');
        
        const alertBox = document.getElementById('alertBox');
        const alertMessage = document.getElementById('alertMessage');
        
        const resultsCard = document.getElementById('resultsCard');
        const verdictCard = document.getElementById('verdictCard');
        const verdictIcon = document.getElementById('verdictIcon');
        const verdictTitle = document.getElementById('verdictTitle');
        const verdictConfidence = document.getElementById('verdictConfidence');
        
        const aiProbValue = document.getElementById('aiProbValue');
        const aiProbBar = document.getElementById('aiProbBar');
        const humanProbValue = document.getElementById('humanProbValue');
        const humanProbBar = document.getElementById('humanProbBar');
        
        const detectionMethods = document.getElementById('detectionMethods');
        const traitsList = document.getElementById('traitsList');
        const detailedAnalysis = document.getElementById('detailedAnalysis');
        
        const historyList = document.getElementById('historyList');
        const sessionDetections = document.getElementById('sessionDetections');
        const avgConfidence = document.getElementById('avgConfidence');
        const totalDetections = document.getElementById('totalDetections');

        // Tab Switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.dataset.tab;
                
                // Update buttons
                document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
                button.classList.add('active');
                
                // Update content
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.querySelector(`[data-content="${tabName}"]`).classList.add('active');
                
                // Hide results when switching tabs
                resultsCard.classList.remove('show');
                hideAlert();
            });
        });

        // Text Input Handler
        textInput.addEventListener('input', () => {
            const text = textInput.value.trim();
            const words = text.split(/\s+/).filter(w => w.length > 0).length;
            
            charCount.textContent = `${text.length} characters • ${words} words`;
            
            analyzeTextBtn.disabled = text.length < 50;
        });

        // File Upload Handlers
        fileUploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadArea.classList.add('drag-over');
        });

        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.classList.remove('drag-over');
        });

        fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadArea.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        removeFile.addEventListener('click', () => {
            currentFile = null;
            selectedFile.classList.remove('show');
            fileInput.value = '';
            analyzeFileBtn.disabled = true;
        });

        function handleFileSelect(file) {
            const validTypes = ['.txt', '.pdf', '.doc', '.docx'];
            const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
            
            if (!validTypes.includes(fileExtension)) {
                showAlert('Invalid file type. Please upload TXT, PDF, DOC, or DOCX files.', 'error');
                return;
            }
            
            if (file.size > 10 * 1024 * 1024) {
                showAlert('File too large. Maximum size is 10MB.', 'error');
                return;
            }
            
            currentFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = formatFileSize(file.size);
            selectedFile.classList.add('show');
            analyzeFileBtn.disabled = false;
            hideAlert();
        }

        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }

        // Analyze Text
        analyzeTextBtn.addEventListener('click', async () => {
            const text = textInput.value.trim();
            
            if (text.length < 50) {
                showAlert('Text must be at least 50 characters long.', 'error');
                return;
            }
            
            await analyzeText(text, 'text');
        });

        // Analyze File
        analyzeFileBtn.addEventListener('click', async () => {
            if (!currentFile) {
                showAlert('Please select a file first.', 'error');
                return;
            }
            
            await analyzeFile(currentFile);
        });

        // API Call - Analyze Text
        async function analyzeText(text, source) {
            setLoading(true, source);
            hideAlert();
            
            try {
                const formData = new FormData();
                formData.append('text', text);
                formData.append('session_id', getSessionId());
                
                const response = await fetch(`${API_URL}/detect/text`, {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                    addToHistory(data, 'text');
                    updateStats();
                } else {
                    showAlert(data.detail || 'Analysis failed. Please try again.', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showAlert('Network error. ', 'error');
            } finally {
                setLoading(false, source);
            }
        }

        // API Call - Analyze File
        async function analyzeFile(file) {
            setLoading(true, 'file');
            hideAlert();
            
            try {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('session_id', getSessionId());
                
                const response = await fetch(`${API_URL}/detect/file`, {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                    addToHistory(data, 'file', file.name);
                    updateStats();
                } else {
                    showAlert(data.detail || 'File analysis failed. Please try again.', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showAlert('Network error.', 'error');
            } finally {
                setLoading(false, 'file');
            }
        }

        // Display Results
        function displayResults(data) {
            // Show results card
            resultsCard.classList.add('show');
            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
            
            // Verdict
            const verdict = data.verdict;
            let verdictClass = '';
            let icon = '';
            
            if (verdict === 'AI-Generated') {
                verdictClass = 'ai-generated';
                icon = '❌';
            } else if (verdict === 'Human-Written') {
                verdictClass = 'human-written';
                icon = '✅';
            } else {
                verdictClass = 'uncertain';
                icon = '⚠️';
            }
            
            verdictCard.className = `verdict-card ${verdictClass}`;
            verdictIcon.textContent = icon;
            verdictTitle.textContent = verdict;
            verdictConfidence.textContent = `Confidence: ${data.confidence_score.toFixed(1)}%`;
            
            // Probabilities
            const aiProb = (data.ai_probability * 100).toFixed(1);
            const humanProb = (data.human_probability * 100).toFixed(1);
            
            aiProbValue.textContent = `${aiProb}%`;
            aiProbBar.style.width = `${aiProb}%`;
            
            humanProbValue.textContent = `${humanProb}%`;
            humanProbBar.style.width = `${humanProb}%`;
            
            // Detection Methods
            detectionMethods.innerHTML = '';
            data.detection_methods.forEach(method => {
                const badge = document.createElement('div');
                badge.className = 'method-badge';
                badge.innerHTML = `<span>✓</span> ${method.replace(/_/g, ' ').toUpperCase()}`;
                detectionMethods.appendChild(badge);
            });
            
            // Traits
            traitsList.innerHTML = '';
            if (data.traits_detected && data.traits_detected.length > 0) {
                data.traits_detected.forEach(trait => {
                    const li = document.createElement('li');
                    li.className = 'trait-item';
                    li.textContent = trait;
                    traitsList.appendChild(li);
                });
            } else {
                traitsList.innerHTML = '<li class="trait-item">Standard analysis completed</li>';
            }
            
            // Detailed Analysis
            detailedAnalysis.innerHTML = '';
            if (data.analysis_details) {
                Object.entries(data.analysis_details).forEach(([key, value]) => {
                    const detail = document.createElement('div');
                    detail.className = 'analysis-detail';
                    detail.innerHTML = `<span class="analysis-key">${key.replace(/_/g, ' ').toUpperCase()}:</span> ${JSON.stringify(value, null, 2)}`;
                    detailedAnalysis.appendChild(detail);
                });
            }
        }

        // History Management
        function addToHistory(data, type, filename = null) {
            const historyItem = {
                verdict: data.verdict,
                confidence: data.confidence_score,
                timestamp: new Date(),
                type: type,
                filename: filename,
                data: data
            };
            
            sessionHistory.unshift(historyItem);
            
            if (sessionHistory.length > 10) {
                sessionHistory.pop();
            }
            
            updateHistoryDisplay();
        }

        function updateHistoryDisplay() {
            if (sessionHistory.length === 0) {
                historyList.innerHTML = '<li style="text-align: center; color: var(--gray); padding: 2rem;">No analyses yet</li>';
                return;
            }
            
            historyList.innerHTML = '';
            
            sessionHistory.slice(0, 5).forEach((item, index) => {
                const li = document.createElement('li');
                li.className = 'history-item';
                li.onclick = () => displayResults(item.data);
                
                let verdictColor = '';
                if (item.verdict === 'AI-Generated') verdictColor = 'var(--danger)';
                else if (item.verdict === 'Human-Written') verdictColor = 'var(--success)';
                else verdictColor = 'var(--warning)';
                
                li.innerHTML = `
                    <div class="history-header">
                        <span class="history-verdict" style="color: ${verdictColor}">${item.verdict}</span>
                        <span class="history-confidence">${item.confidence.toFixed(1)}%</span>
                    </div>
                    <div class="history-timestamp">
                        ${item.type === 'file' ? `📄 ${item.filename}` : '📝 Text input'} • ${formatTimestamp(item.timestamp)}
                    </div>
                `;
                
                historyList.appendChild(li);
            });
        }

        function formatTimestamp(date) {
            const now = new Date();
            const diff = Math.floor((now - date) / 1000);
            
            if (diff < 60) return 'Just now';
            if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
            if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
            return date.toLocaleDateString();
        }

        // Stats Management
        function updateStats() {
            sessionStats.totalDetections++;
            sessionStats.totalConfidence += sessionHistory[0].confidence;
            
            sessionDetections.textContent = sessionStats.totalDetections;
            totalDetections.textContent = sessionStats.totalDetections;
            
            const avgConf = sessionStats.totalConfidence / sessionStats.totalDetections;
            avgConfidence.textContent = `${avgConf.toFixed(1)}%`;
        }

        // Utility Functions
        function setLoading(isLoading, source) {
            if (source === 'text') {
                analyzeTextBtn.disabled = isLoading;
                analyzeTextBtnText.style.display = isLoading ? 'none' : 'inline';
                analyzeTextBtnSpinner.style.display = isLoading ? 'inline-block' : 'none';
            } else if (source === 'file') {
                analyzeFileBtn.disabled = isLoading;
                analyzeFileBtnText.style.display = isLoading ? 'none' : 'inline';
                analyzeFileBtnSpinner.style.display = isLoading ? 'inline-block' : 'none';
            }
        }

        function showAlert(message, type) {
            alertMessage.textContent = message;
            alertBox.className = `alert alert-${type} show`;
            
            setTimeout(() => {
                hideAlert();
            }, 5000);
        }

        function hideAlert() {
            alertBox.classList.remove('show');
        }

        function getSessionId() {
            let sessionId = localStorage.getItem('ai_detector_session_id');
            if (!sessionId) {
                sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                localStorage.setItem('ai_detector_session_id', sessionId);
            }
            return sessionId;
        }

        function toggleAccordion(header) {
            header.classList.toggle('active');
            const content = header.nextElementSibling;
            content.classList.toggle('show');
        }

        // Sample Texts for Demo
        const sampleTexts = {
            ai: "It is important to note that artificial intelligence has become increasingly prevalent in modern society. Furthermore, machine learning algorithms have demonstrated remarkable capabilities across various domains. Consequently, the implications of these technological advancements warrant careful consideration. Moreover, the ethical dimensions of AI deployment require thorough examination. Therefore, stakeholders must engage in meaningful dialogue to ensure responsible development.",
            human: "I've been thinking a lot about AI lately. Like, it's everywhere now, you know? My phone has it, my car has it, even my fridge apparently has some kind of AI. It's pretty wild when you think about it. I'm not sure if it's all good though. What happens when these systems make mistakes? Who's responsible then? These are the questions that keep me up at night, honestly."
        };

        // Demo functionality
        window.loadSampleAI = () => {
            textInput.value = sampleTexts.ai;
            textInput.dispatchEvent(new Event('input'));
            document.querySelector('[data-tab="text"]').click();
        };

        window.loadSampleHuman = () => {
            textInput.value = sampleTexts.human;
            textInput.dispatchEvent(new Event('input'));
            document.querySelector('[data-tab="text"]').click();
        };

        // Initialize
        console.log('AI Text Detector initialized');
        console.log(' API URL:', API_URL);
        console.log('Tip: Load sample texts by calling loadSampleAI() or loadSampleHuman() in console');
        
        // Check API connection on load
        fetch(`${API_URL}/health`)
            .then(response => response.json())
            .then(data => {
                console.log(' API connection successful:', data);
            })
            .catch(error => {
                console.warn('⚠️ API not reachable. Make sure backend is running on', API_URL);
                showAlert('Backend API not reachable. Please start the backend server.', 'error');
            });
    