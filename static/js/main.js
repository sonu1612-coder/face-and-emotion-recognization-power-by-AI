document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    // Tabs
    const navBtns = document.querySelectorAll('.btn-nav');
    const tabs = document.querySelectorAll('.tab-pane');
    const topbarTitle = document.getElementById('topbarTitle');

    // Camera
    const cameraSelect = document.getElementById('cameraSelect');
    const videoFeed = document.getElementById('videoFeed');

    // Tab 1: Recognize
    const btnRecognize = document.getElementById('btnRecognize');
    const recognitionResult = document.getElementById('recognitionResult');

    // Tab 2: Capture
    const btnCaptureStart = document.getElementById('btnCaptureStart');
    const captureName = document.getElementById('captureName');
    const captureShots = document.getElementById('captureShots');
    const captureStatus = document.getElementById('captureStatus');

    // Tab 3: Database Gallery
    const galleryContainer = document.getElementById('galleryContainer');
    const btnRefreshLogs = document.getElementById('btnRefreshLogs');
    const btnWipeDB = document.getElementById('btnWipeDB');

    // Navigation Logic
    navBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            navBtns.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            
            tabs.forEach(t => t.classList.add('hidden'));
            const targetId = e.target.getAttribute('data-tab');
            document.getElementById(targetId).classList.remove('hidden');

            if(targetId === 'tab-recognize') { 
                topbarTitle.innerText = "Live AI Recon"; 
                const wrapper = document.getElementById('recognizeVideoWrapper');
                if(wrapper) wrapper.appendChild(videoFeed);
            }
            if(targetId === 'tab-capture') { 
                topbarTitle.innerText = "Data Collection Wizard"; 
                const wrapper = document.getElementById('captureVideoWrapper');
                if(wrapper) wrapper.appendChild(videoFeed);
            }
            if(targetId === 'tab-database') { 
                topbarTitle.innerText = "Identity Database"; 
                fetchGallery(); // Auto fetch
            }
        });
    });

    // Camera Switch Logic
    cameraSelect.addEventListener('change', async (e) => {
        const camId = e.target.value;
        try {
            const res = await fetch('/api/camera', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ camera_index: camId })
            });
            const data = await res.json();
            if(data.success) {
                videoFeed.src = `/video_feed?t=${new Date().getTime()}`;
            }
        } catch (err) { console.error("Failed to switch camera", err); }
    });

    // Camera Toggle Logic
    const btnStreamToggle = document.getElementById('btnStreamToggle');
    let isCameraOn = true;
    if (btnStreamToggle) {
        btnStreamToggle.addEventListener('click', async () => {
            const icon = btnStreamToggle.querySelector('i');
            const toggleText = document.getElementById('streamToggleText');
            if (isCameraOn) {
                // Turn OFF
                try {
                    await fetch('/api/camera/stop', { method: 'POST' });
                    videoFeed.src = 'data:image/svg+xml;charset=UTF-8,%3Csvg%20width%3D%22640%22%20height%3D%22480%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Crect%20width%3D%22100%25%22%20height%3D%22100%25%22%20fill%3D%22%23050c17%22%2F%3E%3Ctext%20x%3D%2250%25%22%20y%3D%2250%25%22%20font-family%3D%22Orbitron%2Csans-serif%22%20font-size%3D%2224%22%20font-weight%3D%22bold%22%20fill%3D%22%23555%22%20dominant-baseline%3D%22middle%22%20text-anchor%3D%22middle%22%3ECAMERA%20OFFLINE%3C%2Ftext%3E%3C%2Fsvg%3E';
                    isCameraOn = false;
                    icon.className = 'fa-solid fa-video';
                    icon.style.color = '#00e5ff';
                    toggleText.innerText = 'Turn Camera On';
                    btnStreamToggle.style.borderColor = 'rgba(0,229,255,0.5)';
                } catch (e) { console.error(e); }
            } else {
                // Turn ON
                try {
                    await fetch('/api/camera/start', { method: 'POST' });
                    videoFeed.src = `/video_feed?t=${new Date().getTime()}`;
                    isCameraOn = true;
                    icon.className = 'fa-solid fa-video-slash';
                    icon.style.color = '#ea2e33';
                    toggleText.innerText = 'Turn Camera Off';
                    btnStreamToggle.style.borderColor = 'rgba(234,46,51,0.5)';
                } catch (e) { console.error(e); }
            }
        });
    }

    // Recognize On-Demand
    btnRecognize.addEventListener('click', async () => {
        btnRecognize.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
        btnRecognize.disabled = true;
        
        try {
            const res = await fetch('/api/recognize', { method: 'POST' });
            const data = await res.json();
            
            if(data.success && data.faces && data.faces.length > 0) {
                const f = data.faces[0]; // Take largest face
                recognitionResult.innerHTML = `
                    <h3>${f.identity}</h3>
                    <p>Emotion: <strong>${f.emotion}</strong> | Mask: <strong>${f.mask}</strong></p>
                `;
            } else {
                recognitionResult.innerHTML = `<h3>No Face Detected</h3><p>Please adjust your position and lighting.</p>`;
            }
            recognitionResult.classList.remove('hidden');
        } catch (e) {
            recognitionResult.innerHTML = `<h3 style="color:var(--danger)">Network Error</h3>`;
        }
        
        // Force reflow to replay CSS animation
        recognitionResult.classList.add('hidden');
        void recognitionResult.offsetWidth; 
        recognitionResult.classList.remove('hidden');
        
        btnRecognize.innerHTML = '<i class="fa-solid fa-brain"></i> Recognize Face Now';
        btnRecognize.disabled = false;
    });

    // Chatbot AI Logic
    const btnSendChat = document.getElementById('btnSendChat');
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');

    async function sendChatMessage() {
        const text = chatInput.value.trim();
        if(!text) return;

        // Append user msg
        const userDiv = document.createElement('div');
        userDiv.className = 'message user-message';
        userDiv.innerHTML = `<div class="msg-bubble">${text}</div>`;
        chatMessages.appendChild(userDiv);
        chatInput.value = '';
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Show typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai-message loading-msg';
        typingDiv.innerHTML = `<div class="msg-bubble"><i class="fa-solid fa-ellipsis fa-fade"></i></div>`;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            
            chatMessages.removeChild(typingDiv); // remove typing indicator

            const aiDiv = document.createElement('div');
            aiDiv.className = 'message ai-message';
            aiDiv.innerHTML = `
                <div class="msg-bubble">${data.reply || "I didn't quite catch that."}</div>
                <div class="msg-caption"><i class="fa-solid fa-face-smile"></i> Sensing: ${data.emotion_detected}</div>
            `;
            chatMessages.appendChild(aiDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;

        } catch (err) {
            chatMessages.removeChild(typingDiv);
            const errDiv = document.createElement('div');
            errDiv.className = 'message ai-message';
            errDiv.innerHTML = `<div class="msg-bubble" style="color: var(--danger);">Network error contacting AI.</div>`;
            chatMessages.appendChild(errDiv);
        }
    }

    btnSendChat.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', (e) => {
        if(e.key === 'Enter') sendChatMessage();
    });

    // Capture Data Collection
    btnCaptureStart.addEventListener('click', async () => {
        const nameVal = captureName.value.trim();
        if(!nameVal) {
            captureStatus.style.color = 'var(--danger)';
            captureStatus.innerText = 'Please enter an identity name!';
            return;
        }
        
        btnCaptureStart.disabled = true;
        captureStatus.style.color = 'var(--accent)';
        captureStatus.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Bursting... Please look directly at the camera.';
        
        try {
            const emotionVal = document.getElementById('captureEmotion').value;
            const res = await fetch('/api/capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ identity: nameVal, shots: captureShots.value, emotion: emotionVal })
            });
            const data = await res.json();
            
            if(data.success && data.saved > 0) {
                captureStatus.style.color = 'var(--success)';
                captureStatus.innerText = `Success! Saved ${data.saved} image(s) linked to '${nameVal}'. Model retrained.`;
                captureName.value = '';
            } else {
                captureStatus.style.color = 'var(--danger)';
                captureStatus.innerText = 'Failed. No face was visibly detected during sequence.';
            }
        } catch (e) {
            captureStatus.style.color = 'var(--danger)';
            captureStatus.innerText = 'Network error occurred.';
        }
        btnCaptureStart.disabled = false;
    });

    // Database Gallery fetch & render
    btnRefreshLogs.addEventListener('click', fetchGallery);

    if(btnWipeDB) {
        btnWipeDB.addEventListener('click', async () => {
            const pass = confirm("WARNING: This will permanently delete ALL face records and images from the server. Proceed?");
            if(!pass) return;
            
            try {
                await fetch('/api/database/wipe', { method: 'POST' });
                alert("Database Wiped Completed.");
                fetchGallery();
            } catch (err) {
                console.error(err);
                alert("Error wiping database.");
            }
        });
    }

    async function fetchGallery() {
        galleryContainer.innerHTML = '<p class="loading-text text-center" style="grid-column: 1/-1;">Loading gallery...</p>';
        try {
            const res = await fetch('/api/history');
            const data = await res.json();
            
            galleryContainer.innerHTML = '';
            
            if(data.records && data.records.length > 0) {
                data.records.forEach(r => {
                    const el = document.createElement('div');
                    el.className = 'gallery-item';
                    
                    // Decode URL paths safely back to front end string
                    let decPath = decodeURIComponent(r.image_path);
                    const webPath = decPath.replace(/\\/g, '/');
                    
                    el.innerHTML = `
                        <img src="/${webPath}" class="gallery-img" onerror="this.src='https://via.placeholder.com/150?text=Camera+File'">
                        <div class="gallery-info">
                            <input type="text" value="${r.identity}" id="inp-id-${r.id}" title="Edit Name">
                            <div class="gallery-stats">
                                <span>${r.timestamp.split(' ')[0]}</span>
                            </div>
                        </div>
                        <div class="gallery-actions">
                            <button class="btn-edit" onclick="updateRecord(${r.id})" title="Save Name Change"><i class="fa-solid fa-save"></i> Edit</button>
                            <button class="btn-delete" onclick="deleteRecord(${r.id})" title="Delete Image"><i class="fa-solid fa-trash"></i> Del</button>
                        </div>
                    `;
                    galleryContainer.appendChild(el);
                });
            } else {
                galleryContainer.innerHTML = '<p class="text-center" style="grid-column: 1/-1; color:var(--text-secondary)">No images exist in the database.</p>';
            }
        } catch (err) {
            galleryContainer.innerHTML = '<p class="text-center" style="grid-column: 1/-1; color:var(--danger)">Failed to load data from server.</p>';
        }
    }
    
    // Global functions for inline onclick binding generated dynamically
    window.deleteRecord = async function(id) {
        if(!confirm("Permanently delete this face record?")) return;
        try {
            await fetch(`/api/database/${id}`, { method: 'DELETE' });
            fetchGallery();
        } catch(e) { console.error(e); }
    };
    
    window.updateRecord = async function(id) {
        const newName = document.getElementById(`inp-id-${id}`).value;
        try {
            await fetch(`/api/database/${id}`, { 
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ identity: newName })
            });
            alert('Identity successfully updated and AI model immediately retrained.');
            fetchGallery();
        } catch(e) { console.error(e); }
    };

    /* ----------------------------------------------------
       THREE.JS DASHBOARD 3D BACKGROUND (DATA STREAM)
    ---------------------------------------------------- */
    function init3DBackground() {
        const container = document.getElementById('dashboard-canvas');
        if(!container) return;

        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x0d1117, 0.05);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 10;
        camera.position.y = 2;

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Cybernetic Grid Line floor
        const gridHelper = new THREE.GridHelper(100, 100, 0x00e1ff, 0x002b40);
        gridHelper.position.y = -3;
        scene.add(gridHelper);

        // Particle Stream
        const particleGeo = new THREE.BufferGeometry();
        const pCount = 1000;
        const posArr = new Float32Array(pCount * 3);
        
        for(let i=0; i < pCount; i++) {
            posArr[i*3] = (Math.random() - 0.5) * 40; // x
            posArr[i*3+1] = Math.random() * 20 - 3; // y
            posArr[i*3+2] = (Math.random() - 0.5) * 40; // z
        }
        
        particleGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
        const pMat = new THREE.PointsMaterial({
            color: 0x58a6ff,
            size: 0.1,
            transparent: true,
            opacity: 0.6,
            blending: THREE.AdditiveBlending
        });
        
        const particleMesh = new THREE.Points(particleGeo, pMat);
        scene.add(particleMesh);

        let mouseX = 0;
        let mouseY = 0;
        document.addEventListener('mousemove', (e) => {
            mouseX = (e.clientX - window.innerWidth/2) * 0.0005;
            mouseY = (e.clientY - window.innerHeight/2) * 0.0005;
        });

        function animate() {
            requestAnimationFrame(animate);
            
            // Move stream forward
            const positions = particleMesh.geometry.attributes.position.array;
            for(let i=2; i < pCount * 3; i+=3) {
                positions[i] += 0.05;
                if(positions[i] > 20) {
                    positions[i] = -20; // reset
                }
            }
            particleMesh.geometry.attributes.position.needsUpdate = true;
            
            // Subtle camera parallax
            camera.position.x += (mouseX * 5 - camera.position.x) * 0.05;
            camera.position.y += (-mouseY * 5 + 2 - camera.position.y) * 0.05;
            camera.lookAt(0, 0, 0);

            // Rotate floor slightly for motion feel
            gridHelper.position.z += 0.02;
            if(gridHelper.position.z > 0.5) gridHelper.position.z = 0;

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    init3DBackground();
});
