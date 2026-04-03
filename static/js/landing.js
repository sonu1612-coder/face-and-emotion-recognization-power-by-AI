/* ============================================================
   CORE AI — LANDING JS
   Three.js 3D · Scroll Reveal · Particles · Countdown
   ============================================================ */
document.addEventListener("DOMContentLoaded", () => {

    /* =========================================================
       1. THREE.JS — ANIMATED 3D BACKGROUND
       ========================================================= */
    const canvas = document.getElementById('three-canvas');
    if (canvas && window.THREE) {
        const scene    = new THREE.Scene();
        const W = window.innerWidth, H = window.innerHeight;
        const camera   = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
        renderer.setSize(W, H);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setClearColor(0x030710, 1);
        camera.position.z = 5;

        /* Lights */
        scene.add(new THREE.AmbientLight(0x0a1020, 0.8));
        const cyanLight   = new THREE.PointLight(0x00e5ff, 2.5, 18);
        cyanLight.position.set(3, 3, 3);
        scene.add(cyanLight);
        const purpleLight = new THREE.PointLight(0xa259ff, 1.8, 15);
        purpleLight.position.set(-3, -2, 2);
        scene.add(purpleLight);

        /* Central crystal */
        const crystal = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1.3, 1),
            new THREE.MeshStandardMaterial({ color: 0x050e1e, metalness: 0.9, roughness: 0.08, emissive: 0x001a28, emissiveIntensity: 0.3 })
        );
        scene.add(crystal);

        /* Wire overlay */
        const wire = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1.32, 1),
            new THREE.MeshBasicMaterial({ color: 0x00e5ff, wireframe: true, opacity: 0.14, transparent: true })
        );
        scene.add(wire);

        /* Rings */
        const mkRing = (r, thick, color, opacity) => {
            const m = new THREE.Mesh(
                new THREE.TorusGeometry(r, thick, 8, 90),
                new THREE.MeshBasicMaterial({ color, opacity, transparent: true })
            );
            scene.add(m); return m;
        };
        const ring1 = mkRing(2.2, 0.013, 0x00e5ff, 0.3);
        const ring2 = mkRing(2.8, 0.009, 0xa259ff, 0.18);
        const ring3 = mkRing(1.6, 0.010, 0x00e5ff, 0.22);
        ring1.rotation.x = Math.PI / 3;
        ring2.rotation.x = Math.PI / 2.2; ring2.rotation.z = Math.PI / 5;
        ring3.rotation.x = Math.PI / 4; ring3.rotation.y = Math.PI / 6;

        /* Star field */
        const starPos = new Float32Array(3000 * 3);
        for (let i = 0; i < starPos.length; i++) starPos[i] = (Math.random() - 0.5) * 50;
        const starGeo = new THREE.BufferGeometry();
        starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
        const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.055, sizeAttenuation: true, transparent: true, opacity: 0.55 }));
        scene.add(stars);

        /* Orbiting dot cloud */
        const dotGroup = new THREE.Group();
        for (let i = 0; i < 80; i++) {
            const a   = (i / 80) * Math.PI * 2;
            const r   = 1.9 + Math.sin(i * 0.6) * 0.4;
            const sz  = 0.022 + Math.random() * 0.028;
            const dot = new THREE.Mesh(
                new THREE.SphereGeometry(sz, 4, 4),
                new THREE.MeshBasicMaterial({ color: i % 3 === 0 ? 0xa259ff : 0x00e5ff, transparent: true, opacity: 0.55 + Math.random() * 0.45 })
            );
            dot.position.set(Math.cos(a) * r, (Math.random() - 0.5) * 0.5, Math.sin(a) * r);
            dot.userData = { a, r, y: dot.position.y, spd: 0.003 + Math.random() * 0.004 };
            dotGroup.add(dot);
        }
        scene.add(dotGroup);

        /* Mouse parallax */
        let mx = 0, my = 0;
        document.addEventListener('mousemove', e => {
            mx = (e.clientX / window.innerWidth  - 0.5) * 2;
            my = (e.clientY / window.innerHeight - 0.5) * 2;
        });

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        let t = 0;
        function animate() {
            requestAnimationFrame(animate);
            t += 0.005;

            crystal.rotation.y += 0.003; crystal.rotation.x += 0.0012;
            wire.rotation.y    += 0.003; wire.rotation.x    += 0.0012;

            dotGroup.children.forEach(d => {
                d.userData.a += d.userData.spd;
                d.position.set(
                    Math.cos(d.userData.a) * d.userData.r,
                    d.userData.y + Math.sin(t * 0.9) * 0.06,
                    Math.sin(d.userData.a) * d.userData.r
                );
            });
            dotGroup.rotation.y += 0.0008;
            dotGroup.rotation.x  = Math.sin(t * 0.3) * 0.12;

            ring1.rotation.z += 0.004;
            ring2.rotation.z -= 0.003;
            ring3.rotation.y += 0.006;

            stars.rotation.y += 0.00015;

            cyanLight.intensity   = 2.5 + Math.sin(t * 1.5) * 0.6;
            purpleLight.intensity = 1.8 + Math.cos(t * 1.2) * 0.5;

            camera.position.x += (mx * 0.7 - camera.position.x) * 0.05;
            camera.position.y += (-my * 0.4 - camera.position.y) * 0.05;
            camera.lookAt(scene.position);

            renderer.render(scene, camera);
        }
        animate();
    }

    /* =========================================================
       2. CSS PARTICLE FIELD
       ========================================================= */
    const field = document.getElementById('particle-field');
    if (field) {
        for (let i = 0; i < 70; i++) {
            const p  = document.createElement('div');
            p.classList.add('particle');
            const sz = 1.5 + Math.random() * 3;
            p.style.cssText = `
                left: ${Math.random() * 100}%;
                bottom: ${Math.random() * 15}%;
                width: ${sz}px; height: ${sz}px;
                --dx: ${(Math.random() - 0.5) * 140}px;
                animation-duration: ${7 + Math.random() * 14}s;
                animation-delay: ${Math.random() * -18}s;
                background: ${Math.random() > 0.45 ? '#00e5ff' : '#a259ff'};
                box-shadow: 0 0 ${sz * 2}px currentColor;
            `;
            field.appendChild(p);
        }
    }

    /* =========================================================
       3. SCROLL RAIL & SECTION TRACKING
       ========================================================= */
    const wrapper     = document.getElementById('scrollWrapper');
    const railFill    = document.getElementById('railFill');
    const railDots    = document.querySelectorAll('.rail-dot');
    const sections    = document.querySelectorAll('.snap-section');
    const totalH      = () => wrapper.scrollHeight - wrapper.clientHeight;

    function updateRail() {
        const pct = (wrapper.scrollTop / totalH()) * 100;
        if (railFill) railFill.style.height = pct + '%';

        let activeIdx = 0;
        sections.forEach((sec, i) => {
            const rect = sec.getBoundingClientRect();
            if (rect.top <= window.innerHeight * 0.5) activeIdx = i;
        });
        railDots.forEach((d, i) => d.classList.toggle('active', i === activeIdx));
    }

    wrapper.addEventListener('scroll', updateRail, { passive: true });

    /* Rail dot click → scroll to section */
    railDots.forEach((dot, i) => {
        dot.addEventListener('click', () => {
            sections[i].scrollIntoView({ behavior: 'smooth' });
        });
    });

    /* =========================================================
       4. SCROLL REVEAL (Intersection Observer)
       ========================================================= */
    const revealEls = document.querySelectorAll('.scroll-reveal');

    const revealObs = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const delay = parseInt(entry.target.dataset.delay || 0);
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, delay);
                revealObs.unobserve(entry.target);
            }
        });
    }, { threshold: 0.15, rootMargin: '0px 0px -60px 0px' });

    revealEls.forEach(el => revealObs.observe(el));

    /* =========================================================
       5. COUNTDOWN (triggers when section 5 is visible)
       ========================================================= */
    const timerEl   = document.getElementById('countdownTimer');
    const progEl    = document.getElementById('countdownProg');
    const statusEl  = document.getElementById('countdownStatus');
    let tMinus      = 10;
    let cdInterval  = null;
    let cdStarted   = false;

    function startCountdown() {
        if (cdStarted) return;
        cdStarted = true;
        tMinus = 10;
        if (timerEl) timerEl.textContent = '10';
        if (progEl) progEl.style.width = '0%';

        cdInterval = setInterval(() => {
            tMinus--;
            if (tMinus < 0) tMinus = 0;
            if (timerEl) timerEl.textContent = String(tMinus).padStart(2,'0');
            if (progEl) progEl.style.width = ((10 - tMinus) * 10) + '%';
            if (tMinus <= 0) {
                clearInterval(cdInterval);
                if (statusEl) {
                    statusEl.textContent = 'SYSTEM ONLINE — READY';
                    statusEl.style.color = '#00ff88';
                }
            }
        }, 1000);
    }

    /* Watch for launch section */
    const launchSec = document.getElementById('sec-launch');
    if (launchSec) {
        new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) startCountdown();
        }, { threshold: 0.5 }).observe(launchSec);
    }

    /* =========================================================
       6. BUTTON INTERACTIONS
       ========================================================= */
    const btnLaunch    = document.getElementById('btnLaunch');
    const btnQuit      = document.getElementById('btnQuit');
    const flashOverlay = document.getElementById('flash-overlay');

    if (btnLaunch) {
        btnLaunch.addEventListener('click', e => {
            e.preventDefault();
            if (cdInterval) clearInterval(cdInterval);
            // Pre-warm camera hardware instantly so it's ready on the next page
            fetch('/api/camera/start', { method: 'POST' }).catch(() => {});

            flashOverlay && flashOverlay.classList.add('flash');
            setTimeout(() => {
                flashOverlay && flashOverlay.classList.remove('flash');
                const txt = document.getElementById('launchBtnText');
                if (txt) txt.textContent = 'INITIALIZING...';
                setTimeout(() => { window.location.href = btnLaunch.href; }, 500);
            }, 120);
        });
    }

    if (btnQuit) {
        btnQuit.addEventListener('click', async () => {
            if (!confirm('Shut down CORE AI Python server engine?')) return;
            if (cdInterval) clearInterval(cdInterval);
            const txt = document.getElementById('quitBtnText');
            if (txt) txt.textContent = 'TERMINATING...';
            btnQuit.style.pointerEvents = 'none';
            if (statusEl) {
                statusEl.textContent = 'SHUTTING DOWN...';
                statusEl.style.color = '#ff3b3b';
            }
            try { await fetch('/api/shutdown', { method: 'POST' }); } catch (_) {}
            showDeath();
        });
    }

    function showDeath() {
        document.body.innerHTML = `
        <div style="display:flex;flex-direction:column;height:100vh;width:100vw;
                    justify-content:center;align-items:center;background:#030710;
                    color:#fff;font-family:'Orbitron',sans-serif;gap:20px;text-align:center">
            <div style="font-size:.75rem;letter-spacing:4px;color:rgba(0,229,255,.4)">— CORE AI —</div>
            <h1 style="font-size:2.8rem;color:#00e5ff;text-shadow:0 0 30px rgba(0,229,255,.6)">ENGINE TERMINATED</h1>
            <p style="opacity:.6;font-family:'Inter',sans-serif;letter-spacing:1px">
                The Python Flask server has shut down completely.
            </p>
            <p style="opacity:.3;font-size:.8rem;font-family:'Inter',sans-serif">
                It is safe to close this window.
            </p>
        </div>`;
        setTimeout(() => { try { window.close(); } catch(_){} }, 2000);
    }

});
