/**
 * ARROHA — Real-Time Streaming Multilingual Voice Assistant
 * Hacker House Goa 2026 Edition
 * Features:
 * - Web Audio API Low-Latency Sequential Streaming Player
 * - Audio-Reactive Waveform Canvas (AnalyserNode)
 * - Instant Barge-in / Interruption (<100ms cancellation)
 * - Server-Sent Events (SSE) Delta Token Streaming
 * - 15-Language Router & Telemetry Instrumentation
 */

(() => {
  'use strict';

  // State Management
  const state = {
    mode: 'voice', // 'voice' | 'text'
    voiceState: 'ready', // 'ready' | 'listening' | 'thinking' | 'speaking' | 'interrupted' | 'error'
    currentSessionId: null,
    targetLanguage: 'auto',
    activeAudioSource: null,
    audioQueue: [],
    isPlayingAudio: false,
    audioContext: null,
    analyserNode: null,
    audioDataArray: null,
    canvasCtx: null,
    animFrameId: null,
    mediaRecorder: null,
    audioChunks: [],
    speechRecognizer: null,
    isRecording: false,
    currentAssistantCard: null,
    currentStreamTokens: [],
    conversationHistory: [],
  };

  // DOM Elements
  const DOM = {
    voiceStatePill: document.getElementById('voice-state-pill'),
    voiceStateLabel: document.getElementById('voice-state-label'),
    ttfaQuickPill: document.getElementById('ttfa-quick-pill'),
    ttfaQuickVal: document.getElementById('ttfa-quick-val'),
    langSelect: document.getElementById('lang-select'),
    newChatBtn: document.getElementById('new-chat-btn'),
    debugToggleBtn: document.getElementById('debug-toggle-btn'),
    chatFeed: document.getElementById('chat-feed'),
    welcomeHero: document.getElementById('welcome-hero'),
    visualizerStage: document.getElementById('visualizer-stage'),
    visualizerText: document.getElementById('visualizer-text'),
    waveformCanvas: document.getElementById('audio-waveform-canvas'),
    micTriggerBtn: document.getElementById('mic-trigger-btn'),
    stopSpeechBtn: document.getElementById('stop-speech-btn'),
    textQueryInput: document.getElementById('text-query-input'),
    modeToggleBtn: document.getElementById('mode-toggle-btn'),
    sendQueryBtn: document.getElementById('send-query-btn'),
    diagnosticsDrawer: document.getElementById('diagnostics-drawer'),
    drawerBackdrop: document.getElementById('drawer-backdrop'),
    closeDrawerBtn: document.getElementById('close-drawer-btn'),
    quickChips: document.querySelectorAll('.quick-chip'),
    
    // Telemetry fields
    diagTtfa: document.getElementById('diag-ttfa'),
    diagTtft: document.getElementById('diag-ttft'),
    diagStt: document.getElementById('diag-stt'),
    diagVector: document.getElementById('diag-vector'),
    diagBm25: document.getElementById('diag-bm25'),
    diagFusion: document.getElementById('diag-fusion'),
    diagTts: document.getElementById('diag-tts'),
    diagTotal: document.getElementById('diag-total'),
    budgetBadge: document.getElementById('budget-badge'),
  };

  // Initialize Web Audio API
  function initAudioContext() {
    if (!state.audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      state.audioContext = new AudioCtx({ sampleRate: 24000 });
      state.analyserNode = state.audioContext.createAnalyser();
      state.analyserNode.fftSize = 64;
      state.analyserNode.smoothingTimeConstant = 0.8;
      const bufferLength = state.analyserNode.frequencyBinCount;
      state.audioDataArray = new Uint8Array(bufferLength);
      
      // Canvas setup
      if (DOM.waveformCanvas) {
        state.canvasCtx = DOM.waveformCanvas.getContext('2d');
      }
    }
    if (state.audioContext.state === 'suspended') {
      state.audioContext.resume();
    }
  }

  // Draw Audio-Reactive Waveform on Canvas
  function drawVisualizer() {
    if (!state.canvasCtx || !DOM.waveformCanvas) return;
    const canvas = DOM.waveformCanvas;
    const ctx = state.canvasCtx;
    const width = canvas.width;
    const height = canvas.height;

    state.animFrameId = requestAnimationFrame(drawVisualizer);

    if (state.voiceState === 'speaking' && state.analyserNode) {
      state.analyserNode.getByteFrequencyData(state.audioDataArray);
      ctx.clearRect(0, 0, width, height);

      const barCount = 28;
      const barWidth = (width / barCount) - 3;
      let x = 3;

      for (let i = 0; i < barCount; i++) {
        const binIndex = Math.floor((i / barCount) * state.audioDataArray.length);
        const value = state.audioDataArray[binIndex] || 0;
        const percent = value / 255;
        const barHeight = Math.max(4, percent * height * 0.85);

        // Gradient from Gold to Emerald to Hibiscus Pink
        const grad = ctx.createLinearGradient(0, height - barHeight, 0, height);
        grad.addColorStop(0, '#ffdd00');
        grad.addColorStop(0.5, '#10b981');
        grad.addColorStop(1, '#ff007f');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, (height - barHeight) / 2, barWidth, barHeight, 3);
        ctx.fill();

        x += barWidth + 3;
      }
    } else if (state.voiceState === 'listening') {
      // Gentle animated sine wave for listening
      ctx.clearRect(0, 0, width, height);
      const time = Date.now() * 0.005;
      ctx.strokeStyle = '#ff007f';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let x = 0; x < width; x += 4) {
        const y = height / 2 + Math.sin(x * 0.05 + time) * 8 * Math.sin(time * 0.5);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    } else {
      // Idle / Minimal baseline
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
    }
  }

  // Set Global UI Voice State
  function setVoiceState(newState, labelOverride = null) {
    state.voiceState = newState;
    const pill = DOM.voiceStatePill;
    const label = DOM.voiceStateLabel;
    const stage = DOM.visualizerStage;
    const vText = DOM.visualizerText;
    const micBtn = DOM.micTriggerBtn;
    const stopBtn = DOM.stopSpeechBtn;

    // Reset classes
    pill.className = 'state-pill';
    stage.className = 'visualizer-stage';
    micBtn.classList.remove('listening');

    switch (newState) {
      case 'listening':
        pill.classList.add('state-listening');
        label.textContent = 'LISTENING...';
        stage.classList.add('listening', 'active');
        vText.textContent = 'Listening to your speech...';
        micBtn.classList.add('listening');
        stopBtn.classList.add('hidden');
        break;

      case 'thinking':
        pill.classList.add('state-thinking');
        label.textContent = 'THINKING...';
        stage.classList.add('thinking', 'active');
        vText.textContent = 'Searching 50,400 chunks & synthesizing...';
        stopBtn.classList.add('hidden');
        break;

      case 'speaking':
        pill.classList.add('state-speaking');
        label.textContent = 'SPEAKING';
        stage.classList.add('speaking', 'active');
        vText.textContent = 'Streaming spoken response...';
        stopBtn.classList.remove('hidden');
        break;

      case 'interrupted':
        pill.classList.add('state-interrupted');
        label.textContent = 'INTERRUPTED';
        stage.classList.add('idle');
        vText.textContent = 'Speech halted (Barge-in triggered)';
        stopBtn.classList.add('hidden');
        setTimeout(() => {
          if (state.voiceState === 'interrupted') setVoiceState('ready');
        }, 1200);
        break;

      case 'ready':
      default:
        pill.classList.add('state-ready');
        label.textContent = labelOverride || 'READY';
        stage.classList.add('idle');
        vText.textContent = 'Click microphone or type below';
        stopBtn.classList.add('hidden');
        break;
    }
  }

  // Sequential Streaming Audio Playback Queue
  async function enqueueAudioChunk(base64Data, chunkIndex) {
    if (!base64Data) return;
    initAudioContext();

    try {
      const binaryString = atob(base64Data);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Convert 16-bit PCM (24kHz Mono) or WAV to AudioBuffer
      let audioBuffer;
      if (binaryString.startsWith('RIFF')) {
        audioBuffer = await state.audioContext.decodeAudioData(bytes.buffer);
      } else {
        // Raw 16-bit signed PCM 24000 Hz
        const int16 = new Int16Array(bytes.buffer);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
          float32[i] = int16[i] / 32768.0;
        }
        audioBuffer = state.audioContext.createBuffer(1, float32.length, 24000);
        audioBuffer.getChannelData(0).set(float32);
      }

      state.audioQueue.push({ buffer: audioBuffer, chunkIndex });
      if (!state.isPlayingAudio) {
        playNextAudioInQueue();
      }
    } catch (err) {
      console.warn('Audio decoding error:', err);
    }
  }

  function playNextAudioInQueue() {
    if (state.audioQueue.length === 0) {
      state.isPlayingAudio = false;
      if (state.voiceState === 'speaking') {
        setVoiceState('ready');
      }
      return;
    }

    state.isPlayingAudio = true;
    setVoiceState('speaking');

    const item = state.audioQueue.shift();
    const source = state.audioContext.createBufferSource();
    source.buffer = item.buffer;

    // Connect to AnalyserNode for reactive visualizer, then to Destination (Speakers)
    source.connect(state.analyserNode);
    state.analyserNode.connect(state.audioContext.destination);

    state.activeAudioSource = source;

    source.onended = () => {
      state.activeAudioSource = null;
      playNextAudioInQueue();
    };

    source.start(0);
  }

  // Natural Browser Speech Synthesis (Crystal-Clear Human Voice)
  function speakTextWithBrowserTTS(text, langCode) {
    if (!('speechSynthesis' in window)) return;
    if (state.mode !== 'voice') return;

    window.speechSynthesis.cancel(); // Cancel any prior speech

    const cleanText = text.replace(/\[Source \d+[^\]]*\]/g, '').replace(/https?:\/\/\S+/g, '').trim();
    if (!cleanText) return;

    const utterance = new SpeechSynthesisUtterance(cleanText);
    
    // Map ISO language codes to browser BCP-47 locale tags
    const localeMap = {
      'en': 'en-IN',
      'hi': 'hi-IN',
      'bn': 'bn-IN',
      'ta': 'ta-IN',
      'te': 'te-IN',
      'mr': 'mr-IN',
      'gu': 'gu-IN',
      'kn': 'kn-IN',
      'ml': 'ml-IN',
      'pa': 'pa-IN',
      'or': 'or-IN',
      'as': 'as-IN',
      'ne': 'ne-NP',
      'ur': 'ur-PK'
    };

    const targetLocale = localeMap[langCode] || langCode || 'en-IN';
    utterance.lang = targetLocale;
    utterance.rate = 1.05;
    utterance.pitch = 1.0;

    // Pick best matching natural voice
    const voices = window.speechSynthesis.getVoices();
    if (voices && voices.length > 0) {
      const matchedVoice = voices.find(v => v.lang.startsWith(targetLocale.split('-')[0])) ||
                           voices.find(v => v.lang.includes('IN') || v.lang.includes('en'));
      if (matchedVoice) {
        utterance.voice = matchedVoice;
      }
    }

    utterance.onstart = () => {
      setVoiceState('speaking');
    };

    utterance.onend = () => {
      if (state.voiceState === 'speaking') {
        setVoiceState('ready');
      }
    };

    utterance.onerror = (e) => {
      console.warn('SpeechSynthesis note:', e);
      if (state.voiceState === 'speaking') {
        setVoiceState('ready');
      }
    };

    window.speechSynthesis.speak(utterance);
  }

  // Instant Barge-In / Interruption Handler
  async function triggerBargeIn() {
    console.log('[Barge-in] Triggering immediate audio cancellation...');
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    if (state.activeAudioSource) {
      try {
        state.activeAudioSource.stop();
      } catch (e) {}
      state.activeAudioSource = null;
    }
    state.audioQueue = [];
    state.isPlayingAudio = false;

    if (state.currentSessionId) {
      try {
        fetch(`/voice/interrupt?session_id=${state.currentSessionId}`, { method: 'POST' });
      } catch (e) {}
    }

    setVoiceState('interrupted');
  }

  // Message Card UI Rendering
  function appendUserMessage(text, isVoice = false) {
    if (DOM.welcomeHero) DOM.welcomeHero.style.display = 'none';

    const card = document.createElement('div');
    card.className = 'message-card user-msg';
    
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    card.innerHTML = `
      <div class="user-bubble">
        <div class="user-text">${escapeHtml(text)}</div>
      </div>
      <div class="user-meta">
        ${isVoice ? '<span class="voice-badge-icon">🎙️ Spoken Voice</span> • ' : ''}
        <span>${timeStr}</span>
      </div>
    `;
    DOM.chatFeed.appendChild(card);
    DOM.chatFeed.scrollTop = DOM.chatFeed.scrollHeight;
  }

  function createAssistantCard() {
    if (DOM.welcomeHero) DOM.welcomeHero.style.display = 'none';

    const card = document.createElement('div');
    card.className = 'message-card assistant-msg';
    card.innerHTML = `
      <div class="assistant-bubble-wrap">
        <div class="assistant-avatar">A</div>
        <div class="assistant-content-card">
          <div class="assistant-text-body"><span class="streaming-text"></span><span class="streaming-cursor"></span></div>
          <div class="sources-accordion hidden">
            <button class="sources-toggle-btn">
              <span>📚 View Grounded Evidence Sources (0)</span>
            </button>
            <div class="sources-drawer-list hidden"></div>
          </div>
          <div class="assistant-footer">
            <div class="footer-badges">
              <span class="badge-ttfa-live">⚡ TTFA: --</span>
              <span class="badge-lang-tag">Lang: --</span>
            </div>
            <div class="footer-actions">
              <button class="msg-action-btn copy-btn" title="Copy Answer">📋 Copy</button>
            </div>
          </div>
        </div>
      </div>
    `;

    DOM.chatFeed.appendChild(card);
    DOM.chatFeed.scrollTop = DOM.chatFeed.scrollHeight;

    // Attach copy event
    const copyBtn = card.querySelector('.copy-btn');
    copyBtn.addEventListener('click', () => {
      const text = card.querySelector('.streaming-text').innerText;
      navigator.clipboard.writeText(text);
      copyBtn.textContent = '✅ Copied!';
      setTimeout(() => (copyBtn.textContent = '📋 Copy'), 2000);
    });

    // Attach sources toggle event
    const sourcesBtn = card.querySelector('.sources-toggle-btn');
    const sourcesList = card.querySelector('.sources-drawer-list');
    sourcesBtn.addEventListener('click', () => {
      sourcesList.classList.toggle('hidden');
    });

    state.currentAssistantCard = card;
    state.currentStreamTokens = [];
    return card;
  }

  function appendStreamToken(token) {
    if (!state.currentAssistantCard) return;
    state.currentStreamTokens.push(token);
    const span = state.currentAssistantCard.querySelector('.streaming-text');
    if (span) {
      span.textContent = state.currentStreamTokens.join('');
      DOM.chatFeed.scrollTop = DOM.chatFeed.scrollHeight;
    }
  }

  function finalizeAssistantCard(doneData) {
    if (!state.currentAssistantCard) return;
    const card = state.currentAssistantCard;
    const cursor = card.querySelector('.streaming-cursor');
    if (cursor) cursor.remove();

    const span = card.querySelector('.streaming-text');
    if (span && (!span.textContent || !span.textContent.trim()) && doneData.text) {
      span.textContent = doneData.text;
    }

    const ttfaBadge = card.querySelector('.badge-ttfa-live');
    const langBadge = card.querySelector('.badge-lang-tag');

    if (doneData.latency && doneData.latency.first_audio_latency_ms) {
      const ttfa = doneData.latency.first_audio_latency_ms;
      ttfaBadge.textContent = `⚡ TTFA: ${ttfa} ms`;
      DOM.ttfaQuickVal.textContent = `${ttfa} ms`;
    }

    if (doneData.language) {
      langBadge.textContent = `Lang: ${doneData.language.toUpperCase()}`;
    }

    // Populate Sources if available
    if (doneData.sources && doneData.sources.length > 0) {
      const acc = card.querySelector('.sources-accordion');
      acc.classList.remove('hidden');
      acc.querySelector('.sources-toggle-btn span').textContent = `📚 View Grounded Evidence Sources (${doneData.sources.length})`;
      const list = card.querySelector('.sources-drawer-list');
      list.innerHTML = doneData.sources
        .map(
          (s, idx) => `
        <div class="source-item-card">
          <div class="source-item-header">
            <span>Source #${idx + 1} (${(s.language || 'en').toUpperCase()})</span>
            <span>Score: ${(s.score || 1.0).toFixed(3)}</span>
          </div>
          <div>${escapeHtml(s.text || '')}</div>
        </div>
      `
        )
        .join('');
    }

    // Update Diagnostics Drawer
    if (doneData.latency) {
      updateDiagnostics(doneData.latency);
    }
  }

  function updateDiagnostics(lat) {
    if (!lat) return;
    if (DOM.diagTtfa) DOM.diagTtfa.textContent = `${lat.first_audio_latency_ms || 0} ms`;
    if (DOM.diagTtft) DOM.diagTtft.textContent = `${lat.llm_ttft_ms || 0} ms`;
    if (DOM.diagStt) DOM.diagStt.textContent = `${lat.stt_ms || 0} ms`;
    if (DOM.diagVector) DOM.diagVector.textContent = `${lat.vector_retrieval_ms || 0} ms`;
    if (DOM.diagBm25) DOM.diagBm25.textContent = `${lat.bm25_retrieval_ms || 0} ms`;
    if (DOM.diagFusion) DOM.diagFusion.textContent = `${lat.hybrid_fusion_ms || 0} ms`;
    if (DOM.diagTts) DOM.diagTts.textContent = `${lat.tts_first_chunk_ms || 0} ms`;
    if (DOM.diagTotal) DOM.diagTotal.textContent = `${lat.total_ms || 0} ms`;

    if (DOM.budgetBadge) {
      if ((lat.first_audio_latency_ms || 0) <= 200.0) {
        DOM.budgetBadge.className = 'budget-badge pass';
        DOM.budgetBadge.textContent = '⚡ PASS (< 200 ms TARGET ACHIEVED)';
      } else {
        DOM.budgetBadge.className = 'budget-badge fail';
        DOM.budgetBadge.textContent = '⚠️ TARGET MISSED';
      }
    }
  }

  // Execute Voice Query Stream via SSE
  async function executeStreamingVoiceQuery(queryText, audioBase64 = null, isVoiceInput = false) {
    initAudioContext();
    const sessionId = 'sess_' + Date.now();
    state.currentSessionId = sessionId;

    appendUserMessage(queryText || 'Voice Query Payload', isVoiceInput);
    createAssistantCard();
    setVoiceState('thinking');

    const selectedLang = DOM.langSelect.value === 'auto' ? null : DOM.langSelect.value;
    const reqBody = {
      query: queryText,
      audio_base64: audioBase64,
      language: selectedLang,
      mode: state.mode,
      stream: true,
      session_id: sessionId,
    };

    try {
      const response = await fetch('/voice/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // Keep last partial chunk

        for (const line of lines) {
          if (!line.trim()) continue;
          let eventType = 'message';
          let dataStr = '';

          for (const rawLine of line.split('\n')) {
            if (rawLine.startsWith('event: ')) {
              eventType = rawLine.substring(7).trim();
            } else if (rawLine.startsWith('data: ')) {
              dataStr = rawLine.substring(6).trim();
            }
          }

          if (dataStr) {
            try {
              const data = JSON.parse(dataStr);
              handleStreamChunk(eventType, data);
            } catch (err) {
              console.warn('Malformed SSE data packet:', dataStr);
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming request failed:', error);
      setVoiceState('ready', 'ERROR');
      if (state.currentAssistantCard) {
        appendStreamToken(`\n[Connection Error: ${error.message}]`);
      }
    }
  }

  function handleStreamChunk(event, data) {
    if (data.session_id && data.session_id !== state.currentSessionId) {
      return; // Ignore older session chunks
    }

    if (event === 'status') {
      if (data.text === 'THINKING') setVoiceState('thinking');
      else if (data.text === 'SPEAKING') setVoiceState('speaking');
      else if (data.text === 'INTERRUPTED') setVoiceState('interrupted');
    } else if (event === 'token') {
      if (data.delta) {
        appendStreamToken(data.delta);
      }
    } else if (event === 'audio_chunk') {
      if (data.audio_base64 && state.mode === 'voice') {
        enqueueAudioChunk(data.audio_base64, data.chunk_index);
      }
    } else if (event === 'done') {
      finalizeAssistantCard(data);
      if (state.mode === 'voice' && data.text) {
        speakTextWithBrowserTTS(data.text, data.language || state.targetLanguage);
      } else if (!state.isPlayingAudio) {
        setVoiceState('ready');
      }
    } else if (event === 'error') {
      console.error('Backend voice error:', data.text);
      setVoiceState('ready', 'ERROR');
      if (state.currentAssistantCard) {
        appendStreamToken(`\n[Error: ${data.text}]`);
      }
    }
  }

  // Voice Recording & Speech Recognition Setup
  function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      state.speechRecognizer = new SpeechRecognition();
      state.speechRecognizer.continuous = false;
      state.speechRecognizer.interimResults = false;

      state.speechRecognizer.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        console.log('[STT] Recognized speech transcript:', transcript);
        if (transcript.trim()) {
          executeStreamingVoiceQuery(transcript.trim(), null, true);
        }
      };

      state.speechRecognizer.onerror = (err) => {
        console.warn('Web Speech STT error:', err);
        setVoiceState('ready');
      };

      state.speechRecognizer.onend = () => {
        state.isRecording = false;
        if (state.voiceState === 'listening') {
          setVoiceState('thinking');
        }
      };
    }
  }

  async function startMicrophoneRecording() {
    // If AI is speaking, clicking mic triggers instant barge-in!
    if (state.isPlayingAudio || state.voiceState === 'speaking') {
      await triggerBargeIn();
    }

    initAudioContext();
    initSpeechRecognition();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.mediaRecorder = new MediaRecorder(stream);
      state.audioChunks = [];

      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.audioChunks.push(e.data);
      };

      state.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64Audio = reader.result.split(',')[1];
          if (!state.speechRecognizer) {
            executeStreamingVoiceQuery('Spoken Audio', base64Audio, true);
          }
        };
        reader.readAsDataURL(audioBlob);
      };

      state.mediaRecorder.start();
      state.isRecording = true;
      setVoiceState('listening');

      if (state.speechRecognizer) {
        const targetLang = DOM.langSelect.value;
        state.speechRecognizer.lang = targetLang === 'auto' ? 'en-IN' : targetLang;
        state.speechRecognizer.start();
      }
    } catch (err) {
      console.warn('Microphone access denied / unavailable:', err);
      alert('Microphone access is required for real-time speech input.');
      setVoiceState('ready');
    }
  }

  function stopMicrophoneRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
      state.mediaRecorder.stop();
    }
    if (state.speechRecognizer) {
      try {
        state.speechRecognizer.stop();
      } catch (e) {}
    }
    state.isRecording = false;
  }

  // Event Listeners
  function setupEventListeners() {
    // Microphone Button Click
    DOM.micTriggerBtn.addEventListener('click', () => {
      if (state.isRecording) {
        stopMicrophoneRecording();
      } else {
        startMicrophoneRecording();
      }
    });

    // Instant Interrupt / Stop Speech Button
    DOM.stopSpeechBtn.addEventListener('click', () => {
      triggerBargeIn();
    });

    // Send Query Button Click
    DOM.sendQueryBtn.addEventListener('click', () => {
      const text = DOM.textQueryInput.value.trim();
      if (text) {
        DOM.textQueryInput.value = '';
        executeStreamingVoiceQuery(text, null, false);
      }
    });

    // Text Input Keydown (Enter to send)
    DOM.textQueryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const text = DOM.textQueryInput.value.trim();
        if (text) {
          DOM.textQueryInput.value = '';
          executeStreamingVoiceQuery(text, null, false);
        }
      }
    });

    // Mode Toggle (Voice vs Text)
    DOM.modeToggleBtn.addEventListener('click', () => {
      if (state.mode === 'voice') {
        state.mode = 'text';
        DOM.modeToggleBtn.classList.remove('active');
        DOM.modeToggleBtn.querySelector('.mode-icon').textContent = '📝';
        DOM.modeToggleBtn.querySelector('.mode-label').textContent = 'Text Only';
      } else {
        state.mode = 'voice';
        DOM.modeToggleBtn.classList.add('active');
        DOM.modeToggleBtn.querySelector('.mode-icon').textContent = '🎙️';
        DOM.modeToggleBtn.querySelector('.mode-label').textContent = 'Voice ON';
      }
    });

    // Quick Prompt Chips
    DOM.quickChips.forEach((chip) => {
      chip.addEventListener('click', () => {
        const prompt = chip.dataset.prompt;
        if (prompt) {
          executeStreamingVoiceQuery(prompt, null, false);
        }
      });
    });

    // Diagnostics Drawer Toggle
    DOM.debugToggleBtn.addEventListener('click', () => {
      DOM.diagnosticsDrawer.classList.toggle('open');
      DOM.drawerBackdrop.classList.toggle('open');
    });

    DOM.closeDrawerBtn.addEventListener('click', () => {
      DOM.diagnosticsDrawer.classList.remove('open');
      DOM.drawerBackdrop.classList.remove('open');
    });

    DOM.drawerBackdrop.addEventListener('click', () => {
      DOM.diagnosticsDrawer.classList.remove('open');
      DOM.drawerBackdrop.classList.remove('open');
    });

    // New Chat Button
    DOM.newChatBtn.addEventListener('click', () => {
      triggerBargeIn();
      DOM.chatFeed.innerHTML = '';
      if (DOM.welcomeHero) {
        DOM.welcomeHero.style.display = 'block';
        DOM.chatFeed.appendChild(DOM.welcomeHero);
      }
      setVoiceState('ready');
    });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Application Entry Point
  function init() {
    setupEventListeners();
    initAudioContext();
    drawVisualizer();
    setVoiceState('ready');
    console.log('🌴 ARROHA Real-Time Streaming Voice Assistant initialized (HH Goa 2026).');
  }

  window.addEventListener('DOMContentLoaded', init);
})();
