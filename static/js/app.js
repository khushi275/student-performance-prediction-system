// StudentAI — app.js
// Handles: simulation (What-If analysis) + PDF generation

const CLR = {
  blue:   '#00d4ff',
  purple: '#7b2fff',
  green:  '#00ff88',
  red:    '#ff4136',
  yellow: '#ffc107'
};

let simChartInst = null;

// ─────────────────────────────────────────────────────────────────────────────
// SIMULATION  (What-If Analysis)
//
// How it works:
//   1. Takes the student's original form data (stored in window.baseForm)
//   2. Picks one parameter the user wants to test (e.g. "What if attendance changes?")
//   3. Sends several different values of that parameter to the server
//   4. Server returns predicted grade + pass probability for each value
//   5. Results are drawn as a line chart so you can see the trend
// ─────────────────────────────────────────────────────────────────────────────
async function runSimulation(btnEl) {
  const btn         = (btnEl instanceof HTMLElement) ? btnEl : document.getElementById('runSimBtn');
  const paramSelect = document.getElementById('simParam');
  const simDesc     = document.getElementById('simDesc');
  const canvas      = document.getElementById('simChart');
  if (!paramSelect || !canvas) return;

  const param = paramSelect.value;

  // Show what this simulation is testing
  const descriptions = {
    attendance_percentage:       'Testing how attendance % (30→95%) affects predicted grade',
    study_hours_per_day:         'Testing how daily study hours (1→8 hrs) affect predicted grade',
    social_media_hours:          'Testing how social media usage (0→6 hrs/day) affects grade',
    sleep_hours:                 'Testing how sleep duration (5→9 hrs/night) affects grade',
    concept_understanding_score: 'Testing how concept understanding score (20→100) affects grade'
  };
  if (simDesc) simDesc.textContent = descriptions[param] || '';

  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Running...'; }

  try {
    // Test values to try for each parameter
    const testValues = {
      attendance_percentage:       [30, 50, 70, 85, 95],
      study_hours_per_day:         [1, 2, 4, 6, 8],
      social_media_hours:          [0, 2, 4, 6],
      sleep_hours:                 [5, 6, 7, 8, 9],
      concept_understanding_score: [20, 40, 60, 80, 100]
    };
    const values = testValues[param] || [30, 50, 70, 90];

    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ param, base_form: window.baseForm || {}, values })
    });

    if (!res.ok) throw new Error('Server error ' + res.status);
    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.message || 'Simulation failed');

    const sims = data.simulations;
    const labels = values.map(v =>
      param.includes('score') ? v + ' pts' :
      param.includes('hours') ? v + ' hrs' : v + '%'
    );

    if (simChartInst) { simChartInst.destroy(); simChartInst = null; }

    simChartInst = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Pass Probability (%)',
          data: sims.map(s => s.pass_probability),
          borderColor: CLR.blue,
          backgroundColor: 'rgba(0,212,255,0.08)',
          borderWidth: 2.5,
          tension: 0.4,
          fill: true,
          pointRadius: 7,
          pointHoverRadius: 9,
          pointBackgroundColor: sims.map(s =>
            s.performance === 'High' ? CLR.green :
            s.performance === 'Medium' ? CLR.yellow : CLR.red
          ),
          pointBorderColor: '#0d0d1a',
          pointBorderWidth: 2
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: '#aaa', font: { size: 12 } } },
          tooltip: {
            callbacks: {
              afterLabel: ctx => {
                const s = sims[ctx.dataIndex];
                return `  Grade: ${s.predicted_grade}   Performance: ${s.performance}`;
              }
            }
          }
        },
        scales: {
          y: {
            min: 0, max: 100,
            ticks: { color: '#888', callback: v => v + '%' },
            grid:  { color: 'rgba(255,255,255,0.05)' }
          },
          x: { ticks: { color: '#aaa' }, grid: { display: false } }
        }
      }
    });

    if (btn) {
      btn.innerHTML = '<i class="fas fa-check me-1"></i>Done';
      setTimeout(() => { btn.innerHTML = '<i class="fas fa-play me-1"></i>Run'; btn.disabled = false; }, 1500);
    }

  } catch (err) {
    console.error('Simulation error:', err);
    const errEl = document.getElementById('simError');
    if (errEl) {
      errEl.textContent = 'Error: ' + err.message;
      errEl.style.display = 'block';
      setTimeout(() => { errEl.style.display = 'none'; }, 6000);
    }
    if (btn) { btn.innerHTML = '<i class="fas fa-play me-1"></i>Run'; btn.disabled = false; }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// PDF GENERATION
// Captures the full result page as an image then saves as PDF.
// Using html2canvas so charts, colours and layout all render exactly as seen.
// ─────────────────────────────────────────────────────────────────────────────
async function downloadPDF(btnEl) {
  const btn = (btnEl instanceof HTMLElement) ? btnEl : document.querySelector('.btn-pdf');
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Generating PDF...'; }

  try {
    // 1. Capture the visible page
    const body = document.body;
    const canvas = await html2canvas(body, {
      scale: 1.5,           // balance quality vs file size
      useCORS: true,
      backgroundColor: '#0d0d1a',
      logging: false,
      windowWidth: 1200,    // consistent desktop-width capture
      scrollX: 0,
      scrollY: 0,
      height: body.scrollHeight,
      width: body.scrollWidth
    });

    // 2. Build PDF — A4 portrait, fit full width, paginate height
    const { jsPDF } = window.jspdf;
    const doc  = new jsPDF('p', 'mm', 'a4');
    const pageW  = doc.internal.pageSize.getWidth();   // 210 mm
    const pageH  = doc.internal.pageSize.getHeight();  // 297 mm

    const imgData   = canvas.toDataURL('image/jpeg', 0.92);
    const imgW      = pageW;                           // fill page width
    const imgH      = (canvas.height * imgW) / canvas.width;  // keep aspect ratio
    const totalPages = Math.ceil(imgH / pageH);

    for (let page = 0; page < totalPages; page++) {
      if (page > 0) doc.addPage();
      // offset image upward for each page
      doc.addImage(imgData, 'JPEG', 0, -(page * pageH), imgW, imgH);
    }

    // 3. Save
    const name = (document.querySelector('.result-student-name')?.textContent?.trim() || 'Student')
                   .replace(/[^a-z0-9]/gi, '_');
    doc.save(`StudentAI_${name}.pdf`);

    if (btn) {
      btn.innerHTML = '<i class="fas fa-check me-2"></i>Downloaded!';
      setTimeout(() => { btn.disabled = false; btn.innerHTML = '<i class="fas fa-file-pdf me-2"></i>Download PDF'; }, 2000);
    }

  } catch (err) {
    console.error('PDF error:', err);
    alert('PDF generation failed: ' + err.message);
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-file-pdf me-2"></i>Download PDF'; }
  }
}

// Auto-run simulation on result page load
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('simChart')) {
    setTimeout(() => runSimulation(document.getElementById('runSimBtn')), 600);
  }
});

window.runSimulation = runSimulation;
window.downloadPDF   = downloadPDF;
