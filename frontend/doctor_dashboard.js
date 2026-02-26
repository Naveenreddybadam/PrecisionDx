const BACKEND_URL = "http://127.0.0.1:5000";

/* =========================
   LOAD PATIENTS (OP LIST)
========================= */
function loadPatients() {

    fetch(`${BACKEND_URL}/patients`)
    .then(res => res.json())
    .then(data => {

        const select = document.getElementById("patientSelect");
        select.innerHTML = "<option value=''>Select OP Number</option>";

        if (!data.length) {
            select.innerHTML += "<option disabled>No Patients Found</option>";
            return;
        }

        data.forEach(p => {
            select.innerHTML += `
                <option value="${p.id}">
                    OP-${p.id} | ${p.patient_name}
                </option>`;
        });
    })
    .catch(() => alert("Cannot load patients"));
}


/* =========================
   DETECT TUMOR
========================= */
function detectTumor() {

    const patientId = document.getElementById("patientSelect").value;
    const file = document.getElementById("imageInput").files[0];

    if (!patientId) {
        alert("Select Patient (OP Number)");
        return;
    }

    if (!file) {
        alert("Upload MRI image");
        return;
    }

    const formData = new FormData();
    formData.append("image", file);
    formData.append("patient_id", patientId);

    document.getElementById("result").innerHTML = "Processing...";

    fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {

        if (!data.tumor_detected) {
            document.getElementById("result").innerHTML = "No Tumor Detected";
            return;
        }

        let resultHTML = `
            <b>Tumor:</b> ${data.tumor_type}<br>
            <b>Grade:</b> ${data.grade}<br>
            <b>Confidence:</b> ${data.confidence.toFixed(2)}<br>
            <b>Risk:</b> ${data.risk_level}<br>
<b>Trend:</b> ${
    data.trend === "WORSENING" ? "🔺 Worsening" :
    data.trend === "IMPROVING" ? "🔻 Improving" :
    data.trend === "STABLE" ? "➖ Stable" :
    "-"
}
        `;

        if (data.previous_confidence !== null) {
    resultHTML += `
        <hr>
        <b>Previous Confidence:</b> ${data.previous_confidence.toFixed(2)}<br>
        <b>Previous Risk:</b> ${data.previous_risk}<br>
        <b>Trend:</b> ${data.trend}
    `;
}

        document.getElementById("result").innerHTML = resultHTML;

        const img = document.getElementById("previewImage");
        img.src = BACKEND_URL + data.gradcam_image;
        img.style.display = "block";

        loadHistory();
    })
    .catch(() => alert("Detection Failed"));
}
function updateSummary(data) {

    let high = 0, medium = 0, low = 0;

    data.forEach(r => {
        if (r.risk_level === "HIGH") high++;
        else if (r.risk_level === "MEDIUM") medium++;
        else low++;
    });

    const box = document.getElementById("summaryBox");

    box.innerHTML = `
        <div class="badge badge-high">HIGH: ${high}</div>
        <div class="badge badge-medium">MEDIUM: ${medium}</div>
        <div class="badge badge-low">LOW: ${low}</div>
    `;
}


function loadHistory() {

    const filter = document.getElementById("filterSelect")?.value || "all";
    fetch(`${BACKEND_URL}/patient_trend_all?filter=${filter}`)
    .then(res => res.json())
    .then(data => {

    updateSummary(data);

    const table = document.getElementById("historyTable");
        table.innerHTML = "";

        if (!data.length) {
            table.innerHTML =
                "<tr><td colspan='10'>No Reports Found</td></tr>";
            return;
        }

        data.forEach(r => {

            let rowClass = "";
            if (r.risk_level === "HIGH") rowClass = "class='high'";
            else if (r.risk_level === "MEDIUM") rowClass = "class='medium'";

            table.innerHTML += `
            <tr ${rowClass}>
                <td>${r.report_id}</td>
                <td>${r.patient_name}</td>
                <td>${r.age !== null && r.age !== undefined ? r.age : "-"}</td>
                <td>${r.tumor_type}</td>
                <td>${r.grade}</td>
                <td>${r.confidence ? r.confidence.toFixed(2) : "-"}</td>
                <td>
    <span class="badge 
        ${r.risk_level === "HIGH" ? "badge-high" :
          r.risk_level === "MEDIUM" ? "badge-medium" :
          "badge-low"}">
        ${r.risk_level}
    </span>
</td>
                <td>${r.created_at}</td>
                <td>
                    ${r.gradcam_path ? 
                    `<img src="${BACKEND_URL}${r.gradcam_path}" width="70">`
                    : "-"}
                </td>
                <td>
                    <button onclick="downloadReport(${r.report_id})">
                        PDF
                    </button>
                </td>
            </tr>`;
        });

    })
    .catch(err => {
        console.error(err);
        alert("Cannot load history");
    });
}

/* =========================
   DOWNLOAD PDF
========================= */
function downloadReport(reportId) {
    window.open(`${BACKEND_URL}/generate_report/${reportId}`, "_blank");
}


/* =========================
   SEARCH
========================= */
function filterTable() {

    const input = document.getElementById("searchInput").value.toLowerCase();
    const rows = document.querySelectorAll("#historyTable tr");

    rows.forEach(row => {
        const name = row.cells[1].innerText.toLowerCase();
        row.style.display = name.includes(input) ? "" : "none";
    });
}
function applyFilter() {
    loadHistory();
}
function loadMetrics() {
    fetch("http://127.0.0.1:5000/model_metrics")
        .then(res => res.json())
        .then(data => {

            if (data.error) {
                document.getElementById("metricsBox").innerHTML = "Metrics not available";
                return;
            }

            const accuracy = (data.accuracy * 100).toFixed(2);
            const precision = (data["weighted avg"].precision * 100).toFixed(2);
            const recall = (data["weighted avg"].recall * 100).toFixed(2);
            const f1 = (data["weighted avg"]["f1-score"] * 100).toFixed(2);

            document.getElementById("metricsBox").innerHTML = `
                <b>Accuracy:</b> ${accuracy}% <br>
                <b>Precision:</b> ${precision}% <br>
                <b>Recall:</b> ${recall}% <br>
                <b>F1 Score:</b> ${f1}%
            `;
        });
}

/* =========================
   INIT
========================= */
window.onload = function() {
    loadPatients();
    loadHistory();
    loadMetrics();
};